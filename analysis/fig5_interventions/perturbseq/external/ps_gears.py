"""GEARS (Roohani et al., Nat Biotechnol 2024; pip cell-gears 0.1.2) on the Replogle-Nadig Perturb-seq lines, using the
SAME features (3,000 = targets + HVGs), the SAME log-normalised expression, and the SAME by-gene 70/30 split as
analysis/fig5_interventions/perturbseq/ft_eval.py. Only perturbations whose target is in the GEARS perturbation graph (GO graph over GEARS'
default essential/all-data gene list) can be trained or predicted; coverage is reported.

Cells: all non-targeting cells (controls) + up to CAP cells per covered perturbation (random, seed 0).
Split: custom GEARS split -- train = covered TRAIN-split genes minus a random 10% (val, for GEARS' best-epoch
selection), test = covered TEST-split genes. Default GEARS hyper-parameters (hidden 64, 1 GO + 1 co-expression GNN
layer, direction_lambda 0.1, lr 1e-3, wd 5e-4, batch 32), EPOCHS epochs.
Prediction: for each covered gene (train + test), the trained best model is applied to the SAME fixed set of NPRED
control cells; predicted response = mean(pred) - mean(input control cells) (paired, so control sampling noise cancels),
in log-normalised units. Output npz (names, <name>|genes, <name>|lfc) is scored by ft_eval_bench.py part 'ext'.

usage: python ps_gears.py LINE SPLITSEED [EPOCHS]
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, time, pickle
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, torch
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
LINE, SPLIT = sys.argv[1], int(sys.argv[2]); EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 15
CAP, NPRED = int(os.environ.get("GEARS_CAP", "64")), 1000
REPO = f"{_CMC_ROOT}"; D = f"{_CMC_RESULTS}/replogle/data"
B = f"{_CMC_RESULTS}/paper/analyses/benchmark"; SCR = f"{_CMC_SCRATCH}/nov_benchmark_scratch"
sys.path.insert(0, f"{SCR}/pylib")
from gears import PertData, GEARS
from torch_geometric.data import Data, DataLoader
np.random.seed(0); torch.manual_seed(0)

FEATS = json.load(open(f"{D}/features.json"))["features"]; G = len(FEATS)
SPL = json.load(open(f"{B}/splits/ps_split_s{SPLIT}.json")); TRS, TES = set(SPL["train_genes"]), set(SPL["test_genes"])
de = np.load(f"{D}/{LINE}_de.npz"); FI = {g: i for i, g in enumerate(FEATS)}
# evaluable KOs: identical criterion to ft_eval.build_ctx (>=3 DE features excluding the target, MW p<0.05 & |lfc|>=0.10)
EV = []
for g, lfc, pv in zip(de["genes"].astype(str), de["lfc"].astype(float), de["pvals"].astype(float)):
    keep = np.array(FEATS) != g
    if (((pv < 0.05) & (np.abs(lfc) >= 0.10)) & keep).sum() >= 3: EV.append(g)
TRAIN = [g for g in EV if g in TRS]; TEST = [g for g in EV if g in TES]

TAGG = os.environ.get("GEARS_TAG", ""); WD = f"{SCR}/gears/{LINE}_split{SPLIT}{TAGG}"; os.makedirs(WD, exist_ok=True)
pdat = PertData(WD)                                     # downloads gene2go_all.pkl
pdat.adata = None; pdat.default_pert_graph = True
# coverage: genes in GEARS' default perturbation graph (gene2go restricted to essential_all_data_pert_genes)
from gears.utils import dataverse_download
path_ = os.path.join(WD, "essential_all_data_pert_genes.pkl")
dataverse_download("https://dataverse.harvard.edu/api/access/datafile/6934320", path_)
ess = pickle.load(open(path_, "rb")); PERTS = set(g for g in ess if g in pdat.gene2go)
cov_tr = [g for g in TRAIN if g in PERTS]; cov_te = [g for g in TEST if g in PERTS]
COV = dict(n_train=len(TRAIN), n_test=len(TEST), n_train_cov=len(cov_tr), n_test_cov=len(cov_te),
           frac_test_cov=len(cov_te) / len(TEST), test_not_covered=sorted(set(TEST) - set(cov_te)))
log("coverage", {k: v for k, v in COV.items() if k != "test_not_covered"})

obs = pd.read_csv(f"{D}/{LINE}_obs.csv"); rs = np.random.RandomState(0)
nt = np.where(obs.is_nt.values)[0]; rows = [nt]; cond = ["ctrl"] * len(nt)
for g in cov_tr + cov_te:
    idx = np.where((obs.gene == g).values)[0]
    if len(idx) > CAP: idx = np.sort(rs.choice(idx, CAP, replace=False))
    rows.append(idx); cond += [f"{g}+ctrl"] * len(idx)
rows = np.concatenate(rows); order = np.argsort(rows); rows = rows[order]; cond = np.array(cond)[order]
X = np.load(f"{D}/{LINE}_X.npy", mmap_mode="r")[rows].astype(np.float32)
A = ad.AnnData(sp.csr_matrix(X), obs=pd.DataFrame(dict(condition=cond, cell_type=LINE), index=[f"c{i}" for i in rows]),
               var=pd.DataFrame(dict(gene_name=FEATS), index=FEATS))
log("adata", A.shape, "conditions", A.obs.condition.nunique())
pdat.new_data_process(dataset_name=f"{LINE}_s{SPLIT}", adata=A)
rv = np.random.RandomState(1); trc = list(cov_tr); rv.shuffle(trc); nval = max(1, round(0.1 * len(trc)))
split = dict(train=["ctrl"] + [f"{g}+ctrl" for g in sorted(trc[nval:])], val=[f"{g}+ctrl" for g in sorted(trc[:nval])],
             test=[f"{g}+ctrl" for g in cov_te])
sp_path = f"{WD}/split.pkl"; pickle.dump(split, open(sp_path, "wb"))
pdat.prepare_split(split="custom", split_dict_path=sp_path); pdat.get_dataloader(batch_size=32, test_batch_size=128)
dev = "cuda" if torch.cuda.is_available() else "cpu"
gm = GEARS(pdat, device=dev); gm.model_initialize(hidden_size=64)
log("training GEARS on", dev, "epochs", EPOCHS)
gm.train(epochs=EPOCHS, lr=1e-3)
gm.save_model(f"{WD}/model")

# ---- predictions on a fixed control-cell set (paired difference)
ctrl = gm.ctrl_adata; ci = np.random.RandomState(2).choice(ctrl.n_obs, min(NPRED, ctrl.n_obs), replace=False)
Xc = ctrl.X[ci].toarray().astype(np.float32); base = Xc.mean(0)
pl = list(gm.pert_list); bm = gm.best_model.to(dev).eval()
out_genes, out = [], []
for g in cov_tr + cov_te:
    pi = [pl.index(g)]; preds = []
    cg = [Data(x=torch.Tensor(x[None]).T, pert_idx=pi, pert=[g]) for x in Xc]
    for b in DataLoader(cg, 250, shuffle=False):
        b = b.to(dev)
        with torch.no_grad(): preds.append(bm(b).detach().cpu().numpy())
    out_genes.append(g); out.append(np.concatenate(preds).mean(0) - base)
L_ = np.stack(out).astype(np.float32)
o = f"{B}/ext/gears{TAGG}_{LINE}_split{SPLIT}.npz"; os.makedirs(os.path.dirname(o), exist_ok=True)
np.savez_compressed(o, names=np.array(["gears"]), **{"gears|genes": np.array(out_genes), "gears|lfc": L_})  # same name "gears" regardless of device tag
json.dump(dict(coverage=COV, epochs=EPOCHS, cap=CAP, n_cells=int(A.n_obs), n_val=nval, device=dev,
               n_pred_ctrl=int(len(ci)), elapsed_s=time.time() - T0), open(o.replace(".npz", ".json"), "w"), indent=1)
log("DONE", o, L_.shape)
