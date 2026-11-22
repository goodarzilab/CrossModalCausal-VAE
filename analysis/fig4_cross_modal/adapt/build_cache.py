"""Build the harness eps caches (mm_core.load_dataset format) for any model dir WITHOUT loading the 809k training corpus:
align + raw DE are model-independent and are copied from the existing sp_dec_gl1000 s42 cache; eps = encoder posterior means
of the same 4,000 control cells as P.encode_control_cells (MAX_CONTROL_CELLS=4000, RandomState(42) subsample of
align['control_mask'] over the obs-sorted, RNA/protein-aligned perturbation cells), encoded with THIS model.
usage: python -I build_cache.py RD [RD ...]      (env AD_CACHE = cache dir; AD_CHECK=1 compares with the original cache)"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, glob, pickle, hashlib
import numpy as np, torch, scanpy as sc
REPO = f"{_CMC_ROOT}"
sys.path.insert(0, os.path.join(_CMC_ROOT, "analysis", "common")); sys.path.insert(0, os.path.join(REPO, "src"))
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE
CACHE = os.environ.get("AD_CACHE", f"{_CMC_SCRATCH}/nov_adapt/cache"); os.makedirs(CACHE, exist_ok=True)
DATA = f"{_CMC_DATA}"
DATASETS = {"frangieh": (f"{DATA}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad", f"{DATA}/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad"),
            "iccite": (f"{DATA}/chen_perturb_icciteseq/export/iccite_rna.h5ad", f"{DATA}/chen_perturb_icciteseq/export/iccite_prot.h5ad"),
            "arce": (f"{DATA}/arce_perturb_citeseq/arce_rna.h5ad", f"{DATA}/arce_perturb_citeseq/arce_prot.h5ad")}
X_CACHE = {}
def ctrl_inputs(tag, al):
    if tag in X_CACHE: return X_CACHE[tag]
    pr = sc.read_h5ad(DATASETS[tag][0], backed="r"); pp = sc.read_h5ad(DATASETS[tag][1], backed="r")
    common_obs = sorted(set(pr.obs_names) & set(pp.obs_names)); cm = np.asarray(al["control_mask"].values, bool)
    assert len(cm) == len(common_obs)
    ctrl_idx = np.where(cm)[0]
    if len(ctrl_idx) > 4000: ctrl_idx = np.random.RandomState(42).choice(ctrl_idx, 4000, replace=False)
    names = [common_obs[i] for i in ctrl_idx]
    ri = pr.obs_names.get_indexer(names); pi = pp.obs_names.get_indexer(names)
    o1 = np.argsort(ri); Xr = pr.X[ri[o1]]; Xr = np.asarray(Xr.toarray() if hasattr(Xr, "toarray") else Xr)[np.argsort(o1)]
    o2 = np.argsort(pi); Xp = pp.X[pi[o2]]; Xp = np.asarray(Xp.toarray() if hasattr(Xp, "toarray") else Xp)[np.argsort(o2)]
    pr.file.close(); pp.file.close()
    NG, NP = len(al["train_gene_names"]), len(al["train_prot_names"])
    xr = np.zeros((len(names), NG), np.float32); xp = np.zeros((len(names), NP), np.float32)
    for p_idx, t_idx in zip(al["rna_eval_perturb_indices"], al["rna_eval_train_indices"]):
        xr[:, t_idx] = (Xr[:, p_idx] - al["train_rna_mean"][t_idx]) / al["train_rna_std"][t_idx]
    for p_idx, t_idx in zip(al["prot_eval_perturb_indices"], al["prot_eval_train_indices"]):
        xp[:, t_idx] = (Xp[:, p_idx] - al["train_prot_mean"][t_idx]) / al["train_prot_std"][t_idx]
    X_CACHE[tag] = (xr, xp); return xr, xp

for rd in sys.argv[1:]:
    rd = os.path.abspath(rd); cfg = json.load(open(f"{rd}/config.json")); ck = P.resolve_ckpt(rd)
    sd0 = torch.load(ck, map_location="cpu"); ng, np_ = sd0["dec_rna.weight"].shape[0], sd0["dec_prot.weight"].shape[0]; del sd0
    m = P.load_ctx_model(CrossModalCausalLDVAE, rd, ng, np_, cfg); m.eval()
    h = hashlib.md5(open(ck, "rb").read()).hexdigest()[:12]
    for tag in ["frangieh", "iccite", "arce"]:
        out = f"{CACHE}/rc_{tag}_{os.path.basename(rd)}_{h}_0.1_0.05.pkl"
        if os.path.exists(out): print("exists", out, flush=True); continue
        ref = pickle.load(open(glob.glob(f"{_CMC_RESULTS}/_ftcache/rc_{tag}_sp_dec_gl1000_3a5d257c9323_0.1_0.05.pkl")[0], "rb"))
        al = ref["align"]; xr, xp = ctrl_inputs(tag, al)
        with torch.no_grad():
            er, ep = [], []
            for s in range(0, len(xr), 256):
                a, _ = m.enc_rna(torch.as_tensor(xr[s:s + 256], device=P.DEVICE)); b, _ = m.enc_prot(torch.as_tensor(xp[s:s + 256], device=P.DEVICE))
                er.append(a.cpu()); ep.append(b.cpu())
        eps = torch.cat([torch.cat(er), torch.cat(ep)], 1).numpy()[:4000]
        if os.environ.get("AD_CHECK") == "1":
            print(tag, "max |eps - original cache eps| =", float(np.abs(eps - ref["eps"]).max()), "shape", eps.shape, ref["eps"].shape, flush=True)
        pickle.dump({"align": al, "de": ref["de"], "eps": eps}, open(out + ".tmp", "wb")); os.replace(out + ".tmp", out)
        print("wrote", out, flush=True)
