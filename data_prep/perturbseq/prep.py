"""Replogle 2022 (CRISPRi Perturb-seq) preprocessing for the RNA-only CMC-VAE fine-tuning test.

.X in replogle.h5ad is already log1p(counts / UMI_count * 1e4) (verified: smallest nonzero value per cell
== log1p(1e4/UMI_count)), i.e. exactly our pipeline's library-size-normalise-to-1e4 + log1p, so it is used as is.

Outputs (results/replogle/data/):
  split.json               global by-gene 70/30 split (seed 42) over the UNION of jurkat+k562 targets
  features.json            shared feature set: every jurkat/k562 target present in var + top HVGs -> N_FEAT
  {line}_X.npy             float32 [cells, N_FEAT] log-normalised expression (all cells of the line)
  {line}_obs.csv           gene, gem_group, is_nt, is_train (obs-training cell = NT or training-gene KO)
  {line}_stats.npz         per-feature mean / std over the line's observational-training cells
  {line}_de.npz            per KO (target in features, >=30 cells): lfc = mean(KO)-mean(NT) [log-norm],
                           Mann-Whitney p (identical statistic to perturb_validation_analysis.fast_mannwhitney_pvals:
                           average-tie ranks, normal approx, no tie correction) computed vs ALL NT cells
  {line}_corr.npy          feature x feature Pearson correlation over the observational-training cells
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, time
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
from scipy.stats import norm, rankdata

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

SRC = f"{_CMC_DATA}/replogle/replogle.h5ad"
OUT = f"{_CMC_RESULTS}/replogle/data"; os.makedirs(OUT, exist_ok=True)
LINES = ["jurkat", "k562"]; N_FEAT = 3000; SEED = 42; MIN_CELLS = 30

a = ad.read_h5ad(SRC, backed="r"); obs = a.obs.copy(); var = np.array(a.var_names.astype(str))
obs["gene"] = obs["gene"].astype(str); obs["cell_line"] = obs["cell_line"].astype(str)
targets = sorted(set(obs.gene[obs.cell_line.isin(LINES)]) - {"non-targeting"})
perm = np.random.RandomState(SEED).permutation(len(targets)); n_test = round(0.30 * len(targets))
test = sorted(targets[i] for i in perm[:n_test]); train = sorted(targets[i] for i in perm[n_test:])
json.dump(dict(seed=SEED, n_targets=len(targets), train_genes=train, test_genes=test), open(f"{OUT}/split.json", "w"))
log(f"targets union {len(targets)}: train {len(train)} test {len(test)}")
trset = set(train)

rows = np.where(obs.cell_line.isin(LINES).values)[0]
log("reading X ...")
X = np.asarray(a.X[:], dtype=np.float32)[rows]; obs = obs.iloc[rows].reset_index()
a.file.close()
log("X", X.shape)
is_nt = (obs.gene == "non-targeting").values; is_tr = is_nt | obs.gene.isin(trset).values

# ---- shared feature set: targets in var + HVGs (seurat, per-line batch_key) on observational-training cells only
tg_in_var = sorted(set(targets) & set(var))
hv = ad.AnnData(X[is_tr], obs=pd.DataFrame({"cell_line": obs.cell_line.values[is_tr]}), var=pd.DataFrame(index=var))
sc.pp.highly_variable_genes(hv, flavor="seurat", batch_key="cell_line", n_top_genes=N_FEAT)
v = hv.var.copy(); v["is_target"] = v.index.isin(tg_in_var)
v = v.sort_values(["highly_variable_nbatches", "dispersions_norm"], ascending=False)
hvg_extra = [g for g in v.index if g not in set(tg_in_var)][:N_FEAT - len(tg_in_var)]
feats = sorted(set(tg_in_var) | set(hvg_extra)); del hv
fidx = np.array([np.where(var == g)[0][0] for g in feats])
json.dump(dict(features=feats, n_target_features=len(tg_in_var), n_hvg_extra=len(hvg_extra)), open(f"{OUT}/features.json", "w"))
log(f"features {len(feats)} = {len(tg_in_var)} targets + {len(hvg_extra)} HVGs")
X = X[:, fidx]

def mw_p(Xp, ctrl_sorted):
    """== fast_mannwhitney_pvals(Xp, Xc): U1 = sum_i #(ctrl<x_i) + .5 #(ctrl==x_i) (+ within-group term cancels)."""
    n1, n2 = Xp.shape[0], ctrl_sorted.shape[0]; U = np.empty(Xp.shape[1])
    for j in range(Xp.shape[1]):
        c = ctrl_sorted[:, j]; lo = np.searchsorted(c, Xp[:, j], "left"); hi = np.searchsorted(c, Xp[:, j], "right")
        U[j] = lo.sum() + 0.5 * (hi - lo).sum()
    mu = n1 * n2 / 2; sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return 2 * norm.sf(np.abs((U - mu) / (sigma + 1e-10)))

def ref_mw(Xp, Xc):   # verbatim reference implementation (for an equivalence check)
    n1, n2 = Xp.shape[0], Xc.shape[0]; Xa = np.vstack([Xp, Xc]); r = np.apply_along_axis(rankdata, 0, Xa)
    U1 = r[:n1].sum(0) - n1 * (n1 + 1) / 2; mu = n1 * n2 / 2; s = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return 2 * norm.sf(np.abs((U1 - mu) / (s + 1e-10)))

fset = {g: i for i, g in enumerate(feats)}
for line in LINES:
    m = (obs.cell_line == line).values; Xl = X[m]; ol = obs[m].reset_index(drop=True)
    nt = (ol.gene == "non-targeting").values; tr = nt | ol.gene.isin(trset).values
    np.save(f"{OUT}/{line}_X.npy", Xl)
    pd.DataFrame(dict(cell=ol["cell_barcode"].values, gene=ol.gene.values, gem_group=ol.gem_group.astype(str).values,
                      is_nt=nt, is_train=tr, UMI_count=ol.UMI_count.values)).to_csv(f"{OUT}/{line}_obs.csv", index=False)
    mu = Xl[tr].mean(0).astype(np.float64); sd = Xl[tr].std(0).astype(np.float64)
    np.savez(f"{OUT}/{line}_stats.npz", mean=mu, std=sd, n_train_cells=int(tr.sum()), n_nt=int(nt.sum()))
    Xz = (Xl[tr] - mu) / (sd + 1e-8); C = (Xz.T @ Xz) / Xz.shape[0]; np.save(f"{OUT}/{line}_corr.npy", C.astype(np.float32)); del Xz
    Xc = Xl[nt]; Xc_sorted = np.sort(Xc, 0); cm = Xc.mean(0)
    kos = [g for g in sorted(set(ol.gene) - {"non-targeting"}) if g in fset]
    G, L, PV, NC = [], [], [], []
    for k, g in enumerate(kos):
        idx = np.where((ol.gene == g).values)[0]
        if len(idx) < MIN_CELLS: continue
        Xp = Xl[idx]; G.append(g); L.append(Xp.mean(0) - cm); PV.append(mw_p(Xp, Xc_sorted)); NC.append(len(idx))
        if k == 0:
            ref = ref_mw(Xp, Xc); log(f"  MW equivalence check ({g}): max|dp| = {np.abs(ref - PV[-1]).max():.2e}")
    np.savez(f"{OUT}/{line}_de.npz", genes=np.array(G), lfc=np.stack(L).astype(np.float32), pvals=np.stack(PV).astype(np.float32),
             n_cells=np.array(NC), split=np.array(["train" if g in trset else "test" for g in G]))
    log(f"{line}: cells {m.sum()} (NT {nt.sum()}, obs-train {tr.sum()}), KOs with target in features & >=30 cells: {len(G)}")
log("done")
