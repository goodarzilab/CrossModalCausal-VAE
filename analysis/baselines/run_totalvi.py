"""totalVI baseline for CMC-VAE (gap G1 — 'is the SCM beyond a multimodal embedding + correlation?').

Trains totalVI on the combined_v4 (2-cohort) train split, then produces totalVI's analogues of the
three CMC-VAE metrics. NO metric is scored here — this script only emits the raw matrices/vectors that
`score_totalvi.py` funnels through the SAME harness copied from analysis/common/causal_eval_suite.py.

Protocol (mirrors src/train.py / analysis/fig2_networks/crossmodal_imputation.py split, seed 42):
  idx_train (80%) / idx_val (10%) / idx_test (10%)   via nested train_test_split(test_size=0.2, then 0.5)

Outputs (results/baselines/totalvi_raw.npz):
  1. IMPUTATION (held-out, no leakage): totalVI trained on idx_train; protein imputed for idx_test with
     protein MASKED (scArches query, protein set to 0) -> per-protein & per-cell Pearson/Spearman vs
     observed (CLR, the exact CMC-VAE target array) and vs raw counts.
  2. TRANSLATION MAP (G_rp analogue): totalVI has no explicit W_rp -> correlation between denoised RNA and
     denoised protein across a train subsample -> G_rp_tv [ng x npr].  Scored with the identical
     C_translation cognate enrichment in score_totalvi.py.
  3. GRN: totalVI denoised RNA-RNA Pearson correlation -> G_rr_tv [ng x ng] (symmetric).  Scored vs
     CollecTRI/DoRothEA/coexpr/STRING (recovery AUROC) + orientation asymmetry in score_totalvi.py.

Env: $SCVI_PY   (GPU).  Absolute paths only.
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
import numpy as np, scanpy as sc
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _CMC_ROOT
CV4 = f"{_CMC_DATA}/combined_v4"
TVI_INPUT = os.environ.get("TVI_INPUT", f"{CV4}/totalvi_input.h5ad")
OUTDIR = f"{_CMC_RESULTS}/baselines"; os.makedirs(OUTDIR, exist_ok=True)

SEED = 42
BATCH_KEY = "donor"          # matches CMC-VAE config batch_col="donor"
MAX_EPOCHS = int(os.environ.get("TVI_MAX_EPOCHS", "50"))
CORR_SUBSAMPLE = int(os.environ.get("TVI_CORR_SUBSAMPLE", "60000"))  # train cells for denoised-corr matrices
np.random.seed(SEED)

import torch, scvi
print(f"[env] scvi={scvi.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
scvi.settings.seed = SEED

SMOKE = os.environ.get("TVI_SMOKE", "0") == "1"
t0 = time.time()
print(f"[1] load totalVI input h5ad ... (SMOKE={SMOKE})", flush=True)
adata = sc.read_h5ad(TVI_INPUT)
if SMOKE:
    rs = np.random.RandomState(0).choice(adata.n_obs, 4000, replace=False)
    adata = adata[np.sort(rs)].copy()
    MAX_EPOCHS = 2; CORR_SUBSAMPLE = 1500
N = adata.n_obs; ng = adata.n_vars; npr = adata.obsm["protein_expression"].shape[1]
genes = list(adata.var_names.astype(str))
prots = list(adata.obsm["protein_expression"].columns.astype(str))
print(f"    N={N} ng={ng} npr={npr}", flush=True)

# ----- observed protein CLR (the exact CMC-VAE imputation target) -----
apc = sc.read_h5ad(f"{CV4}/citeseq_combined.h5ad")
apc = apc[list(adata.obs_names)]  # align to (possibly subsampled) adata
assert list(apc.obs_names) == list(adata.obs_names), "CLR/counts obs mismatch"
Xp_clr = apc.X.toarray() if hasattr(apc.X, "toarray") else np.asarray(apc.X)  # [N, 144] CLR
Xp_raw = np.asarray(adata.obsm["protein_expression"].values, dtype=np.float32)
del apc

# ----- split (identical to train.py) -----
idx = np.arange(N)
idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=SEED)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=SEED)
print(f"    split: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}", flush=True)

# ================= train totalVI on TRAIN split =================
print("[2] setup + train totalVI on train split ...", flush=True)
adata_train = adata[idx_train].copy()
scvi.model.TOTALVI.setup_anndata(adata_train, layer="counts",
                                 protein_expression_obsm_key="protein_expression", batch_key=BATCH_KEY)
model = scvi.model.TOTALVI(adata_train)
model.train(max_epochs=MAX_EPOCHS, early_stopping=True, batch_size=1024,
            early_stopping_patience=8, check_val_every_n_epoch=1)
hist = model.history
def _last(df, alts):
    for k in alts:
        if df is not None and k in df: return float(df[k].dropna().iloc[-1])
    return None
conv = {
    "epochs_run": int(len(hist.get("elbo_train", hist.get("elbo_validation", [])))) if hist else None,
    "elbo_train_last": _last(hist.get("elbo_train"), ["elbo_train"]) if hist else None,
    "elbo_val_last": _last(hist.get("elbo_validation"), ["elbo_validation"]) if hist else None,
    "recon_rna_val_last": _last(hist.get("reconstruction_loss_validation"), ["reconstruction_loss_validation"]) if hist else None,
    "max_epochs_cap": MAX_EPOCHS,
}
print(f"    convergence: {conv}", flush=True)

# ================= 1. IMPUTATION on held-out test (protein masked) =================
print("[3] impute protein for held-out TEST cells (protein masked, scArches query) ...", flush=True)
adata_q = adata[idx_test].copy()
adata_q.obsm["protein_expression"] = adata_q.obsm["protein_expression"] * 0.0  # mask protein (missing)
scvi.model.TOTALVI.prepare_query_anndata(adata_q, model)
q = scvi.model.TOTALVI.load_query_data(adata_q, model)
imp_mode = "scarches_surgery"
try:
    q.train(max_epochs=max(1, MAX_EPOCHS // 2), early_stopping=True, batch_size=1024,
            early_stopping_patience=5, plan_kwargs={"weight_decay": 0.0})
except Exception as e:
    imp_mode = "scarches_nosurgery"
    print(f"    (surgery train failed: {str(e)[:120]} -> using reference weights, no surgery)", flush=True)
_, prot_imp = q.get_normalized_expression(adata_q, n_samples=25, return_mean=True, return_numpy=True)
prot_imp = np.asarray(prot_imp, dtype=np.float32)  # [n_test, 144] imputed protein (foreground)
print(f"    imputed protein shape {prot_imp.shape} (mode={imp_mode})", flush=True)

def _pear(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9: return np.nan
    return float(np.corrcoef(a, b)[0, 1])
from scipy.stats import spearmanr
Xp_clr_te = Xp_clr[idx_test]; Xp_raw_te = Xp_raw[idx_test]
perprot_pear_clr = np.array([_pear(prot_imp[:, j], Xp_clr_te[:, j]) for j in range(npr)])
perprot_pear_raw = np.array([_pear(prot_imp[:, j], np.log1p(Xp_raw_te[:, j])) for j in range(npr)])
perprot_spear_clr = np.array([spearmanr(prot_imp[:, j], Xp_clr_te[:, j]).correlation for j in range(npr)])
# per-cell Pearson (z-score proteins across cells first, as in benchmarks)
def zc(M):
    return (M - M.mean(0)) / (M.std(0) + 1e-8)
zi, zt = zc(prot_imp), zc(Xp_clr_te)
percell_pear = np.array([_pear(zi[i], zt[i]) for i in range(min(20000, len(idx_test)))])
imp = {
    "per_protein_pearson_clr_mean": float(np.nanmean(perprot_pear_clr)),
    "per_protein_pearson_logcounts_mean": float(np.nanmean(perprot_pear_raw)),
    "per_protein_spearman_clr_mean": float(np.nanmean(perprot_spear_clr)),
    "per_cell_pearson_mean": float(np.nanmean(percell_pear)),
    "n_test_cells": int(len(idx_test)), "n_proteins": int(npr), "impute_mode": imp_mode,
}
print(f"    imputation: {imp}", flush=True)

# ================= denoised outputs on TRAIN subsample (for translation + GRN) =================
print("[4] denoised RNA + protein on train subsample -> corr matrices ...", flush=True)
sub = np.random.RandomState(SEED).choice(len(idx_train), size=min(CORR_SUBSAMPLE, len(idx_train)), replace=False)
adata_sub = adata_train[sub].copy()
rna_den, prot_den = model.get_normalized_expression(adata_sub, n_samples=1, return_numpy=True)
rna_den = np.asarray(rna_den, dtype=np.float32); prot_den = np.asarray(prot_den, dtype=np.float32)
print(f"    denoised rna {rna_den.shape} prot {prot_den.shape}", flush=True)

def zcols(M):
    return (M - M.mean(0)) / (M.std(0) + 1e-8)
Zr = zcols(rna_den); Zp = zcols(prot_den); n = Zr.shape[0]
# translation map: gene x protein correlation
G_rp_tv = (Zr.T @ Zp) / (n - 1)                       # [ng, npr]
# GRN: gene x gene correlation (denoised co-expression)
G_rr_tv = (Zr.T @ Zr) / (n - 1)                       # [ng, ng]
np.fill_diagonal(G_rr_tv, 0.0)
print(f"    G_rp_tv {G_rp_tv.shape} |mean|={np.abs(G_rp_tv).mean():.4f} ; "
      f"G_rr_tv {G_rr_tv.shape} |mean|={np.abs(G_rr_tv).mean():.4f}", flush=True)

# ================= save raw artifacts =================
np.savez_compressed(f"{OUTDIR}/totalvi_raw.npz",
    genes=np.array(genes), prots=np.array(prots),
    G_rp_tv=G_rp_tv.astype(np.float32), G_rr_tv=G_rr_tv.astype(np.float32),
    perprot_pear_clr=perprot_pear_clr, perprot_pear_logcounts=perprot_pear_raw,
    perprot_spear_clr=perprot_spear_clr)
meta = {"convergence": conv, "imputation": imp,
        "n_corr_subsample": int(n), "seed": SEED, "batch_key": BATCH_KEY,
        "scvi_version": scvi.__version__, "runtime_min": round((time.time()-t0)/60, 1)}
json.dump(meta, open(f"{OUTDIR}/totalvi_run_meta.json", "w"), indent=2)
try:
    model.save(f"{OUTDIR}/totalvi_model", overwrite=True)
except Exception as e:
    print("    (model.save skipped:", str(e)[:80], ")", flush=True)
print(f"[DONE] runtime {meta['runtime_min']} min -> {OUTDIR}/totalvi_raw.npz", flush=True)
