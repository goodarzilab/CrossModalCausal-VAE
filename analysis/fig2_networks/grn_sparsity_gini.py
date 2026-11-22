"""Sparsity + Gini concentration of the inferred GRN (G_rr) for every baseline, on the common 2692-gene
vocab (off-diagonal |edge weights|). Gini: 0=uniform (dense, all edges equal), 1=maximally concentrated
(few dominant edges). Sparsity: fraction of |w| below 1e-6 x max (effective zeros). SIGFPE-safe.
Writes results/baselines/grn_sparsity_gini.json."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, numpy as np, pandas as pd
import ctypes as _ct
for _l in ("libm.so.6", None):
    try: _ct.CDLL(_l).fedisableexcept(0x3f); break
    except Exception: pass
R = f"{_CMC_ROOT}"

def gini(x):
    x = np.sort(np.abs(np.asarray(x, float)).ravel())
    n = x.size
    if n == 0 or x.sum() == 0: return float("nan")
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1.0) / n)

def stats(M):
    M = np.asarray(M, float).copy(); np.fill_diagonal(M, 0.0)
    a = np.abs(M[~np.eye(M.shape[0], dtype=bool)])
    mx = a.max() if a.size else 0.0
    spars = float((a < 1e-6 * mx).mean()) if mx > 0 else 1.0          # legacy: frac < 1e-6*max
    frac_lt_1e3 = float((a < 1e-3 * mx).mean()) if mx > 0 else 1.0     # cleaner sparsity: frac < 0.1%*max
    top1 = float(np.sort(a)[::-1][:max(1, int(0.01 * a.size))].sum() / a.sum()) if a.sum() > 0 else float("nan")
    return {"gini": round(gini(a), 4), "sparsity": round(spars, 4), "frac_lt_1e3": round(frac_lt_1e3, 4),
            "mass_top1pct": round(top1, 4), "n_genes": int(M.shape[0])}

out = {}

# --- CMC-VAE projected G_rr (top model = sp_dec_gl1000, decoder group-lasso sparse) ---
try:
    Grr = pd.read_csv(f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000/GRN_inferred.csv", index_col=0).values
    out["CMC-VAE"] = stats(Grr)
except Exception as e:
    out["CMC-VAE"] = {"error": str(e)}

# --- GRNBoost2 (prefer 809k, else largest available) ---
import glob
g3 = None
for tag in ["809k", "200k", "100k"]:
    p = f"{_CMC_RESULTS}/baselines/genie3_matrix_grnboost2_all_{tag}.npz"
    if os.path.exists(p): g3 = (p, tag); break
if g3:
    d = np.load(g3[0], allow_pickle=True); out["GRNBoost2"] = stats(d["matrix"]); out["GRNBoost2"]["src"] = g3[1]

# --- DeepSEM W (809k stable) ---
d = np.load(f"{_CMC_RESULTS}/baselines/deepsem_W_809k_clamp09.npz", allow_pickle=True)
out["DeepSEM"] = stats(d["W"])

# --- totalVI G_rr_tv (809k) ---
d = np.load(f"{_CMC_RESULTS}/baselines/totalvi_raw.npz", allow_pickle=True)
out["totalVI"] = stats(d["G_rr_tv"])

# --- correlation |Pearson| on the 100k subset (size-invariant; fast) ---
try:
    import scanpy as sc
    a = sc.read_h5ad(f"{_CMC_DATA}/combined_v4_100k/rna_combined.h5ad")
    X = (a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)).astype(np.float32)
    Xc = X - X.mean(0, keepdims=True); del X
    sd = np.sqrt((Xc * Xc).sum(0)); sd[sd < 1e-9] = 1e-9
    Xs = Xc / sd; del Xc
    Cm = np.nan_to_num((Xs.T @ Xs)); del Xs
    out["correlation"] = stats(Cm); out["correlation"]["src"] = "100k (size-invariant)"
except Exception as e:
    out["correlation"] = {"error": str(e)}

json.dump(out, open(f"{_CMC_RESULTS}/baselines/grn_sparsity_gini.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print("\n%-12s %8s %10s %12s" % ("model", "Gini", "sparsity", "mass_top1%"))
for m in ["CMC-VAE", "GRNBoost2", "DeepSEM", "totalVI", "correlation"]:
    s = out.get(m, {})
    if "gini" in s: print("%-12s %8.3f %10.3f %12.3f" % (m, s["gini"], s["sparsity"], s["mass_top1pct"]))
