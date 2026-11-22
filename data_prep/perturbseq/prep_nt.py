"""Control-only (non-targeting) per-feature statistics for the control-only pretraining arm and the canonical scale.

Reads results/replogle/data (read-only); writes paper_nov_revisions/analyses/perturbseq/data/{line}_ntstats.npz with
  mean, std        per-feature mean / sd over ALL non-targeting cells of the line (log1p-normalised units)
  n_nt             number of NT cells
These are (i) the z-scoring statistics of the control-only CMC-VAE / DeepSEM models and (ii) the canonical scale
(control-cell sd) on which all predictors are compared (rr_*_canon metrics in ft_eval.py).
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
import os, numpy as np, pandas as pd
D = f"{_CMC_RESULTS}/replogle/data"
OUT = f"{_CMC_RESULTS}/paper/analyses/perturbseq/data"; os.makedirs(OUT, exist_ok=True)
for line in ["jurkat", "k562"]:
    obs = pd.read_csv(f"{D}/{line}_obs.csv"); nt = np.where(obs.is_nt.values)[0]
    X = np.load(f"{D}/{line}_X.npy", mmap_mode="r")[nt].astype(np.float64)
    np.savez(f"{OUT}/{line}_ntstats.npz", mean=X.mean(0), std=X.std(0), n_nt=len(nt))
    print(line, "NT cells", len(nt), "median sd", np.median(X.std(0)), "n sd<1e-3:", int((X.std(0) < 1e-3).sum()))
