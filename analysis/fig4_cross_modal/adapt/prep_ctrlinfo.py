"""Precompute (once) the information-matched control-cell objects used by mm/mm_transfer.py and mm/mm_translate.py:
B = RidgeCV(per-protein alpha, logspace(0,8,17)) raw log-RNA -> raw CLR protein on the 90% control-cell split, C = control-cell
gene-gene correlation on the same cells. Identical code to mm_transfer._ctrl_info (kept in sync)."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, pickle, numpy as np
from sklearn.linear_model import RidgeCV
REPO = f"{_CMC_ROOT}"; ADOUT = f"{_CMC_SCRATCH}/nov_adapt/diag"
for t in ["frangieh", "iccite", "arce"]:
    f = f"{ADOUT}/ctrlinfo_{t}.npz"
    if os.path.exists(f): print("exists", f); continue
    pb = pickle.load(open(f"{_CMC_RESULTS}/paper/analyses/stats/de/pb_{t}.pkl", "rb"))
    Xc = pb["ct"]["cells_r"].astype(np.float64); Yc = pb["ct"]["cells_p"].astype(np.float64)
    prm = np.random.RandomState(12345).permutation(len(Xc)); tr = np.sort(prm[int(round(0.1 * len(Xc))):])
    Xc, Yc = Xc[tr], Yc[tr]
    B = RidgeCV(alphas=np.logspace(0, 8, 17), alpha_per_target=True).fit(Xc - Xc.mean(0), Yc - Yc.mean(0)).coef_.T
    sd = Xc.std(0); Z = (Xc - Xc.mean(0)) / np.where(sd > 0, sd, 1); C = (Z.T @ Z) / len(Z)
    np.savez(f + ".tmp.npz", B=B, C=C); os.replace(f + ".tmp.npz", f); print("wrote", f, B.shape, C.shape, flush=True)
