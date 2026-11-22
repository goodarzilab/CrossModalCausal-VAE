"""Fig. 4c significance: adapted vs PBMC-model protein-from-RNA imputation in screen control cells.
Per protein, the seed-mean (42/1/2) RNA-only imputation Pearson r of the adapted model (variant i) is compared with the
unadapted model by a two-sided Wilcoxon signed-rank test; Benjamini-Hochberg correction across the three screens.
Inputs: <results>/paper/analyses/adapt/diag/sp_{i,none}_<screen>_<seed>.json (written by adapt/adapt.py).
Output: <results>/paper/figures/adapt/fig4b_imputation_tests.json (read by analysis/figures/fig4.py)."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, os
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

DIAG = f"{_CMC_RESULTS}/paper/analyses/adapt/diag"
OUT = f"{_CMC_RESULTS}/paper/figures/adapt/fig4b_imputation_tests.json"
res = {}
for s in ("frangieh", "iccite", "arce"):
    A = np.nanmean([json.load(open(f"{DIAG}/sp_i_{s}_{k}.json"))["diag"]["rna_only_perprot"] for k in ("s42", "s1", "s2")], 0)
    U = np.nanmean([json.load(open(f"{DIAG}/sp_none_{s}_{k}.json"))["diag"]["rna_only_perprot"] for k in ("s42", "s1", "s2")], 0)
    m = np.isfinite(A) & np.isfinite(U); A, U = A[m], U[m]          # proteins with undefined r (constant) are dropped
    res[s] = dict(n=int(m.sum()), p=float(wilcoxon(A, U, alternative="two-sided").pvalue),
                  median_a=float(np.median(A)), median_u=float(np.median(U)), frac_up=float(np.mean(A > U)))
q = multipletests([res[s]["p"] for s in res], method="fdr_bh")[1]
for s, qq in zip(res, q): res[s]["q"] = float(qq)
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
