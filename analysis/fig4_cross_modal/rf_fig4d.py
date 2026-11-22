#!/usr/bin/env python
"""Item 2 (Fig 4d): CIs for control-cell-adapted RNA->protein translation vs baselines, from the adapt workstream's
per-KO outputs (analyses/adapt/eval/translate/{sp_none,sp_i}.json; 204 KOs with >=3 DE proteins, 3 screens).
Hierarchical bootstrap (10,000 replicates): knockouts resampled with replacement within screen, and the 3 model seeds
resampled with replacement (CMC arms: per-seed per-KO metrics 'CMC-enc:s42/s1/s2'; baselines are seed-independent).
Point estimate = mean over KOs of the seed-averaged per-KO metric. Paired contrasts use the same KO draws.
Two-sided bootstrap P (2*min tail, floor 1/(B+1)) and Wilcoxon on seed-averaged per-KO differences; BH within panel.
usage: python -I rf_fig4d.py OUT_JSON"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import sys, json, numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
E = f"{_CMC_RESULTS}/paper/analyses/adapt/eval/translate"; OUT = sys.argv[1]
TN, TI = json.load(open(f"{E}/sp_none.json")), json.load(open(f"{E}/sp_i.json"))
SEEDS = ["s42", "s1", "s2"]; MET = ["pe_cent", "retrieval"]
def units(js, meth, m): return {(r["cohort"], r["gene"]): r.get(m) for r in js["per_ko"][meth]}
KEYS = sorted(units(TN, "corr", "pe_cent"))
ARMS = {"CMC-VAE, unadapted": ("seed", TN), "CMC-VAE, adapted": ("seed", TI), "Corpus correlation map": ("base", TN, "corr"),
        "Cognate transcript": ("base", TN, "cognate"), "Control-cell ridge (same cells)": ("base", TN, "ctrlridge90")}
def mat(arm, m):   # [n_KO, 3] (baselines repeated)
    a = ARMS[arm]
    if a[0] == "seed":
        cols = [units(a[1], f"CMC-enc:{s}", m) for s in SEEDS]; return np.array([[c.get(k, np.nan) if c.get(k) is not None else np.nan for c in cols] for k in KEYS], float)
    u = units(a[1], a[2], m); v = np.array([u.get(k) if u.get(k) is not None else np.nan for k in KEYS], float); return np.repeat(v[:, None], 3, 1)
scr = np.array([k[0] for k in KEYS]); idx = {c: np.where(scr == c)[0] for c in sorted(set(scr))}
rng = np.random.default_rng(1); Bn = 10000
draws = [(np.concatenate([rng.choice(v, len(v)) for v in idx.values()]), rng.choice(3, 3)) for _ in range(Bn)]
res = dict(n_ko=len(KEYS), n_per_screen={c: int(len(v)) for c, v in idx.items()}, means={}, contrasts=[], figure_values={})
for m in MET:
    M = {a: mat(a, m) for a in ARMS}
    for a, X in M.items():
        pt = float(np.nanmean(np.nanmean(X, 1))); bs = np.array([np.nanmean(np.nanmean(X[i][:, s], 1)) for i, s in draws])
        res["means"][f"{a}|{m}"] = dict(mean=pt, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(np.isfinite(np.nanmean(X, 1)).sum()))
        key = "CMC-enc" if ARMS[a][0] == "seed" else ARMS[a][2]
        res["figure_values"][f"{a}|{m}"] = (TI if a == "CMC-VAE, adapted" else TN)["summary"][key]["overall"][m]
    fam = []
    for a, b in [("CMC-VAE, adapted", "CMC-VAE, unadapted"), ("CMC-VAE, adapted", "Corpus correlation map"), ("CMC-VAE, adapted", "Cognate transcript"),
                 ("CMC-VAE, adapted", "Control-cell ridge (same cells)")]:
        D = M[a] - M[b]; dv = np.nanmean(D, 1); f = np.isfinite(dv)
        bs = np.array([np.nanmean(np.nanmean(D[i][:, s], 1)) for i, s in draws])
        pb = float(max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / (Bn + 1)))
        fam.append(dict(metric=m, a=a, b=b, n=int(f.sum()), diff=float(dv[f].mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        p_boot=pb, p_wilcoxon=float(wilcoxon(dv[f]).pvalue), per_seed=[float(np.nanmean(D[:, j])) for j in range(3)]))
    q = multipletests([x["p_boot"] for x in fam], method="fdr_bh")[1]
    for x, qq in zip(fam, q): x["q_boot_bh"] = float(qq)
    res["contrasts"] += fam
json.dump(res, open(OUT, "w"), indent=1)
for k, v in res["means"].items(): print(k, round(v["mean"], 3), [round(x, 3) for x in v["ci"]], "fig", round(res["figure_values"][k], 3))
for x in res["contrasts"]: print(x["metric"], x["a"], "-", x["b"], round(x["diff"], 3), [round(c, 3) for c in x["ci"]], "pboot", x["p_boot"], "q", round(x["q_boot_bh"], 4))
