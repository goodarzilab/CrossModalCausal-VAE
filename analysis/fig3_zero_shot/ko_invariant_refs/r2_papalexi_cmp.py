"""Task 1, Papalexi: paired CMC-VAE (sp_dec_gl1000, reviewfix/papalexi) vs RNA-only CMC-VAE (round2/papalexi), signed clamp,
3 seeds. Per DE definition (cell_raw, cell_bh, pb): per-KO seed-averaged difference, two-sided Wilcoxon + KO bootstrap CI
(10,000); for pb also the joint KO x control-cell bootstrap (999 replicates, identical RNG streams default_rng(100003+b) in
both runs, evaluable units do not depend on the model -> replicate-wise difference of means). BH over the family.
usage: python -I r2_papalexi_cmp.py OUT_JSON"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
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
PA = f"{_CMC_RESULTS}/paper/analyses/reviewfix/papalexi"; RO = f"{_CMC_SCRATCH}/nov_round2/papalexi"
S = ["s42", "s1", "s2"]; MET = ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]
L = lambda d, s: json.load(open(f"{d}/papalexi_{s}.json"))
P = {s: L(PA, s) for s in S}; R = {s: L(RO, s) for s in S}
rng = np.random.default_rng(0); out = dict(rows=[])
for mode in ["cell_raw", "cell_bh", "pb"]:
    units = P["s42"]["modes"][mode]["units"]
    for m in MET:
        def v(D, g): 
            x = [D[s]["modes"][mode]["per_ko"].get(f"cmc_clamp|{g}", {}).get(m, np.nan) for s in S]
            x = np.array([np.nan if y is None else y for y in x], float); return np.nanmean(x) if np.isfinite(x).any() else np.nan
        a = np.array([v(P, g) for g in units]); b = np.array([v(R, g) for g in units]); f = np.isfinite(a) & np.isfinite(b)
        d = (a - b)[f]; n = len(d)
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)]) if n > 2 else None
        e = dict(mode=mode, metric=m, n=int(n), paired=float(a[f].mean()), rnaonly=float(b[f].mean()), diff=float(d.mean()),
                 ci_ko=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs is not None else None,
                 p_wilcoxon=float(wilcoxon(d).pvalue) if n > 2 and np.any(d != 0) else None)
        if mode == "pb":
            try:
                BP = {s: json.load(open(f"{PA}/papalexi_boot_{s}.json")) for s in S}; BR = {s: json.load(open(f"{RO}/papalexi_boot_{s}.json")) for s in S}
                k = f"mean|cmc_clamp|{m}"
                X = np.array([[np.nanmean([BP[s][i].get(k, np.nan) for s in S]) - np.nanmean([BR[s][i].get(k, np.nan) for s in S])] for i in range(len(BP["s42"]))]).ravel()
                X = X[np.isfinite(X)]; xc = X + (e["diff"] - np.median(X))
                e.update(ci_joint=[float(np.percentile(xc, 2.5)), float(np.percentile(xc, 97.5))],
                         p_boot_joint=float(max(2 * min((xc <= 0).mean(), (xc >= 0).mean()), 1 / (len(xc) + 1))), n_boot=int(len(xc)))
            except FileNotFoundError: pass
        out["rows"].append(e)
ps = [e.get("p_boot_joint", e["p_wilcoxon"]) for e in out["rows"]]; ok = [i for i, p in enumerate(ps) if p is not None]
q = multipletests([ps[i] for i in ok], method="fdr_bh")[1]
for i, qq in zip(ok, q): out["rows"][i]["q"] = float(qq)
json.dump(out, open(sys.argv[1], "w"), indent=1)
for e in out["rows"]:
    print(f"{e['mode']:8s} {e['metric']:13s} n={e['n']:2d} paired {e['paired']:.3f} rnaonly {e['rnaonly']:.3f} diff {e['diff']:+.3f} CI {e.get('ci_joint', e['ci_ko'])} p {e.get('p_boot_joint', e['p_wilcoxon'])} q {e.get('q')}")
