"""Review2 M4a: paired-bootstrap comparison of full sp_dec_gl1000 vs retrained A_rr==0 (per screen, per-KO rr_auroc).
Writes analyses/review2/arr0/results.json."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon

REPO = f"{_CMC_ROOT}"
AR = f"{_CMC_RESULTS}/paper/analyses/review2/arr0"
STATS = f"{_CMC_RESULTS}/paper/analyses/stats/runs"
SCREENS = ["frangieh", "iccite", "arce"]
METRICS = ["rr_auroc", "rp_auroc"]
B = 10000

FULL = {"s42": f"{STATS}/rc_sp_dec_gl1000_pb_bh.json", "s1": f"{STATS}/rc_sp_dec_gl1000_s1_pb_bh.json",
        "s2": f"{STATS}/rc_sp_dec_gl1000_s2_pb_bh.json"}
ARR0 = {s: f"{AR}/runs/arr0_{s}.json" for s in ["s42", "s1", "s2"]}
RUNDIR_FULL = {"s42": "sweep_sparse/sp_dec_gl1000", "s1": "sweep_sparse/sp_dec_gl1000_s1", "s2": "sweep_sparse/sp_dec_gl1000_s2"}


def per_ko(path, metric):
    r = json.load(open(path))
    out = {t: {} for t in SCREENS}
    for row in r["fig3_obs"]["per_ko"]:
        v = row.get(metric, np.nan)
        if isinstance(v, float) and np.isfinite(v):
            out[row["cohort"]][row["gene"]] = v
    return out, r["fig3_obs"]["summary"]


def paired(a, b, seed=0):
    """a, b: dict gene->value. Delta = a - b over common genes."""
    g = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in g], float)
    if len(d) < 3:
        return dict(n=len(d))
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    try:
        wp = float(wilcoxon(d).pvalue)
    except ValueError:
        wp = None
    return dict(n=int(len(d)), mean_full=float(np.mean([a[k] for k in g])), mean_arr0=float(np.mean([b[k] for k in g])),
                delta=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                p_one_sided_boot=float((bs <= 0).mean()), p_two_sided_boot=float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))),
                wilcoxon_p=wp, frac_ko_full_better=float((d > 0).mean()))


def stats_json(rd):
    p = f"{_CMC_RESULTS}/{rd}/final_stats.json"
    m = f"{_CMC_RESULTS}/{rd}/metrics.json"
    out = {}
    if os.path.exists(p):
        out["final_val"] = {k: v for k, v in json.load(open(p)).items() if k in ("rec", "kl_r", "kl_p", "rho", "total")}
    if os.path.exists(m):
        mm = json.load(open(m))
        out["test"] = {k: v for k, v in mm.get("test", {}).items() if k in ("rec", "kl_r", "kl_p", "rho", "total")}
    return out


res = dict(plan="PLAN.md (pre-registered)", harness="code/rc_stats_arr0.py (copy of analysis/fig3_zero_shot/stats/rc_stats.py; ST_DE=pb_bh, RC_FIG3_ONLY=1)",
           verification={}, per_seed={}, seed_mean={}, summaries={}, recon={})
# gate 0: verification
for name in ["verify_s42_existingcache", "verify_s42_freshcache"]:
    p = f"{AR}/runs/{name}.json"
    if os.path.exists(p):
        pk, sm = per_ko(p, "rr_auroc"); ref, _ = per_ko(FULL["s42"], "rr_auroc")
        res["verification"][name] = {t: dict(rr_auroc=sm[t]["rr_auroc"], n=sm[t]["n_rr_auroc"],
                                             max_abs_diff_vs_ref_per_ko=float(max(abs(pk[t][g] - ref[t][g]) for g in ref[t])) if set(ref[t]) == set(pk[t]) else "KO set differs")
                                     for t in SCREENS}

avail = [s for s in ["s42", "s1", "s2"] if os.path.exists(ARR0[s])]
for s in ["s42", "s1", "s2"]:
    res["recon"][f"full_{s}"] = stats_json(RUNDIR_FULL[s])
    res["recon"][f"arr0_{s}"] = stats_json(f"review2_arr0_{s}")
for s in avail:
    res["per_seed"][s] = {}
    for m in METRICS:
        f, fs = per_ko(FULL[s], m); a, as_ = per_ko(ARR0[s], m)
        res["per_seed"][s][m] = {t: paired(f[t], a[t]) for t in SCREENS}
        res["summaries"].setdefault(s, {})[m] = {t: dict(full=fs[t].get(m), arr0=as_[t].get(m)) for t in SCREENS}
if len(avail) > 1:
    for m in METRICS:
        F = [per_ko(FULL[s], m)[0] for s in avail]; A = [per_ko(ARR0[s], m)[0] for s in avail]
        res["seed_mean"][m] = {}
        for t in SCREENS:
            gs = set.intersection(*[set(x[t]) for x in F + A])
            fm = {g: float(np.mean([x[t][g] for x in F])) for g in gs}; am = {g: float(np.mean([x[t][g] for x in A])) for g in gs}
            res["seed_mean"][m][t] = paired(fm, am)
    res["seed_mean"]["seeds"] = avail
json.dump(res, open(f"{AR}/results.json", "w"), indent=2)
for s in avail:
    for t in SCREENS:
        r = res["per_seed"][s]["rr_auroc"][t]
        print(f"{s} {t:9s} n={r['n']:3d} full {r['mean_full']:.3f} arr0 {r['mean_arr0']:.3f} d={r['delta']:+.3f} CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p1={r['p_one_sided_boot']:.4f}")
if res["seed_mean"]:
    for t in SCREENS:
        r = res["seed_mean"]["rr_auroc"][t]
        print(f"seedmean {t:9s} n={r['n']:3d} full {r['mean_full']:.3f} arr0 {r['mean_arr0']:.3f} d={r['delta']:+.3f} CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p1={r['p_one_sided_boot']:.4f}")
print(json.dumps(res["verification"], indent=1)); print(json.dumps(res["recon"], indent=1))
