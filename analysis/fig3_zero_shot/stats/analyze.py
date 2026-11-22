#!/usr/bin/env python
"""Collect point estimates (rep 0) + joint-bootstrap replicates -> tests, CIs, effect sizes, eligibility tables.
Writes analyses/stats/stats_summary.json and markdown tables (tables_*.md).  usage: python -I analyze.py [pb_bh]"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, glob, pickle
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pblimma import bh
ST = f"{_CMC_RESULTS}/paper/analyses/stats"
DEMODE = sys.argv[1] if len(sys.argv) > 1 else "pb_bh"
TAGS = ["frangieh", "iccite", "arce"]; SC = ["overall"] + TAGS
PT = pickle.load(open(f"{ST}/boot/point_{DEMODE}.pkl", "rb")); T = PT["T"]
BOOT = []
for f in sorted(glob.glob(f"{ST}/boot/boot_{DEMODE}_*.json")):
    BOOT += [r for r in json.load(open(f)) if r["_b"] != 0]
log = print
log("bootstrap replicates:", len(BOOT))

def vals(st, m, scope):
    out = {}
    for (s, t, g), r in T.items():
        if s == st and (scope == "overall" or t == scope):
            v = r.get(m, np.nan)
            if v is not None and np.isfinite(v): out[(t, g)] = float(v)
    return out

REP0 = [r for f in sorted(glob.glob(f"{ST}/boot/boot_{DEMODE}_000.json")) for r in json.load(open(f)) if r["_b"] == 0][0]
BIAS = {}
def bci(key):
    """Median-centred percentile CI: the bootstrap distribution is shifted so that its median equals the point
    estimate (re-calling DE inside each replicate yields fewer DE features and biases set-based metrics; the shift
    removes that bias, keeping the bootstrap spread).  Returns CI, centred replicate values, n."""
    x = np.array([r[key] for r in BOOT if r.get(key) is not None], float)
    x = x[np.isfinite(x)]
    if len(x) < 20 or REP0.get(key) is None: return None, None, len(x)
    sh = REP0[key] - np.median(x); BIAS[key] = float(-sh); xc = x + sh
    return [float(np.percentile(xc, 2.5)), float(np.percentile(xc, 97.5))], xc, len(x)

def pboot(x):
    if x is None or len(x) == 0: return None
    return float(max(2 * min((x <= 0).mean(), (x >= 0).mean()), 1.0 / (len(x) + 1)))

def mean_entry(st, m, scope):
    v = vals(st, m, scope); a = np.array(list(v.values()))
    ci, _, nb = bci(f"mean|{st}|{m}|{scope}")
    return dict(mean=float(a.mean()) if len(a) else None, n=len(a), sem=float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None, ci95_boot=ci)

def comp(a, ma, b, mb, scope):
    A, B = vals(a, ma, scope), vals(b, mb, scope)
    ks = sorted(set(A) & set(B)); x = np.array([A[k] for k in ks]); y = np.array([B[k] for k in ks])
    r = dict(a=a, ma=ma, b=b, mb=mb, scope=scope, n=len(ks))
    if len(ks) < 3: return r
    d = x - y
    r.update(mean_a=float(x.mean()), mean_b=float(y.mean()), mean_diff=float(d.mean()),
             dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None,
             ci95_t=[float(d.mean() - stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))),
                     float(d.mean() + stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))],
             p_t=float(stats.ttest_rel(x, y).pvalue) if not np.allclose(d, 0) else 1.0,
             p_wilcoxon=float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
    ci, xb, nb = bci(f"diff|{a}|{ma}|{b}|{mb}|{scope}")
    r.update(ci95_boot=ci, p_boot=pboot(xb), n_boot=nb)
    return r

RES = dict(demode=DEMODE, n_boot=len(BOOT))
# ---------------- eligibility ----------------
EL = {}
for t in TAGS:
    pb = pickle.load(open(f"{ST}/de/pb_{t}.pkl", "rb"))
    ev = PT["evaluable"][t]; hp = PT["has_p"][t]; nde = PT["n_de"][t]; ndp = PT["n_dep"][t]
    gl = list(nde)
    EL[t] = dict(**pb["eligibility"], in_vocab=len(gl), rna_ge3=len(ev), rna_ge5=int(sum(nde[g] >= 5 for g in ev)),
                 prot_ge3=int(sum(hp[g] for g in ev)), prot_ge5=int(sum(ndp[g] >= 5 for g in ev if hp[g])),
                 prot_auroc_defined=len(vals("f3:cmc_s42", "rp_auroc", t)), n_control_cells=pb["n_control_cells"], n_control_pb=pb["n_control_pb"],
                 pb_strata=pb["pb_strata_counts"], extra=pb["extra"],
                 median_n_de_rna=float(np.median([nde[g] for g in ev])) if ev else None,
                 median_pb_per_unit=float(np.median([pb["de"][g]["n_pb"] for g in gl])))
RES["eligibility"] = EL
# ---------------- Fig 3 ----------------
F3SETS = ["f3:cmc_s42", "f3:cmc_mean", "f3:corr", "f3:totalvi_ko", "f3:deepsem_ko"]
M3 = ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rr_pe_meanresp", "rr_auroc_meanresp", "rp_auroc", "rp_auroc_ge5", "rp_pe", "rp_pe_cent", "rp_retrieval", "rp_pe_meanresp", "rp_auroc_meanresp"]
RES["fig3_means"] = {st: {m: {sc: mean_entry(st, m, sc) for sc in SC} for m in M3} for st in F3SETS}
f3c = []
for b in ["f3:corr", "f3:totalvi_ko", "f3:deepsem_ko"]:
    for m in ["rr_auroc", "rr_pe", "rp_auroc", "rp_pe", "rr_pe_cent", "rr_retrieval", "rp_pe_cent", "rp_retrieval"]:
        for sc in TAGS:
            r = comp("f3:cmc_s42", m, b, m, sc)
            if r.get("n", 0) >= 3: f3c.append(r)
fam = [r for r in f3c if r["ma"] in ("rr_auroc", "rr_pe", "rp_auroc", "rp_pe")]
for r, q in zip(fam, bh([r["p_t"] for r in fam])): r["p_bh_fig3"] = float(q)
sec = [r for r in f3c if r["ma"] not in ("rr_auroc", "rr_pe", "rp_auroc", "rp_pe")]
for r, q in zip(sec, bh([r["p_t"] for r in sec])): r["p_bh_fig3_secondary"] = float(q)
for m, mm in [("rr_auroc", "rr_auroc_meanresp"), ("rr_pe", "rr_pe_meanresp"), ("rp_auroc", "rp_auroc_meanresp"), ("rp_pe", "rp_pe_meanresp")]:
    for sc in SC:
        f3c.append(comp("f3:cmc_s42", m, "f3:cmc_s42", mm, sc))
RES["fig3_comparisons"] = f3c
# ---------------- Fig 4 ----------------
f4 = []
PAN = {"b": ("rr_auroc", ["Arr_transposed", "Arr_shuffled", "Arr_zeroed"]), "c": ("rp_auroc", ["Wrp_from_WprT", "Wrp_shuffled", "Wrp_zeroed"]),
       "d": ("rr_pe", ["Arr_transposed", "Arr_shuffled", "Arr_zeroed"]), "e": ("rp_pe", ["Wrp_from_WprT", "Wrp_shuffled"]),
       "d_cent": ("rr_pe_cent", ["Arr_transposed", "Arr_shuffled", "Arr_zeroed"]), "retr_rr": ("rr_retrieval", ["Arr_transposed", "Arr_shuffled", "Arr_zeroed"]),
       "e_cent": ("rp_pe_cent", ["Wrp_from_WprT", "Wrp_shuffled"])}
RES["fig4_means"] = {}
for pan, (m, vs) in PAN.items():
    rows = []
    for v in vs:
        for sc in TAGS:
            r = comp("f3:cmc_s42", m, f"f4:{v}", m, sc); r["panel"] = pan
            if r.get("n", 0) >= 3: rows.append(r)
            RES["fig4_means"].setdefault(f"f4:{v}", {}).setdefault(m, {})[sc] = mean_entry(f"f4:{v}", m, sc)
    for r, q in zip(rows, bh([r["p_t"] for r in rows])): r["p_bh_panel"] = float(q)
    pbs = [r.get("p_boot") or 1.0 for r in rows]
    for r, q in zip(rows, bh(pbs)): r["p_boot_bh_panel"] = float(q)
    f4 += rows
RES["fig4_comparisons"] = f4
# ---------------- Fig 5 ----------------
V5 = ["ft_baseline", "ft_standard", "ft_labelshuffle", "ft_meantarget", "ft_signflip"]
M5 = ["rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc", "rp_pe", "rp_pe_cent", "rp_auroc", "rp_retrieval", "rr_pe_meanresp", "rp_pe_meanresp"]
RES["fig5_means"] = {}
for v in V5:
    for s in ["mean", "s42", "s1", "s2"]:
        st = f"f5:{v}|{s}"
        RES["fig5_means"][st] = {m: {sc: mean_entry(st, m, sc) for sc in SC} for m in M5}
f5 = []
for pan, m in [("b", "rr_pe"), ("c", "rp_pe"), ("b_cent", "rr_pe_cent"), ("retr", "rr_retrieval"), ("c_cent", "rp_pe_cent")]:
    rows = []
    for a, b in [("ft_standard", "ft_baseline"), ("ft_standard", "ft_labelshuffle"), ("ft_standard", "ft_meantarget"),
                 ("ft_labelshuffle", "ft_baseline"), ("ft_meantarget", "ft_baseline")]:
        for sc in SC:
            r = comp(f"f5:{a}|mean", m, f"f5:{b}|mean", m, sc); r["panel"] = pan
            if r.get("n", 0) >= 3: rows.append(r)
    for r, q in zip(rows, bh([r["p_t"] for r in rows])): r["p_bh_panel"] = float(q)
    f5 += rows
RES["fig5_comparisons"] = f5
RES["fig5_seedwise"] = [comp(f"f5:ft_standard|{s}", m, f"f5:{b}|{s}", m, sc) for s in ["s42", "s1", "s2"] for m in ["rr_pe", "rp_pe", "rr_pe_cent"]
                        for b in ["ft_baseline", "ft_labelshuffle", "ft_meantarget"] for sc in SC]
# per-seed pooled values (micro-average over held-out units) -> seed mean / sd, the quantity plotted in Fig 5b-d
SEEDSUM = {}
for v in V5:
    for m in M5:
        for sc in SC:
            xs = [mean_entry(f"f5:{v}|{s}", m, sc)["mean"] for s in ["s42", "s1", "s2"]]
            xs = [x for x in xs if x is not None]
            SEEDSUM[f"{v}|{m}|{sc}"] = dict(per_seed=xs, mean=float(np.mean(xs)) if xs else None, sd=float(np.std(xs, ddof=1)) if len(xs) > 1 else None)
RES["fig5_seed_summary"] = SEEDSUM
# ---------------- pooled protein ----------------
PP = {}
for k in [k for k in REP0 if k.startswith("pooled|")]:
    ci, xb, nb = bci(k)
    PP[k] = dict(point=REP0[k], ci95_boot=ci)
RES["boot_n_eval"] = {t: [float(np.percentile([r["_n_eval"][t] for r in BOOT], q)) for q in (2.5, 50, 97.5)] for t in TAGS}
RES["pooled_protein_boot"] = PP
RES["bootstrap_median_minus_point"] = dict(median_abs=float(np.median(np.abs(list(BIAS.values())))), by_key=BIAS)
json.dump(RES, open(f"{ST}/stats_summary_{DEMODE}.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else o)
print("WROTE", f"{ST}/stats_summary_{DEMODE}.json")
