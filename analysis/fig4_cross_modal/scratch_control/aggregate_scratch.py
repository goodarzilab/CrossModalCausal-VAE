"""Review-2 M7a: aggregate scratch-control results per PLAN.md -> results.json (+ printed digest).
Statistics copied from analysis/fig4_cross_modal/adapt/aggregate.py (Wilcoxon two-sided, 2,000 bootstrap over units seed 0, BH within panel)."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, json, glob
import numpy as np
from scipy.stats import wilcoxon
HERE = f"{_CMC_RESULTS}/paper/analyses/review2/adapt_scratch"
O = f"{_CMC_SCRATCH}/nov_adapt"
SEEDS = ["s42", "s1", "s2"]; TAGS = ["frangieh", "iccite", "arce"]
OUT = {"plan_sha256": open(f"{HERE}/PLAN.sha256").read().split()[0]}; DIG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); DIG.append(s)
def bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0: return p
    o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1); r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q
def boot(d, B=2000, seed=0):
    d = np.asarray(d, float); r = np.random.RandomState(seed); bs = d[r.randint(0, len(d), (B, len(d)))].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def ptest(a, b, na, nb, metric, scope, per_seed=None):
    ks = sorted(k for k in set(a) & set(b) if np.isfinite(a[k]) and np.isfinite(b[k]))
    r = dict(a=na, b=nb, metric=metric, scope=scope, n=len(ks))
    if len(ks) < 2: return r
    d = np.array([a[k] - b[k] for k in ks])
    r.update(mean_a=float(np.mean([a[k] for k in ks])), mean_b=float(np.mean([b[k] for k in ks])), diff=float(d.mean()), ci95=boot(d))
    if len(ks) >= 5: r["p"] = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    if per_seed is not None: r["per_seed_diff"] = per_seed; r["n_seeds_same_sign"] = int(sum(np.sign(x) == np.sign(r["diff"]) for x in per_seed))
    return r
def panel(tests, name):
    ok = [t for t in tests if "p" in t]; q = bh([t["p"] for t in ok])
    for t, x in zip(ok, q): t["q"] = float(x)
    OUT.setdefault("tests", {})[name] = tests
    say(f"--- panel {name}")
    for t in tests:
        if "diff" not in t: say(f"  {t['a']} - {t['b']} {t['metric']} [{t['scope']}]: n={t['n']} too few"); continue
        ps = "" if "per_seed_diff" not in t else " seeds " + "/".join(f"{x:+.3f}" for x in t["per_seed_diff"])
        pq = f" p={t['p']:.2g} q={t['q']:.2g}" if "p" in t else " (descriptive)"
        say(f"  {t['a']:>11s} - {t['b']:<11s} {t['metric']:9s} [{t['scope']:8s}] {t['mean_a']:.3f} vs {t['mean_b']:.3f} d={t['diff']:+.3f} "
            f"[{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] n={t['n']}{pq}{ps}")

# ------------------------------------------------------------------ E2 imputation (held-out 10% control cells)
def diag_file(arm, dom, sk):
    if arm == "adapted_i": return f"{O}/diag/sp_i_{dom}_{sk}.json"
    if arm == "unadapted": return f"{O}/diag/sp_none_{dom}_{sk}.json"
    return f"{HERE}/diag/{arm}_{dom}_{sk}.json"
ARMS = ["unadapted", "adapted_i", "repro_i", "scratch", "scratch_x3", "adapt_x3", "scratch_gen"]
IMP = {}; IMPS = {}
for arm in ARMS:
    for dom in TAGS:
        fs = [diag_file(arm, dom, s) for s in SEEDS]
        if not all(os.path.exists(f) for f in fs): continue
        js = [json.load(open(f)) for f in fs]
        pp = np.array([j["diag"]["rna_only_perprot"] for j in js], float)       # seeds x proteins
        IMP.setdefault(arm, {})[dom] = {i: float(np.nanmean(pp[:, i])) if np.isfinite(pp[:, i]).any() else np.nan for i in range(pp.shape[1])}
        IMP[arm][dom + ":perseed"] = pp
        IMPS.setdefault(arm, {})[dom] = dict(rna_only_seedmean_of_medians=float(np.mean([j["diag"]["rna_only"] for j in js])),
                                             per_seed=[float(j["diag"]["rna_only"]) for j in js],
                                             with_protein=float(np.mean([j["diag"]["with_protein"] for j in js])),
                                             rna_recon=float(np.mean([j["diag"]["rna_recon_r_median"] for j in js])),
                                             best_step=[j.get("best_step") for j in js], n_steps_run=[j.get("n_steps_run") for j in js],
                                             crit_best=[j.get("crit_best") for j in js], ridge=js[0].get("ridge_median"),
                                             init_rna_only=[j["diag_init"]["rna_only"] for j in js] if "diag_init" in js[0] else None)
OUT["imputation_summary"] = IMPS
say("=== E2 imputation (seed-mean of per-seed median per-protein r, RNA-only input; held-out 10%)")
for arm in ARMS:
    if arm in IMPS:
        say(f"  {arm:12s} " + "  ".join(f"{d} {IMPS[arm][d]['rna_only_seedmean_of_medians']:.3f} (steps {IMPS[arm][d]['best_step']})" for d in TAGS if d in IMPS[arm]))
def imp_units(arm, scope):
    return {(d, i): v for d in (TAGS if scope == "pooled" else [scope]) for i, v in IMP[arm].get(d, {}).items()} if arm in IMP else {}
def imp_perseed(a, b, scope):
    out = []
    for si in range(3):
        diffs = []
        for d in (TAGS if scope == "pooled" else [scope]):
            if d not in IMP.get(a, {}) or d not in IMP.get(b, {}): return None
            diffs += list(IMP[a][d + ":perseed"][si] - IMP[b][d + ":perseed"][si])
        out.append(float(np.nanmean(diffs)))
    return out

# ------------------------------------------------------------------ E1 translation
TRF = {"adapted_i": f"{O}/eval/translate/sp_i.json", "unadapted": f"{O}/eval/translate/sp_none.json",
       "orig_rescored": f"{HERE}/eval/translate_orig_sp_i.json"}
for arm in ["repro_i", "scratch", "scratch_x3", "adapt_x3", "scratch_gen"]: TRF[arm] = f"{HERE}/eval/translate_{arm}.json"
TR = {k: json.load(open(f)) for k, f in TRF.items() if os.path.exists(f)}
def tr_units(arm, metric, scope, method="CMC-enc"):
    js = TR[arm]; return {(r["cohort"], r["gene"]): (r[metric] if r.get(metric) is not None else np.nan) for r in js["per_ko"][method]
                          if scope == "pooled" or r["cohort"] == scope}
def tr_perseed(a, b, metric, scope, mb=None):
    out = []
    for s in SEEDS:
        if f"CMC-enc:{s}" not in TR[a]["per_ko"]: return None
        ua = tr_units(a, metric, scope, f"CMC-enc:{s}")
        ub = tr_units(b, metric, scope, mb if mb else f"CMC-enc:{s}"); ks = [k for k in ua if k in ub]
        out.append(float(np.nanmean([ua[k] - ub[k] for k in ks])))
    return out
say("=== E1 translation (CMC-enc, seed-mean prediction; KOs with >=3 DE proteins)")
TS = {}
for arm, js in TR.items():
    TS[arm] = {sc: {k: js["summary"]["CMC-enc"][sc].get(k) for k in ["pe_cent", "retrieval", "pe", "auroc", "n_pe_cent"]} for sc in ["overall"] + TAGS}
    TS[arm]["perprotein_r"] = js["summary_perprot"]["CMC-enc"]["overall"]["r"]
    TS[arm]["per_seed"] = {s: {k: js["summary"][f"CMC-enc:{s}"]["overall"][k] for k in ["pe_cent", "retrieval"]} for s in SEEDS if f"CMC-enc:{s}" in js["summary"]}
    s = TS[arm]["overall"]
    say(f"  {arm:13s} pooled cent {s['pe_cent']:+.3f} retr {s['retrieval']:.3f} pe {s['pe']:+.3f} auroc {s['auroc']:.3f} n={s['n_pe_cent']} | "
        + " ".join(f"{t}: {TS[arm][t]['pe_cent']:+.3f}/{TS[arm][t]['retrieval']:.3f}" for t in TAGS))
if "adapted_i" in TR:
    for mth in ["ctrlridge90", "cognate", "corr"]:
        s = TR["adapted_i"]["summary"][mth]["overall"]; say(f"  {mth:13s} pooled cent {s['pe_cent']:+.3f} retr {s['retrieval']:.3f}")
OUT["translate_summary"] = TS

# ------------------------------------------------------------------ step 0: reproduction
REP = {}
if "orig_rescored" in TR and "adapted_i" in TR:
    a, b = tr_units("orig_rescored", "pe_cent", "pooled"), tr_units("adapted_i", "pe_cent", "pooled")
    REP["scorer_rerun_max_abs_diff_pe_cent"] = float(np.nanmax([abs(a[k] - b[k]) for k in a]))
    REP["scorer_rerun_pooled"] = TS["orig_rescored"]["overall"]
if "repro_i" in TR:
    REP["repro_i_pooled"] = TS["repro_i"]["overall"]; REP["adapted_i_pooled"] = TS["adapted_i"]["overall"]
    REP["repro_i_minus_adapted_i"] = {k: TS["repro_i"]["overall"][k] - TS["adapted_i"]["overall"][k] for k in ["pe_cent", "retrieval"]}
if "repro_i" in IMPS:
    REP["imputation_repro_vs_orig"] = {d: [IMPS["repro_i"][d]["rna_only_seedmean_of_medians"], IMPS["adapted_i"][d]["rna_only_seedmean_of_medians"]] for d in TAGS}
    mx = 0.0
    for d in TAGS:
        for s in SEEDS:
            mx = max(mx, abs(json.load(open(diag_file("repro_i", d, s)))["diag"]["rna_only"] - json.load(open(diag_file("adapted_i", d, s)))["diag"]["rna_only"]))
    REP["imputation_max_abs_diff_per_model"] = mx
REP["passed"] = bool(REP.get("repro_i_minus_adapted_i") and all(abs(v) <= 0.01 for v in REP["repro_i_minus_adapted_i"].values())
                     and REP.get("scorer_rerun_max_abs_diff_pe_cent", 1) < 1e-6)
OUT["reproduction"] = REP; say("=== step 0 reproduction:", json.dumps(REP))

# ------------------------------------------------------------------ tests
def e1(a, b, metric, scope, mb=None):
    if a not in TR or (b not in TR and not mb): return dict(a=a, b=b, metric=metric, scope=scope, n=0)
    ub = tr_units("adapted_i", metric, scope, mb) if mb else tr_units(b, metric, scope)
    return ptest(tr_units(a, metric, scope), ub, a, b if not mb else mb, metric, scope,
                 tr_perseed(a, b if not mb else "adapted_i", metric, scope, mb))
def e2(a, b, scope):
    if a not in IMP or b not in IMP: return dict(a=a, b=b, metric="imp_r", scope=scope, n=0)
    return ptest(imp_units(a, scope), imp_units(b, scope), a, b, "imp_r", scope, imp_perseed(a, b, scope))
P1 = []
for b in ["scratch", "scratch_x3"]:
    for m in ["pe_cent", "retrieval"]: P1.append(e1("adapted_i", b, m, "pooled"))
    P1.append(e2("adapted_i", b, "pooled"))
panel(P1, "P1_primary")
P2 = []
for t in TAGS:
    for b in ["scratch", "scratch_x3"]:
        for m in ["pe_cent", "retrieval"]: P2.append(e1("adapted_i", b, m, t))
        P2.append(e2("adapted_i", b, t))
panel(P2, "P2_per_screen")
P3 = []
for a, b in [("adapt_x3", "scratch_x3"), ("adapted_i", "scratch_gen"), ("adapt_x3", "scratch_gen")]:
    for m in ["pe_cent", "retrieval"]: P3.append(e1(a, b, m, "pooled"))
    P3.append(e2(a, b, "pooled"))
for a in ["scratch", "scratch_x3", "scratch_gen"]:
    for m in ["pe_cent", "retrieval"]:
        P3.append(e1(a, "unadapted", m, "pooled")); P3.append(e1(a, None, m, "pooled", mb="ctrlridge90"))
panel(P3, "P3_secondary")
P3s = []
for t in TAGS:
    for a, b in [("adapt_x3", "scratch_x3"), ("adapted_i", "scratch_gen")]:
        for m in ["pe_cent", "retrieval"]: P3s.append(e1(a, b, m, t))
        P3s.append(e2(a, b, t))
panel(P3s, "P3_per_screen")

# ------------------------------------------------------------------ decision (PLAN.md)
def get(name, a, b, metric, scope="pooled"):
    return next((t for t in OUT["tests"][name] if t["a"] == a and t["b"] == b and t["metric"] == metric and t["scope"] == scope), {})
def pos(t): return bool(t.get("q", 1) < 0.05 and t.get("diff", 0) > 0 and t.get("n_seeds_same_sign", 0) >= 2)
dec = {}
for m in ["pe_cent", "retrieval", "imp_r"]:
    dec[m] = dict(vs_scratch=pos(get("P1_primary", "adapted_i", "scratch", m)), vs_scratch_x3=pos(get("P1_primary", "adapted_i", "scratch_x3", m)))
e1_help = any(dec[m]["vs_scratch"] and dec[m]["vs_scratch_x3"] for m in ["pe_cent", "retrieval"])
e1_conv = (not e1_help) and any(dec[m]["vs_scratch"] for m in ["pe_cent", "retrieval"])
dec["E1_pretraining_helps"] = e1_help; dec["E1_only_speeds_convergence"] = e1_conv
dec["E2_pretraining_helps"] = dec["imp_r"]["vs_scratch"] and dec["imp_r"]["vs_scratch_x3"]
OUT["decision"] = dec; say("=== decision:", json.dumps(dec))
json.dump(OUT, open(f"{HERE}/results.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else (o.tolist() if hasattr(o, "tolist") else str(o)))
open(f"{HERE}/digest.txt", "w").write("\n".join(DIG) + "\n")
