"""adapt workstream: aggregate every endpoint, run the pre-registered tests (PREREGISTRATION.md), write summary.json + digest.
Per-KO metrics are averaged over seeds before testing; two-sided Wilcoxon; 2,000-resample bootstrap-over-KO 95% CI; BH within panel.
usage: python -I aggregate.py"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, glob
import numpy as np
from scipy.stats import wilcoxon
O = f"{_CMC_SCRATCH}/nov_adapt"; E = f"{O}/eval"
SEEDS = ["s42", "s1", "s2"]; TAGS = ["frangieh", "iccite", "arce"]
OUT = {}; DIG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); DIG.append(s)
def bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0: return p
    o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1); r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q
def boot(d, B=2000, seed=0):
    d = np.asarray(d, float); r = np.random.RandomState(seed); bs = d[r.randint(0, len(d), (B, len(d)))].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def ptest(a, b, name_a, name_b, metric, per_seed=None):
    """a, b: dict key->value (seed-mean per unit). diff = a - b."""
    ks = sorted(k for k in set(a) & set(b) if np.isfinite(a[k]) and np.isfinite(b[k]))
    if len(ks) < 5: return dict(a=name_a, b=name_b, metric=metric, n=len(ks))
    d = np.array([a[k] - b[k] for k in ks])
    r = dict(a=name_a, b=name_b, metric=metric, n=len(ks), mean_a=float(np.mean([a[k] for k in ks])), mean_b=float(np.mean([b[k] for k in ks])),
             diff=float(d.mean()), ci95=boot(d), p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
    if per_seed is not None: r["per_seed_diff"] = per_seed
    return r
def panel(tests, name):
    ok = [t for t in tests if "p" in t]; q = bh([t["p"] for t in ok])
    for t, x in zip(ok, q): t["q"] = float(x)
    OUT.setdefault("tests", {})[name] = tests
    for t in tests:
        if "p" in t:
            ps = "" if "per_seed_diff" not in t else " per-seed " + "/".join(f"{x:+.3f}" if x == x else "nan" for x in t["per_seed_diff"])
            say(f"  [{name}] {t['a']} - {t['b']} {t['metric']}: {t['mean_a']:.3f} vs {t['mean_b']:.3f} d={t['diff']:+.3f} "
                f"[{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] n={t['n']} p={t['p']:.2g} q={t['q']:.2g}{ps}")
        else: say(f"  [{name}] {t['a']} - {t['b']} {t['metric']}: n={t['n']} (too few)")

# =============================================================== (a) control-cell imputation diagnostic
say("=== (a) held-out control-cell protein-from-RNA imputation (median per-protein r; seed mean [min,max])")
A = {}
for f in sorted(glob.glob(f"{O}/diag/*_*_*_s*.json")):
    b = os.path.basename(f)[:-5]; parts = b.split("_"); sk = parts[-1]; dom = parts[-2]; base = parts[0]; var = "_".join(parts[1:-2])
    if dom not in TAGS: continue
    d = json.load(open(f)); dg = d["diag"]
    A.setdefault((base, var, dom), {})[sk] = dict(rna_only=dg["rna_only"], with_protein=dg["with_protein"], share=dg["var_share_rna_path"],
                                                rna_rec=dg["rna_recon_r_median"], perprot=dg["rna_only_perprot"], ridge=d["ridge_median"],
                                                best_step=d.get("best_step"), crit0=d.get("crit0"), crit_best=d.get("crit_best"))
RG = {t: json.load(open(f"{O}/diag/ridge_{t}.json")) for t in TAGS if os.path.exists(f"{O}/diag/ridge_{t}.json")}
TA = []
for (base, var, dom), v in sorted(A.items()):
    ks = [s for s in SEEDS if s in v]
    row = dict(base=base, variant=var, domain=dom, n_seeds=len(ks), ridge=RG[dom]["median"] if dom in RG else None)
    for m in ["rna_only", "with_protein", "share", "rna_rec"]:
        x = [v[s][m] for s in ks]; row[m] = float(np.mean(x)); row[m + "_range"] = [float(min(x)), float(max(x))]
    row["best_step"] = [v[s]["best_step"] for s in ks]
    pp = np.nanmean([v[s]["perprot"] for s in ks], 0); rp = np.array(RG[dom]["perprot"], float)
    row["rna_only_seedmean_perprot_median"] = float(np.nanmedian(pp))
    f_ = np.isfinite(pp) & np.isfinite(rp)
    if f_.sum() >= 5:
        dd = pp[f_] - rp[f_]; row["vs_ridge_perprot"] = dict(n=int(f_.sum()), median_diff=float(np.median(dd)), mean_diff=float(dd.mean()), ci95=boot(dd),
                                                              p=float(wilcoxon(dd).pvalue))
    row["ratio_to_ridge"] = row["rna_only"] / row["ridge"] if row["ridge"] else None
    TA.append(row)
    say(f"  {base:2s} {var:6s} {dom:8s} rna_only {row['rna_only']:+.3f} [{row['rna_only_range'][0]:+.3f},{row['rna_only_range'][1]:+.3f}] "
        f"with_prot {row['with_protein']:.3f} share {row['share']:.3f} rna_rec {row['rna_rec']:.3f} | ridge {row['ridge']:.3f} ratio {row['ratio_to_ridge']:.2f} steps {row['best_step']}")
OUT["imputation"] = TA
mech = {dom: next((r["ratio_to_ridge"] for r in TA if r["base"] == "sp" and r["variant"] == "i" and r["domain"] == dom), None) for dom in TAGS}
OUT["mechanism_restored"] = dict(ratios=mech, rule=">=0.5 x ridge in both iccite and arce",
                                 met=bool(mech.get("iccite") and mech.get("arce") and mech["iccite"] >= 0.5 and mech["arce"] >= 0.5))
say("mechanism restored (pre-specified rule):", OUT["mechanism_restored"])

# GW adaptation reconstruction (no paired data): validation criterion before/after
GWA = []
for f in sorted(glob.glob(f"{O}/diag/*_gw*_s*.json")):
    d = json.load(open(f)); GWA.append(dict(file=os.path.basename(f), crit0=d["crit0"], crit_best=d["crit_best"], best_step=d["best_step"],
                                            parts0=d["hist"][0]["parts"], parts_best=[h for h in d["hist"] if h["step"] == d["best_step"]][0]["parts"]))
OUT["gw_adaptation"] = GWA

# =============================================================== (b1) translation
def load_translate(lab):
    f = f"{E}/translate/{lab}.json"
    return json.load(open(f)) if os.path.exists(f) else None
TR = {lab: load_translate(lab) for lab in ["sp_none", "sp_i", "sp_ii", "sp_iii", "sp_i_xm1", "xm_none", "xm_i"]}
def tr_units(js, method, metric, scope="overall"):
    return {(r["cohort"], r["gene"]): r[metric] for r in js["per_ko"][method] if (scope == "overall" or r["cohort"] == scope) and r.get(metric) is not None}
if TR["sp_i"] and TR["sp_none"]:
    say("=== (b1) translation: measured RNA -> protein (pooled; seed-mean predictions)")
    summ = {}
    for lab, js in TR.items():
        if js is None: continue
        summ[lab] = {m: js["summary"][m] for m in js["summary"] if m.startswith("CMC-enc") or m in ("ctrlridge", "ctrlridge90", "corr", "cognate", "koridge", "totalVI", "cridge", "CMC-Wrp")}
        s = js["summary"]["CMC-enc"]["overall"]; pp = js["summary_perprot"]["CMC-enc"]["overall"]
        say(f"  {lab:9s} CMC-enc pe {s['pe']:+.3f} cent {s['pe_cent']:+.3f} retr {s['retrieval']:.3f} auroc {s['auroc']:.3f} per-protein r {pp['r']:+.3f} (n={s['n_pe_cent']})")
    for mth in ["ctrlridge90", "ctrlridge", "corr", "cognate", "koridge"]:
        s = TR["sp_none"]["summary"][mth]["overall"]; pp = TR["sp_none"]["summary_perprot"][mth]["overall"]
        say(f"  {mth:11s}       pe {s['pe']:+.3f} cent {s['pe_cent']:+.3f} retr {s['retrieval']:.3f} auroc {s['auroc']:.3f} per-protein r {pp['r']:+.3f}")
    OUT["translate_summary"] = summ
    def per_seed_tr(jsA, jsB, mB, metric):
        out = []
        for s in SEEDS:
            a = tr_units(jsA, f"CMC-enc:{s}", metric) if f"CMC-enc:{s}" in jsA["per_ko"] else None
            if a is None: return None
            b = tr_units(jsB, f"CMC-enc:{s}" if mB == "CMC-enc" else mB, metric); ks = set(a) & set(b)
            out.append(float(np.nanmean([a[k] - b[k] for k in ks])))
        return out
    T = []
    for metric in ["pe_cent", "retrieval"]:
        a = tr_units(TR["sp_i"], "CMC-enc", metric)
        T.append(ptest(a, tr_units(TR["sp_none"], "CMC-enc", metric), "adapted(i) CMC-enc", "unadapted CMC-enc", metric, per_seed_tr(TR["sp_i"], TR["sp_none"], "CMC-enc", metric)))
        T.append(ptest(a, tr_units(TR["sp_none"], "ctrlridge90", metric), "adapted(i) CMC-enc", "ctrlridge90", metric, per_seed_tr(TR["sp_i"], TR["sp_none"], "ctrlridge90", metric)))
    panel(T, "TR_primary")
    T2 = []
    for lab in ["sp_ii", "sp_iii", "sp_i_xm1", "xm_i"]:
        if TR.get(lab) is None: continue
        ref = "xm_none" if lab.startswith("xm") else "sp_none"
        for metric in ["pe_cent", "retrieval"]:
            a = tr_units(TR[lab], "CMC-enc", metric)
            T2.append(ptest(a, tr_units(TR[ref], "CMC-enc", metric), f"{lab} CMC-enc", f"{ref} CMC-enc", metric))
            T2.append(ptest(a, tr_units(TR["sp_none"], "ctrlridge90", metric), f"{lab} CMC-enc", "ctrlridge90", metric))
            T2.append(ptest(a, tr_units(TR["sp_none"], "corr", metric), f"{lab} CMC-enc", "corpus corr map", metric))
    panel(T2, "TR_secondary")
    T3 = []
    for scope in TAGS:
        for metric in ["pe_cent", "retrieval"]:
            a = tr_units(TR["sp_i"], "CMC-enc", metric, scope)
            T3.append(ptest(a, tr_units(TR["sp_none"], "CMC-enc", metric, scope), f"adapted(i) [{scope}]", "unadapted", metric))
            T3.append(ptest(a, tr_units(TR["sp_none"], "ctrlridge90", metric, scope), f"adapted(i) [{scope}]", "ctrlridge90", metric))
    panel(T3, "TR_per_screen")
    # per-protein r across KOs
    def pp_units(js, m): return {(r["cohort"], r["protein"]): r["r"] for r in js["per_protein"][m] if r.get("r") is not None}
    T4 = [ptest(pp_units(TR["sp_i"], "CMC-enc"), pp_units(TR["sp_none"], "CMC-enc"), "adapted(i) per-protein r", "unadapted", "r_across_KOs"),
          ptest(pp_units(TR["sp_i"], "CMC-enc"), pp_units(TR["sp_none"], "ctrlridge90"), "adapted(i) per-protein r", "ctrlridge90", "r_across_KOs")]
    panel(T4, "TR_perprotein")

# =============================================================== (b2) CITE-seq transfer (per-screen runs)
def load_transfer(lab, op):
    R = {}
    for t in TAGS:
        for s in SEEDS:
            f = f"{E}/transfer/{lab}/{op}_{t}_{s}.json"
            if not os.path.exists(f): continue
            js = json.load(open(f))
            if "main" not in js: continue
            R[(t, s)] = js
    return R
def arm_units(R, arm, metric, scope="overall", seeds=SEEDS, section="main"):
    """seed-mean per (cohort, gene)."""
    acc = {}
    for (t, s), js in R.items():
        if s not in seeds or (scope != "overall" and t != scope): continue
        sec = js.get(section, {})
        if arm not in sec: continue
        for r in sec[arm]["per_ko"]:
            v = r.get(metric)
            if v is None or not np.isfinite(v): continue
            acc.setdefault((r["cohort"], r["gene"]), []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}
def per_seed_arm(RA, armA, RB, armB, metric):
    out = []
    for s in SEEDS:
        a = arm_units(RA, armA, metric, seeds=[s]); b = arm_units(RB, armB, metric, seeds=[s]); ks = set(a) & set(b)
        out.append(float(np.mean([a[k] - b[k] for k in ks])) if ks else float("nan"))
    return out
TF = {(lab, op): load_transfer(lab, op) for lab in ["sp_none", "sp_i", "sp_ii", "sp_iii", "sp_i_xm1", "xm_none", "xm_i"] for op in ["clamp_zero", "cutclamp_zero"]}
TF = {k: v for k, v in TF.items() if v}
OUT["transfer_runs"] = {f"{k[0]}|{k[1]}": sorted(f"{t}:{s}" for t, s in v) for k, v in TF.items()}
ARMS = ["obs", "ftR", "ftR_shuf", "ftP", "ftP_shuf", "ftRP", "ftR+ctrl", "obs+ctrl", "ridgeR_ctrlcx+ctrl", "ridgeR_cx+ctrl", "zs_ctrlcorr+ctrl",
        "measRNA+ctrl", "measRNA+corr", "measRNA+cog", "ftR+corr", "ridgeR+corr", "ridgeP_cx", "ridgeP_ctrlcx", "ridgeP_fp", "ridgeR_ctrlcx+corr"]
MET = ["rp_pe_cent", "rp_retrieval", "rp_auroc", "rp_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc", "rr_pe"]
summ = {}
for (lab, op), R in TF.items():
    for arm in ARMS:
        for metric in MET:
            u = arm_units(R, arm, metric)
            if u: summ.setdefault(f"{lab}|{op}", {}).setdefault(arm, {})[metric] = dict(mean=float(np.mean(list(u.values()))), n=len(u))
OUT["transfer_summary"] = summ
if ("sp_i", "clamp_zero") in TF and ("sp_none", "clamp_zero") in TF:
    say("=== (b2) CITE-seq transfer, per-screen protocol, clamp (seed-mean; pooled over screens)")
    for key in ["sp_none|clamp_zero", "sp_i|clamp_zero", "sp_ii|clamp_zero", "sp_iii|clamp_zero", "sp_i_xm1|clamp_zero", "xm_none|clamp_zero", "xm_i|clamp_zero",
                "sp_none|cutclamp_zero", "sp_i|cutclamp_zero"]:
        if key not in summ: continue
        for arm in ["obs", "ftR", "ftR_shuf", "ftP", "ftR+ctrl"]:
            s = summ[key].get(arm, {})
            if s: say(f"  {key:22s} {arm:9s} " + " ".join(f"{m} {s[m]['mean']:+.3f}" for m in MET if m in s) + f" (n_rp={s.get('rp_retrieval', {}).get('n')}, n_rr={s.get('rr_retrieval', {}).get('n')})")
    for arm in ["ridgeR_ctrlcx+ctrl", "ridgeR_cx+ctrl", "zs_ctrlcorr+ctrl", "measRNA+ctrl", "measRNA+corr", "ridgeP_cx", "ridgeP_ctrlcx"]:
        s = summ["sp_none|clamp_zero"].get(arm, {})
        if s: say(f"  baseline {arm:20s} " + " ".join(f"{m} {s[m]['mean']:+.3f}" for m in MET if m in s))
    Ri, R0 = TF[("sp_i", "clamp_zero")], TF[("sp_none", "clamp_zero")]
    T = []
    for metric in ["rp_pe_cent", "rp_retrieval"]:
        a = arm_units(Ri, "ftR", metric)
        T.append(ptest(a, arm_units(Ri, "ftR_shuf", metric), "adapted ftR", "adapted ftR_shuf", metric, per_seed_arm(Ri, "ftR", Ri, "ftR_shuf", metric)))
        T.append(ptest(a, arm_units(R0, "ftR", metric), "adapted ftR", "unadapted ftR", metric, per_seed_arm(Ri, "ftR", R0, "ftR", metric)))
        T.append(ptest(a, arm_units(R0, "ridgeR_ctrlcx+ctrl", metric), "adapted ftR", "ridgeR_ctrlcx+ctrl", metric, per_seed_arm(Ri, "ftR", R0, "ridgeR_ctrlcx+ctrl", metric)))
    panel(T, "XM_primary")
    T = []
    for metric in ["rr_pe_cent", "rr_retrieval"]:
        a = arm_units(Ri, "ftR", metric)
        T.append(ptest(a, arm_units(R0, "ftR", metric), "adapted ftR", "unadapted ftR", metric, per_seed_arm(Ri, "ftR", R0, "ftR", metric)))
        T.append(ptest(a, arm_units(R0, "ridgeR_ctrlcx", metric) or arm_units(R0, "ridgeP_ctrlcx", metric), "adapted ftR", "ridgeR_ctrlcx", metric,
                       per_seed_arm(Ri, "ftR", R0, "ridgeP_ctrlcx", metric)))
    panel(T, "RNA_primary")
    # secondary: obs arm, other variants/bases, cutclamp, per-screen, measRNA reference, ftR - ftR+ctrl
    T = []
    for (lab, op), R in TF.items():
        if lab in ("sp_none", "xm_none"): continue
        ref = TF.get(("xm_none" if lab.startswith("xm") else "sp_none", op))
        for metric in ["rp_pe_cent", "rp_retrieval", "rp_auroc"]:
            a = arm_units(R, "ftR", metric)
            if (lab, op) != ("sp_i", "clamp_zero"):
                T.append(ptest(a, arm_units(R, "ftR_shuf", metric), f"{lab}|{op} ftR", "same ftR_shuf", metric))
                if ref: T.append(ptest(a, arm_units(ref, "ftR", metric), f"{lab}|{op} ftR", "unadapted ftR", metric))
            if ref: T.append(ptest(arm_units(R, "obs", metric), arm_units(ref, "obs", metric), f"{lab}|{op} obs", "unadapted obs", metric))
            T.append(ptest(a, arm_units(R, "ftR+ctrl", metric), f"{lab}|{op} ftR", "ftR(own RNA)+ctrl map", metric))
            T.append(ptest(a, arm_units(TF[("sp_none", "clamp_zero")], "measRNA+ctrl", metric), f"{lab}|{op} ftR", "measRNA+ctrl (ref)", metric))
    panel(T, "XM_secondary")
    T = []
    for (lab, op), R in TF.items():
        if lab in ("sp_none", "xm_none") or (lab, op) == ("sp_i", "clamp_zero"): continue
        ref = TF.get(("xm_none" if lab.startswith("xm") else "sp_none", op))
        for metric in ["rr_pe_cent", "rr_retrieval", "rr_auroc"]:
            a = arm_units(R, "ftR", metric)
            if ref: T.append(ptest(a, arm_units(ref, "ftR", metric), f"{lab}|{op} ftR", "unadapted ftR", metric))
            T.append(ptest(a, arm_units(TF[("sp_none", "clamp_zero")], "ridgeP_ctrlcx", metric), f"{lab}|{op} ftR", "ridgeR_ctrlcx", metric))
            if ref: T.append(ptest(arm_units(R, "obs", metric), arm_units(ref, "obs", metric), f"{lab}|{op} obs", "unadapted obs", metric))
    panel(T, "RNA_secondary")
    T = []
    for scope in TAGS:
        for metric in ["rp_pe_cent", "rp_retrieval", "rr_pe_cent", "rr_retrieval"]:
            T.append(ptest(arm_units(Ri, "ftR", metric, scope), arm_units(Ri, "ftR_shuf", metric, scope), f"adapted ftR [{scope}]", "adapted ftR_shuf", metric))
            T.append(ptest(arm_units(Ri, "ftR", metric, scope), arm_units(R0, "ftR", metric, scope), f"adapted ftR [{scope}]", "unadapted ftR", metric))
    panel(T, "XM_per_screen")
    T = []   # [exploratory, added after unblinding of RNA_primary: the ridge comparators fall below chance (0.37-0.41 retrieval,
             #  ridge collapses to the mean response and LOO centring anti-correlates it), so compare with zero-shot control co-expression]
    Z = "zs_ctrlcorr+ctrl"
    for metric in ["rr_pe_cent", "rr_retrieval", "rr_auroc"]:
        for lab_, arm_ in [("sp_i", "ftR"), ("sp_i", "obs"), ("sp_none", "obs"), ("sp_none", "ftR")]:
            R_ = TF[(lab_, "clamp_zero")]
            T.append(ptest(arm_units(R_, arm_, metric), arm_units(R0, Z, metric), f"{lab_} {arm_}", "zero-shot ctrl co-expression", metric))
        T.append(ptest(arm_units(Ri, "obs", metric), arm_units(R0, "obs", metric), "adapted obs", "unadapted obs", metric))
    for metric in ["rp_pe_cent", "rp_retrieval"]:
        T.append(ptest(arm_units(Ri, "ftR", metric), arm_units(R0, Z, metric), "adapted ftR", "zero-shot ctrl co-expr.+ctrl map", metric))
    panel(T, "RNA_vs_zeroshot_exploratory")
    T = []   # within-modality protein-label arm (ftP) adapted vs unadapted and vs shuffled
    for metric in ["rp_pe_cent", "rp_retrieval"]:
        T.append(ptest(arm_units(Ri, "ftP", metric), arm_units(Ri, "ftP_shuf", metric), "adapted ftP", "adapted ftP_shuf", metric))
        T.append(ptest(arm_units(Ri, "ftP", metric), arm_units(R0, "ftP", metric), "adapted ftP", "unadapted ftP", metric))
        T.append(ptest(arm_units(R0, "ftP", metric), arm_units(R0, "ftP_shuf", metric), "unadapted ftP", "unadapted ftP_shuf", metric))
    panel(T, "ftP_secondary")
    # protein-label ridge learning curve (references): how many protein-labelled KOs does ridge need to match adapted ftR?
    LC = {}
    for feat in ["cx", "ctrlcx"]:
        for n in [3, 6, 12, 25, 50, 1000000]:
            vals = {m: [] for m in ["rp_pe_cent", "rp_retrieval", "rp_auroc"]}
            for d in range(3):
                for m in vals:
                    u = arm_units(R0, f"ridgeP_{feat}|{n}|{d}", m, section="ridgelc")
                    if u: vals[m].append(float(np.mean(list(u.values()))))
            LC.setdefault(feat, {})[n] = {m: float(np.mean(v)) if v else None for m, v in vals.items()}
    OUT["ridgeP_learning_curve"] = LC
    say("  protein-label ridge LC (rp_retrieval):", {f: {n: (round(v["rp_retrieval"], 3) if v["rp_retrieval"] is not None else None) for n, v in d.items()} for f, d in LC.items()})

# =============================================================== (b3) GWCD4i -> SCITO
GW = {}
for lab in ["sp_none", "sp_i", "sp_ii", "sp_iii", "xm_none", "xm_i"]:
    for cond in ["Stim48hr", "Stim8hr"]:
        f = f"{E}/gwscito/{lab}/score_clamp_zero_{cond}.json"
        if os.path.exists(f): GW[(lab, cond)] = json.load(open(f))
if GW:
    say("=== (b3) GWCD4i -> SCITO (pre-registered metrics)")
    gs = {}
    for (lab, cond), js in sorted(GW.items()):
        for arm in ["obs", "ftR", "ftR_shuf"]:
            r = js["methods"][arm]
            gs[f"{lab}|{cond}|{arm}"] = dict(pair_auroc=r["pair_auroc"], pair_auroc_ci=r.get("pair_auroc_ci"), retrieval=r["retrieval"], wt_auroc=r["wt_auroc"],
                                             sign_acc=r["signed"]["sign_acc"], cog_up=r["cognate"]["frac_up"],
                                             per_seed=[js["methods"].get(f"{arm}{s}", {}).get("pair_auroc") for s in ["_s42", "_s1", "_s2"]])
            say(f"  {lab:8s} {cond:8s} {arm:8s} M1 {r['pair_auroc']:.3f} {r.get('pair_auroc_ci', '')} M2 {r['wt_auroc']:.3f} M4 {r['retrieval']:.3f} "
                f"M3 {r['signed']['sign_acc']:.3f} M5 {r['cognate']['frac_up']:.2f} | per-seed M1 {gs[f'{lab}|{cond}|{arm}']['per_seed']}")
        if lab == "sp_none":
            for b in ["ridgeP_cx|1000000", "zs_corr", "zs_cognate", "measRNA+corr", "ridgeR+corr"]:
                if b in js["methods"]: say(f"  baseline {cond} {b:18s} M1 {js['methods'][b]['pair_auroc']:.3f} M4 {js['methods'][b]['retrieval']:.3f}")
    OUT["gwscito_summary"] = gs
    tw = {}
    for (lab, cond), js in GW.items(): tw[f"{lab}|{cond}"] = js["tests_ftR_vs"]
    OUT["gwscito_within_tests"] = tw
    # primary panel GW: adapted FT - unadapted FT (compare file, batch-stratified bootstrap), adapted FT - adapted obs (score file)
    T = []
    for lab in ["sp_i"]:
        cf = f"{E}/gwscito/{lab}/compare_old_new_clamp_zero_Stim48hr.json"
        if os.path.exists(cf):
            C = json.load(open(cf))
            for metric, mname in [("pair_auroc", "M1"), ("retrieval", "M4")]:
                t = [x for x in C[metric] if x["new"] == "ftR" and x["old"] == "ftR"][0]
                T.append(dict(a=f"{lab} FT", b="unadapted FT", metric=mname, n=t.get("n", 1529), mean_a=t["new_val"], mean_b=t["old_val"], diff=t["diff"], ci95=t["ci95"], p=t["p"]))
        if (lab, "Stim48hr") in GW:
            for metric, mname in [("pair_auroc", "M1"), ("retrieval", "M4")]:
                t = [x for x in GW[(lab, "Stim48hr")]["tests_ftR_vs"][metric] if x["b"] == "obs"][0]
                ma = GW[(lab, "Stim48hr")]["methods"]["ftR"][metric]; mb = GW[(lab, "Stim48hr")]["methods"]["obs"][metric]
                T.append(dict(a=f"{lab} FT", b=f"{lab} obs", metric=mname, n=t.get("n", 1529), mean_a=ma, mean_b=mb, diff=t["diff"], ci95=t["ci95"], p=t["p"]))
    panel(T, "GW_primary")
    T = []
    for lab in ["sp_ii", "sp_iii", "xm_i", "sp_i"]:
        for cond in ["Stim48hr", "Stim8hr"]:
            if lab == "sp_i" and cond == "Stim48hr": continue
            cf = f"{E}/gwscito/{lab}/compare_old_new_clamp_zero_{cond}.json"
            if not os.path.exists(cf): continue
            C = json.load(open(cf))
            for metric, mname in [("pair_auroc", "M1"), ("retrieval", "M4")]:
                for arm in ["ftR", "obs"]:
                    t = [x for x in C[metric] if x["new"] == arm and x["old"] == arm][0]
                    T.append(dict(a=f"{lab} {cond} {arm}", b="unadapted", metric=mname, n=t.get("n", 1529), mean_a=t["new_val"], mean_b=t["old_val"], diff=t["diff"], ci95=t["ci95"], p=t["p"]))
    panel(T, "GW_secondary")

# =============================================================== (b4) GWCD4i learning curves (fixpert protocol)
def load_lc(lab, cond, size):
    U = {}; info = {}
    for s in SEEDS:
      for f in [f"{O}/fx/runs/lc_gw_{cond}_s{s[1:]}_{size}_{lab}.json", f"{O}/fx/runs/lc_gw_{cond}_s{s[1:]}_{size}_{lab}_cc.json"]:
        # main file = all methods except cutclamp (CPU, after the batched-solve hang); *_cc = cutclamp only (GPU)
        if not os.path.exists(f): continue
        js = json.load(open(f)); tg = js["test_genes"]
        for sub, d in js.items():
            if not (sub.startswith("sub") and sub[3:].isdigit()): continue
            for meth, r in d.items():
                if meth == "cutclamp" and not f.endswith("_cc.json"): continue
                for metric in ["rr_retrieval", "rr_pe_cent_canon"]:
                    for g, v in zip(tg, r["per_ko"][metric]):
                        if v is not None: U.setdefault((meth, metric), {}).setdefault(g, []).append(v)
                info.setdefault(meth, []).append(r["summary"])
    return {k: {g: float(np.mean(v)) for g, v in d.items()} for k, d in U.items()}
LCR = {}
for lab in ["sp_none", "sp_i", "sp_ii", "sp_iii"]:
    for size in ["5", "10", "25", "50", "100", "200", "all"]:
        u = load_lc(lab, "Stim8hr", size)
        if u: LCR[(lab, size)] = u
    u = load_lc(lab, "Stim48hr", "all")
    if u: LCR[(lab, "48all")] = u
if LCR:
    say("=== (b4) GWCD4i learning curves (retrieval / centred pearDE canonical; mean over subsets x seeds)")
    lcs = {}
    for (lab, size), u in sorted(LCR.items()):
        for (meth, metric), d in u.items():
            lcs.setdefault(f"{lab}|{size}", {}).setdefault(meth, {})[metric] = float(np.mean(list(d.values())))
    OUT["lc_summary"] = lcs
    for key in sorted(lcs):
        d = lcs[key]
        say(f"  {key:14s} " + " ".join(f"{m}:{d[m].get('rr_retrieval', np.nan):.3f}/{d[m].get('rr_pe_cent_canon', np.nan):+.3f}" for m in
                                    ["base_shift", "F_sym_shift_short_pm", "F_sym_global_lbfgs_ls", "cutclamp", "ridge_ntcorr", "ridge_dec", "linear_ae_ctrlpca", "corr_zeroshot"] if m in d))
    T = []
    for size in ["5", "10", "25"]:
        if ("sp_i", size) not in LCR: continue
        u = LCR[("sp_i", size)]; u0 = LCR.get(("sp_none", size), {})
        for meth in ["base_shift", "F_sym_shift_short_pm"]:
            T.append(ptest(u[(meth, "rr_retrieval")], u[("ridge_ntcorr", "rr_retrieval")], f"adapted {meth} n={size}", "ridge_ntcorr", "retrieval"))
    panel(T, "LC_primary")
    T = []
    for lab in ["sp_i", "sp_ii", "sp_iii"]:
        for size in ["5", "10", "25", "50", "100", "200", "all", "48all"]:
            if (lab, size) not in LCR or ("sp_none", size) not in LCR: continue
            u, u0 = LCR[(lab, size)], LCR[("sp_none", size)]
            for meth in ["base_shift", "F_sym_shift_short_pm", "F_sym_global_lbfgs_ls"]:
                for metric in ["rr_retrieval", "rr_pe_cent_canon"]:
                    if (meth, metric) in u and (meth, metric) in u0:
                        T.append(ptest(u[(meth, metric)], u0[(meth, metric)], f"{lab} {meth} n={size}", "unadapted", metric))
                    if lab == "sp_i" and size not in ("5", "10", "25") or (lab == "sp_i" and metric == "rr_pe_cent_canon"):
                        T.append(ptest(u[(meth, metric)], u[("ridge_ntcorr", metric)], f"{lab} {meth} n={size}", "ridge_ntcorr", metric))
    panel(T, "LC_secondary")

json.dump(OUT, open(f"{O}/summary.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else (o.item() if hasattr(o, "item") else str(o)))
open(f"{O}/digest.txt", "w").write("\n".join(DIG) + "\n")
print("wrote summary.json / digest.txt")
