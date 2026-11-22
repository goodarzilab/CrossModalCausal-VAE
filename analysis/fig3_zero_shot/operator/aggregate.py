"""Aggregate operator-workstream runs (3 seeds) into summary.json + source-data CSVs."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, os, glob, numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon
R = f"{_CMC_ROOT}"; A = f"{_CMC_RESULTS}/paper/analyses/operator"; RUNS = f"{A}/runs"
SEEDS = ["s42", "s1", "s2"]; TAGS = ["frangieh", "iccite", "arce"]; SC = ["overall"] + TAGS
def ld(p): return json.load(open(p)) if os.path.exists(p) else None
D34 = {s: ld(f"{RUNS}/op_d34_s{'_s42' if s == 's42' else '_' + s}.json") for s in SEEDS}
FT = {}
for s in SEEDS:
    for op in ["clamp_zero", "cutclamp_zero", "edgecut"]:
        j = ld(f"{RUNS}/op_ft2_{op}_s{'_s42' if s == 's42' else '_' + s}.json")
        if j and "fig5" in j:
            for k, v in j["fig5"].items(): FT.setdefault(k, {})[s] = v
def nz(x): return None if x is None or (isinstance(x, float) and x != x) else x
def msd(vals):
    v = np.array([x for x in vals if x is not None and x == x], float)
    return dict(mean=float(v.mean()) if len(v) else None, sd=float(v.std(ddof=1)) if len(v) > 1 else None, vals=[float(x) for x in v])
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q
OUT = dict(seeds=[s for s in SEEDS if D34[s]])
# ---------------- diagnostics
dg = {}
ok = [s for s in SEEDS if D34[s] and "diag" in D34[s]]
dg["eps_norms"] = {s: D34[s]["diag"]["eps_norms"] for s in ok}
dg["eps_zero_neg"] = {m: {s: dict(corr_rna_median=D34[s]["diag"][f"edgecut_eps_{m}"]["corr_with_full_rna_median"],
                                  per_cohort=D34[s]["diag"][f"edgecut_eps_{m}"]["per_cohort_corr_rna"],
                                  summary={sc: {k: D34[s]["diag"][f"edgecut_eps_{m}"]["summary"][sc].get(k) for k in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe", "rp_pe_cent"]} for sc in SC},
                                  full={sc: {k: D34[s]["diag"]["edgecut_full"]["summary"][sc].get(k) for k in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe", "rp_pe_cent"]} for sc in SC})
                         for s in ok} for m in ["zero", "neg"]}
dg["shift"] = {k: {s: dict(agg=D34[s]["diag"]["shift_contribution"][k]["agg"], overall=D34[s]["diag"]["shift_contribution"][k]["summary_overall"]) for s in ok}
               for k in D34[ok[0]]["diag"]["shift_contribution"]} if ok else {}
ont = {}
for opn in (D34[ok[0]]["diag"]["on_target"]["by_operator"] if ok else []):
    ont[opn] = {sc: {q: msd([D34[s]["diag"]["on_target"]["by_operator"][opn][sc].get(q) for s in ok]) for q in
                     ["frac_neg", "r_ctrl_xhat_z", "r_ctrl_mean_log", "rho_ctrl_mean_log", "r_meas_z", "rho_meas_z", "frac_sign_agree_meas", "median_dx_over_ctrl_dist_to_z0", "sign_test_p"]}
                 for sc in SC}
    for sc in SC: ont[opn][sc]["s42_detail"] = D34[ok[0]]["diag"]["on_target"]["by_operator"][opn][sc]
dg["on_target"] = ont; dg["measured_on_target"] = D34[ok[0]]["diag"]["on_target"]["measured"] if ok else None
dg["kd_vs_oe"] = {s: D34[s]["diag"]["on_target"]["kd_vs_oe_corr"] for s in ok}
OUT["diagnostics"] = dg
rows = []
for s in ok:
    for r in D34[s]["diag"]["on_target"]["per_ko"]: rows.append(dict(seed=s, **{k: v for k, v in r.items()}))
pd.DataFrame(rows).to_csv(f"{A}/source_data_ontarget_per_ko.csv", index=False)
# ---------------- Fig 3
K3 = ["rr_auroc", "rr_auroc_meanresp", "rr_pe", "rr_pe_cent", "rr_pe_meanresp", "rr_pe_shuf", "rr_retrieval",
      "rp_auroc", "rp_auroc_meanresp", "rp_pe", "rp_pe_cent", "rp_pe_meanresp", "rp_pe_shuf", "rp_retrieval"]
ok3 = [s for s in SEEDS if D34[s] and "fig3" in D34[s]]
f3 = {}
for opn in (D34[ok3[0]]["fig3"] if ok3 else []):
    f3[opn] = {sc: {k: msd([D34[s]["fig3"][opn]["summary"][sc].get(k) for s in ok3]) for k in K3} for sc in SC}
    f3[opn]["similarity_s42"] = D34[ok3[0]]["fig3"][opn]["similarity"]
    if "paired_vs_edgecut" in D34[ok3[0]]["fig3"][opn]:
        f3[opn]["paired_vs_edgecut_s42"] = D34[ok3[0]]["fig3"][opn]["paired_vs_edgecut"]
rc = ld(f"{_CMC_RESULTS}/reviewer_controls/rc_sp_dec_gl1000.json")
f3["_corr_baseline_s42"] = {sc: {k: rc["fig3_baselines"]["corr"]["summary"][sc].get(k) for k in K3} for sc in SC}
OUT["fig3"] = f3
rows = []
for s in ok3:
    for opn, v in D34[s]["fig3"].items():
        for r in v["per_ko"]: rows.append(dict(seed=s, operator=opn, **{k: r.get(k) for k in ["cohort", "gene", "dx_target"] + K3}))
pd.DataFrame(rows).to_csv(f"{A}/source_data_fig3_per_knockout.csv", index=False)
# ---------------- Fig 4
ok4 = [s for s in SEEDS if D34[s] and "fig4" in D34[s]]
f4 = {"corr_Arr_ArrT": {s: D34[s]["fig4"]["corr_Arr_ArrT"] for s in ok4}}
K4 = ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval"]
for opn in [k for k in (D34[ok4[0]]["fig4"] if ok4 else {}) if k != "corr_Arr_ArrT"]:
    f4[opn] = {}
    for vn in D34[ok4[0]]["fig4"][opn]:
        f4[opn][vn] = {sc: {k: msd([D34[s]["fig4"][opn][vn]["summary"][sc].get(k) for s in ok4]) for k in K4} for sc in ["overall"] + TAGS}
        if vn != "learned":
            f4[opn][vn]["paired_s"] = {s: {m: D34[s]["fig4"][opn][vn]["paired_variant_minus_learned"][m].get("overall") for m in K4} for s in ok4}
OUT["fig4"] = f4
# ---------------- Fig 5
FK = ["rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval", "rr_pe_meanresp", "rp_pe_meanresp", "dx_target"]
f5 = {}; rows = []
for key, bys in FT.items():
    seeds = sorted(bys); e = dict(seeds=seeds, arms={}, tests_pooled={}, tests_per_seed={})
    arms = [a for a in ["baseline", "real", "shuffle_avg", "shuffle_0", "shuffle_1", "shuffle_2", "mean", "infonce"] if all(a in bys[s] for s in seeds)]
    for a in arms:
        e["arms"][a] = {sc: {k: msd([bys[s][a]["summary"][sc].get(k) for s in seeds]) for k in FK} for sc in SC}
        e["arms"][a]["info"] = {s: bys[s][a].get("info") for s in seeds}
        for s in seeds:
            for r in bys[s][a]["per_ko"]: rows.append(dict(seed=s, setting=key, arm=a, **{k: r.get(k) for k in ["cohort", "gene"] + FK}))
    # pooled across seeds: average each KO over seeds, then two-sided paired tests
    for a, b in [("real", "shuffle_avg"), ("real", "mean"), ("real", "baseline"), ("infonce", "shuffle_avg"), ("infonce", "baseline"), ("shuffle_avg", "baseline")]:
        if a not in arms or b not in arms: continue
        res = {}; ps = []
        for m in ["rr_pe", "rr_pe_cent", "rr_retrieval", "rp_pe", "rp_pe_cent", "rp_retrieval"]:
            for sc in SC:
                da, db = {}, {}
                for s in seeds:
                    for r in bys[s][a]["per_ko"]:
                        if sc == "overall" or r["cohort"] == sc: da.setdefault((r["cohort"], r["gene"]), []).append(r.get(m))
                    for r in bys[s][b]["per_ko"]:
                        if sc == "overall" or r["cohort"] == sc: db.setdefault((r["cohort"], r["gene"]), []).append(r.get(m))
                pr = []
                for k in da:
                    if k in db:
                        x = [v for v in da[k] if v is not None and v == v]; y = [v for v in db[k] if v is not None and v == v]
                        if x and y: pr.append((np.mean(x), np.mean(y)))
                pr = np.array(pr).reshape(-1, 2)
                if len(pr) < 3 or np.allclose(pr[:, 0], pr[:, 1]): res[f"{m}|{sc}"] = dict(n=len(pr)); continue
                d = pr[:, 0] - pr[:, 1]; rng = np.random.RandomState(0)
                bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(4000)]
                res[f"{m}|{sc}"] = dict(n=len(pr), mean_a=float(pr[:, 0].mean()), mean_b=float(pr[:, 1].mean()), diff=float(d.mean()),
                                        ci95_boot=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                        p_t=float(ttest_rel(pr[:, 0], pr[:, 1]).pvalue), p_wilcoxon=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
        keys = [k for k in res if "p_t" in res[k] and k.endswith("|overall")]
        q = bh([res[k]["p_t"] for k in keys])
        for k, qq in zip(keys, q): res[k]["q_bh_overall_panel"] = float(qq)
        e["tests_pooled"][f"{a}_vs_{b}"] = res
        e["tests_per_seed"][f"{a}_vs_{b}"] = {s: {m: bys[s]["tests"].get(f"{a}_vs_{b}", {}).get(m, {}).get("overall") for m in ["rr_pe", "rr_pe_cent", "rr_retrieval", "rp_pe", "rp_pe_cent"]} for s in seeds if "tests" in bys[s]}
    f5[key] = e
OUT["fig5"] = f5
pd.DataFrame(rows).to_csv(f"{A}/source_data_fig5_per_knockout.csv", index=False)
# ---------------- synthetic
sy = ld(f"{A}/synthetic_operator.json")
if sy:
    OUT["synthetic"] = dict(config=sy["config"], results=[dict(scenario=r["scenario"], summary=r["summary"]) for r in sy["results"]])
# ---------------- pseudobulk-DE rescoring (stats workstream DE; OP_DE=pb_bh)
PB = {s: ld(f"{RUNS}/op_pb_s{'_s42' if s == 's42' else '_' + s}.json") for s in SEEDS}
okp = [s for s in SEEDS if PB[s] and "fig3" in PB[s]]
if okp:
    pbo = dict(seeds=okp, fig3={}, on_target={}, fig5={})
    for opn in PB[okp[0]]["fig3"]:
        pbo["fig3"][opn] = {sc: {k: msd([PB[s]["fig3"][opn]["summary"][sc].get(k) for s in okp]) for k in K3} for sc in SC}
    if "diag" in PB[okp[0]]:
        for opn, v in PB[okp[0]]["diag"]["on_target"]["by_operator"].items():
            pbo["on_target"][opn] = {sc: {q: msd([PB[s]["diag"]["on_target"]["by_operator"][opn][sc].get(q) for s in okp if "diag" in PB[s]]) for q in ["frac_neg", "r_ctrl_xhat_z", "r_meas_z", "rho_meas_z"]} for sc in SC}
        pbo["measured_on_target"] = PB[okp[0]]["diag"]["on_target"]["measured"]
    okf = [s for s in okp if "fig5" in PB[s]]
    for key in (PB[okf[0]]["fig5"] if okf else {}):
        bys = {s: PB[s]["fig5"][key] for s in okf if key in PB[s]["fig5"]}; e = dict(seeds=sorted(bys), arms={}, tests_pooled={})
        for a in ["baseline", "real", "shuffle_avg", "mean", "infonce"]:
            if all(a in bys[s] for s in bys): e["arms"][a] = {sc: {k: msd([bys[s][a]["summary"][sc].get(k) for s in bys]) for k in FK} for sc in SC}
        for a, b in [("real", "shuffle_avg"), ("infonce", "shuffle_avg"), ("real", "baseline")]:
            res = {}
            for m in ["rr_pe", "rr_pe_cent", "rr_retrieval", "rp_pe", "rp_pe_cent"]:
                da, db = {}, {}
                for s in bys:
                    if a not in bys[s] or b not in bys[s]: continue
                    for r in bys[s][a]["per_ko"]: da.setdefault((r["cohort"], r["gene"]), []).append(r.get(m))
                    for r in bys[s][b]["per_ko"]: db.setdefault((r["cohort"], r["gene"]), []).append(r.get(m))
                pr = np.array([(np.nanmean([x for x in da[k] if x is not None]), np.nanmean([x for x in db[k] if x is not None])) for k in da if k in db
                               if any(x is not None and x == x for x in da[k]) and any(x is not None and x == x for x in db[k])]).reshape(-1, 2)
                if len(pr) >= 3 and not np.allclose(pr[:, 0], pr[:, 1]):
                    d = pr[:, 0] - pr[:, 1]; rng = np.random.RandomState(0); bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(4000)]
                    res[m] = dict(n=len(pr), diff=float(d.mean()), ci95_boot=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p_t=float(ttest_rel(pr[:, 0], pr[:, 1]).pvalue))
            e["tests_pooled"][f"{a}_vs_{b}"] = res
        pbo["fig5"][key] = e
    OUT["pseudobulk_de"] = pbo
json.dump(OUT, open(f"{A}/summary.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else str(o))
print("wrote summary.json; fig5 settings:", list(f5), "seeds d34:", ok)
