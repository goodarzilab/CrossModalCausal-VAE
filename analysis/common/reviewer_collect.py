"""Aggregate reviewer_controls outputs across the three base-model seeds (42/1/2) and export Source Data.
Writes results/reviewer_controls/summary.json + source_data_*.csv and prints the tables used in the response."""
import json, os, numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon, ttest_1samp

D = "results/reviewer_controls"
MODELS = {"42": "sp_dec_gl1000", "1": "sp_dec_gl1000_s1", "2": "sp_dec_gl1000_s2"}
TAGS = ["frangieh", "iccite", "arce"]; SC = ["overall"] + TAGS
R = {s: json.load(open(f"{D}/rc_{m}.json")) for s, m in MODELS.items() if os.path.exists(f"{D}/rc_{m}.json")}
print("models found:", list(R))
OUT = {}

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q

def per_ko_df(key, seed):
    return pd.DataFrame(R[seed][key]["per_ko"]).assign(seed=seed)

def seedavg(key, metric):
    """per-KO metric averaged over the available seed models (KO sets are identical across seeds)."""
    dfs = [per_ko_df(key, s)[["cohort", "gene", metric]].rename(columns={metric: s}) for s in R if key in R[s]]
    m = dfs[0]
    for d in dfs[1:]: m = m.merge(d, on=["cohort", "gene"], how="outer")
    m[metric] = m[[s for s in R if s in m]].mean(1, skipna=True); return m[["cohort", "gene", metric]]

def paired_tests(key_a, key_b, metric, alt="greater"):
    a = seedavg(key_a, metric).rename(columns={metric: "a"}); b = seedavg(key_b, metric).rename(columns={metric: "b"})
    m = a.merge(b, on=["cohort", "gene"]).dropna()
    res = {}
    for sc in SC:
        x = m if sc == "overall" else m[m.cohort == sc]
        if len(x) < 3: res[sc] = dict(n=len(x)); continue
        d = x.b - x.a
        res[sc] = dict(n=int(len(x)), mean_a=float(x.a.mean()), mean_b=float(x.b.mean()), diff=float(d.mean()),
                       p_t=float(ttest_rel(x.b, x.a, alternative=alt).pvalue),
                       p_w=float(wilcoxon(d, alternative=alt).pvalue) if (d != 0).any() else 1.0)
    q = bh([res[s].get("p_t", 1.0) for s in SC])
    for s, qq in zip(SC, q): res[s]["q_t_BH"] = float(qq)
    return res

def mean_sd(key, metric, sc="overall"):
    v = np.array([R[s][key]["summary"][sc][metric] for s in R if key in R[s]], float)
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0, [float(x) for x in v]

def fmt(t): return f"{t[0]:+.3f}±{t[1]:.3f}"

# ---------- 1. Fig 5 headline: per-model, 3-model mean±SD, paired tests on seed-averaged per-KO scores
print("\n=== Fig 5 (held-out genes) — orig scale; per seed + mean±SD ===")
F5 = {}
for blk in ["rr", "rp"]:
    for sc in SC:
        b = mean_sd("fig5_baseline", f"{blk}_pe", sc); t = mean_sd("ft_standard", f"{blk}_pe", sc)
        F5[f"{blk}_{sc}"] = dict(baseline=b, tuned=t, n=R["42"]["ft_standard"]["summary"][sc][f"n_{blk}_pe"])
        print(f"  {blk} {sc:9s} n={F5[f'{blk}_{sc}']['n']:3d}  baseline {fmt(b)} {b[2]}  ->  tuned {fmt(t)} {t[2]}")
F5["paired_seedavg"] = {f"{blk}_pe": paired_tests("fig5_baseline", "ft_standard", f"{blk}_pe") for blk in ["rr", "rp"]}
OUT["fig5"] = F5

# ---------- 2. scale robustness (Fig 5 and Fig 3)
print("\n=== Response-scale check (3-model mean) ===")
SCL = {}
for key in ["fig3_obs", "fig5_baseline", "ft_standard", "ft_zscale"]:
    for blk in ["rr", "rp"]:
        for v in ["pe", "pe_z", "pe_log", "auroc", "auroc_log"]:
            mtr = f"{blk}_{v}"
            if mtr not in R["42"][key]["summary"]["overall"]: continue
            SCL[f"{key}|{mtr}"] = {sc: mean_sd(key, mtr, sc)[:2] for sc in SC}
for key in ["fig3_obs", "fig5_baseline", "ft_standard", "ft_zscale"]:
    for blk in ["rr", "rp"]:
        print(f"  {key:14s} {blk}: " + "  ".join(f"{v}={SCL[f'{key}|{blk}_{v}']['overall'][0]:+.3f}" for v in ["pe", "pe_z", "pe_log", "auroc", "auroc_log"] if f"{key}|{blk}_{v}" in SCL))
SCL["paired_ft_vs_base"] = {m: paired_tests("fig5_baseline", "ft_standard", m) for m in ["rr_pe_z", "rr_pe_log", "rp_pe_z", "rp_pe_log"]}
SCL["paired_ftz_vs_base"] = {m: paired_tests("fig5_baseline", "ft_zscale", m) for m in ["rr_pe", "rr_pe_z", "rr_pe_log", "rp_pe_z", "rp_pe_log"]}
OUT["scale"] = SCL

# ---------- 3. shared-response controls
print("\n=== Shared-response controls (held-out, 3-model mean; overall) ===")
CT = {}
for key in ["fig5_baseline", "ft_standard", "ft_zscale", "ft_labelshuffle_0", "ft_labelshuffle_1", "ft_labelshuffle_2",
            "ft_signflip", "ft_meantarget", "ft_nested"]:
    if key not in R["42"]: continue
    row = {}
    for m in ["rr_pe", "rr_pe_cent", "rr_pe_cent_z", "rr_pe_shuf", "rr_pe_meanresp", "rr_pe_meanpred", "rp_pe", "rp_pe_cent",
              "rp_pe_shuf", "rp_pe_meanresp", "rp_pe_nocog", "rr_auroc", "rp_auroc"]:
        row[m] = {sc: mean_sd(key, m, sc)[:2] for sc in SC}
    row["structure"] = {k: [R[s][key]["structure"].get(k) for s in R if key in R[s]] for k in
                        ["corr_Arr_ArrT", "corr_Grr_GrrT", "collectri_fwd_over_rev_median", "collectri_frac_fwd_gt_rev", "rel_change_A"]}
    row["specificity"] = {k: [R[s][key].get("specificity_matched_minus_shuffled", {}).get(k, {}).get("mean") for s in R if key in R[s]]
                          for k in ["rr_overall", "rp_overall"] + [f"rr_{t}" for t in TAGS] + [f"rp_{t}" for t in TAGS]}
    CT[key] = row
    o = lambda m: f"{row[m]['overall'][0]:+.3f}"
    print(f"  {key:18s} RR pe {o('rr_pe')} cent {o('rr_pe_cent')} shuf {o('rr_pe_shuf')} meanresp {o('rr_pe_meanresp')} | "
          f"RP pe {o('rp_pe')} cent {o('rp_pe_cent')} shuf {o('rp_pe_shuf')} meanresp {o('rp_pe_meanresp')} nocog {o('rp_pe_nocog')} | "
          f"Gsym {np.round(row['structure']['corr_Grr_GrrT'],4)} CTRI {np.round(row['structure']['collectri_fwd_over_rev_median'],3)}")
CT["paired_cent"] = {m: paired_tests("fig5_baseline", "ft_standard", m) for m in ["rr_pe_cent", "rr_pe_cent_z", "rp_pe_cent", "rp_pe_nocog"]}
CT["paired_ft_vs_meantarget"] = {m: paired_tests("ft_meantarget", "ft_standard", m, alt="two-sided") for m in ["rr_pe", "rr_pe_cent", "rp_pe", "rp_pe_cent"]}
CT["paired_ft_vs_labelshuffle0"] = {m: paired_tests("ft_labelshuffle_0", "ft_standard", m, alt="two-sided") for m in ["rr_pe", "rr_pe_cent", "rp_pe", "rp_pe_cent"]}
# matched vs mean-response predictor (does the tuned model beat simply predicting the cohort mean response?)
CT["tuned_vs_meanresp"] = {}
for blk in ["rr", "rp"]:
    a = seedavg("ft_standard", f"{blk}_pe_meanresp").rename(columns={f"{blk}_pe_meanresp": "a"})
    b = seedavg("ft_standard", f"{blk}_pe").rename(columns={f"{blk}_pe": "b"}); m = a.merge(b, on=["cohort", "gene"]).dropna()
    CT["tuned_vs_meanresp"][blk] = {sc: (lambda x: dict(n=len(x), meanresp=float(x.a.mean()), tuned=float(x.b.mean()),
                                                         p_two=float(ttest_rel(x.b, x.a).pvalue)) if len(x) > 2 else {})
                                    (m if sc == "overall" else m[m.cohort == sc]) for sc in SC}
OUT["controls"] = CT

# ---------- 4. learning curve (seed 42 model has the curve; others too)
print("\n=== Learning curve (overall held-out; mean over subsets; per seed model) ===")
LC = {}
for s in R:
    if "learning_curve" not in R[s]: continue
    for size, v in R[s]["learning_curve"].items():
        LC.setdefault(size, {})[s] = {m: v["agg"][m] for m in ["rr_pe", "rr_pe_cent", "rr_pe_shuf", "rp_pe", "rp_pe_cent", "rr_auroc"]}
        LC[size][s]["Gsym"] = v["agg"]["corr_Grr_GrrT"]["mean"]; LC[size][s]["ctri"] = v["agg"]["collectri_fwd_over_rev_median"]["mean"]
for size in LC:
    ss = LC[size]
    print(f"  k={size:>4}: " + " | ".join(f"s{s} RR {ss[s]['rr_pe']['mean']:+.3f}±{ss[s]['rr_pe']['sd']:.3f} cent {ss[s]['rr_pe_cent']['mean']:+.3f} "
                                          f"shuf {ss[s]['rr_pe_shuf']['mean']:+.3f} RP {ss[s]['rp_pe']['mean']:+.3f} cent {ss[s]['rp_pe_cent']['mean']:+.3f}" for s in ss))
OUT["learning_curve"] = LC

# ---------- 5. Fig 3 observational: scale + baselines with knockdown signs, paired
if "fig3_baselines" in R["42"]:
    print("\n=== Fig 3 (all evaluable): CMC-VAE vs baselines (knockdown-consistent signs), seed-42 model ===")
    FB = {}
    cmc = R["42"]["fig3_obs"]["summary"]
    for name, v in R["42"]["fig3_baselines"].items():
        s = v["summary"]; FB[name] = dict(summary={sc: {m: s[sc].get(m) for m in ["rr_auroc", "rr_pe", "rr_pe_z", "rp_auroc", "rp_pe", "rp_pe_z", "rp_pe_meanresp"]} for sc in SC},
                                          paired=v["paired_vs_cmc"])
        print(f"  {name:20s} " + " ".join(f"{t}: RRau {s[t]['rr_auroc']:.3f} RRpe {s[t]['rr_pe']:+.3f} RPau {s[t]['rp_auroc']:.3f} RPpe {s[t]['rp_pe']:+.3f}" for t in TAGS))
    print(f"  {'CMC-VAE':20s} " + " ".join(f"{t}: RRau {cmc[t]['rr_auroc']:.3f} RRpe {cmc[t]['rr_pe']:+.3f} RPau {cmc[t]['rp_auroc']:.3f} RPpe {cmc[t]['rp_pe']:+.3f}" for t in TAGS))
    print("  CMC-VAE extra (Fig3 set): " + " ".join(f"{t}: RP meanresp {cmc[t]['rp_pe_meanresp']:+.3f} RP nocog {cmc[t]['rp_pe_nocog']:+.3f} RR meanresp {cmc[t]['rr_pe_meanresp']:+.3f} RR cent {cmc[t]['rr_pe_cent']:+.3f} RP cent {cmc[t]['rp_pe_cent']:+.3f}" for t in TAGS))
    FB["cmc"] = {sc: cmc[sc] for sc in SC}
    OUT["fig3_baselines"] = FB
    # source data
    rows = []
    for k in R["42"]["fig3_obs"]["per_ko"]: rows.append(dict(model="CMC-VAE", **k))
    for name, v in R["42"]["fig3_baselines"].items():
        for k in v["per_ko"]: rows.append(dict(model=name, **k))
    pd.DataFrame(rows).to_csv(f"{D}/source_data_fig3_per_knockout.csv", index=False)

# ---------- 6. BH-FDR DE
if "bh" in R["42"]:
    print("\n=== BH-FDR DE ground truth ===")
    BH = {}
    for s in R:
        b = R[s].get("bh")
        if not b: continue
        BH[s] = dict(n_units=b["n_units"], n_test=b["n_test"], fig3=b["fig3_obs"]["overall"], base=b["fig5_baseline"]["overall"], ft=b["ft"]["overall"],
                     per=dict(fig3={t: b["fig3_obs"][t] for t in TAGS}, base={t: b["fig5_baseline"][t] for t in TAGS}, ft={t: b["ft"][t] for t in TAGS}),
                     paired=b["paired"])
        print(f"  seed {s}: n_units {b['n_units']} test {b['n_test']} | Fig3 RRau {b['fig3_obs']['overall']['rr_auroc']:.3f} RPpe {b['fig3_obs']['overall']['rp_pe']:+.3f} | "
              f"Fig5 RR {b['fig5_baseline']['overall']['rr_pe']:+.3f}->{b['ft']['overall']['rr_pe']:+.3f} (cent {b['fig5_baseline']['overall']['rr_pe_cent']:+.3f}->{b['ft']['overall']['rr_pe_cent']:+.3f}) "
              f"RP {b['fig5_baseline']['overall']['rp_pe']:+.3f}->{b['ft']['overall']['rp_pe']:+.3f}")
    OUT["bh"] = BH

# ---------- 7. nested
if "ft_nested" in R["42"]:
    OUT["nested"] = {s: dict(selected=R[s]["ft_nested"]["selected"], heldout=R[s]["ft_nested"]["summary"]["overall"]) for s in R if "ft_nested" in R[s]}
    for s, v in OUT["nested"].items():
        print(f"  nested seed {s}: selected w_dir={v['selected']['w_dir']} w_mag={v['selected']['w_mag']} -> held-out RR {v['heldout']['rr_pe']:+.3f} RP {v['heldout']['rp_pe']:+.3f}")

# ---------- source data, Fig 5
rows = []
for s in R:
    for key, lab in [("fig5_baseline", "observational"), ("ft_standard", "fine-tuned")]:
        for k in R[s][key]["per_ko"]: rows.append(dict(base_model_seed=s, condition=lab, **k))
pd.DataFrame(rows).to_csv(f"{D}/source_data_fig5_per_knockout.csv", index=False)
json.dump(OUT, open(f"{D}/summary.json", "w"), indent=1, default=str)
print("\nwrote", f"{D}/summary.json", "and source_data_*.csv")
