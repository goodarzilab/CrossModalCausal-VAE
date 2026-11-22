"""Aggregate paper_nov_revisions/analyses/perturbseq/runs/*.json -> tables.md + summary.json (seed-level statistics).

Conventions: two-sided paired tests; BH within panel; effect = mean difference over held-out KOs with 95% CIs from
(i) a bootstrap over KOs of the seed-averaged per-KO differences and (ii) a hierarchical bootstrap (seeds, then KOs).
Seed-level: per-seed mean differences, sign consistency, paired t-test across seeds (n = 3; low power, reported as is).
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, os, glob, numpy as np
from scipy.stats import wilcoxon, ttest_rel, ttest_1samp
from statsmodels.stats.multitest import multipletests

AN = f"{_CMC_RESULTS}/paper/analyses/perturbseq"; R = f"{AN}/runs"
LINES = ["jurkat", "k562"]; PRES = ["obs", "ctrl", "ctrl_ep1000"]; SEEDS = [42, 1, 2]
def J(f):
    try: return json.load(open(f))
    except Exception: return None
def runs(line, pre, suffix=""):
    return {s: r for s, r in ((s, J(f"{R}/{line}_{pre}_s{s}{suffix}.json")) for s in SEEDS) if r}

def kovec(rows, m):
    return {r["gene"]: r.get(m) for r in rows if r.get(m) is not None and np.isfinite(r.get(m))}

def seed_contrast(rows_a_by_seed, rows_b_by_seed, m, rng=np.random.RandomState(0), nb=4000):
    """b - a. rows_*_by_seed: {seed: per-KO rows}. Returns per-seed stats + seed-averaged + hierarchical bootstrap."""
    seeds = [s for s in rows_a_by_seed if s in rows_b_by_seed]
    per = {}; Dmat = []
    genes = None
    for s in seeds:
        a, b = kovec(rows_a_by_seed[s], m), kovec(rows_b_by_seed[s], m); g = sorted(set(a) & set(b))
        genes = g if genes is None else sorted(set(genes) & set(g))
    for s in seeds:
        a, b = kovec(rows_a_by_seed[s], m), kovec(rows_b_by_seed[s], m)
        d = np.array([b[g] - a[g] for g in genes]); Dmat.append(d)
        per[s] = dict(mean_a=float(np.mean([a[g] for g in genes])), mean_b=float(np.mean([b[g] for g in genes])), mean_diff=float(d.mean()),
                      p_wilcoxon=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
    Dmat = np.array(Dmat); n = Dmat.shape[1]; dbar = Dmat.mean(0)
    bs = dbar[rng.randint(0, n, (nb, n))].mean(1)
    hb = np.array([Dmat[rng.randint(0, len(seeds), len(seeds))][:, rng.randint(0, n, n)].mean() for _ in range(1000)])
    sd = Dmat.mean(1)
    out = dict(n_ko=n, n_seeds=len(seeds), per_seed=per, mean_diff=float(dbar.mean()),
               ci95_ko=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               ci95_hier=[float(np.percentile(hb, 2.5)), float(np.percentile(hb, 97.5))],
               p_wilcoxon_seedavg=float(wilcoxon(dbar).pvalue) if np.any(dbar != 0) else 1.0,
               p_t_seedavg=float(ttest_1samp(dbar, 0).pvalue),
               seed_diffs=[float(x) for x in sd], sign_consistent=bool(np.all(sd > 0) or np.all(sd < 0)),
               p_t_seedlevel=float(ttest_1samp(sd, 0).pvalue) if len(sd) > 1 and np.std(sd) > 0 else None,
               max_p_per_seed=float(max(v["p_wilcoxon"] for v in per.values())),
               mean_a=float(np.mean([v["mean_a"] for v in per.values()])), mean_b=float(np.mean([v["mean_b"] for v in per.values()])),
               sd_a=float(np.std([v["mean_a"] for v in per.values()], ddof=1)) if len(per) > 1 else 0.0,
               sd_b=float(np.std([v["mean_b"] for v in per.values()], ddof=1)) if len(per) > 1 else 0.0)
    return out

def bh(panel):
    """panel: {name: contrast dict with p_wilcoxon_seedavg} -> add q_bh."""
    ks = [k for k in panel if panel[k] and "p_wilcoxon_seedavg" in panel[k]]
    if not ks: return panel
    q = multipletests([panel[k]["p_wilcoxon_seedavg"] for k in ks], method="fdr_bh")[1]
    for k, v in zip(ks, q): panel[k]["q_bh"] = float(v)
    return panel

def ftrows(r, arm):
    F = r.get("ft", {}).get("shift", {})
    if arm == "obs": return r["obs"]["cmc_shift"]["per_ko"]
    return F.get(arm, {}).get("per_ko")

ARMS = ["obs", "real", "shuffle_avg", "mean", "centered", "centered_shuffle_avg", "offset", "offset_shuffle_avg", "infonce", "infonce_shuffle_avg",
        "centered_infonce", "centered_infonce_shuffle_avg"]
CONTR = [("shuffle_avg", "real"), ("obs", "real"), ("infonce_shuffle_avg", "infonce"), ("centered_shuffle_avg", "centered"),
         ("centered_infonce_shuffle_avg", "centered_infonce"), ("obs", "infonce")]
MET = ["rr_retrieval", "rr_pe_cent", "rr_pe_cent_canon", "rr_pcor", "rr_systema_cacc", "rr_systema_cacc_hvg", "rr_pe"]
S = dict(ft={}, ft_contrasts={}, ctrl_vs_obs={}, nb={}, ds={}, lc={}, xctx2={}, data={}, latent={})
md = []
def f3(x, sgn=True): return "-" if x is None else (f"{x:+.3f}" if sgn else f"{x:.3f}")
def pfmt(p): return "-" if p is None else f"{p:.1g}"

# ------------------------------------------------ fine-tuning arms (Fig 5e) per line x pretraining
for line in LINES:
    for pre in PRES:
        rs = runs(line, pre); rs = {s: r for s, r in rs.items() if "ft" in r}
        if not rs: continue
        key = f"{line}_{pre}"; r0 = next(iter(rs.values())); S["data"][key] = r0["data"]
        S["latent"][key] = dict(latent=192, n_adjacency_positions=192 * 192 - 192, n_scm_params=2 * 192 * 192)
        tab = {}
        for arm in ARMS:
            vals = {m: [r["obs"]["cmc_shift"]["summary"][m] if arm == "obs" else r["ft"]["shift"].get(arm, {}).get("summary", {}).get(m) for r in rs.values()] for m in MET}
            tab[arm] = {m: dict(mean=float(np.mean([v for v in vals[m] if v is not None])) if any(v is not None for v in vals[m]) else None,
                                sd=float(np.std([v for v in vals[m] if v is not None], ddof=1)) if sum(v is not None for v in vals[m]) > 1 else None,
                                per_seed=vals[m]) for m in MET}
        S["ft"][key] = dict(seeds=list(rs), arms=tab)
        md.append(f"\n### Fine-tuning, {line}, pretraining = {pre} (seeds {list(rs)}; n held-out = {r0['data']['n_test']}; mean ± s.d. across seeds)\n"
                  "| arm | retrieval | centred pearDE | centred pearDE (canonical) | partial r(p,o|m) | Systema cacc | Systema cacc (HVG) | raw pearDE |\n|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            t = tab[arm]
            md.append(f"| {arm} | " + " | ".join("-" if t[m]["mean"] is None else f"{t[m]['mean']:.3f} ± {t[m]['sd'] or 0:.3f}" for m in MET) + " |")
        panel = {}
        for a, b in CONTR:
            ra = {s: ftrows(r, a) for s, r in rs.items()}; rb = {s: ftrows(r, b) for s, r in rs.items()}
            if any(v is None for v in list(ra.values()) + list(rb.values())): continue
            for m in ["rr_retrieval", "rr_pe_cent", "rr_pe_cent_canon", "rr_pcor"]:
                panel[f"{b}_vs_{a}|{m}"] = seed_contrast(ra, rb, m)
        bh(panel); S["ft_contrasts"][key] = panel
        md.append("\nSeed-level contrasts (b − a; two-sided; seed-averaged per-KO Wilcoxon, BH within panel; per-seed max p; seed-level t over 3 seed means)\n"
                  "| contrast | metric | Δ mean | 95% CI (KO boot) | 95% CI (hier.) | per-seed Δ | P (seed-avg) | q BH | max P per seed | P seed-level t |\n|---|---|---|---|---|---|---|---|---|---|")
        for k, v in panel.items():
            c, m = k.split("|")
            md.append(f"| {c} | {m.replace('rr_','')} | {v['mean_diff']:+.3f} | [{v['ci95_ko'][0]:+.3f}, {v['ci95_ko'][1]:+.3f}] | [{v['ci95_hier'][0]:+.3f}, {v['ci95_hier'][1]:+.3f}] | "
                      f"{', '.join(f'{x:+.3f}' for x in v['seed_diffs'])} | {pfmt(v['p_wilcoxon_seedavg'])} | {pfmt(v.get('q_bh'))} | {pfmt(v['max_p_per_seed'])} | {pfmt(v['p_t_seedlevel'])} |")

# ------------------------------------------------ control-only vs original pretraining (same KOs, seed-averaged)
for line, ctag in [(l, c) for l in LINES for c in ["ctrl", "ctrl_ep1000"]]:
    ro, rc = runs(line, "obs"), runs(line, ctag)
    ro = {s: r for s, r in ro.items() if "ft" in r}; rc = {s: r for s, r in rc.items() if "ft" in r}
    if not ro or not rc: continue
    out = {}
    for arm in ["obs", "real", "infonce", "centered", "centered_infonce"]:
        for m in ["rr_retrieval", "rr_pe_cent", "rr_pe_cent_canon"]:
            # gain (arm - shuffled control) per KO, seed-averaged within each pretraining, then ctrl - obs
            def gain(rs_, arm=arm, m=m):
                shuf = {"obs": None, "real": "shuffle_avg", "infonce": "infonce_shuffle_avg", "centered": "centered_shuffle_avg", "centered_infonce": "centered_infonce_shuffle_avg"}[arm]
                G = []
                for s, r in rs_.items():
                    a = kovec(ftrows(r, arm), m)
                    if shuf: b = kovec(ftrows(r, shuf), m); a = {g: a[g] - b[g] for g in a if g in b}
                    G.append(a)
                gs = sorted(set.intersection(*[set(x) for x in G])); return {g: np.mean([x[g] for x in G]) for g in gs}
            go, gc = gain(ro), gain(rc); gs = sorted(set(go) & set(gc)); d = np.array([gc[g] - go[g] for g in gs])
            bs = d[np.random.RandomState(0).randint(0, len(d), (4000, len(d)))].mean(1)
            out[f"{arm}|{m}"] = dict(n=len(d), obs_pre=float(np.mean([go[g] for g in gs])), ctrl_pre=float(np.mean([gc[g] for g in gs])),
                                     diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p_wilcoxon=float(wilcoxon(d).pvalue))
    S["ctrl_vs_obs"][f"{line}_{ctag}"] = out
    md.append(f"\n### Control-only ({ctag}) minus original pretraining, {line}: FT gain over the label-shuffled control (arm 'obs' = untuned level), seed-averaged per KO\n"
              "| arm | metric | original | control-only | Δ (ctrl − orig) | 95% CI | P |\n|---|---|---|---|---|---|---|")
    for k, v in out.items():
        a, m = k.split("|"); md.append(f"| {a} | {m.replace('rr_','')} | {v['obs_pre']:+.3f} | {v['ctrl_pre']:+.3f} | {v['diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {pfmt(v['p_wilcoxon'])} |")

# ------------------------------------------------ ridge / RRR / SCM (C7)
NBK = ["mean_train_response", "knn10_ntcorr", "knn10_footprint", "ridge_ntcorr", "ridge_footprint", "rrr_spanD_r192", "rrr_spanD_r64", "rrr_spanD_r32", "rrr_spanD_r8",
       "rrr_spanD_shiftscaled", "ridge_ntcorr_out_spanD"]
for line in LINES:
    for pre in PRES:
        rs = {s: r for s, r in runs(line, pre).items() if "nb" in r}
        if not rs: continue
        key = f"{line}_{pre}"; tab = {}
        for k in NBK + ["SCM:real", "SCM:infonce", "SCM:centered_infonce"]:
            if k.startswith("SCM:"):
                vals = {m: [r["ft"]["shift"][k[4:]]["summary"][m] for r in rs.values() if "ft" in r] for m in MET}
            else:
                vals = {m: [r["nb"][k]["summary"][m] for r in rs.values() if k in r["nb"]] for m in MET}
            tab[k] = {m: dict(mean=float(np.mean(v)) if v else None, sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, per_seed=v) for m, v in vals.items()}
        panel = {}
        for a, b in [("SCM:real", "rrr_spanD_r192"), ("SCM:infonce", "rrr_spanD_r192"), ("rrr_spanD_r192", "ridge_footprint"), ("ridge_ntcorr", "ridge_footprint"),
                     ("SCM:infonce", "ridge_footprint"), ("rrr_spanD_r192", "rrr_spanD_shiftscaled"), ("SCM:real", "rrr_spanD_shiftscaled")]:
            get = lambda r, k: r["ft"]["shift"][k[4:]]["per_ko"] if k.startswith("SCM:") else r["nb"][k]["per_ko"]
            ra = {s: get(r, a) for s, r in rs.items()}; rb = {s: get(r, b) for s, r in rs.items()}
            for m in ["rr_retrieval", "rr_pe_cent", "rr_pe_cent_canon"]:
                panel[f"{b}_vs_{a}|{m}"] = seed_contrast(ra, rb, m)
        bh(panel); S["nb"][key] = dict(table=tab, contrasts=panel,
                                       alpha={k: [r["nb"][k].get("alpha") for r in rs.values() if k in r["nb"]] for k in NBK})
        md.append(f"\n### Supervised gene-feature predictors vs SCM fine-tuning, {line}, pretraining = {pre} (seeds {list(rs)})\n"
                  "| predictor | retrieval | centred pearDE | centred (canonical) | partial r | Systema cacc | Systema cacc (HVG) | raw pearDE |\n|---|---|---|---|---|---|---|---|")
        for k in tab:
            t = tab[k]; md.append(f"| {k} | " + " | ".join("-" if t[m]["mean"] is None else f"{t[m]['mean']:.3f} ± {t[m]['sd'] or 0:.3f}" for m in MET) + " |")
        md.append("\n| contrast (b − a) | metric | Δ | 95% CI (hier.) | per-seed Δ | P (seed-avg) | q BH |\n|---|---|---|---|---|---|---|")
        for k, v in panel.items():
            c, m = k.split("|"); md.append(f"| {c} | {m.replace('rr_','')} | {v['mean_diff']:+.3f} | [{v['ci95_hier'][0]:+.3f}, {v['ci95_hier'][1]:+.3f}] | {', '.join(f'{x:+.3f}' for x in v['seed_diffs'])} | {pfmt(v['p_wilcoxon_seedavg'])} | {pfmt(v.get('q_bh'))} |")

# ------------------------------------------------ learning curves
for line in LINES:
    for pre in PRES:
        rs = {s: r for s, r in runs(line, pre, "_lc").items() if "lc" in r}
        if not rs: continue
        key = f"{line}_{pre}"; full = {s: r for s, r in runs(line, pre).items() if "ft" in r and "nb" in r}
        V = ["real", "shuffle", "centered", "centered_shuffle", "infonce", "infonce_shuffle", "ridge_footprint", "ridge_ntcorr", "rrr_spanD_r192"]
        out = {}
        for sz in ["5", "25", "100", "400", "all"]:
            out[sz] = {}
            for v in V:
                for m in ["rr_retrieval", "rr_pe_cent", "rr_pe_cent_canon"]:
                    if sz == "all":
                        fk = {"real": ("ft", "real"), "shuffle": ("ft", "shuffle_avg"), "centered": ("ft", "centered"), "centered_shuffle": ("ft", "centered_shuffle_avg"),
                              "infonce": ("ft", "infonce"), "infonce_shuffle": ("ft", "infonce_shuffle_avg"), "ridge_footprint": ("nb", "ridge_footprint"),
                              "ridge_ntcorr": ("nb", "ridge_ntcorr"), "rrr_spanD_r192": ("nb", "rrr_spanD_r192")}[v]
                        vals = [(r[fk[0]]["shift"] if fk[0] == "ft" else r[fk[0]])[fk[1]]["summary"][m] for r in full.values()]
                    else:
                        vals = [r["lc"]["shift"][sz][v]["mean"][m] for r in rs.values() if sz in r["lc"]["shift"] and v in r["lc"]["shift"][sz]]
                    if vals: out[sz].setdefault(v, {})[m] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, n_models=len(vals))
            if sz != "all":
                out[sz]["paired_per_seed"] = {s: {k: {m: {x: vv[m].get(x) for x in ["mean_diff", "p_wilcoxon", "ci95"]} for m in vv} for k, vv in r["lc"]["shift"][sz]["paired"].items()}
                                              for s, r in rs.items() if sz in r["lc"]["shift"]}
        S["lc"][key] = out
        md.append(f"\n### Learning curve, {line}, pretraining = {pre} (mean ± s.d. across {len(rs)} models of the per-model subset means)\n| n genes | " + " | ".join(V) + " |\n|---|" + "---|" * len(V))
        for m in ["rr_retrieval", "rr_pe_cent"]:
            md.append(f"| **{m.replace('rr_','')}** |" + " |" * len(V))
            for sz in out:
                md.append(f"| {sz} | " + " | ".join(f"{out[sz][v][m]['mean']:.3f}±{out[sz][v][m]['sd']:.3f}" if v in out[sz] and m in out[sz][v] else "" for v in V) + " |")

# ------------------------------------------------ DeepSEM
for line in LINES:
    for pre in PRES:
        rs = {s: r for s, r in runs(line, pre, "_ds").items() if "deepsem" in r}
        cm = {s: r for s, r in runs(line, pre).items() if "ft" in r}
        if not rs: continue
        key = f"{line}_{pre}"; X0 = next(iter(rs.values()))["deepsem"]; out = dict(cmc_n_param=X0["cmc_n_param"], cmc_n_positions=X0["cmc_n_positions"], variants={})
        for mode in [k for k in X0 if k.startswith("top") or k.startswith("lr")]:
            seeds = [s for s in rs if mode in rs[s]["deepsem"] and s in cm]
            vt = dict(n_param=X0[mode]["n_param"], n_positions=X0[mode]["n_positions"], arms={}, contrasts={})
            for arm in ["obs", "real", "shuffle_avg", "centered", "centered_shuffle_avg", "infonce", "infonce_shuffle_avg", "centered_infonce", "centered_infonce_shuffle_avg"]:
                g = lambda s: rs[s]["deepsem"]["obs"] if arm == "obs" else rs[s]["deepsem"][mode].get(arm)
                if any(g(s) is None for s in seeds): continue
                vt["arms"][arm] = {m: dict(mean=float(np.mean([g(s)["summary"][m] for s in seeds])), sd=float(np.std([g(s)["summary"][m] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0) for m in MET}
            panel = {}
            for arm, sh in [("real", "shuffle_avg"), ("infonce", "infonce_shuffle_avg"), ("centered", "centered_shuffle_avg"), ("centered_infonce", "centered_infonce_shuffle_avg")]:
                if arm not in vt["arms"] or sh not in vt["arms"]: continue
                for m in ["rr_retrieval", "rr_pe_cent", "rr_pe_cent_canon", "rr_pcor"]:
                    dsr = {s: rs[s]["deepsem"][mode][arm]["per_ko"] for s in seeds}; cmr = {s: ftrows(cm[s], arm) for s in seeds}
                    panel[f"CMC_minus_DS:{arm}|{m}"] = seed_contrast(dsr, cmr, m)
                    dss = {s: rs[s]["deepsem"][mode][sh]["per_ko"] for s in seeds}; cms = {s: ftrows(cm[s], sh) for s in seeds}
                    panel[f"DS_real_minus_shuffle:{arm}|{m}"] = seed_contrast(dss, dsr, m)
                    # difference-in-differences: (CMC arm - CMC shuffled) - (DS arm - DS shuffled)
                    did_a = {s: [dict(gene=r["gene"], **{m: (r.get(m) - q.get(m)) if r.get(m) is not None and q.get(m) is not None else None}) for r, q in zip(dsr[s], dss[s])] for s in seeds}
                    did_b = {s: [dict(gene=r["gene"], **{m: (r.get(m) - q.get(m)) if r.get(m) is not None and q.get(m) is not None else None}) for r, q in zip(cmr[s], cms[s])] for s in seeds}
                    panel[f"DiD_CMCgain_minus_DSgain:{arm}|{m}"] = seed_contrast(did_a, did_b, m)
            bh(panel); vt["contrasts"] = panel; vt["seeds"] = seeds; out["variants"][mode] = vt
            md.append(f"\n### DeepSEM [{mode}: {vt['n_param']:,} params, {vt['n_positions']:,} positions] trained on the same cells as CMC ({line}, {pre}; seeds {seeds})\n"
                      "| arm | retrieval | centred | centred (canonical) | partial r | Systema | Systema (HVG) | raw pearDE |\n|---|---|---|---|---|---|---|---|")
            for arm, t in vt["arms"].items():
                md.append(f"| DS {arm} | " + " | ".join(f"{t[m]['mean']:.3f} ± {t[m]['sd']:.3f}" for m in MET) + " |")
            md.append("\n| contrast | metric | Δ | 95% CI (hier.) | per-seed Δ | P (seed-avg) | q BH | max P per seed |\n|---|---|---|---|---|---|---|---|")
            for k, v in panel.items():
                c, m = k.split("|"); md.append(f"| {c} | {m.replace('rr_','')} | {v['mean_diff']:+.3f} | [{v['ci95_hier'][0]:+.3f}, {v['ci95_hier'][1]:+.3f}] | {', '.join(f'{x:+.3f}' for x in v['seed_diffs'])} | {pfmt(v['p_wilcoxon_seedavg'])} | {pfmt(v.get('q_bh'))} | {pfmt(v['max_p_per_seed'])} |")
        S["ds"][key] = out

# ------------------------------------------------ cross-context
for line in LINES:
    for pre in PRES:
        rs = {s: r for s, r in runs(line, pre).items() if "xctx2" in r}
        if not rs: continue
        key = f"{line}_{pre}"; out = {}
        for reg in ["R1_prev_perturbed", "R2_never_perturbed"]:
            X = {s: r["xctx2"][reg] for s, r in rs.items() if reg in r["xctx2"]}
            if not X: continue
            arms = list(next(iter(X.values()))["summary"])
            tab = {a: {m: dict(mean=float(np.mean([x["summary"][a][m] for x in X.values()])), sd=float(np.std([x["summary"][a][m] for x in X.values()], ddof=1)) if len(X) > 1 else 0.0) for m in MET} for a in arms}
            panel = {}
            for a, b in [("scm_shuffle_avg", "scm_real"), ("ridge_footprint_T", "scm_real"), ("rrr_spanD_T", "scm_real"), ("ridge_coexpr_T", "scm_real"),
                         ("ridge_coexpr_S_to_T", "scm_real"), ("scm_real", "within_scm_real"), ("ridge_footprint_T", "within_ridge_footprint"),
                         ("scm_real", "copy_source_response"), ("ridge_footprint_T", "copy_source_response")]:
                if a not in arms or b not in arms: continue
                for m in ["rr_retrieval", "rr_pe_cent", "rr_pe_cent_canon"]:
                    panel[f"{b}_vs_{a}|{m}"] = seed_contrast({s: x["per_ko"][a] for s, x in X.items()}, {s: x["per_ko"][b] for s, x in X.items()}, m)
            bh(panel); n_eval = next(iter(X.values()))["n_eval"]; n_tr = next(iter(X.values()))["n_train"]
            out[reg] = dict(n_eval=n_eval, n_train=n_tr, seeds=list(X), table=tab, contrasts=panel)
            other = {"jurkat": "K562", "k562": "Jurkat"}[line]
            md.append(f"\n### Cross-context {other} → {line} ({reg}; source-line training KOs {n_tr}, evaluated target KOs {n_eval}; pretraining {pre}; seeds {list(X)})\n"
                      "| predictor | retrieval | centred | centred (canonical) | partial r | Systema | Systema (HVG) | raw pearDE |\n|---|---|---|---|---|---|---|---|")
            for a in arms:
                md.append(f"| {a} | " + " | ".join(f"{tab[a][m]['mean']:.3f} ± {tab[a][m]['sd']:.3f}" for m in MET) + " |")
            md.append("\n| contrast (b − a) | metric | Δ | 95% CI (hier.) | per-seed Δ | P (seed-avg) | q BH |\n|---|---|---|---|---|---|---|")
            for k, v in panel.items():
                c, m = k.split("|"); md.append(f"| {c} | {m.replace('rr_','')} | {v['mean_diff']:+.3f} | [{v['ci95_hier'][0]:+.3f}, {v['ci95_hier'][1]:+.3f}] | {', '.join(f'{x:+.3f}' for x in v['seed_diffs'])} | {pfmt(v['p_wilcoxon_seedavg'])} | {pfmt(v.get('q_bh'))} |")
        S["xctx2"][key] = out

open(f"{AN}/tables.md", "w").write("# Perturb-seq revision tables (auto-generated by analysis/fig5_interventions/perturbseq/aggregate_rev.py)\n" + "\n".join(md) + "\n")
json.dump(S, open(f"{AN}/summary.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else str(o))
print("wrote tables.md / summary.json;", len(md), "lines")
