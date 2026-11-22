#!/usr/bin/env python
"""Regenerated Fig 3 (b-e), Fig 4 (b-e) and Fig 5 (b-d) under the condition-matched pseudobulk DE definition.
Style follows paper/figscripts/rev_fig5_parts.py (Carlito, same palette).  Error bars: joint (knockout + control-cell)
bootstrap 95% CI for Figs 3-4; s.d. across the three independently trained models for Fig 5 (bars = mean of the three
per-seed pooled values, the same summary in b, c and d).  Brackets: two-sided paired t-tests on per-unit values,
BH within panel (Fig 3: across all Fig 3 comparisons).  Per-unit points are drawn for every panel (small n visible).
Outputs PDF+PNG + source-data CSV/JSON in paper_nov_revisions/figures/stats/."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, json, pickle, csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from scipy import stats
for f in os.listdir(os.path.expanduser("~/.fonts")):
    if f.startswith("Carlito"): fm.fontManager.addfont(os.path.expanduser(f"~/.fonts/{f}"))
plt.rcParams.update({"font.family": "Carlito", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False,
    "mathtext.fontset": "custom", "mathtext.rm": "Carlito", "mathtext.it": "Carlito:italic"})
ST = f"{_CMC_RESULTS}/paper/analyses/stats"; OUT = f"{_CMC_RESULTS}/paper/figures/stats"
os.makedirs(OUT, exist_ok=True)
S = json.load(open(f"{ST}/stats_summary_pb_bh.json")); T = pickle.load(open(f"{ST}/boot/point_pb_bh.pkl", "rb"))["T"]
TAGS = ["frangieh", "iccite", "arce"]; NAME = {"frangieh": "Frangieh", "iccite": "Chen", "arce": "Arce", "overall": "Overall"}
EK = dict(lw=0.7, capsize=1.8, ecolor="#333")
OBS, FT, SH, MEAN, DS, DSL = "#a0aec0", "#2b6cb0", "#e2e8f0", "#9ae6b4", "#6b46c1", "#b7a6e0"
def stars(p): return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
def bracket(ax, x1, x2, y, s, h):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.6, c="k"); ax.text((x1 + x2) / 2, y + h, s, ha="center", va="bottom", fontsize=6)
def lab(ax, s, x=-0.2, y=1.04): ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
def units(st, m, t):
    return {g: r[m] for (s, tt, g), r in T.items() if s == st and tt == t and r.get(m) is not None and np.isfinite(r.get(m, np.nan))}
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300); plt.close(fig)
SRC = []   # long-format source data rows

def bar_panel(ax, sets, m, chance, ylab, title, comps, ci_from="f3", hatch_metric=None, ref_metric=None):
    """sets: [(setkey, label, colour)]; comps: list of comparison dicts (from stats_summary) for brackets."""
    n_s = len(sets); w = 0.8 / n_s; X = np.arange(3); top = {}
    for i, (st, lb, col) in enumerate(sets):
        for j, t in enumerate(TAGS):
            v = units(st, m, t); xs = X[j] + (i - (n_s - 1) / 2) * w
            if not v: continue
            a = np.array(list(v.values())); mu = a.mean()
            src = S["fig3_means"].get(st) or S["fig4_means"].get(st) or {}
            ci = (src.get(m, {}).get(t, {}) or {}).get("ci95_boot")
            err = np.array([[mu - ci[0]], [ci[1] - mu]]) if ci else None
            ax.bar(xs, mu, w * 0.92, color=col, label=lb if j == 0 else None, yerr=err, error_kw=EK, zorder=1)
            if len(a) <= 30:   # per-unit points on small panels (Arce, Frangieh protein pearDE)
                jit = (np.random.default_rng(j * 10 + i).random(len(a)) - 0.5) * w * 0.6
                ax.scatter(xs + jit, a, s=3, c="#1a202c", alpha=0.7, lw=0, zorder=2)
            top[(i, j)] = max(mu + (err[1][0] if err is not None else 0), 0)
            SRC.append(dict(figure=title, metric=m, set=st, label=lb, cohort=t, n=len(a), mean=float(mu), ci_lo=ci[0] if ci else None, ci_hi=ci[1] if ci else None))
            for g, val in v.items(): SRC.append(dict(figure=title, metric=m, set=st, label=lb, cohort=t, unit=g, value=val))
            if hatch_metric:
                vc = units(st, hatch_metric, t)
                if vc:
                    ax.bar(xs, np.mean(list(vc.values())), w * 0.5, color="none", edgecolor="#1a202c", hatch="////", lw=0.4, zorder=3)
        if ref_metric and i == 0:
            for j, t in enumerate(TAGS):
                v = units(st, ref_metric, t)
                if v: ax.hlines(np.mean(list(v.values())), X[j] - 0.42, X[j] + 0.42, colors="#2f855a", lw=1.0, ls="--", zorder=4,
                                label="Mean-response reference" if j == 0 else None)
    ax.axhline(chance, ls=":", lw=0.8, c="#4a5568")
    ns = []
    for t in TAGS:
        v = units(sets[0][0], m, t); ns.append(len(v))
    ax.set_xticks(X); ax.set_xticklabels([f"{NAME[t]}\n(n = {n})" for t, n in zip(TAGS, ns)])
    ax.set_ylabel(ylab); ax.set_title(title)
    yl = ax.get_ylim(); span = yl[1] - yl[0]; k = {}; cur = {}
    for c in comps:
        j = TAGS.index(c["scope"]); i2 = [s[0] for s in sets].index(c["b"])
        p = c.get("p_bh_panel", c.get("p_bh_fig3", c["p_t"]))
        y = max(top.get((0, j), 0), top.get((i2, j), 0)) + 0.04 * span
        if j in cur: y = max(y, cur[j] + 0.08 * span)
        cur[j] = y; k[j] = k.get(j, 0) + 1
        bracket(ax, X[j] - (n_s - 1) / 2 * w, X[j] + (i2 - (n_s - 1) / 2) * w, y, stars(p), 0.015 * span)
    ax.set_ylim(yl[0], max([yl[1]] + [v + 0.12 * span for v in cur.values()]))

# ============================ Fig 3 ============================
C3 = S["fig3_comparisons"]
def c3(m, bases): return [c for c in C3 if c["a"] == "f3:cmc_s42" and c["ma"] == m and c["b"] in bases and c["mb"] == m and c["scope"] in TAGS and "p_bh_fig3" in c]
fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.4)); plt.subplots_adjust(hspace=0.55, wspace=0.28)
S3 = [("f3:cmc_s42", "CMC-VAE", FT), ("f3:corr", "Correlation", "#dd6b20"), ("f3:totalvi_ko", "totalVI", "#f6ad55"), ("f3:deepsem_ko", "DeepSEM", DS)]
bar_panel(axs[0, 0], S3, "rr_auroc", 0.5, "AUROC", "RNA-to-RNA", c3("rr_auroc", ["f3:corr", "f3:totalvi_ko", "f3:deepsem_ko"]), ref_metric="rr_auroc_meanresp")
bar_panel(axs[0, 1], S3, "rr_pe", 0.0, "Signed pearDE", "RNA-to-RNA", c3("rr_pe", ["f3:corr", "f3:totalvi_ko", "f3:deepsem_ko"]), hatch_metric="rr_pe_cent", ref_metric="rr_pe_meanresp")
bar_panel(axs[1, 0], S3[:3], "rp_auroc", 0.5, "AUROC", "RNA-to-protein", c3("rp_auroc", ["f3:corr", "f3:totalvi_ko"]), ref_metric="rp_auroc_meanresp")
bar_panel(axs[1, 1], S3[:3], "rp_pe", 0.0, "Signed pearDE", "RNA-to-protein", c3("rp_pe", ["f3:corr", "f3:totalvi_ko"]), hatch_metric="rp_pe_cent", ref_metric="rp_pe_meanresp")
for a, l in zip(axs.flat, "bcde"): lab(a, l)
h, l = axs[0, 1].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, -0.01, "Pseudobulk DE (condition-matched, BH q<0.05). Bars: mean over units, 95% joint-bootstrap CI; dots: units (panels with n <= 30); hatched: mean-centred pearDE; dashed: mean-response reference.\n"
         "Brackets: CMC-VAE vs baseline, two-sided paired t, BH across Fig. 3 comparisons.", ha="center", fontsize=6)
save(fig, "Figure3_bcde_pb")

# ============================ Fig 4 ============================
C4 = S["fig4_comparisons"]
def c4(pan): return [c for c in C4 if c["panel"] == pan]
fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.4)); plt.subplots_adjust(hspace=0.55, wspace=0.28)
SA = [("f3:cmc_s42", "Learned", FT), ("f4:Arr_transposed", r"Transposed ($A_{rr}^{\top}$)", "#c53030"), ("f4:Arr_shuffled", "Shuffled", "#a0aec0"), ("f4:Arr_zeroed", "Zeroed", "#cbd5e0")]
SW = [("f3:cmc_s42", "Learned", FT), ("f4:Wrp_from_WprT", r"Swap ($W_{rp}\leftarrow W_{pr}^{\top}$)", "#c53030"), ("f4:Wrp_shuffled", "Shuffled", "#a0aec0"), ("f4:Wrp_zeroed", "Zeroed", "#cbd5e0")]
bar_panel(axs[0, 0], SA, "rr_auroc", 0.5, "AUROC", r"$A_{rr}$ (RNA block)", c4("b"))
bar_panel(axs[0, 1], SW, "rp_auroc", 0.5, "AUROC", r"$W_{rp}$ (RNA-to-protein block)", c4("c"))
bar_panel(axs[1, 0], SA, "rr_pe", 0.0, "Signed pearDE", r"$A_{rr}$ (RNA block)", c4("d"))
bar_panel(axs[1, 1], SW[:3], "rp_pe", 0.0, "Signed pearDE", r"$W_{rp}$ (RNA-to-protein block)", c4("e"))
for a, l in zip(axs.flat, "bcde"): lab(a, l)
for a in axs[0]: a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.09), fontsize=6, ncol=4, handlelength=1.0, columnspacing=0.8)
fig.text(0.5, -0.01, "Learned vs each variant: two-sided paired t, BH within panel (ns shown). Bars: mean, 95% joint knockout + control-cell bootstrap CI; dots: units (n <= 30).\n"
         "The swap replaces W_rp by the transposed protein-to-RNA block (r = 0.71 with W_rp); it is a matrix swap, not a reversal of direction.", ha="center", fontsize=6)
save(fig, "Figure4_bcde_pb")

# ============================ Fig 5 b-d ============================
C5 = S["fig5_comparisons"]; SS = S["fig5_seed_summary"]
ARMS = [("ft_baseline", "Observational", OBS, ""), ("ft_standard", "Fine-tuned (real targets)", FT, ""),
        ("ft_labelshuffle", "Label-shuffled targets", SH, "////"), ("ft_meantarget", "Screen-mean targets", MEAN, "")]
def f5panel(ax, m, title, pan):
    X = np.arange(4); w = 0.2; tops = {}
    sc = ["overall"] + TAGS
    for i, (v, lb, col, hat) in enumerate(ARMS):
        mu = [SS[f"{v}|{m}|{s}"]["mean"] for s in sc]; sd = [SS[f"{v}|{m}|{s}"]["sd"] for s in sc]
        xs = X + (i - 1.5) * w
        ax.bar(xs, mu, w * 0.92, color=col, hatch=hat, edgecolor="#718096" if hat else col, lw=0.4, yerr=sd, error_kw=EK, label=lb)
        for j, s in enumerate(sc):
            ax.scatter(np.full(3, xs[j]) + np.array([-0.04, 0, 0.04]), SS[f"{v}|{m}|{s}"]["per_seed"], s=2.5, c="#1a202c", zorder=3, lw=0)
            tops[(i, j)] = mu[j] + (sd[j] or 0)
            SRC.append(dict(figure=f"Fig5{pan}", metric=m, set=v, cohort=s, mean_of_seeds=mu[j], sd_seeds=sd[j], per_seed=SS[f"{v}|{m}|{s}"]["per_seed"]))
        # per-unit (seed-averaged) points for Arce
        ua = units(f"f5:{v}|mean", m, "arce")
        ax.scatter(np.full(len(ua), xs[3]) + 0.05, list(ua.values()), s=4, marker="x", c="#c05621", lw=0.6, zorder=4, label="Arce units (seed mean)" if i == 0 else None)
    ax.axhline(0, ls=":", lw=0.8, c="#4a5568")
    ns = [len(units("f5:ft_standard|mean", m, t)) for t in TAGS]
    ax.set_xticks(X); ax.set_xticklabels([f"Overall\n(n = {sum(ns)})"] + [f"{NAME[t]}\n(n = {n})" for t, n in zip(TAGS, ns)])
    ax.set_ylabel("Held-out signed pearDE"); ax.set_title(title)
    yl = ax.get_ylim(); span = yl[1] - yl[0]
    for j, s in enumerate(sc):
        y0 = max(tops[(i, j)] for i in range(4)) + 0.04 * span
        for k_, (b, i2) in enumerate([("ft_baseline", 0), ("ft_labelshuffle", 2)]):
            c = [c for c in C5 if c["panel"] == pan and c["a"] == "f5:ft_standard|mean" and c["b"] == f"f5:{b}|mean" and c["scope"] == s]
            if c: bracket(ax, X[j] + (1 - 1.5) * w, X[j] + (i2 - 1.5) * w, y0 + k_ * 0.08 * span, stars(c[0]["p_bh_panel"]), 0.015 * span)
    ax.set_ylim(yl[0], yl[1] + 0.2 * span)
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.6)); plt.subplots_adjust(wspace=0.25)
f5panel(axs[0], "rr_pe", "RNA-to-RNA", "b"); f5panel(axs[1], "rp_pe", "RNA-to-protein", "c")
lab(axs[0], "b"); lab(axs[1], "c")
axs[0].legend(loc="upper left", ncol=2, fontsize=5.8, handlelength=1.0)
fig.text(0.5, -0.06, "Bars: mean of the three per-model pooled values (seeds 42, 1, 2); error bars: s.d. across models; dots: models. Brackets: fine-tuned vs observational (lower) and vs label-shuffled (upper),\n"
         "two-sided paired t on per-unit values averaged over models, BH within panel. Pseudobulk DE definition.", ha="center", fontsize=6)
save(fig, "Figure5_bc_pb")

def fig5d():
    # Fig 5d: same summary (mean of per-seed pooled values) for CMC and DeepSEM
    DSJ = [json.load(open(f"{ST}/runs/rc_sp_dec_gl1000{s}_pb_bh_ds.json")) for s in ["", "_s1", "_s2"]]
    vals = {}
    for m in ["rr_pe", "rr_pe_cent"]:
        vals[m] = dict(cmc_obs=SS[f"ft_baseline|{m}|overall"]["per_seed"], cmc_ft=SS[f"ft_standard|{m}|overall"]["per_seed"],
                       cmc_shuf=SS[f"ft_labelshuffle|{m}|overall"]["per_seed"],
                       ds_obs=[d["deepsem_matched"]["baseline"]["summary"]["overall"][m] for d in DSJ],
                       ds_ft=[d["deepsem_matched"]["ft_cmc_matched_real"]["summary"]["overall"][m] for d in DSJ],
                       ds_shuf=[d["deepsem_matched"].get("ft_cmc_matched_shuffle", {}).get("summary", {}).get("overall", {}).get(m, np.nan) for d in DSJ])
    # paired test CMC fine-tuned vs DeepSEM fine-tuned on per-unit values averaged over seeds
    def per_unit_mean(rowsets, m):
        acc = {}
        for rows in rowsets:
            for r in rows:
                x = r.get(m)
                if x is not None and np.isfinite(x): acc.setdefault((r["cohort"], r["gene"]), []).append(x)
        return {k: np.mean(v) for k, v in acc.items()}
    p5d = {}
    for m in ["rr_pe", "rr_pe_cent"]:
        a = {(t, g): v for t in TAGS for g, v in units("f5:ft_standard|mean", m, t).items()}
        b = per_unit_mean([d["deepsem_matched"]["ft_cmc_matched_real"]["per_ko"] for d in DSJ], m)
        ks = sorted(set(a) & set(b)); x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks])
        p5d[m] = dict(n=len(ks), mean_diff=float((x - y).mean()), p_two=float(stats.ttest_rel(x, y).pvalue))
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    keys = [("cmc_obs", "CMC-VAE, observational", OBS, ""), ("cmc_ft", "CMC-VAE, fine-tuned", FT, ""), ("cmc_shuf", "CMC-VAE, label-shuffled", SH, "////"),
            ("ds_obs", "DeepSEM, observational", DSL, ""), ("ds_ft", "DeepSEM, fine-tuned", DS, ""), ("ds_shuf", "DeepSEM, label-shuffled", "#e9d8fd", "////")]
    w = 0.13; X = np.array([0, 1.1])
    for i, (k, name, col, hat) in enumerate(keys):
        mu = [np.nanmean(vals[m][k]) for m in ["rr_pe", "rr_pe_cent"]]; sd = [np.nanstd(vals[m][k], ddof=1) for m in ["rr_pe", "rr_pe_cent"]]
        ax.bar(X + (i - 2.5) * w, mu, w * 0.92, color=col, hatch=hat, edgecolor="#718096" if hat else col, lw=0.4, yerr=sd, error_kw=EK, label=name)
        for j, m in enumerate(["rr_pe", "rr_pe_cent"]):
            ax.scatter(np.full(3, X[j] + (i - 2.5) * w), vals[m][k], s=2.5, c="#1a202c", zorder=3, lw=0)
    ax.axhline(0, ls=":", lw=0.8, c="#4a5568"); ax.set_xticks(X); ax.set_xticklabels(["Raw pearDE", "Centred pearDE"])
    ax.set_ylabel("Held-out signed pearDE")
    n5 = sum(len(units("f5:ft_standard|mean", "rr_pe", t)) for t in TAGS)
    ax.set_title(f"Matched fine-tuning, CITE-seq screens (n = {n5})")
    yl = ax.get_ylim(); span = yl[1] - yl[0]
    for j, m in enumerate(["rr_pe", "rr_pe_cent"]):
        y = max(np.nanmax(vals[m]["cmc_ft"]), np.nanmax(vals[m]["ds_ft"])) + 0.05 * span
        bracket(ax, X[j] + (1 - 2.5) * w, X[j] + (4 - 2.5) * w, y, stars(p5d[m]["p_two"]), 0.02 * span)
    ax.set_ylim(yl[0], yl[1] + 0.15 * span); ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=5.8)
    lab(ax, "d", x=-0.17)
    save(fig, "Figure5_d_pb")
    json.dump(dict(vals=vals, paired_cmcft_vs_dsft_two_sided=p5d), open(f"{OUT}/Figure5_d_pb_data.json", "w"), indent=1)
    return p5d
try:
    p5d = fig5d()
except KeyError as e:
    p5d = None; print('Fig 5d skipped (DeepSEM runs incomplete):', e)
keys_ = sorted(set().union(*[r.keys() for r in SRC]))
with open(f"{OUT}/source_data_fig3_4_5_pb.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=keys_); wr.writeheader(); [wr.writerow(r) for r in SRC]
print("figures written", p5d)
