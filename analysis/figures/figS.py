"""New supplementary figures (clean, same style). The original Supplementary Figs 1-8 are kept unchanged."""
import json, numpy as np, pandas as pd
from common import *

SD = pd.read_csv(f"{NR}/figures/stats/source_data_fig3_4_5_pb.csv")
ST = json.load(open(f"{NR}/analyses/stats/stats_summary_pb_bh.json"))
EXP = json.load(open(f"{NR}/figures/round2/R2_Fig3b_koinvariant_stratified_source_data.json"))
PS = json.load(open(f"{NR}/analyses/perturbseq/summary.json"))


def per_ko(set_, scr, metric):
    x = SD[(SD.set == set_) & (SD.cohort == scr) & (SD.metric == metric) & SD.unit.notna()]
    return x.groupby("unit")["value"].mean().astype(float).dropna()


def grouped(ax, models, metric, ylab, ylim, zero=False, legend=True):
    x = np.arange(3); slot = 0.84 / len(models); w = slot * 0.92
    for i, (m, s) in enumerate(models):
        st = [per_ko(s, c, metric) for c in SCREENS]
        ax.bar(x - 0.42 + slot * (i + 0.5), [v.mean() for v in st], w,
               yerr=[v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in st],
               color=COL[m], label=m, error_kw=EK)
    if zero:
        ax.axhline(0, ls=":", lw=0.9, c="#4a5568")
    else:
        chance(ax)
    n = [len(per_ko(models[0][1], c, metric)) for c in SCREENS]
    ax.set_xticks(x); ax.set_xticklabels([f"{SLBL[c]}\n(n = {k})" for c, k in zip(SCREENS, n)])
    ax.set_ylim(*ylim); ax.set_ylabel(ylab)
    if legend:
        ax.legend(loc="upper left", ncol=4, fontsize=7, handlelength=1.2, columnspacing=0.8)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig); print("wrote", name)


M4 = [("CMC-VAE", "f3:cmc_s42"), ("totalVI", "f3:totalvi_ko"), ("Correlation", "f3:corr"), ("DeepSEM", "f3:deepsem_ko")]
M3 = M4[:3]

# ---- S-A: signed and protein metrics for Fig. 3 ------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.38))
grouped(axs[0], M4, "rr_pe", "RNA-response signed pearDE", (-0.45, 0.6), zero=True, legend=False)
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 1.1))
axs[0].set_title("RNA-to-RNA, signed", fontsize=8.5)
grouped(axs[1], M3, "rp_auroc", "Protein-response AUROC", (0.3, 0.85), legend=False)
axs[1].set_title("RNA-to-protein, ranking", fontsize=8.5)
grouped(axs[2], M3, "rp_pe", "Protein-response signed pearDE", (-0.45, 0.6), zero=True, legend=False)
axs[2].set_title("RNA-to-protein, signed", fontsize=8.5)
for a, l in zip(axs, "abc"):
    panel(a, l, x=-0.3)
save(fig, "SupplementaryFig_perturbation_extended")

# ---- S-B: expression-based reference rankings -----------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 2.6))
refs = [("CMC-VAE", None, COL["CMC-VAE"]), ("Correlation", "corr", COL["Correlation"]),
        ("Control-cell mean", "ctrl_mean", "#4a5568"), ("Control-cell s.d.", "ctrl_sd", "#718096")]
x = np.arange(3); slot = 0.84 / len(refs); w = slot * 0.92
for i, (lb, k, c) in enumerate(refs):
    if k is None:
        vals = [per_ko("f3:cmc_s42", s, "rr_auroc").mean() for s in SCREENS]
        err = [per_ko("f3:cmc_s42", s, "rr_auroc").sem() for s in SCREENS]
    else:
        vals = [EXP[f"rr_auroc|{k}|{s}"]["point"] for s in SCREENS]
        err = [(EXP[f"rr_auroc|{k}|{s}"]["ci"][1] - EXP[f"rr_auroc|{k}|{s}"]["ci"][0]) / 3.92 for s in SCREENS]
    ax.bar(x - 0.42 + slot * (i + 0.5), vals, w, yerr=err, color=c, label=lb, error_kw=EK)
chance(ax); ax.set_ylim(0.4, 1.02)
ax.set_xticks(x); ax.set_xticklabels([SLBL[s] for s in SCREENS]); ax.set_ylabel("RNA-response AUROC")
ax.legend(loc="upper left", ncol=2, fontsize=7, handlelength=1.2)
save(fig, "SupplementaryFig_expression_references")

# ---- S-C: CITE-seq fine-tuning arms ------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.3))
arms = [("ft_baseline", "Observational", COL["obs"], None), ("ft_standard", "Fine-tuned", COL["ft"], None),
        ("ft_labelshuffle", "Label-shuffled", "white", "////"), ("ft_meantarget", "Screen-mean targets", "#90cdf4", None)]
coh = ["overall", "frangieh", "iccite", "arce"]; cl = ["Overall", "Frangieh", "Chen", "Arce"]
for ax, figname, ylab in [(axs[0], "Fig5b", "Held-out RNA pearDE"), (axs[1], "Fig5c", "Held-out protein pearDE")]:
    slot = 0.84 / len(arms); w = slot * 0.92
    for i, (s, lb, c, h) in enumerate(arms):
        r = SD[(SD.figure == figname) & (SD.set == s) & SD.unit.isna()].set_index("cohort")
        ax.bar(np.arange(4) - 0.42 + slot * (i + 0.5), [r.loc[k, "mean_of_seeds"] for k in coh], w,
               yerr=[r.loc[k, "sd_seeds"] for k in coh], color=c, edgecolor="#4a5568" if h else c, hatch=h,
               lw=0.7, label=lb, error_kw=EK)
    ax.axhline(0, ls=":", lw=0.9, c="#4a5568"); ax.set_xticks(range(4)); ax.set_xticklabels(cl)
    ax.set_ylabel(ylab); ax.set_ylim(-0.35, 0.75)
axs[0].legend(loc="upper left", ncol=2, fontsize=7, handlelength=1.2)
for a, l in zip(axs, "ab"):
    panel(a, l, x=-0.16)
save(fig, "SupplementaryFig_citeseq_finetuning")

# ---- S-D: Perturb-seq, latent-graph fine-tuning vs supervised predictors ------------------------------
EXTV = {"jurkat": {"GEARS": (0.625, 0.085), "Linear (Ahlmann-Eltze)": (0.625, 0.151), "scGPT + ridge": (0.666, 0.191)},
        "k562": {"GEARS": (0.603, 0.080), "Linear (Ahlmann-Eltze)": (0.623, 0.149), "scGPT + ridge": (0.677, 0.225)}}
meths = [("Latent-graph FT", "SCM:real", COL["CMC-VAE"]), ("Latent-graph FT + InfoNCE", "SCM:infonce", COL["ftnce"]),
         ("Reduced-rank ridge", "rrr_spanD_r192", "#319795"), ("Ridge (decoder loadings)", "ridge_footprint", COL["totalVI"]),
         ("GEARS", None, "#d69e2e"), ("Linear (Ahlmann-Eltze)", None, "#a0aec0"), ("scGPT + ridge", None, "#6b46c1")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.3))
for ax, (mi, ylab, yl) in zip(axs, [(0, "Held-out retrieval", (0.45, 0.72)), (1, "Held-out centred pearDE", (0, 0.25))]):
    slot = 0.86 / len(meths); w = slot * 0.92
    for i, (lb, k, c) in enumerate(meths):
        v = []
        for ln in ("jurkat", "k562"):
            if k:
                t = PS["nb"][f"{ln}_obs"]["table"][k]; v.append(t["rr_retrieval" if mi == 0 else "rr_pe_cent"]["mean"])
            else:
                v.append(EXTV[ln][lb][mi])
        ax.bar(np.arange(2) - 0.43 + slot * (i + 0.5), v, w, color=c, label=lb)
    if mi == 0:
        chance(ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Jurkat", "K562"]); ax.set_ylabel(ylab); ax.set_ylim(*yl)
h_, l_ = axs[1].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.13), handlelength=1.1, columnspacing=0.8)
for a, l in zip(axs, "ab"):
    panel(a, l, x=-0.16)
save(fig, "SupplementaryFig_perturbseq_predictors")

# ---- S-E: knockdown operators: does the in silico knockout lower its own target? ------------------------
fig, ax = plt.subplots(figsize=(4.2, 2.5))
vals = {"Soft do (edge scaling)": [0.933, 0.604, 0.609], "Signed knockdown shift": [0.977, 1.00, 1.00], "Measured": [0.920, 0.787, 1.00]}  # condition-matched set (174/122/23), mean of three models (analyses/operator/runs/op_pb_s_*)
cols = ["#2b6cb0", "#63b3ed", "#a0aec0"]; slot = 0.84 / 3; w = slot * 0.92
for i, ((lb, v), c) in enumerate(zip(vals.items(), cols)):
    ax.bar(np.arange(3) - 0.42 + slot * (i + 0.5), v, w, color=c, label=lb)
ax.set_xticks(range(3)); ax.set_xticklabels([SLBL[s] for s in SCREENS]); ax.set_ylim(0, 1.25)
ax.set_ylabel("Fraction of knockouts\nlowering the target gene"); ax.axhline(0.5, ls=":", lw=0.9, c="#4a5568")
ax.legend(loc="upper left", ncol=2, fontsize=7, handlelength=1.2)
save(fig, "SupplementaryFig_knockdown_operators")
