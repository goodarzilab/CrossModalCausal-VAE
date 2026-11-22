"""Figure 5: interventions refine the latent graph. Original schematic (panel a) and original Perturb-seq
panel styles; adds the synthetic known-target recovery, the matched DeepSEM comparison (one DeepSEM:
the 73,728-entry mask matching CMC-VAE's parameter count) and control-only pretraining."""
import json, os, re, numpy as np, pandas as pd, fitz
from matplotlib.patches import Patch
from common import *

PS = json.load(open(f"{NR}/analyses/perturbseq/summary.json"))
SYN = pd.read_csv(f"{NR}/figures/identifiability/SuppFig_synthetic_interventional_recovery_source.csv")
TAB = open(f"{NR}/analyses/perturbseq/tables.md").read().splitlines()


def did_q(line, pre, arm):
    """q of CMC gain minus DeepSEM(top73728) gain in retrieval, from the perturbseq tables."""
    hdr = f"### DeepSEM [top73728: 73,728 params, 73,728 positions] trained on the same cells as CMC ({line}, {pre};"
    i = next(k for k, l in enumerate(TAB) if l.startswith(hdr))
    for l in TAB[i:i + 60]:
        if l.startswith(f"| DiD_CMCgain_minus_DSgain:{arm} | retrieval |"):
            return float(l.split("|")[7])
    return 1.0


fig = plt.figure(figsize=(7.2, 7.4))
gs = fig.add_gridspec(3, 2, width_ratios=[1.12, 1], height_ratios=[1.0, 1.05, 1.0], hspace=0.62, wspace=0.36)

# ---- a: schematic placeholder (filled from the original PDF at composition) -----------------------
axa = fig.add_subplot(gs[0, 0]); axa.axis("off")

# ---- b: synthetic system with a known graph ---------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
S = SYN[SYN.regime == "all"]
grp = [("G_rr_orient", "before", "Observational", COL["obs"]), ("G_rr_orient", "after", "+ known-target\ninterventions", COL["ft"])]
xs = np.arange(2); w = 0.36
for i, (m, col, lb, c) in enumerate(grp):
    vals = [S[(S.l_r == L) & (S.metric == m)][col] for L in (8, 16)]
    ax.bar(xs + (i - 0.5) * w, [v.mean() for v in vals], w * 0.95,
           yerr=[v.std(ddof=1) / np.sqrt(len(v)) for v in vals], color=c, label=lb, error_kw=EK)
chance(ax); ax.set_ylim(0.3, 1.16)
ax.set_xticks(xs); ax.set_xticklabels(["8 RNA latents", "16 RNA latents"])
ax.set_ylabel("Edge-orientation accuracy")
ax.set_title("Synthetic latent graph with known structure", fontsize=8.5)
for j in range(2):
    a = S[(S.l_r == (8, 16)[j]) & (S.metric == "G_rr_orient")]
    bracket(ax, j - w / 2, j + w / 2, a.after.mean() + 0.06, "***")
ax.legend(loc="upper left", fontsize=7, handlelength=1.2, ncol=2, columnspacing=1.0)
panel(ax, "b", x=-0.2)

# ---- c: Perturb-seq held-out genes -----------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
arms = [("obs", "Observational", COL["obs"], None), ("real", "Fine-tuned", COL["ft"], None),
        ("shuffle_avg", "Label-shuffled", "white", "////"), ("infonce", "FT + InfoNCE", COL["ftnce"], None),
        ("infonce_shuffle_avg", "InfoNCE, shuffled", "white", "xxxx")]
lines = [("jurkat", "Jurkat\n(n = 428)"), ("k562", "K562\n(n = 399)")]
slot = 0.84 / len(arms); w = slot * 0.9
for i, (k, lb, c, h) in enumerate(arms):
    for j, (ln, _) in enumerate(lines):
        r = PS["ft"][f"{ln}_obs"]["arms"][k]["rr_pe_cent"]
        xx = j - 0.42 + slot * (i + 0.5)
        ax.bar(xx, r["mean"], w, color=c, edgecolor="#4a5568" if h else c, hatch=h, lw=0.7,
               label=lb if j == 0 else None)
        if "per_seed" in r:
            ax.scatter([xx] * len(r["per_seed"]), r["per_seed"], s=6, color="#1a202c", zorder=3, lw=0)
    # FT vs shuffled, InfoNCE vs its shuffled (joint bootstrap, q <= 7e-4)
for j in range(2):
    for a_, b_ in [(1, 2), (3, 4)]:
        xa = j - 0.42 + slot * (a_ + 0.5); xb = j - 0.42 + slot * (b_ + 0.5)
        top = max(PS["ft"][f"{lines[j][0]}_obs"]["arms"][arms[a_][0]]["rr_pe_cent"]["per_seed"])
        bracket(ax, xa, xb, top + 0.006, "***", h=0.002)
ax.axhline(0, ls=":", lw=0.9, c="#4a5568")
ax.set_xticks([0, 1]); ax.set_xticklabels([l for _, l in lines])
ax.set_ylabel("Held-out centred pearDE"); ax.set_ylim(-0.02, 0.118)
ax.set_title("Perturb-seq (essential-gene screens), held-out genes", fontsize=8.5)
ax.legend(loc="upper left", fontsize=6.8, ncol=3, handlelength=1.2, columnspacing=0.7)
panel(ax, "c", x=-0.15)

# ---- d: learning curve --------------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
ns = ["5", "25", "100", "400", "all"]; xs = np.arange(len(ns))
for ln, c, mk in [("jurkat", "#2b6cb0", "o"), ("k562", "#63b3ed", "s")]:
    L = PS["lc"][f"{ln}_obs"]
    for arm, ls, cc in [("real", "-", c), ("shuffle", "--", "#cbd5e0" if ln == "k562" else "#a0aec0")]:
        m = [L[n][arm]["rr_pe_cent"]["mean"] for n in ns]; s = [L[n][arm]["rr_pe_cent"]["sd"] for n in ns]
        lab_ = f"{'Jurkat' if ln == 'jurkat' else 'K562'}, {'fine-tuned' if arm == 'real' else 'label-shuffled'}"
        ax.errorbar(xs, m, yerr=s, color=cc, ls=ls, marker=mk, ms=3.5, lw=1.2, capsize=1.5, label=lab_)
ax.axhline(0, ls=":", lw=0.9, c="#4a5568")
ax.set_xticks(xs); ax.set_xticklabels(["5", "25", "100", "400", "all\n(~1,000)"])
ax.set_xlabel("Number of training perturbations"); ax.set_ylabel("Held-out centred pearDE")
ax.set_ylim(-0.02, 0.105); ax.set_title("Training-set size", fontsize=8.5)
ax.legend(loc="upper left", fontsize=6.8, handlelength=1.6, ncol=2, columnspacing=0.8)
panel(ax, "d", x=-0.2)

# ---- e: CMC-VAE vs matched DeepSEM -------------------------------------------------------------------
ax = fig.add_subplot(gs[2, 0])
groups = [("jurkat", "real", "Jurkat\nstandard"), ("jurkat", "infonce", "Jurkat\nInfoNCE"),
          ("k562", "real", "K562\nstandard"), ("k562", "infonce", "K562\nInfoNCE")]
w = 0.36
for j, (ln, arm, _) in enumerate(groups):
    sh = "shuffle_avg" if arm == "real" else "infonce_shuffle_avg"
    cm = PS["ft"][f"{ln}_obs"]["arms"]
    cg = np.array(cm[arm]["rr_retrieval"]["per_seed"]) - np.array(cm[sh]["rr_retrieval"]["per_seed"])
    ds = PS["ds"][f"{ln}_obs"]["variants"]["top73728"]["arms"]
    dg = ds[arm]["rr_retrieval"]["mean"] - ds[sh]["rr_retrieval"]["mean"]
    dsd = np.hypot(ds[arm]["rr_retrieval"]["sd"], ds[sh]["rr_retrieval"]["sd"])
    ax.bar(j - w / 2, cg.mean(), w * 0.95, color=COL["CMC-VAE"], label="CMC-VAE" if j == 0 else None)
    ax.scatter([j - w / 2] * 3, cg, s=6, color="#1a202c", zorder=3, lw=0)
    ax.bar(j + w / 2, dg, w * 0.95, yerr=dsd, color=COL["DeepSEM"], error_kw=EK,
           label="DeepSEM (same cells)" if j == 0 else None)
    q = did_q(ln, "obs", arm)
    bracket(ax, j - w / 2, j + w / 2, max(cg.max(), dg + dsd) + 0.006, lab(q), h=0.002)
ax.axhline(0, ls=":", lw=0.9, c="#4a5568")
ax.set_xticks(range(4)); ax.set_xticklabels([g[2] for g in groups])
ax.set_ylabel("Retrieval gain over\nlabel-shuffled")
ax.set_ylim(-0.01, 0.16); ax.set_title("Latent-graph vs gene-graph refinement", fontsize=8.5)
ax.legend(loc="upper left", fontsize=6.8, handlelength=1.2, ncol=2, columnspacing=0.8)
panel(ax, "e", x=-0.15)

# ---- f: control-only pretraining --------------------------------------------------------------------
ax = fig.add_subplot(gs[2, 1])
w = 0.36
for j, (ln, lb) in enumerate([("jurkat", "Jurkat"), ("k562", "K562")]):
    for i, (pre, name, c) in enumerate([("obs", "Controls + training perturbations", COL["CMC-VAE"]),
                                        ("ctrl", "Control cells only", "#63b3ed")]):
        a = PS["ft"][f"{ln}_{pre}"]["arms"]
        g = np.array(a["real"]["rr_pe_cent"]["per_seed"]) - np.array(a["shuffle_avg"]["rr_pe_cent"]["per_seed"])
        xx = j + (i - 0.5) * w
        ax.bar(xx, g.mean(), w * 0.95, color=c, label=name if j == 0 else None)
        ax.scatter([xx] * 3, g, s=6, color="#1a202c", zorder=3, lw=0)
ax.axhline(0, ls=":", lw=0.9, c="#4a5568")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Jurkat", "K562"])
ax.set_ylabel("Centred pearDE gain\nover label-shuffled")
ax.set_ylim(0, 0.08); ax.set_title("Pretraining data", fontsize=8.5)
ax.legend(loc="upper left", fontsize=6.8, title="Pretrained on", title_fontsize=6.8, handlelength=1.2)
panel(ax, "f", x=-0.2)

fig.savefig(f"{OUT}/_parts/fig5_data.pdf", bbox_inches="tight", pad_inches=0.02)
bb = axa.get_position(); figsize = fig.get_size_inches()
plt.close(fig)

# ---- compose: original schematic into panel a -----------------------------------------------------
src = fitz.open(f"{R}/Figures_Main/Figure5.pdf"); sp = src[0]
for r in sp.search_for("(b–g)"):
    sp.add_redact_annot(r, fill=(1, 1, 1))
sp.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
clip = fitz.Rect(0, 0, 252, 160)
data = fitz.open(f"{OUT}/_parts/fig5_data.pdf"); dp = data[0]
out = fitz.open(); W = dp.rect.width; H = dp.rect.height
pg = out.new_page(width=W, height=H)
pg.show_pdf_page(pg.rect, data, 0)
# panel a occupies the top-left cell: place schematic scaled to the left column width
tw = W * 0.50; th = tw * clip.height / clip.width
pg.draw_rect(fitz.Rect(0, 0, tw + 4, th + 4), color=None, fill=(1, 1, 1))
pg.show_pdf_page(fitz.Rect(0, 0, tw, th), src, 0, clip=clip)
out.save(f"{OUT}/Figure5.pdf")
out[0].get_pixmap(dpi=300).save(f"{OUT}/Figure5.png")
print("Figure5 written")
