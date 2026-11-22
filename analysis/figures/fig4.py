"""Figure 4: a cognate RNA-to-protein translation map and cross-modal translation of knockout responses.
a, schematic of the cross-modal path (cognate map -> fit to a context's control cells -> translate knockout RNA responses);
b, cognate edge strength vs decodability; c, control-cell imputation (adapted vs PBMC, Wilcoxon over proteins, BH);
d,e, translation of measured knockout RNA responses into protein responses."""
import json, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from common import *

D = json.load(open(f"{R}/results/translation_decodability.json"))["analysis1_decodability_map"]
T = json.load(open(f"{NR}/figures/reviewfix/fig4d_ci.json"))
A = json.load(open(f"{NR}/figures/adapt/source_data_adapt.json"))["A_imputation"]
QB = json.load(open(f"{NR}/figures/adapt/fig4b_imputation_tests.json"))

RNA, PROT, XM, RED, GREY, INK = "#2b6cb0", "#2f855a", "#dd6b1f", "#c53030", "#a0aec0", "#1a202c"

fig = plt.figure(figsize=(7.2, 6.8))
gs = fig.add_gridspec(2, 2, top=0.70, bottom=0.0, hspace=0.42, wspace=0.32)

# =============================================================================== a: schematic
ax = fig.add_axes([0.0, 0.775, 1.0, 0.215]); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
ax.text(-1.0, 29.4, "a", fontsize=11, fontweight="bold", va="top", ha="left")

def header(x0, x1, txt):
    ax.text((x0 + x1) / 2, 29.2, txt, ha="center", va="top", fontsize=8.3, fontweight="bold", color=INK)
    ax.plot([x0, x1], [26.6, 26.6], color="#cbd5e0", lw=0.8)

def arrow(x0, y0, x1, y1, c="#718096", lw=1.0, ms=7, ls="-", cs="arc3"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, ls=ls,
                                 connectionstyle=cs, shrinkA=0, shrinkB=0))

def node(x, y, c, r=1.15):
    ax.add_patch(Circle((x, y), r, color=c, zorder=4))

# ---- block 1: cognate translation map learned from PBMC CITE-seq
header(0.5, 29, "Cognate translation map")
genes = ["CD19", "CD14", "CD8A", "CD79B"]; prots = ["CD19", "CD14", "CD8", "CD79b"]
ys = [20.6, 16.6, 12.6, 8.6]
for i, ya in enumerate(ys):
    for j, yb in enumerate(ys):
        if i != j:
            ax.plot([9.6, 19.4], [ya, yb], color="#e2e8f0", lw=0.6, zorder=1)
for y in ys:
    ax.plot([9.6, 19.4], [y, y], color=XM, lw=2.2, zorder=2, solid_capstyle="round")
for g, p, y in zip(genes, prots, ys):
    node(9.6, y, RNA, 0.95); node(19.4, y, PROT, 0.95)
    ax.text(8.0, y, g, ha="right", va="center", fontsize=7, color=INK, style="italic")
    ax.text(21.0, y, p, ha="left", va="center", fontsize=7, color=INK)
ax.text(9.6, 23.3, "transcript", ha="center", fontsize=6.8, color=RNA)
ax.text(19.4, 23.3, "protein", ha="center", fontsize=6.8, color=PROT)
ax.text(14.5, 4.6, "PBMC CITE-seq, 809,130 cells", ha="center", fontsize=6.8, color="#4a5568")
ax.text(14.5, 1.6, r"cognate edges strongest in $G_{rp}$", ha="center", fontsize=6.8, color="#4a5568")

# ---- block 2: fit to a new context with control cells only
header(33, 56, "Fit to a new cellular context")
for k, (cx, cy) in enumerate([(37.4, 20.5), (40.6, 22.4), (39.6, 18.2), (43.0, 19.9), (36.6, 16.6), (42.0, 16.0)]):
    ax.add_patch(Circle((cx, cy), 1.45, facecolor="#edf2f7", edgecolor="#718096", lw=0.7, zorder=3))
    ax.add_patch(Circle((cx - 0.45, cy), 0.42, color=RNA, zorder=4)); ax.add_patch(Circle((cx + 0.45, cy), 0.42, color=PROT, zorder=4))
ax.text(39.8, 11.8, "unperturbed control cells\npaired RNA + protein", ha="center", va="center", fontsize=6.8, color="#4a5568",
        linespacing=1.15)
arrow(45.6, 19.2, 49.0, 19.2)
ax.add_patch(FancyBboxPatch((49.4, 15.2), 6.4, 8.0, boxstyle="round,pad=0.25,rounding_size=0.8", facecolor="white",
                            edgecolor=RNA, lw=1.1, zorder=3))
ax.text(52.6, 19.2, "CMC-\nVAE", ha="center", va="center", fontsize=7, fontweight="bold", color=RNA, linespacing=1.0, zorder=5)
ax.text(44.5, 5.6, "no perturbation labels", ha="center", fontsize=6.8, color=RED)
ax.text(44.5, 2.6, "no knockout cells", ha="center", fontsize=6.8, color=RED)

# ---- block 3: translate knockout RNA responses through the cross-modal path
header(61, 99.5, "Translate knockout responses across modalities")
def bars(x0, y0, vals, c, w=0.75, s=2.4):
    for k, v in enumerate(vals):
        ax.add_patch(Rectangle((x0 + k * (w + 0.35), y0), w, v * s, color=c, lw=0, zorder=3))
    ax.plot([x0 - 0.3, x0 + len(vals) * (w + 0.35)], [y0, y0], color="#4a5568", lw=0.6, zorder=3)
bars(61.6, 17.5, [0.9, -0.7, 1.3, 0.4, -1.1, 0.6, -0.4], RNA)
ax.text(65.0, 9.6, "measured\nknockout\nRNA response", ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.1)
arrow(69.6, 17.5, 72.4, 17.5)
zr = [(74.4, 21.4), (74.4, 17.5), (74.4, 13.6)]; zp = [(82.4, 19.5), (82.4, 15.5)]
for a_ in zr:
    for b_ in zp:
        ax.add_patch(FancyArrowPatch(a_, b_, arrowstyle="-|>", mutation_scale=6, color=XM, lw=1.3, shrinkA=4.5, shrinkB=4.5, zorder=2))
for (x, y) in zr: node(x, y, RNA)
for (x, y) in zp: node(x, y, PROT)
ax.text(74.4, 23.6, r"$z_{r}$", ha="center", fontsize=7.5, color=RNA)
ax.text(82.4, 21.7, r"$z_{p}$", ha="center", fontsize=7.5, color=PROT)
ax.text(78.4, 9.0, r"$W_{rp}$", ha="center", fontsize=7.5, color=XM, fontweight="bold")
arrow(84.6, 17.5, 87.6, 17.5)
bars(88.2, 17.5, [0.8, -0.9, 1.1, -0.3, 0.7], PROT)
ax.text(91.0, 9.6, "predicted\nprotein\nresponse", ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.1)
ax.text(80.3, 3.6, "scored against measured protein responses (centred pearDE, retrieval)", ha="center", fontsize=6.8,
        color="#4a5568")
arrow(56.6, 19.2, 60.6, 19.2, c="#718096")

# =============================================================================== b: cognate edge strength tracks protein decodability
ax = fig.add_subplot(gs[0, 0])
pp = D["per_protein"]
x = np.array([p["decodability"] for p in pp]); y = np.array([p["edge_abs"] for p in pp])
ax.scatter(x, y, s=14, color=RNA, lw=0.3, edgecolor="white", zorder=3)
ax.set_yscale("log")
OFF = {"CD8": (6, 4), "CD14": (6, -10), "CD19": (-16, 9), "CD79b": (-44, 3)}
for p in pp:
    if p["protein"] in OFF:
        ax.annotate(p["protein"], (p["decodability"], p["edge_abs"]), xytext=OFF[p["protein"]], textcoords="offset points",
                    fontsize=6.8, color="#2d3748", arrowprops=dict(arrowstyle="-", lw=0.5, color="#718096", shrinkA=0, shrinkB=2))
r = D["spearman_edge_vs_decodability"]
ax.text(0.03, 0.96, f"Spearman ρ = {r['rho']:.2f}\nn = {r['n']} cognate pairs", transform=ax.transAxes, va="top", fontsize=7.5)
ax.set_xlabel("Protein predictability from its own transcript (Pearson r)")
ax.set_ylabel(r"Cognate edge strength $|G_{rp}[g,p]|$")
ax.set_title("Cognate translation edges track decodability", fontsize=8.5)
panel(ax, "b", x=-0.2)

# =============================================================================== c: protein-from-RNA imputation in screen control cells
ax = fig.add_subplot(gs[0, 1])
get = {(r_["variant"], r_["domain"]): r_ for r_ in A if r_["base"] == "sp"}
scr = ["frangieh", "iccite", "arce"]; xs = np.arange(3); w = 0.34
tops = np.zeros(3)
for i, (v, lb, c) in enumerate([("none", "CMC-VAE, PBMC model", "#90cdf4"), ("i", "CMC-VAE, adapted to control cells", RNA)]):
    m = np.array([get[(v, s)]["rna_only"] for s in scr])
    lo = m - np.array([get[(v, s)]["rna_only_range"][0] for s in scr])
    hi = np.array([get[(v, s)]["rna_only_range"][1] for s in scr]) - m
    ax.bar(xs + (i - 0.5) * w, m, w * 0.95, yerr=[lo, hi], color=c, label=lb, error_kw=EK)
    tops = np.maximum(tops, m + hi)
for j, s in enumerate(scr):
    bracket(ax, j - w / 2, j + w / 2, tops[j] + 0.025, lab(QB[s]["q"]))
ax.set_xticks(xs); ax.set_xticklabels([SLBL[s] for s in scr])
ax.set_ylabel("Protein predicted from RNA\n(median per-protein r)")
ax.set_ylim(0, 0.5); ax.axhline(0, lw=0.6, c="#4a5568")
ax.legend(loc="upper left", fontsize=7, handlelength=1.3)
ax.set_title("Cross-modal imputation in new cell contexts", fontsize=8.5)
panel(ax, "c", x=-0.2)

# =============================================================================== d,e: translating knockout RNA responses into protein responses
names = [("CMC-VAE, unadapted", "PBMC\nmodel", "#90cdf4"), ("CMC-VAE, adapted", "Adapted\nCMC-VAE", RNA),
         ("Corpus correlation map", "Correlation\nmap", COL["Correlation"]), ("Cognate transcript", "Cognate\ntranscript", COL["GRNBoost2"])]
cmp = {(c_["metric"], c_["b"]): c_["p_boot"] for c_ in T["contrasts"]}
for k, (metric, ylab, yl, pl, ttl) in enumerate([
        ("pe_cent", "Centred protein pearDE", (0, 0.8), "d", "Knockout RNA → protein response (204 knockouts)"),
        ("retrieval", "Protein-response retrieval", (0.4, 1.06), "e", "Knockout RNA → protein response (retrieval)")]):
    ax = fig.add_subplot(gs[1, k])
    for i, (key, lb, c) in enumerate(names):
        r_ = T["means"][f"{key}|{metric}"]
        ax.bar(i, r_["mean"], 0.7, color=c, zorder=2)
        ax.errorbar(i, r_["mean"], yerr=[[r_["mean"] - r_["ci"][0]], [r_["ci"][1] - r_["mean"]]], fmt="none", zorder=4, **EK)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n_[1] for n_ in names], fontsize=7)
    ax.set_ylim(*yl); ax.set_ylabel(ylab)
    chance(ax) if metric == "retrieval" else ax.axhline(0, lw=0.6, c="#4a5568")
    top = max(T["means"][f"{n_[0]}|{metric}"]["ci"][1] for n_ in names)
    yb = top + 0.03 * (yl[1] - yl[0])
    for j, (key, _, _) in enumerate(names):
        if j == 1: continue
        bracket(ax, 1, j, yb, lab(cmp.get((metric, key), 1))); yb += 0.065 * (yl[1] - yl[0])
    ax.set_title(ttl, fontsize=8.5)
    panel(ax, pl, x=-0.2)

fig.savefig(f"{OUT}/Figure4.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}/Figure4.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
print("Figure4 written")
