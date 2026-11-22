"""Figure 3: zero-shot knockout-response ranking. Original layout and style; corrected ground truth
(condition-matched pseudobulk DE, guide-clean controls). Panel a is the original schematic (unit counts updated)."""
import json, os, numpy as np, pandas as pd, fitz
from common import *

SD = pd.read_csv(f"{NR}/figures/stats/source_data_fig3_4_5_pb.csv")
ST = json.load(open(f"{NR}/analyses/stats/stats_summary_pb_bh.json"))
PAP = json.load(open(f"{NR}/analyses/untouched/stats_papalexi.json"))


def per_ko(set_, scr, metric="rr_auroc"):
    x = SD[(SD.set == set_) & (SD.cohort == scr) & (SD.metric == metric) & SD.unit.notna()]
    return x.groupby("unit")["value"].mean().astype(float)


def ms(set_, scr):
    v = per_ko(set_, scr).dropna(); return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)


Q3 = {(c["b"], c["scope"]): c.get("p_bh_fig3", 1.0) for c in ST["fig3_comparisons"]
      if c["ma"] == "rr_auroc" and c["a"] == "f3:cmc_s42"}
Q4 = {(c["b"], c["scope"]): c["p_boot_bh_panel"] for c in ST["fig4_comparisons"] if c["ma"] == "rr_auroc"}
# bootstrap-based q for Fig 3 contrasts (joint knockout x control-cell bootstrap, BH over the figure)
P3 = {(c["b"], c["scope"]): c["p_boot"] for c in ST["fig3_comparisons"] if c["ma"] == "rr_auroc"}

fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1, 1], hspace=0.62, wspace=0.28)

# ---- b: development screens, RNA-response AUROC ------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
models = [("CMC-VAE", "f3:cmc_s42"), ("totalVI", "f3:totalvi_ko"), ("Correlation", "f3:corr"), ("DeepSEM", "f3:deepsem_ko")]
x = np.arange(3); slot = 0.84 / len(models); w = slot * 0.92
tops = {}
for i, (m, s) in enumerate(models):
    st = [ms(s, c) for c in SCREENS]
    xx = x - 0.42 + slot * (i + 0.5)
    ax.bar(xx, [a[0] for a in st], w, yerr=[a[1] for a in st], color=COL[m], label=m, error_kw=EK)
    tops[m] = (xx, [a[0] + a[1] for a in st])
chance(ax); ax.set_ylim(0.4, 1.0)
ax.set_xticks(x); ax.set_xticklabels([f"{SLBL[c]}\n(n = {ms('f3:cmc_s42', c)[2]})" for c in SCREENS])
ax.set_ylabel("RNA-response AUROC")
ax.set_title("Development CRISPR screens", fontsize=8.5)
for j, c in enumerate(SCREENS):
    y = max(max(t[1][j] for t in tops.values()), 0) + 0.02
    for m, s in models[1:]:
        q = P3.get((s, c), 1); qb = Q3.get((s, c), 1)
        sig = lab(max(q, qb))
        x1, x2 = tops["CMC-VAE"][0][j], tops[m][0][j]
        bracket(ax, x1, x2, y, sig); y += 0.045
ax.legend(loc="upper left", ncol=4, fontsize=7, handlelength=1.2, columnspacing=0.9, bbox_to_anchor=(0, 1.0))
panel(ax, "b", x=-0.11)

# ---- c: untouched screen ------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
T = PAP["zeroshot_table"]; Pq = PAP["zeroshot_paired"]
pm = [("CMC-VAE", "CMC-VAE (seed mean)", None), ("totalVI", "totalvi", "CMC-VAE minus totalvi"),
      ("Correlation", "corr", "CMC-VAE minus corr"), ("DeepSEM", "deepsem", "CMC-VAE minus deepsem")]
xs = np.arange(len(pm))
for i, (m, k, _) in enumerate(pm):
    r = T[k]["rr_auroc"]
    ax.bar(i, r["mean"], 0.68, yerr=r["sem"], color=COL[m], error_kw=EK)
chance(ax); ax.set_ylim(0.4, 1.0)
ax.set_xticks(xs); ax.set_xticklabels([m for m, _, _ in pm], rotation=0, fontsize=7)
ax.set_ylabel("RNA-response AUROC")
ax.set_title(f"Untouched screen (Papalexi, THP-1; n = {T['corr']['rr_auroc']['n']})", fontsize=8.5)
y = max(T[k]["rr_auroc"]["mean"] + T[k]["rr_auroc"]["sem"] for _, k, _ in pm) + 0.02
for i, (m, k, cmp) in enumerate(pm[1:], start=1):
    bracket(ax, 0, i, y, lab(Pq[cmp]["rr_auroc"]["q_bh"])); y += 0.045
panel(ax, "c", x=-0.2)

# ---- d: latent-graph ablation -------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
arms = [("Learned", "f3:cmc_s42"), ("Shuffled", "f4:Arr_shuffled"), ("Zeroed", "f4:Arr_zeroed")]
slot = 0.8 / len(arms); w = slot * 0.92; tops = {}
for i, (a, s) in enumerate(arms):
    st = [ms(s, c) for c in SCREENS]; xx = x - 0.4 + slot * (i + 0.5)
    ax.bar(xx, [v[0] for v in st], w, yerr=[v[1] for v in st], color=COL[a], label=a, error_kw=EK)
    tops[a] = (xx, [v[0] + v[1] for v in st])
chance(ax); ax.set_ylim(0.4, 1.0)
ax.set_xticks(x); ax.set_xticklabels([SLBL[c] for c in SCREENS])
ax.set_ylabel("RNA-response AUROC")
ax.set_title(r"Ablating the latent RNA graph $A_{rr}$", fontsize=8.5)
for j, c in enumerate(SCREENS):
    y = max(t[1][j] for t in tops.values()) + 0.02
    for a, s in arms[1:]:
        bracket(ax, tops["Learned"][0][j], tops[a][0][j], y, lab(Q4[(s, c)])); y += 0.045
ax.legend(loc="upper left", ncol=3, fontsize=7, handlelength=1.2, bbox_to_anchor=(0, 1.0))
panel(ax, "d", x=-0.11)

# ---- e: per-knockout comparison with correlation ------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
SC = {"frangieh": "#2b6cb0", "iccite": "#63b3ed", "arce": "#1a365d"}
allx, ally = [], []
for c in SCREENS:
    a = per_ko("f3:cmc_s42", c); b = per_ko("f3:corr", c); g = a.index.intersection(b.index)
    ax.scatter(b[g], a[g], s=6, lw=0, alpha=0.75, color=SC[c], label=SLBL[c])
    allx += list(b[g]); ally += list(a[g])
allx, ally = np.array(allx), np.array(ally)
lim = (0.3, 1.0); ax.plot(lim, lim, ls=":", lw=0.9, c="#4a5568")
ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
ax.set_xlabel("Correlation baseline AUROC"); ax.set_ylabel("CMC-VAE AUROC")
frac = np.mean(ally > allx)
ax.text(0.97, 0.05, f"{100*frac:.0f}% of {len(allx)} knockouts\nabove the diagonal", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7)
ax.legend(loc="upper left", fontsize=7, handletextpad=0.1, markerscale=1.6, frameon=True, framealpha=1, edgecolor="white", borderpad=0.3)
ax.set_title("Per-knockout comparison", fontsize=8.5)
panel(ax, "e", x=-0.2)

fig.savefig(f"{OUT}/_parts/fig3_data.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---- compose with the original schematic (panel a) ----------------------------------------------
src = fitz.open(f"{R}/Figures_Main/Figure3.pdf"); W = src[0].rect.width
page = src[0]
for old, new in [("181", "174"), ("214", "122")]:
    for r in page.search_for(old):
        if r.y1 < 80:                     # only the schematic counts (top of page)
            page.add_redact_annot(r, fill=(1, 1, 1))
for s_ in ("Held-out CRISPR screens", "held out from training"):
    for r in page.search_for(s_):
        page.add_redact_annot(r, fill=(1, 1, 1))
page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
FB, FI = os.path.expanduser("~/.fonts/Carlito-Bold.ttf"), os.path.expanduser("~/.fonts/Carlito-Italic.ttf")
for txt, ff, fn, sz, cx, y, col in [("Development CRISPR screens", FB, "carlb", 8.0, 189.48, 14.93, (0x22/255,) * 3),
                                    ("not used in training", FI, "carli", 7.0, 185.67, 136.47, (0x55/255,) * 3)]:
    wdt = fitz.Font(fontfile=ff).text_length(txt, fontsize=sz)
    x0 = 148.93 if txt.startswith("Development") else cx - wdt / 2
    page.insert_text((x0, y), txt, fontsize=sz, fontname=fn, fontfile=ff, color=col)
for old, new, y in [("181", "174", 31.7), ("214", "122", 62.9)]:
    page.insert_text((223.9, y + 6.15), new, fontsize=7.5, fontname="carlb", fontfile=os.path.expanduser("~/.fonts/Carlito-Bold.ttf"), color=(0.10, 0.13, 0.17))
schem_h = 160.0
data = fitz.open(f"{OUT}/_parts/fig3_data.pdf"); dr = data[0].rect
scale = W / dr.width; dh = dr.height * scale
out = fitz.open(); pg = out.new_page(width=W, height=schem_h + dh + 6)
pg.show_pdf_page(fitz.Rect(0, 0, W, schem_h), src, 0, clip=fitz.Rect(0, 0, W, schem_h))
pg.show_pdf_page(fitz.Rect(0, schem_h + 6, W, schem_h + 6 + dh), data, 0)
out.save(f"{OUT}/Figure3.pdf")
out[0].get_pixmap(dpi=300).save(f"{OUT}/Figure3.png")
print("Figure3 written")
