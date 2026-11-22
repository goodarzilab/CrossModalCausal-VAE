"""Per-protein LEARNED DECODABILITY of the projected RNA->protein translation map.

Turns the single "12x cognate enrichment" summary (analysis/fig2_networks/translation_nulls.py) into a
per-protein story, grounded in the biological fact that <50% of PBMC surface proteins
correlate with their own mRNA (post-transcriptional regulation, shedding, epitope vs
transcript mismatch). Question: does CMC-VAE place STRONG + SPECIFIC cognate translation
edges exactly on the transcript-decodable proteins (CD3/CD19/CD8/CD14) and weak/absent
edges on the post-transcriptionally-controlled ones (CD69/CXCR4/LAG3/...)?

Inputs (all on disk, model-independent except G_rp):
  <run>/Transcription_RNA_to_Prot.csv    projected translation map G_rp [genes x proteins]
  results/baselines/_cmcvae_cognate.json empirical per-protein cognate RNA-protein Pearson
                                          (DATA-derived "decodability" axis) + cognate_gene
  results/coupled_panel_2c.json          provenance / cross-check of cognate correlations

Cognate matcher: canon_prot() and the candidate-loop are COPIED VERBATIM from
analysis/fig2_networks/translation_nulls.py (NOT re-derived). That logic finds 27 pairs in this panel; every
one agrees exactly with the curated cognate_gene in _cmcvae_cognate.json (0 disagreements).
The curated mapping is a strict superset (74 pairs; adds standard non-canonical symbols like
CD16->FCGR3A, CD20->MS4A1) and is the one the empirical decodability axis was computed
against, so the 74-pair curated set is the primary analysis and the 27-pair canon subset is
reported as a robustness check.

Analyses:
  1  DECODABILITY MAP  -- Spearman( |cognate G_rp edge| , empirical cognate Pearson ) and
     Spearman( within-column specificity , empirical cognate Pearson ), bootstrap CI over
     proteins.
  2  SIGN POLARITY     -- fraction of cognate edges with G_rp>0 (central dogma => positive),
     split by decodability; binomial test; post-transcriptional case studies.
  3  OFF-DIAGONAL (trans) STRUCTURE -- for each protein column, top-k non-cognate genes
     tested for enrichment in a curated same-lineage marker-gene set (B/T/Mono/NK), vs the
     matched-degree within-column permutation null (reused from translation_nulls.py Null C).

Outputs:
  results/translation_decodability.json
  paper/figures/_new_translation_decodability.pdf (+ .png)

Usage:  python analysis/fig4_cross_modal/translation_decodability.py [results/sweep_sparse/<run>]
        (default results/sweep_sparse/sp_dec_gl1000)
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _CMC_ROOT
RES = os.path.join(REPO, "results")
sys.path.insert(0, os.path.join(_CMC_ROOT, "analysis", "figures", "original"))
from style import plt, C, panel_label, save

np.random.seed(42)
N_PERM = 2000
N_BOOT = 10000

# ------------------------------------------------------------------ inputs
run_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(RES, "sweep_sparse", "sp_dec_gl1000")
run = os.path.basename(run_dir)
csv_path = os.path.join(run_dir, "Transcription_RNA_to_Prot.csv")
print(f"=== translation_decodability: {run} ===")

G = pd.read_csv(csv_path, index_col=0)
genes = [str(g) for g in G.index]
prots = [str(p) for p in G.columns]
V = G.values.astype(float)          # signed G_rp [ng, npr]
A = np.abs(V)                        # |G_rp|
ng, npr = V.shape
gup = {g.upper(): i for i, g in enumerate(genes)}
print(f"G_rp {V.shape}  |G_rp| mean={A.mean():.2e}")

cj = json.load(open(os.path.join(RES, "baselines", "_cmcvae_cognate.json")))
pnames = cj["protein_names"]
cur_gene = {p: g for p, g in zip(pnames, cj["cognate_gene"])}
cur_corr = {p: c for p, c in zip(pnames, cj["cognate_mrna_pearson"])}
assert prots == pnames, "protein-column order must match _cmcvae_cognate.json"

# provenance cross-check with coupled_panel_2c.json (does not change any result)
try:
    cp = json.load(open(os.path.join(RES, "coupled_panel_2c.json")))
    cp_corr = {m["protein"]: m["corr"] for m in cp["markers"]}
    xchk = [(p, cur_corr[p], cp_corr[p]) for p in cp_corr if p in cur_corr and cur_corr[p] is not None]
    max_dev = max((abs(a - b) for _, a, b in xchk), default=0.0)
    print(f"coupled_panel_2c cross-check: {len(xchk)} overlap, max |dev|={max_dev:.3f}")
except Exception as e:
    max_dev = None; print("coupled_panel_2c cross-check skipped:", e)

# totalVI comparison rho for the decodability axis (honest CMC-vs-totalVI margin in panel A)
try:
    _tvj = json.load(open(os.path.join(RES, "decodability_vs_totalvi.json")))
    totalvi_rho = float(_tvj["totalvi_rho"])
except Exception as e:
    totalvi_rho = None; print("decodability_vs_totalvi load skipped:", e)


# ------------------------------------------------------------------ cognate matcher (VERBATIM from translation_nulls.py)
def canon_prot(x):
    x = re.sub(r'^AB_', '', x); x = re.sub(r'^(human|mouse)', '', x, flags=re.I)
    x = re.sub(r'-\d+$', '', x); return x.split('_')[0].upper()

canon_cognate = []  # (gene_idx, prot_idx) -- exact translation_nulls logic
for j, p in enumerate(prots):
    c = canon_prot(p)
    for cand in (c, c.replace("CD", ""), c + "E", c + "A"):
        if cand in gup:
            canon_cognate.append((gup[cand], j)); break
canon_pairs = {prots[pj]: genes[gi] for gi, pj in canon_cognate}
print(f"canon_prot cognate pairs: {len(canon_cognate)}")

# curated 74-pair superset (empirical decodability axis was computed against these genes)
pairs = []  # dict per protein with a cognate gene present in this panel
n_agree = n_disagree = 0
for p in prots:
    g = cur_gene[p]; c = cur_corr[p]
    if g is None or c is None:
        continue
    gi = gup.get(g.upper())
    if gi is None:
        continue
    pj = prots.index(p)
    if p in canon_pairs:
        if canon_pairs[p].upper() == g.upper(): n_agree += 1
        else: n_disagree += 1
    edge = A[gi, pj]
    rank_from_top = float((A[:, pj] > A[gi, pj]).sum()) / ng   # frac of genes STRONGER than cognate (0=best)
    specificity = 1.0 - rank_from_top                          # frac of column genes cognate beats (1=best)
    pairs.append(dict(protein=p, gene=g, gene_idx=gi, prot_idx=pj,
                      decodability=float(c), edge_abs=float(edge), edge_signed=float(V[gi, pj]),
                      specificity=float(specificity), rank_from_top=float(rank_from_top),
                      in_canon=bool(p in canon_pairs)))
print(f"curated cognate pairs used: {len(pairs)}  (canon subset agree={n_agree} disagree={n_disagree})")
assert n_disagree == 0, "curated mapping must agree with canon_prot on the overlap"

dec = np.array([d["decodability"] for d in pairs])
edge = np.array([d["edge_abs"] for d in pairs])
spec = np.array([d["specificity"] for d in pairs])
sgn = np.array([d["edge_signed"] for d in pairs])
in_canon = np.array([d["in_canon"] for d in pairs])


# ================================================================== ANALYSIS 1: decodability map
def spearman_boot(x, y, nboot=N_BOOT):
    rho = float(spearmanr(x, y).statistic)
    p = float(spearmanr(x, y).pvalue)
    n = len(x); rs = np.empty(nboot)
    for b in range(nboot):
        idx = np.random.randint(0, n, n)
        rs[b] = spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.percentile(rs[np.isfinite(rs)], [2.5, 97.5])
    return dict(rho=rho, p=p, ci95=[float(lo), float(hi)], n=int(n))

edge_dec = spearman_boot(edge, dec)
spec_dec = spearman_boot(spec, dec)
# robustness: canon-only 27-pair subset
edge_dec_canon = spearman_boot(edge[in_canon], dec[in_canon]) if in_canon.sum() > 5 else None
print(f"[A1] Spearman(edge,decodability) rho={edge_dec['rho']:.3f} CI{edge_dec['ci95']} p={edge_dec['p']:.1e}")
print(f"[A1] Spearman(specificity,decodability) rho={spec_dec['rho']:.3f} CI{spec_dec['ci95']} p={spec_dec['p']:.1e}")

analysis1 = dict(n_proteins=len(pairs), spearman_edge_vs_decodability=edge_dec,
                 spearman_specificity_vs_decodability=spec_dec,
                 spearman_edge_vs_decodability_canon27=edge_dec_canon,
                 per_protein=pairs)


# ================================================================== ANALYSIS 2: sign polarity
pos = sgn > 0
frac_pos = float(pos.mean())
binom_p = float(binomtest(int(pos.sum()), len(pos), 0.5).pvalue)
med = np.median(dec)
hi_mask = dec >= med
frac_pos_hi = float(pos[hi_mask].mean())
frac_pos_lo = float(pos[~hi_mask].mean())
# case studies: the least-decodable cognate proteins (post-transcriptional / activation / chemokine)
order_lo = np.argsort(dec)
cases = []
for i in order_lo[:10]:
    d = pairs[i]
    cases.append(dict(protein=d["protein"], gene=d["gene"], decodability=d["decodability"],
                      edge_signed=d["edge_signed"], specificity=d["specificity"], positive=bool(sgn[i] > 0)))
print(f"[A2] frac cognate edges positive = {frac_pos:.3f} (binom p={binom_p:.1e}); "
      f"high-decodable {frac_pos_hi:.2f} vs low-decodable {frac_pos_lo:.2f} positive")
analysis2 = dict(n_cognate_edges=len(pos), frac_positive=frac_pos, binom_p_vs_half=binom_p,
                 frac_positive_high_decodable=frac_pos_hi, frac_positive_low_decodable=frac_pos_lo,
                 least_decodable_case_studies=cases)


# ================================================================== ANALYSIS 3: off-diagonal (trans) lineage structure
LINEAGE_MARKERS = {   # curated lineage -> canonical marker GENES (filtered to panel below)
    "B":    ["MS4A1", "CD79A", "CD79B", "CD19", "CD22", "IGHM", "IGHD", "CD24", "BANK1", "TCL1A", "VPREB3", "FCRL1"],
    "T":    ["CD3D", "CD3E", "CD3G", "CD2", "CD8A", "CD8B", "IL7R", "TRAC", "TRBC2", "CD247", "LTB", "CD27"],
    "Mono": ["CD14", "LYZ", "FCN1", "CSF1R", "FCGR3A", "S100A8", "S100A9", "VCAN", "CD68", "CD36", "FCGR1A", "MS4A7", "CD163"],
    "NK":   ["NCAM1", "NKG7", "GNLY", "KLRD1", "KLRB1", "KLRK1", "NCR1", "NCR3", "KIR2DL1", "KIR2DL3", "KIR3DL1", "GZMB", "PRF1"],
}
PROT_LINEAGE = {  # curated protein -> lineage (only unambiguous surface markers)
    "CD19": "B", "CD20": "B", "CD22": "B", "CD79b": "B", "IgD": "B", "IgM": "B", "CD21": "B", "CD24": "B",
    "CD3-1": "T", "CD2": "T", "CD8": "T", "CD27": "T", "CD28": "T", "CD127": "T", "TCR-1": "T", "TCR-2": "T",
    "CD14": "Mono", "CD64": "Mono", "CD11b-1": "Mono", "CD11c": "Mono", "CD36": "Mono", "CD163": "Mono", "CLEC12A": "Mono",
    "CD56-1": "NK", "CD158": "NK", "CD158b": "NK", "CD158e1": "NK", "CD335": "NK", "CD337": "NK", "CD161": "NK", "CD314": "NK",
}
lin_idx = {L: set(gup[g.upper()] for g in gs if g.upper() in gup) for L, gs in LINEAGE_MARKERS.items()}

def offdiag_enrichment(k):
    cols = []   # (prot_idx, cognate_gene_idx or None, marker_index_set, lineage, protein)
    for p, L in PROT_LINEAGE.items():
        if p not in prots:
            continue
        pj = prots.index(p)
        cg = cur_gene.get(p); cogi = gup.get(cg.upper()) if cg else None
        mset = set(lin_idx[L])
        if cogi is not None:
            mset.discard(cogi)   # exclude self cognate -> pure trans test
        cols.append((pj, cogi, mset, L, p))
    tot = k * len(cols)
    hits = 0; per_prot = []; per_lin = {L: [0, 0] for L in LINEAGE_MARKERS}
    for pj, cogi, mset, L, p in cols:
        col = A[:, pj].copy()
        if cogi is not None: col[cogi] = -1.0
        top = np.argsort(-col)[:k]
        h = sum(1 for g in top if g in mset)
        hits += h; per_lin[L][0] += h; per_lin[L][1] += k
        per_prot.append(dict(protein=p, lineage=L, hits=int(h), k=k,
                             top_markers=[genes[g] for g in top if g in mset]))
    obs = hits / tot
    # matched-degree within-column permutation null (translation_nulls Null C): shuffle gene identity per column
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        nh = 0
        for pj, cogi, mset, L, p in cols:
            perm = np.random.permutation(ng)
            cand = perm[perm != cogi][:k] if cogi is not None else perm[:k]
            nh += sum(1 for g in cand if g in mset)
        null[b] = nh / tot
    p_emp = (1 + int((null >= obs).sum())) / (N_PERM + 1)
    z = float((obs - null.mean()) / (null.std() + 1e-12))
    per_lin_prec = {L: (per_lin[L][0] / per_lin[L][1] if per_lin[L][1] else 0.0) for L in per_lin}
    return dict(k=int(k), n_proteins=len(cols), pooled_precision=float(obs),
                null_mean=float(null.mean()), null_std=float(null.std()),
                null_p95=float(np.percentile(null, 95)), z=z, p_emp=float(p_emp),
                per_lineage_precision=per_lin_prec, per_protein=per_prot)

offdiag = {f"k{k}": offdiag_enrichment(k) for k in (10, 20)}
for k in (10, 20):
    o = offdiag[f"k{k}"]
    print(f"[A3] top-{k}: pooled precision={o['pooled_precision']:.3f} vs null {o['null_mean']:.4f} "
          f"z={o['z']:.1f} p={o['p_emp']:.1e}")
analysis3 = dict(lineage_markers={L: [g for g in gs if g.upper() in gup] for L, gs in LINEAGE_MARKERS.items()},
                 protein_lineage=PROT_LINEAGE, results=offdiag)


# ================================================================== save JSON
out = dict(run=run, csv=os.path.relpath(csv_path, REPO),
           n_genes=ng, n_proteins=npr,
           cognate_matcher="canon_prot copied verbatim from analysis/fig2_networks/translation_nulls.py; curated 74-pair superset from _cmcvae_cognate.json (agrees with canon on 27/27 overlap)",
           n_canon_pairs=len(canon_cognate), n_curated_pairs=len(pairs),
           coupled_panel_2c_max_dev=max_dev,
           n_perm=N_PERM, n_boot=N_BOOT,
           analysis1_decodability_map=analysis1,
           analysis2_sign_polarity=analysis2,
           analysis3_offdiagonal_lineage=analysis3)
out_json = os.path.join(RES, "translation_decodability.json")
json.dump(out, open(out_json, "w"), indent=2)
print("wrote", out_json)


# ================================================================== FIGURE
fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.8))

# ---- Panel A: decodability -> cognate edge strength (log), colored by specificity
axA = axs[0]
sc = axA.scatter(dec, edge, c=spec, cmap="viridis", s=26, edgecolor="k", linewidth=0.3,
                 vmin=0, vmax=1, zorder=3)
axA.set_yscale("log")
axA.set_xlabel("empirical cognate mRNA–protein Pearson\n(data-derived decodability)")
axA.set_ylabel("|cognate translation edge|  $|G_{rp}[\\mathrm{gene},\\mathrm{prot}]|$")
axA.set_title("A  Learned edge tracks decodability", loc="left", fontweight="bold")
cb = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.02)
cb.set_label("within-column\nspecificity", fontsize=7)
cb.ax.tick_params(labelsize=6.5)
# label a few high- and low-decodability proteins
by_dec = sorted(range(len(pairs)), key=lambda i: dec[i])
lab_lo = [i for i in by_dec[:14] if pairs[i]["protein"] in ("CD69", "CD184", "CD223", "CD314", "CD357", "CD207")][:4]
lab_hi = [i for i in by_dec[::-1] if pairs[i]["protein"] in ("CD8", "CD19", "CD3-1", "CD14", "CD20", "CD16")][:4]
for i in lab_hi + lab_lo:
    axA.annotate(pairs[i]["protein"], (dec[i], edge[i]), fontsize=6.2, color=C["ink"],
                 xytext=(3, 2), textcoords="offset points")
_tv_line = f"   (totalVI $\\rho$ = {totalvi_rho:.2f})" if totalvi_rho is not None else ""
axA.text(0.03, 0.97,
         f"$\\rho_{{edge}}$ = {edge_dec['rho']:.2f}  [{edge_dec['ci95'][0]:.2f}, {edge_dec['ci95'][1]:.2f}]{_tv_line}\n"
         f"$\\rho_{{spec}}$ = {spec_dec['rho']:.2f}  [{spec_dec['ci95'][0]:.2f}, {spec_dec['ci95'][1]:.2f}]\n"
         f"n = {len(pairs)} proteins",
         transform=axA.transAxes, ha="left", va="top", fontsize=7,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["muted"], lw=0.6))

# ---- Panel B: sign polarity vs decodability
axB = axs[1]
col_sign = np.where(pos, C["pos"], C["neg"])
axB.axhline(0, color=C["ink"], lw=0.9, zorder=2)
axB.scatter(dec, sgn, c=col_sign, s=26, edgecolor="k", linewidth=0.3, zorder=3)
axB.set_yscale("symlog", linthresh=1e-6)
axB.set_xlabel("empirical cognate decodability (Pearson)")
axB.set_ylabel("signed cognate edge  $G_{rp}$  (symlog)")
axB.set_title("B  Sign polarity (central dogma → +)", loc="left", fontweight="bold")
# annotate only the 4 most-negative (wrong-polarity) cognate edges, decluttered
neg_i = sorted([i for i in range(len(pairs)) if not pos[i]], key=lambda i: sgn[i])[:4]
try:
    from adjustText import adjust_text
    _texts = [axB.text(dec[i], sgn[i], pairs[i]["protein"], fontsize=6.2, color=C["neg"], va="center")
              for i in neg_i]
    adjust_text(_texts, ax=axB, expand=(1.3, 1.6),
                arrowprops=dict(arrowstyle="-", color=C["neg"], lw=0.4))
except Exception:   # fallback: stagger the labels vertically
    for rank, i in enumerate(neg_i):
        axB.annotate(pairs[i]["protein"], (dec[i], sgn[i]), fontsize=6.2, color=C["neg"],
                     xytext=(5, -4 - 7 * rank), textcoords="offset points", va="center")
axB.text(0.97, 0.03,
         f"{frac_pos*100:.0f}% of cognate edges positive\n"
         f"(binom p = {binom_p:.0e})\n"
         f"high-decodable: {frac_pos_hi*100:.0f}% +   low: {frac_pos_lo*100:.0f}% +",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=7,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["muted"], lw=0.6))

# ---- Panel C: off-diagonal (trans) same-lineage enrichment, precision@10
axC = axs[2]
o = offdiag["k10"]
lins = ["B", "T", "Mono", "NK"]
vals = [o["per_lineage_precision"][L] for L in lins] + [o["pooled_precision"]]
labs = lins + ["pooled"]
xpos = np.arange(len(labs))
colors = [C["model"]] * len(lins) + [C["accent"]]
axC.bar(xpos, vals, color=colors, edgecolor="k", linewidth=0.4, zorder=3)
axC.axhline(o["null_mean"], color=C["neg"], lw=1.2, ls="--", zorder=4)
axC.annotate(f"matched-degree null = {o['null_mean']:.3f}", (0.5, o["null_mean"]),
             xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
             fontsize=6.5, color=C["neg"])
axC.set_xticks(xpos); axC.set_xticklabels(labs)
axC.set_ylabel("precision@10  (same-lineage\nmarkers in top-10 off-diagonal genes)")
axC.set_title("C  Trans edges are lineage-coherent", loc="left", fontweight="bold")
axC.text(0.03, 0.97,
         f"pooled {o['pooled_precision']:.2f} vs null {o['null_mean']:.3f}\n"
         f"z = {o['z']:.0f},  p < {1/(N_PERM+1):.0e}\n"
         f"n = {o['n_proteins']} lineage-typed proteins",
         transform=axC.transAxes, ha="left", va="top", fontsize=7,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["muted"], lw=0.6))

fig.suptitle("Per-protein learned decodability of the RNA→protein translation map",
             fontsize=9, fontweight="bold", y=1.02)
fig.tight_layout()
out_pdf = os.path.join(REPO, "paper", "figures", "AppendixFigS6.pdf")
save(fig, out_pdf)
print("DONE")
