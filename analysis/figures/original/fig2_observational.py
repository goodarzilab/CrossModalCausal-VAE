"""Fig 2 — Observational structure recovery & the cognate central-dogma mapping.
A: GRN/PPI recovery AUROC (CMC-VAE vs totalVI vs GENIE3).  B: signaling-pathway recovery (mean±SD, 3 seeds).
C: cognate matched-vs-unmatched translation weight density.  D: permutation nulls + totalVI specificity contrast."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from style import plt, C, panel_label, save
from scipy.stats import mannwhitneyu, gaussian_kde
R = f"{_CMC_ROOT}"; M = f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000"
def J(p): return json.load(open(p)) if os.path.exists(p) else {}

fig = plt.figure(figsize=(7.2, 8.4)); gs = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.34,
                                                            height_ratios=[1, 1, 0.85])

# ---- A: GRN/PPI recovery AUROC (CMC-VAE, totalVI, GENIE3, stable DeepSEM, correlation) ----
axA = fig.add_subplot(gs[0, 0])
nets = ["coexpr", "STRING", "DoRothEA", "CollecTRI"]
cmc = [0.976, 0.756, 0.678, 0.632]; tvi = [0.977, 0.705, 0.603, 0.575]   # CMC=sp_dec_gl1000 (top sparse); totalVI 809k
gen = [0.979, 0.646, 0.536, 0.517]; dsm = [0.932, 0.658, 0.572, 0.553]   # GRNBoost2 809k (patched threaded); DeepSEM 809k
corr = [1.000, 0.748, 0.674, 0.625]   # 809k; coexpr=1.0: the co-expression reference IS correlation
x = np.arange(len(nets)); w = 0.16
series = [("CMC-VAE", cmc, C["cmc"]), ("correlation", corr, C["shuf"]), ("totalVI", tvi, C["totalvi"]),
          ("GRNBoost2", gen, C["genie3"]), ("DeepSEM", dsm, C["accent"])]
for i, (lab, vals, col) in enumerate(series):
    axA.bar(x + (i - 2) * w, vals, w, label=lab, color=col)
axA.axhline(0.5, ls=":", lw=0.8, c=C["neutral"])
axA.set_xticks(x); axA.set_xticklabels(nets, rotation=20, ha="right")
axA.set_ylabel("recovery AUROC"); axA.set_ylim(0.45, 1.0)
axA.legend(loc="upper right", ncol=1, fontsize=6.0, handlelength=1.2, handletextpad=0.4, labelspacing=0.22)
axA.set_title("GRN/PPI recovery ($G_{rr}$)"); panel_label(axA, "A")

# ---- B: signaling-pathway recovery ----
axB = fig.add_subplot(gs[0, 1])
paths = ["MHC-I*","IFN-α","TCR","TNF-α/NF-κB","IFN-γ","Apoptosis","cGAS-STING","Antigen pres.","JAK-STAT"]
au = [0.922,0.817,0.790,0.786,0.757,0.706,0.693,0.672,0.586]   # sp_dec_gl1000 (pathway_recovery.json)
yy = np.arange(len(paths))[::-1]
axB.barh(yy, au, color=C["cmc2"], edgecolor=C["cmc"])
axB.axvline(0.5, ls=":", lw=0.8, c=C["neutral"])
axB.set_yticks(yy); axB.set_yticklabels(paths); axB.set_xlim(0.5, 0.95); axB.set_xlabel("within-pathway AUROC")
axB.set_title("Signaling-pathway recovery (*=pos. ctrl)"); panel_label(axB, "B", x=-0.42)

# ---- C: cognate matched vs unmatched translation weights (ECDF of |weight|, log-x) ----
axC = fig.add_subplot(gs[1, 0])
def norm(s): return str(s).upper().replace("-","").replace("_","").replace(".","")
Tr = pd.read_csv(f"{M}/Transcription_RNA_to_Prot.csv", index_col=0)   # G_rp [gene, prot]
genes=[norm(g) for g in Tr.index]; prots=[norm(p) for p in Tr.columns]; gup={g:i for i,g in enumerate(genes)}
V=Tr.values
cur=J(f"{_CMC_RESULTS}/baselines/_cmcvae_cognate.json"); ppos={p:j for j,p in enumerate(prots)}   # curated 74-pair cognate set
cog=[(gup[norm(gn)], ppos[norm(pn)]) for pn,gn in zip(cur["protein_names"],cur["cognate_gene"])
     if gn is not None and norm(gn) in gup and norm(pn) in ppos]
cbp={}; [cbp.setdefault(p,set()).add(g) for g,p in cog]
matched=np.abs(np.array([V[g,p] for g,p in cog])); bg=np.abs(np.concatenate([np.delete(V[:,p],list(cbp[p])) for g,p in cog]))
fold=np.median(matched)/np.median(bg); pval=mannwhitneyu(matched,bg,alternative="greater")[1]
def ecdf(a): a=np.sort(a[a>0]); return a, np.arange(1,len(a)+1)/len(a)
for arr,lab,col in [(bg,"unmatched gene–protein",C["unmatched"]),(matched,"matched (cognate)",C["matched"])]:
    xs,ys=ecdf(arr); axC.plot(xs,ys,color=col,lw=1.8,label=lab)
axC.axvline(np.median(bg),ls=":",lw=0.8,color=C["unmatched"]); axC.axvline(np.median(matched),ls=":",lw=0.8,color=C["matched"])
axC.set_xscale("log"); axC.set_xlabel("|translation edge weight $G_{rp}[g,p]$|"); axC.set_ylabel("cumulative fraction")
axC.legend(loc="upper left"); axC.set_title("Cognate vs. non-cognate translation weights"); panel_label(axC, "C")

# ---- D: observed cognate enrichment vs totalVI baseline and permutation nulls ----
axD = fig.add_subplot(gs[1, 1])
nj = J(f"{M}/cognate_nulls_csv_74.json")   # 74 curated cognate pairs, exact per-column-background methodology
obs_fold = nj["observed_fold"]   # sp_dec_gl1000 = 8.2x (matches panel C ECDF, same 74 pairs)
tvi_fold = 3.0                    # totalVI transcript->protein cognate fold on the SAME 74 set (Table~\ref{tab:totalvi})
nulls = [("permute\nentries", nj["nullA_permute_entries"]["null_mean"], nj["nullA_permute_entries"]["null_std"], nj["nullA_permute_entries"]["z"]),
         ("permute\ncolumns", nj["nullB_permute_columns"]["null_mean"], nj["nullB_permute_columns"]["null_std"], nj["nullB_permute_columns"]["z"]),
         ("matched\ndegree",  nj["nullC_matched_degree"]["null_mean"], nj["nullC_matched_degree"]["null_std"], nj["nullC_matched_degree"]["z"])]
labels = ["observed\n(CMC)", "totalVI"] + [n[0] for n in nulls]
vals   = [obs_fold, tvi_fold] + [n[1] for n in nulls]
errs   = [0.0, 0.0] + [n[2] for n in nulls]
cols   = [C["cmc"], C["totalvi"]] + [C["gray"]] * 3     # CMC=blue, totalVI=orange (match panels A/E)
xb = np.arange(len(labels))
axD.bar(xb, vals, yerr=errs, width=0.66, color=cols, error_kw=dict(lw=0.8, capsize=2))
axD.annotate(f"{obs_fold:.1f}$\\times$", (0, obs_fold + 0.5), ha="center", fontsize=8.5,
             color=C["cmc"], fontweight="bold")
axD.annotate(f"{tvi_fold:.1f}$\\times$", (1, tvi_fold + 0.5), ha="center", fontsize=7.5,
             color=C["totalvi"], fontweight="bold")
for i, n in enumerate(nulls):                                  # SD above null (z) on each null bar
    axD.annotate(f"z={n[3]:.0f}", (i + 2, n[1] + n[2] + 0.9), ha="center", fontsize=7, color=C["dark"])
axD.set_xticks(xb); axD.set_xticklabels(labels, fontsize=6.6)
axD.set_ylabel("cognate enrichment fold"); axD.set_ylim(0, 10)
axD.text(0.98, 0.60, "N=1000 permutations\nall $p<10^{-3}$", transform=axD.transAxes, fontsize=6.4,
         ha="right", va="top", color=C["ink"])
axD.set_title("Cognate enrichment: CMC vs. totalVI & nulls", fontsize=9.5); panel_label(axD, "D")

# ---- E: inferred-GRN concentration (Gini) & sparsity across all baselines ----
axE = fig.add_subplot(gs[2, :])
gj = J(f"{_CMC_RESULTS}/baselines/grn_sparsity_gini.json")
order = ["CMC-VAE", "GRNBoost2", "DeepSEM", "totalVI", "correlation"]
spars_v = [gj.get(m, {}).get("frac_lt_1e3", np.nan) for m in order]    # frac |w| < 0.1%*max (comparable across models)
conc_v = [gj.get(m, {}).get("mass_top1pct", np.nan) for m in order]    # mass held by top 1% of edges
xe = np.arange(len(order)); we = 0.38
axE.bar(xe - we / 2, spars_v, we, color=C["cmc"], label="sparsity (frac $|w|<0.1\\%\\!\\cdot$max)")
axE.bar(xe + we / 2, conc_v, we, color=C["gray"], label="concentration (mass in top-1% edges)")
for i, (s, cc) in enumerate(zip(spars_v, conc_v)):        # value labels
    axE.annotate(f"{s:.2f}", (i - we / 2, s + 0.02), ha="center", fontsize=6.0, color=C["cmc"])
    axE.annotate(f"{cc:.2f}", (i + we / 2, cc + 0.02), ha="center", fontsize=6.0, color=C["dark"])
axE.set_xticks(xe); axE.set_xticklabels(order, rotation=12, ha="right")
axE.set_ylabel("fraction"); axE.set_ylim(0, 1.12)
axE.legend(loc="upper right", ncol=1, fontsize=6.2, frameon=False, bbox_to_anchor=(1.0, 1.02),
           handlelength=1.1, handletextpad=0.4, labelspacing=0.25)
axE.set_title("Inferred-GRN sparsity & concentration ($G_{rr}$, off-diagonal)", loc="left"); panel_label(axE, "E", x=-0.055)

save(fig, f"{R}/paper/figures/Figure2.pdf")
