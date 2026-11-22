"""Regenerate Fig. 2 (a,b,d,e,f with information-matched baselines + paired tests; c unchanged in content), Supp Fig 1c,
Supp Fig 3 (dashed-line legend issue fixed; sparsity -> concentration), a Fig-3 companion (zero-shot comparators) and a
same-training-cells corpus comparison. Style: analysis/figures/original/style.py. Output: paper_nov_revisions/figures/baselines/."""
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
R = f"{_CMC_ROOT}"; sys.path.insert(0, f"{_CMC_ROOT}/analysis/figures/original")
from style import plt, C, panel_label, save, stars
B = f"{_CMC_RESULTS}/paper/analyses/baselines"; F = f"{_CMC_RESULTS}/paper/figures/baselines"; os.makedirs(F, exist_ok=True)
J = json.load(open(f"{B}/fig2_baselines.json"))
SEEDS = ["42", "1", "2"]
COL = {"CMC-VAE": C["cmc"], "corr": C["shuf"], "partial_corr": C["pos"], "corr_within_ct": C["accent"], "partial_within_ct": "#9ae6b4",
       "totalVI": C["totalvi"], "GRNBoost2": C["genie3"], "DeepSEM": "#553c9a", "screen": "#975a16", "agn": "#cbd5e0"}
LAB = {"CMC-VAE": "CMC-VAE", "corr": "Correlation", "partial_corr": "Partial corr. (LW)", "corr_within_ct": "Within-cell-type corr.",
       "partial_within_ct": "Within-cell-type partial corr.", "totalVI": "totalVI", "GRNBoost2": "GRNBoost2", "DeepSEM": "DeepSEM"}
SRC = {}

def qlab(q): return "***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else "ns"
def seedvals(d, key=lambda s: f"CMC-VAE_s{s}"): return np.array([d[key(s)] for s in SEEDS])

# =============================================================== FIGURE 2
fig = plt.figure(figsize=(7.4, 9.2)); gs = fig.add_gridspec(3, 2, hspace=0.75, wspace=0.42, height_ratios=[1, 1, 0.95])

# ---- a: reference recovery
axA = fig.add_subplot(gs[0, 0]); ra = J["fig2a_reference_recovery"]
nets = ["STRING", "DoRothEA", "CollecTRI"]
old = {"totalVI": [0.705, 0.603, 0.575], "GRNBoost2": [0.646, 0.536, 0.517], "DeepSEM": [0.658, 0.572, 0.553]}   # Fig 2a (809k), unchanged
series = ["CMC-VAE", "corr", "corr_within_ct", "partial_corr", "totalVI", "GRNBoost2", "DeepSEM"]
w = 0.115; x = np.arange(len(nets)); SRC["fig2a"] = {}
for i, m in enumerate(series):
    xs = x + (i - 3) * w
    if m == "CMC-VAE":
        v = np.array([seedvals(ra[n]["auroc"]) for n in nets]); mu = v.mean(1)
        axA.bar(xs, mu, w, color=COL[m], label=LAB[m]); axA.errorbar(xs, mu, yerr=[mu - v.min(1), v.max(1) - mu], fmt="none", ecolor=C["ink"], lw=0.7, capsize=1.5)
        SRC["fig2a"][m] = dict(zip(nets, v.tolist()))
    elif m in old:
        axA.bar(xs, old[m], w, color=COL[m], label=LAB[m]); SRC["fig2a"][m] = dict(zip(nets, old[m]))
    else:
        v = [ra[n]["auroc"][m] for n in nets]; axA.bar(xs, v, w, color=COL[m], label=LAB[m]); SRC["fig2a"][m] = dict(zip(nets, v))
def s3(d, m):   # sign and BH stars for CMC seeds 42/1/2 vs comparator m
    return "/".join(("+" if d[f"s{s_}_vs_{m}"]["diff"] > 0 else "−") + qlab(d[f"s{s_}_vs_{m}"]["q_bh"]).replace("ns", "ns") for s_ in SEEDS)
for j, n in enumerate(nets):   # paired bootstrap over TFs/source genes; 3 CMC-VAE seeds (42/1/2)
    top = max(ra[n]["auroc"].values())
    axA.text(x[j], top + 0.012, "vs corr: " + s3(ra[n]["paired"], "corr") + "\nvs within-ct: " + s3(ra[n]["paired"], "corr_within_ct"), ha="center", va="bottom", fontsize=4.6)
axA.axhline(0.5, ls=":", lw=0.8, c=C["neutral"]); axA.set_xticks(x); axA.set_xticklabels(nets)
axA.set_ylim(0.45, 0.83); axA.set_ylabel("Recovery AUROC"); axA.set_title("Reference-network recovery ($G_{rr}$)")
axA.legend(fontsize=5.4, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), handlelength=1.0, columnspacing=0.6, labelspacing=0.2); panel_label(axA, "a")

# ---- b: immune programs with correlation baselines (dot plot)
axB = fig.add_subplot(gs[0, 1]); pb = J["fig2b_programs"]
short = {"cGAS-STING": "cGAS-STING", "IFN-alpha (Type-I) response": "IFN-α", "IFN-gamma (Type-II) response": "IFN-γ", "JAK-STAT signaling": "JAK-STAT",
         "TNF-alpha / NF-kB signaling": "TNF-α/NF-κB", "Antigen presentation (MHC I & II)": "Antigen pres.", "TCR signaling": "TCR", "Apoptosis": "Apoptosis",
         "MHC-I complex (control)": "MHC-I (ctrl)"}
order = sorted(pb, key=lambda k: -np.mean(seedvals(pb[k]["auroc"]))); yy = np.arange(len(order))[::-1]; SRC["fig2b"] = {}
for i, nm in enumerate(order):
    v = seedvals(pb[nm]["auroc"]); y = yy[i]
    axB.plot([v.min(), v.max()], [y, y], c=COL["CMC-VAE"], lw=1.6, alpha=0.5)
    axB.scatter(v.mean(), y, s=22, c=COL["CMC-VAE"], zorder=3, label="CMC-VAE (3 seeds)" if i == 0 else None)
    for m, mk in [("corr", "s"), ("corr_within_ct", "D"), ("partial_corr", "^")]:
        axB.scatter(pb[nm]["auroc"][m], y, s=14, marker=mk, c=COL[m], zorder=2, label=LAB[m] if i == 0 else None)
    q1 = pb[nm]["paired"]["s42_vs_corr"]; q2 = pb[nm]["paired"]["s42_vs_corr_within_ct"]
    axB.text(1.005, y, s3({k: dict(v, q_bh=v["q_bh"]) for k, v in pb[nm]["paired"].items()}, "corr") + "  |  " + s3(pb[nm]["paired"], "corr_within_ct"), fontsize=4.4, va="center", transform=axB.get_yaxis_transform())
    SRC["fig2b"][nm] = dict(cmc_seeds=v.tolist(), **{m: pb[nm]["auroc"][m] for m in ["corr", "corr_within_ct", "partial_corr"]},
                            q_vs_corr=q1["q_bh"], q_vs_within=q2["q_bh"])
axB.axvline(0.5, ls=":", lw=0.8, c=C["neutral"]); axB.set_yticks(yy); axB.set_yticklabels([short[k] for k in order]); axB.set_xlim(0.5, 1.0)
axB.set_xlabel("Within-program AUROC"); axB.set_title("Immune programs ($|G_{rr}|$) vs. correlation", fontsize=8)
axB.text(1.005, 1.01, "seeds 42/1/2: vs corr | vs within-ct", fontsize=5.2, transform=axB.transAxes, va="bottom")
axB.legend(fontsize=5.2, loc="lower right", handletextpad=0.2, labelspacing=0.2); panel_label(axB, "b", x=-0.36)

# ---- c: cognate specificity vs baselines (fold, top-1 per protein, Gini)
axC = fig.add_subplot(gs[1, 0]); cm = J["fig2c_cognate"]["metrics"]
mets = [("fold", "Fold\n(÷30)", 1 / 30.0), ("top1_col", "Top-1\ntranscript", 1.0), ("top1_row", "Top-1\nprotein", 1.0), ("gini", "Gini", 1.0)]
meth = ["CMC-VAE", "corr", "corr_within_ct", "partial_corr"]; w = 0.2; x = np.arange(len(mets)); SRC["fig2c"] = {}
for i, m in enumerate(meth):
    if m == "CMC-VAE": v = np.array([[cm[f"CMC-VAE_s{s}"][k] * sc for s in SEEDS] for k, _, sc in mets]); mu = v.mean(1); err = [mu - v.min(1), v.max(1) - mu]
    else: mu = np.array([cm[m][k] * sc for k, _, sc in mets]); err = None
    axC.bar(x + (i - 1.5) * w, mu, w, color=COL[m], label=LAB[m], yerr=err, error_kw=dict(lw=0.6, capsize=1.2))
    for j, (k, _, sc) in enumerate(mets):
        if k == "fold": axC.text(x[j] + (i - 1.5) * w, mu[j] + 0.03, f"{mu[j] / sc:.0f}×", ha="center", fontsize=4.8, rotation=90)
    SRC["fig2c"][m] = {k: (cm[m][k] if m != "CMC-VAE" else [cm[f"CMC-VAE_s{s}"][k] for s in SEEDS]) for k, _, _ in mets}
axC.set_xticks(x); axC.set_xticklabels([l for _, l, _ in mets], fontsize=6.4); axC.set_ylim(0, 1.0)
axC.set_ylabel("Value"); axC.set_title("Cognate map: CMC-VAE vs. correlation baselines (n = 74)", fontsize=7.6)
axC.legend(fontsize=5.2, loc="upper right", labelspacing=0.2); panel_label(axC, "c")

# ---- d: nulls with exceedance counts
axD = fig.add_subplot(gs[1, 1]); nd = J["fig2d_nulls"]; obs = nd["observed_fold"]; NN = 10000
names = [("permute_Grp_entries", "Permuted\nentries"), ("shuffle_Wrp_latent_then_project", "Shuffled $W_{rp}$\n(pre-projection)"),
         ("permute_protein_columns", "Permuted\ncolumns"), ("column_norm_preserving", "Column-norm-\npreserving")]
axD.bar(0, obs, 0.62, color=C["cmc"]); axD.text(0, obs + 0.3, f"{obs:.1f}×", ha="center", fontsize=7.5, color=C["cmc"], fontweight="bold")
axD.bar(1, 3.02, 0.62, color=C["totalvi"]); axD.text(1, 3.32, "3.0×", ha="center", fontsize=6.5, color=C["totalvi"])
SRC["fig2d"] = {}
for i, (k, l) in enumerate(names):
    r = nd["nulls"][f"{k}_N{NN}"]; xi = i + 2
    axD.bar(xi, r["null_mean"], 0.62, yerr=r["null_sd"], color=C["gray"], error_kw=dict(lw=0.7, capsize=2))
    axD.scatter([xi], [r["null_max"]], marker="_", s=60, c=C["ink"], zorder=3)
    axD.text(xi, r["null_max"] + 0.35, f"z={r['z']:.0f}\n{r['exceed']}/{NN:,}", ha="center", fontsize=5.4)
    SRC["fig2d"][k] = r
axD.set_xticks(range(6)); axD.set_xticklabels(["CMC-VAE", "totalVI"] + [l for _, l in names], fontsize=5.4, rotation=35, ha="right")
axD.set_ylabel("Cognate enrichment (fold)"); axD.set_ylim(0, 10.5)
axD.text(0.99, 0.98, f"exceedances k of N = 10,000 per null\nP = (k+1)/(N+1) ≤ 1.0×10$^{{-4}}$ for all\n— marks null maximum", transform=axD.transAxes, ha="right", va="top", fontsize=5.4)
axD.set_title("Cognate enrichment vs. randomized nulls"); panel_label(axD, "d")

# ---- e: concentration (Gini & top-1% mass) instead of max-relative sparsity
axE = fig.add_subplot(gs[2, 0]); ce = J["fig2e_concentration"]
ms = ["CMC-VAE", "GRNBoost2", "DeepSEM", "corr", "corr_within_ct", "partial_corr", "totalVI"]; x = np.arange(len(ms)); w = 0.38; SRC["fig2e"] = {}
for k_, (key, lab, col) in enumerate([("gini", "Gini coefficient", C["cmc"]), ("mass_top1pct", "Mass in top 1% of edges", C["gray"])]):
    vals, err = [], [[], []]
    for m in ms:
        if m == "CMC-VAE": v = np.array([ce[f"CMC-VAE_s{s}"][key] for s in SEEDS]); vals.append(v.mean()); err[0].append(v.mean() - v.min()); err[1].append(v.max() - v.mean())
        else: vals.append(ce[m][key]); err[0].append(0); err[1].append(0)
    axE.bar(x + (k_ - 0.5) * w, vals, w, color=col, label=lab, yerr=err, error_kw=dict(lw=0.6, capsize=1.2))
    for i, v in enumerate(vals): axE.text(x[i] + (k_ - 0.5) * w, v + 0.02, f"{v:.2f}", ha="center", fontsize=5.0)
    SRC["fig2e"][key] = dict(zip(ms, vals))
axE.set_xticks(x); axE.set_xticklabels([LAB.get(m, m).replace(" (LW)", "").replace("Within-cell-type corr.", "Within-ct\ncorr.").replace("Partial corr.", "Partial\ncorr.") for m in ms], fontsize=5.8, rotation=35, ha="right")
axE.set_ylim(0, 1.25); axE.set_ylabel("Fraction"); axE.legend(fontsize=5.6, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.04))
axE.set_title("Concentration of $G_{rr}$ edge weights (off-diagonal)"); panel_label(axE, "e")

# ---- f: surface complexes, CMC (3 seeds) vs co-expression, within-lineage, partial correlation
axF = fig.add_subplot(gs[2, 1]); pf = J["fig2f_surface"]
ms = ["corr", "corr_within_ct", "partial_corr_protein_lw", "partial_corr_protein_glasso"]
labs = ["CMC-VAE\n$G_{pp}$", "Co-expr.", "Within-\nlineage", "Partial corr.\n(LW)", "Partial corr.\n(glasso)"]
SRC["fig2f"] = {}
for k_, (sname, off, alpha) in enumerate([("all36", -0.18, 1.0), ("cis15", 0.18, 0.45)]):
    v = seedvals(pf[sname]["auroc"]); xs = [0 + off] + [i + 1 + off for i in range(len(ms))]
    vals = [v.mean()] + [pf[sname]["auroc"][m] for m in ms]
    cols = [C["cmc"], C["shuf"], COL["corr_within_ct"], C["pos"], "#68d391"]
    axF.bar(xs, vals, 0.34, color=cols, alpha=alpha, label=("all 36 edges" if k_ == 0 else "15 cis edges"))
    axF.scatter([xs[0]] * 3, v, s=6, c=C["ink"], zorder=3)
    SRC["fig2f"][sname] = dict(cmc_seeds=v.tolist(), **{m: pf[sname]["auroc"][m] for m in ms}, paired_s42={m: pf[sname]["paired"][f"s42_vs_{m}"] for m in ms})
axF.axhline(0.5, ls="--", lw=0.8, c=C["neutral"]); axF.set_xticks(range(5)); axF.set_xticklabels(labs, fontsize=5.8, rotation=35, ha="right")
axF.set_ylim(0.35, 1.0); axF.set_ylabel("Surface-complex AUROC"); axF.legend(fontsize=5.6, loc="upper left")
axF.set_title("Surface-complex recovery (dots: 3 CMC-VAE seeds)", fontsize=7.8); panel_label(axF, "f")
save(fig, f"{F}/Figure2_revised.pdf"); plt.close(fig)
json.dump(SRC, open(f"{F}/Figure2_revised_source_data.json", "w"), indent=1, default=float)

# =============================================================== SUPP FIG 1c (hubs without IG/TR V(D)J segments)
H = J["hubs"]["no_IGTR_VDJ"]; Hall = J["hubs"]["all_genes"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(wspace=0.55))
for ax, (hh, ttl) in zip(axs, [(Hall, "All genes (as published)"), (H, f"IG/TR V(D)J genes removed (n = {J['n_IGTR_VDJ_genes']})")]):
    top = hh["s42"]["top20"][:14]; stable = set(hh["top50_all_seeds"])
    names = [t[0] for t in top][::-1]; vals = [t[1] for t in top][::-1]; tf = [t[2] for t in top][::-1]
    ax.barh(range(14), vals, color=[C["neg"] if t else C["cmc"] for t in tf])
    ax.set_yticks(range(14)); ax.set_yticklabels([n + (" †" if n in stable else "") for n in names], fontsize=6.4)
    ax.set_xlabel("Summed outgoing weight (row sum of $|G_{rr}|$)"); ax.set_title(ttl, fontsize=7.5)
    ax.text(0.98, 0.03, f"TFs in top 14: {hh['s42']['n_TF_top14']}\nTFs in top 50: {hh['s42']['n_TF_top50']} (seeds 1/2: {hh['s1']['n_TF_top50']}/{hh['s2']['n_TF_top50']})\n"
            f"r(strength, ‖D_r[g,:]‖) = {hh['s42']['r_strength_rownorm']:.2f}\n† in top 50 of all 3 seeds", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6)
panel_label(axs[0], "c"); save(fig, f"{F}/SupplementaryFig1c_revised.pdf"); plt.close(fig)
json.dump(dict(all_genes=Hall, no_IGTR_VDJ=H, no_IGTR_incl_constant=J["hubs"]["no_IGTR_incl_constant"]), open(f"{F}/SupplementaryFig1c_source_data.json", "w"), indent=1)

# =============================================================== SUPP FIG 3 (fixed)
cog = json.load(open(f"{_CMC_RESULTS}/baselines/_cmcvae_cognate.json")); tv = np.load(f"{_CMC_RESULTS}/baselines/totalvi_raw.npz", allow_pickle=True)
st = np.array([v for v in cog["cognate_mrna_pearson"] if v is not None], float); cmcp = np.array(cog["cmcvae_protein_from_rna_pearson"], float); tvp = np.asarray(tv["perprot_pear_clr"], float)
fig = plt.figure(figsize=(7.4, 5.6)); g3 = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.45)
ax = fig.add_subplot(g3[0, 0]); rs = np.random.RandomState(0)
cmc_mean = json.load(open(f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000/crossmodal_imputation.json"))["protein_from_rna_pearson"]   # per-protein values not stored for the headline model
for i, (v, col) in enumerate([(st, C["gray"]), (None, C["cmc"]), (tvp, C["totalvi"])]):
    if v is None:
        ax.bar(i, cmc_mean, 0.6, color=col, alpha=0.85); ax.text(i, cmc_mean + 0.03, f"{cmc_mean:.2f}", ha="center", fontsize=6); continue
    v = v[np.isfinite(v)]; ax.bar(i, v.mean(), 0.6, color=col, alpha=0.85); ax.scatter(i + rs.uniform(-0.2, 0.2, len(v)), v, s=2, c=C["ink"], alpha=0.5, lw=0)
    ax.text(i + 0.33, v.mean(), f"{v.mean():.2f}", ha="left", va="center", fontsize=6)
ax.set_xticks(range(3)); ax.set_xticklabels(["Single-transcript\n(n=74)", "CMC-VAE\n(mean, 144)", "totalVI\n(n=144)"], fontsize=5.8, rotation=25, ha="right")
ax.set_ylabel("Per-protein Pearson r"); ax.set_title("Cross-modal imputation"); panel_label(ax, "a")
ax = fig.add_subplot(g3[0, 1]); v = np.array([J["fig2c_cognate"]["metrics"][f"CMC-VAE_s{s}"]["fold"] for s in SEEDS])
ax.bar(0, v.mean(), 0.6, yerr=v.std(ddof=1), color=C["cmc"], error_kw=dict(lw=0.7, capsize=2)); ax.scatter([0] * 3, v, s=8, c=C["ink"], zorder=3)
ax.bar(1, 3.02, 0.6, color=C["totalvi"]); ax.bar(2, J["fig2c_cognate"]["metrics"]["corr"]["fold"], 0.6, color=C["shuf"]); ax.bar(3, J["fig2c_cognate"]["metrics"]["partial_corr"]["fold"], 0.6, color=C["pos"])
ax.set_xticks(range(4)); ax.set_xticklabels(["CMC-VAE\n(3 seeds)", "totalVI", "Corr.", "Partial\ncorr."], fontsize=6); ax.set_ylabel("Cognate enrichment (fold)")
ax.set_title("Translation specificity"); panel_label(ax, "b")
ax = fig.add_subplot(g3[0, 2]); pf = J["fig2f_surface"]["all36"]["auroc"]
ms = ["CMC-VAE_s42", "corr", "corr_within_ct", "partial_corr_protein_lw", "partial_corr_protein_glasso"]
ax.bar(range(5), [pf[m] for m in ms], 0.65, color=[C["cmc"], C["shuf"], COL["corr_within_ct"], C["pos"], "#68d391"])
for i, m in enumerate(ms): ax.text(i, pf[m] + 0.01, f"{pf[m]:.2f}", ha="center", fontsize=5.8)
ax.axhline(0.5, ls="--", lw=0.8, c=C["neutral"]); ax.set_ylim(0.4, 0.75); ax.set_xticks(range(5))
ax.set_xticklabels(["$G_{pp}$", "Marginal\nco-expr.", "Within-\nlineage", "Partial\n(LW)", "Partial\n(glasso)"], fontsize=5.8, rotation=30, ha="right")
ax.set_ylabel("Surface-complex AUROC"); ax.set_title("PPI, lineage-deconfounded"); panel_label(ax, "c")
# d,e: per-block distribution & concentration (seed 42)
blocks = {}
for nm, fn in [("$G_{rr}$", "GRN_inferred.csv"), ("$G_{rp}$", "Transcription_RNA_to_Prot.csv"), ("$G_{pp}$", "PPI_inferred.csv"), ("$G_{pr}$", "Regulation_Prot_to_RNA.csv")]:
    M = pd.read_csv(f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000/{fn}", index_col=0).values.astype(float)
    if M.shape[0] == M.shape[1]: M = M[~np.eye(M.shape[0], dtype=bool)]
    blocks[nm] = np.abs(M.ravel())
ax = fig.add_subplot(g3[1, 0:2])
for (nm, a), col in zip(blocks.items(), [C["cmc"], C["pos"], C["accent"], C["totalvi"]]):
    r_ = np.log10(a / a.max() + 1e-12); ax.hist(r_[r_ > -6], bins=60, histtype="step", density=True, color=col, lw=1.2, label=nm)
ax.set_xlabel("Relative edge magnitude, log$_{10}$(|w|/max|w|)"); ax.set_ylabel("Density"); ax.legend(fontsize=6, ncol=2); ax.set_title("Per-block edge-weight distribution"); panel_label(ax, "d")
ax = fig.add_subplot(g3[1, 2]); SRC3 = {}
def gini(x):
    x = np.sort(x); n = x.size; i = np.arange(1, n + 1); return 2 * (i * x).sum() / (n * x.sum()) - (n + 1) / n
for i, (nm, a) in enumerate(blocks.items()):
    gg = gini(a); s_ = np.sort(a)[::-1]; t1 = s_[:max(1, int(0.01 * a.size))].sum() / a.sum(); SRC3[nm] = dict(gini=float(gg), mass_top1pct=float(t1))
    ax.bar(i - 0.19, gg, 0.38, color=C["cmc"], label="Gini" if i == 0 else None); ax.bar(i + 0.19, t1, 0.38, color=C["gray"], label="Mass in top 1%" if i == 0 else None)
    ax.text(i - 0.19, gg + 0.02, f"{gg:.2f}", ha="center", fontsize=5.4); ax.text(i + 0.19, t1 + 0.02, f"{t1:.2f}", ha="center", fontsize=5.4)
ax.set_xticks(range(4)); ax.set_xticklabels(list(blocks)); ax.set_ylim(0, 1.05); ax.legend(fontsize=5.6); ax.set_title("Per-block concentration"); panel_label(ax, "e")
save(fig, f"{F}/SupplementaryFig3_revised.pdf"); plt.close(fig)
json.dump(dict(imputation=dict(single_transcript_mean=float(st.mean()), cmc_mean=float(cmc_mean), totalvi_mean=float(np.nanmean(tvp))), per_block=SRC3),
          open(f"{F}/SupplementaryFig3_source_data.json", "w"), indent=1)

# =============================================================== FIG 3 COMPANION: zero-shot comparators
MT = pd.read_csv(f"{B}/zeroshot_means.csv"); T = pd.read_csv(f"{B}/zeroshot_paired_tests.csv")
coh = ["frangieh", "iccite", "arce"]; cname = {"frangieh": "Frangieh", "iccite": "Chen", "arce": "Arce"}
groups = [("CMC-VAE", "CMC-VAE", C["cmc"]), ("cmc_generic", "CMC-VAE, KO-agnostic\n(mean |pred| other KOs)", C["cmc2"]),
          ("corr", "Training corr.", C["shuf"]), ("partial_corr", "Training partial corr.", C["pos"]), ("corr_within_ct", "Training within-ct corr.", C["accent"]),
          ("decoder_row_norm", "Decoder-row norm", "#2c7a7b"), ("train_sd", "Training s.d.", "#a0aec0"),
          ("screen_ctrl_corr", "Screen-control corr.", "#975a16"), ("ctrl_mean", "Screen-control mean", "#dd6b20"), ("ctrl_var", "Screen-control variance", "#c05621")]
fig, axs = plt.subplots(2, 2, figsize=(7.6, 6.2), gridspec_kw=dict(hspace=0.55, wspace=0.28))
for ax, blk, ttl in [(axs[0, 0], "rr", "RNA response: DE-gene AUROC"), (axs[0, 1], "rp", "Protein response: DE-protein AUROC")]:
    w = 0.08; x = np.arange(3)
    for i, (k, lab, col) in enumerate(groups):
        r = MT[(MT.metric == f"{blk}_auroc__{k}")].set_index("cohort").reindex(coh)
        ax.bar(x + (i - 4.5) * w, r.mean_over_seeds, w, yerr=r.sd_over_seeds.fillna(0), color=col, label=lab, error_kw=dict(lw=0.5, capsize=0.8))
    ax.axhline(0.5, ls=":", lw=0.8, c=C["neutral"]); ax.set_xticks(x); ax.set_xticklabels([cname[c] for c in coh]); ax.set_ylim(0.25, 1.0)
    ax.set_ylabel("AUROC (mean over KOs)"); ax.set_title(ttl)
axs[0, 0].legend(fontsize=5.0, ncol=2, loc="upper left", labelspacing=0.15, columnspacing=0.5, handlelength=0.9, bbox_to_anchor=(0, 1.0)); panel_label(axs[0, 0], "a"); panel_label(axs[0, 1], "b")
# c: KO specificity: own - other AUROC
ax = axs[1, 0]; S = pd.read_csv(f"{B}/zeroshot_ko_specificity.csv"); preds = ["CMC-VAE", "corr", "partial_corr", "corr_within_ct", "screen_ctrl_corr"]
pc = [C["cmc"], C["shuf"], C["pos"], C["accent"], "#975a16"]; w = 0.15
for i, (p, col) in enumerate(zip(preds, pc)):
    r = S[(S.block == "rr") & (S.predictor == p)].groupby("cohort")["diff"].agg(["mean", "std"]).reindex(coh + ["overall"])
    ax.bar(np.arange(4) + (i - 2) * w, r["mean"], w, yerr=r["std"].fillna(0), color=col, label=LAB.get(p, "Screen-control corr."), error_kw=dict(lw=0.5, capsize=0.8))
ax.axhline(0, lw=0.8, c=C["ink"]); ax.set_xticks(range(4)); ax.set_xticklabels([cname.get(c, "Overall") for c in coh + ["overall"]])
ax.set_ylabel("AUROC(own KO) − AUROC(other KOs)"); ax.set_title("Knockout specificity of the RNA ranking"); ax.legend(fontsize=5.2, loc="upper left"); panel_label(ax, "c")
# d: retrieval and centred pearDE
ax = axs[1, 1]
for j, (met, lab) in enumerate([("rr_retrieval", "RNA retrieval"), ("rr_pe_cent", "RNA centred pearDE"), ("rp_pe", "Protein pearDE (raw)")]):
    for i, (p, col) in enumerate(zip(preds, pc)):
        r = MT[(MT.metric == f"{met}__{p}") & (MT.cohort == "overall")]
        ax.bar(j + (i - 2) * w, r.mean_over_seeds.values[0], w, yerr=r.sd_over_seeds.fillna(0).values[0], color=col, error_kw=dict(lw=0.5, capsize=0.8))
ax.axhline(0.5, ls=":", lw=0.7, c=C["neutral"], xmax=0.33); ax.axhline(0, lw=0.8, c=C["ink"])
ax.set_xticks(range(3)); ax.set_xticklabels(["RNA retrieval\n(chance 0.5)", "RNA centred\npearDE", "Protein pearDE\n(raw, shared)"], fontsize=6.4)
ax.set_title("Pooled over screens (KO-specific predictors)"); panel_label(ax, "d")
save(fig, f"{F}/Figure3_companion_zeroshot.pdf"); plt.close(fig)

# =============================================================== same-training-cells comparison across corpora
if os.path.exists(f"{B}/corpora_compare.json"):
    CC = json.load(open(f"{B}/corpora_compare.json")); tags = [t for t in CC if t != "summary"]
    fig, axs = plt.subplots(1, 4, figsize=(8.6, 2.4), gridspec_kw=dict(wspace=0.45))
    for ax, k, lab in zip(axs, ["DoRothEA", "CollecTRI", "cog_fold", "cog_top1_col"], ["DoRothEA AUROC", "CollecTRI AUROC", "Cognate fold", "Cognate top-1 (per protein)"]):
        a = np.array([CC[t][f"CMC_{k}"] for t in tags]); b = np.array([CC[t][f"corr_{k}"] for t in tags])
        ax.scatter(b, a, s=10, c=C["cmc"]); lo, hi = min(a.min(), b.min()), max(a.max(), b.max()); pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=0.7, c=C["neutral"]); ax.set_xlabel(f"Correlation (same cells)"); ax.set_ylabel("CMC-VAE")
        s_ = CC.get("summary", {}).get(k, {}); ax.set_title(f"{lab}\nCMC better {s_.get('n_cmc_better', '?')}/{s_.get('n', '?')}, P={s_.get('p_wilcoxon', float('nan')):.2g}", fontsize=6.6)
    save(fig, f"{F}/SupplementaryFig_corpora_same_cells.pdf"); plt.close(fig)
print("figures done")
