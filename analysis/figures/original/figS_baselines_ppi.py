"""AppendixFigS1 — external baselines & the learned four-block structure (one consolidated figure).
Row 1 (external baselines): (A) RNA->protein imputation vs floor/totalVI/ceiling, (B) cognate
translation specificity fold (mean+/-SD over 3 seeds), (C) PPI surface-complex recovery of G_pp.
Row 2 (learned block structure of the headline model): (D) relative edge-magnitude distribution for
all four blocks (G_rr, G_rp, G_pp, G_pr), (E) per-block sparsity & concentration, (F) PPI
lineage-deconfounded added value. Blocks read straight from the saved projected matrices; baseline
scalars from results/ppi_surface_complexes.json + the imputation/cognate JSONs."""
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
from matplotlib.patches import Patch
R = f"{_CMC_ROOT}"; M = f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000"
out = json.load(open(f"{_CMC_RESULTS}/ppi_surface_complexes.json"))
p1a = out["P1_surface_complex_recovery"]["all_curated"]

# ---- learned four-block matrices (off-diagonal for the square blocks) ----
BLK = [("$G_{rr}$", "GRN_inferred.csv", C["cmc"]),        ("$G_{rp}$", "Transcription_RNA_to_Prot.csv", C["pos"]),
       ("$G_{pp}$", "PPI_inferred.csv", C["accent"]),     ("$G_{pr}$", "Regulation_Prot_to_RNA.csv", C["base"])]
def block_vals(fn):
    V = pd.read_csv(f"{M}/{fn}", index_col=0).values.astype(float)
    return V[~np.eye(*V.shape, dtype=bool)] if V.shape[0] == V.shape[1] else V.ravel()
BV = {name: block_vals(fn) for name, fn, _ in BLK}

fig, ax = plt.subplots(2, 3, figsize=(11.2, 6.8)); plt.subplots_adjust(hspace=0.52, wspace=0.34, top=0.9)
BASE = C["base"]; FLOOR = C["shuf"]                       # minimal palette: model=blue, baselines=orange/grey

# A: imputation (floor / CMC / totalVI), aggregate over 80,913 held-out cells
axA = ax[0, 0]
axA.bar(range(3), [0.202, 0.403, 0.502], color=[FLOOR, C["cmc"], BASE], width=0.62)
axA.axhline(0.40, ls="--", lw=0.9, c=C["muted"]); axA.text(2.45, 0.415, "bio. ceiling", fontsize=6.5, c=C["muted"], ha="right")
axA.set_xticks(range(3)); axA.set_xticklabels(["floor", "CMC-VAE", "totalVI"], fontsize=8)
axA.set_ylabel("RNA$\\to$protein Pearson"); axA.set_ylim(0, 0.56)
axA.set_title("Cross-modal imputation", fontsize=9.5); panel_label(axA, "A")

# B: cognate translation specificity fold (74-pair set; CMC seed-SD from Table S1; totalVI single run)
axB = ax[0, 1]
axB.bar([0, 1], [9.8, 3.0], yerr=[1.8, 0.0], color=[C["cmc"], BASE], width=0.58,
        error_kw=dict(lw=0.9, capsize=3, ecolor="#333"))
axB.set_xticks([0, 1]); axB.set_xticklabels(["CMC-VAE", "totalVI"], fontsize=8)
axB.set_ylabel("cognate enrichment fold"); axB.set_ylim(0, 12)
axB.set_title("Translation specificity", fontsize=9.5); panel_label(axB, "B")

# C: PPI surface-complex recovery (G_pp vs marginal co-expr vs totalVI), permutation p annotated
axC = ax[0, 2]; mC = [m for m in ["CMC_VAE_Gpp", "marginal_coexpr", "totalVI"] if m in p1a]
LBLC = {"CMC_VAE_Gpp": "$G_{pp}$", "marginal_coexpr": "co-expr", "totalVI": "totalVI"}
vC = [p1a[m]["auroc"] for m in mC]; pC = [p1a[m]["perm_p"] for m in mC]
axC.bar(range(len(mC)), vC, color=[C["cmc"] if m == "CMC_VAE_Gpp" else BASE for m in mC], width=0.62)
for i, (v, p) in enumerate(zip(vC, pC)):
    star = "n.s." if p > 0.05 else ("$p{<}0.001$" if p < 1e-3 else f"$p{{=}}{p:.3f}$")
    axC.text(i, v + 0.012, f"{v:.2f}\n{star}", ha="center", va="bottom", fontsize=6.6)
axC.axhline(0.5, ls="--", lw=0.9, color=C["muted"]); axC.set_xticks(range(len(mC))); axC.set_xticklabels([LBLC[m] for m in mC], fontsize=8)
axC.set_ylim(0.4, 0.92); axC.set_ylabel("surface-complex AUROC")
axC.set_title(f"PPI recovery ({out['curated']['n_edges']} curated edges)", fontsize=9.5); panel_label(axC, "C")

# D: relative edge-magnitude distribution for all four blocks (|w|/max|w|, log10 x) -- shows block-
#    structured sparsity: G_rr/G_rp/G_pr concentrate near zero, G_pp is comparatively dense.
axD = ax[1, 0]
edges = np.linspace(-6, 0, 61)
for name, _, col in BLK:
    a = np.abs(BV[name]); a = a[a > 0]; r = np.log10(a / a.max())
    h, _ = np.histogram(r, bins=edges, density=True)
    axD.step(0.5 * (edges[:-1] + edges[1:]), h, where="mid", color=col, lw=1.6, label=name)
axD.set_xlabel("relative edge magnitude $\\log_{10}(|w|/\\max|w|)$"); axD.set_ylabel("density")
axD.legend(fontsize=7.5, frameon=False, ncol=2, columnspacing=1.0, handlelength=1.3)
axD.set_title("Per-block edge-weight distribution", fontsize=9.5); panel_label(axD, "D")

# E: per-block sparsity (frac |w|<0.1% of block max) & concentration (mass in top-1% of edges)
axE = ax[1, 1]; xe = np.arange(4); w = 0.38
spar = [float((np.abs(BV[n]) < 1e-3 * np.abs(BV[n]).max()).mean()) for n, _, _ in BLK]
conc = [float(np.sort(np.abs(BV[n]))[::-1][:max(1, int(0.01 * BV[n].size))].sum() / np.abs(BV[n]).sum()) for n, _, _ in BLK]
axE.bar(xe - w/2, spar, w, color=C["cmc"], label="sparsity (frac $|w|{<}0.1\\%$max)")
axE.bar(xe + w/2, conc, w, color=C["gray"], label="concentration (mass top-1%)")
for i, (s, cc) in enumerate(zip(spar, conc)):
    axE.text(i - w/2, s + 0.02, f"{s:.2f}", ha="center", fontsize=6.2, color=C["cmc"])
    axE.text(i + w/2, cc + 0.02, f"{cc:.2f}", ha="center", fontsize=6.2, color=C["dark"])
axE.set_xticks(xe); axE.set_xticklabels([n for n, _, _ in BLK], fontsize=8.5)
axE.set_ylabel("fraction"); axE.set_ylim(0, 1.14)
axE.legend(fontsize=6.4, frameon=False, loc="upper center", ncol=1, handlelength=1.1, labelspacing=0.2)
axE.set_title("Per-block sparsity & concentration", fontsize=9.5); panel_label(axE, "E")

# F: PPI lineage-deconfounded added value (G_pp vs marginal / within-lineage co-expr / partial corr)
axF = ax[1, 2]
LBLF = {"CMC_VAE_Gpp": "$G_{pp}$", "marginal_coexpr": "marg.\nco-expr", "within_lineage_coexpr": "within-lin.\nco-expr", "partial_corr": "partial\ncorr"}
mF = [m for m in ["CMC_VAE_Gpp", "marginal_coexpr", "within_lineage_coexpr", "partial_corr"] if m in p1a]
vF = [p1a[m]["auroc"] for m in mF]
axF.bar(range(len(mF)), vF, color=[C["cmc"] if m == "CMC_VAE_Gpp" else BASE for m in mF], width=0.66)
for i, v in enumerate(vF): axF.text(i, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8)
axF.axhline(0.5, ls="--", lw=0.9, color=C["muted"]); axF.set_xticks(range(len(mF))); axF.set_xticklabels([LBLF[m] for m in mF], fontsize=6.8)
axF.set_ylim(0.4, 0.82); axF.set_ylabel("surface-complex AUROC")
axF.set_title("PPI: lineage-deconfounded", fontsize=9.5); panel_label(axF, "F")

save(fig, f"{R}/paper/figures/AppendixFigS1.pdf"); print("WROTE AppendixFigS1.pdf (baselines + four-block structure)")
