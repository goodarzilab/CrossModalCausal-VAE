"""AppendixFigS7 -- PPI surface-complex validation (protein block G_pp).

Re-plots the figure produced by analysis/fig2_networks/ppi_surface_complexes.py DIRECTLY from the
saved results/ppi_surface_complexes.json (all plotted values are scalars in that
JSON), so it needs no CITE-seq h5ad / graphical-lasso recompute. Plotting code is
copied verbatim from analysis/fig2_networks/ppi_surface_complexes.py; only the data source (JSON) and
the save path (AppendixFigS7.pdf) differ. No model name appears in the figure.
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
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _CMC_ROOT
RES = os.path.join(REPO, "results")
sys.path.insert(0, HERE)
from style import plt, C, panel_label, save  # noqa: E402

out = json.load(open(os.path.join(RES, "ppi_surface_complexes.json")))
p1a = out["P1_surface_complex_recovery"]["all_curated"]
K_mod = out["P3_module_recovery"]["n_modules"]

FIG = os.path.join(REPO, "paper", "figures", "AppendixFigS7.pdf")
os.makedirs(os.path.dirname(FIG), exist_ok=True)
COL = {"CMC_VAE_Gpp": C["model"], "marginal_coexpr": C["muted"], "within_lineage_coexpr": C["accent"],
       "partial_corr": C["pos"], "totalVI": C["base"]}
LBL = {"CMC_VAE_Gpp": "CMC-VAE $G_{pp}$", "marginal_coexpr": "marginal\nco-expr", "within_lineage_coexpr": "within-lineage\nco-expr",
       "partial_corr": "partial\ncorr", "totalVI": "totalVI"}

fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.8))

# Panel A: surface-complex recovery AUROC (G_pp vs marginal co-expr vs totalVI) + chance + perm-p
axA = axs[0]; mA = [m for m in ["CMC_VAE_Gpp", "marginal_coexpr", "totalVI"] if m in p1a]
vA = [p1a[m]["auroc"] for m in mA]; pA = [p1a[m]["perm_p"] for m in mA]
bars = axA.bar(range(len(mA)), vA, color=[COL[m] for m in mA], width=0.66)
for i, (v, p) in enumerate(zip(vA, pA)):
    star = "n.s." if p > 0.05 else ("p<0.001" if p < 1e-3 else f"p={p:.3f}")
    axA.text(i, v + 0.015, f"{v:.2f}\n{star}", ha="center", va="bottom", fontsize=6.5)
axA.axhline(0.5, ls="--", lw=0.9, color=C["muted"]); axA.text(len(mA) - 0.5, 0.505, "chance", fontsize=6.5, ha="right", color=C["muted"])
axA.set_xticks(range(len(mA))); axA.set_xticklabels([LBL[m] for m in mA], fontsize=7)
axA.set_ylim(0.4, max(0.9, max(vA) + 0.12)); axA.set_ylabel("surface-complex AUROC")
axA.set_title(f"A  Surface-complex recovery\n({out['curated']['n_edges']} curated edges)", loc="left")
panel_label(axA, "")

# Panel B: lineage-deconfounded comparison (same positives, stronger baselines)
axB = axs[1]; mB = [m for m in ["CMC_VAE_Gpp", "marginal_coexpr", "within_lineage_coexpr", "partial_corr"] if m in p1a]
vB = [p1a[m]["auroc"] for m in mB]
axB.bar(range(len(mB)), vB, color=[COL[m] for m in mB], width=0.7)
for i, v in enumerate(vB): axB.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8)
axB.axhline(0.5, ls="--", lw=0.9, color=C["muted"]); axB.text(len(mB) - 0.5, 0.505, "chance", fontsize=6.5, ha="right", color=C["muted"])
axB.set_xticks(range(len(mB))); axB.set_xticklabels([LBL[m] for m in mB], fontsize=6.6)
axB.set_ylim(0.4, max(0.9, max(vB) + 0.1)); axB.set_ylabel("surface-complex AUROC")
axB.set_title("B  Lineage-deconfounded\nadded value", loc="left")

# Panel C: module recovery ARI / NMI (grouped) with null (chance ~ 0)
axC = axs[2]; mC = [m for m in ["CMC_VAE_Gpp", "marginal_coexpr", "within_lineage_coexpr"] if m in out["P3_module_recovery"]["by_method"]]
km = str(K_mod); xC = np.arange(len(mC)); w = 0.36
ari = [out["P3_module_recovery"]["by_method"][m][km]["ari"] for m in mC]
nmi = [out["P3_module_recovery"]["by_method"][m][km]["nmi"] for m in mC]
axC.bar(xC - w / 2, ari, w, color=[COL[m] for m in mC], label="ARI")
axC.bar(xC + w / 2, nmi, w, color=[COL[m] for m in mC], alpha=0.5, hatch="//", label="NMI")
for i, m in enumerate(mC):
    p = out["P3_module_recovery"]["by_method"][m][km]["ari_perm_p"]
    axC.text(xC[i] - w / 2, ari[i] + 0.02, f"{ari[i]:.2f}", ha="center", va="bottom", fontsize=6.2)
    if p < 0.05: axC.text(xC[i] - w / 2, -0.06, "*", ha="center", va="top", fontsize=10, color=COL[m])
axC.axhline(0.0, lw=0.8, color=C["ink"])
axC.set_xticks(xC); axC.set_xticklabels([LBL[m] for m in mC], fontsize=6.6)
axC.set_ylabel("cluster agreement"); axC.set_ylim(-0.1, 1.0)
axC.set_title(f"C  Lineage-module recovery\n(spectral, K={K_mod}; * perm p<0.05)", loc="left")
from matplotlib.patches import Patch
axC.legend(handles=[Patch(fc=C["ink"], label="ARI"), Patch(fc=C["ink"], alpha=0.5, hatch="//", label="NMI")],
           loc="upper right", fontsize=6.5)

fig.tight_layout()
save(fig, FIG)
print("DONE")
