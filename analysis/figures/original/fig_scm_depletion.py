"""AppendixFigS9 — conditioning depletes the learned SCM (any target), without orienting it.
A structural variance decomposition of the SCM solve z=(I-A^T)^{-1}eps for the headline model and the
six conditional-iVAE variants (celltype/donor/disease x {joint eps, RNA-only eps}). (A) GRN
self-amplification ||M_rr - I||_F^2 / ||M_rr||_F^2 -- how much the RNA block A_rr reshapes its own
exogenous. (B) cross-modal transmission: max |W_rp| (translation-edge magnitude, log). Both collapse
under every conditional prior -- joint AND RNA-only -- as the per-class prior mean supplies the
class-structured latent variation the SCM would otherwise build. Reads results/cond_prior_rna_summary.json."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from style import plt, C, panel_label, save
R = f"{_CMC_ROOT}"
S = json.load(open(f"{_CMC_RESULTS}/cond_prior_rna_summary.json"))
ORDER = ["top", "celltype_joint", "celltype_rna", "donor_joint", "donor_rna", "disease_joint", "disease_rna"]
LBL = ["top", "celltype\njoint", "celltype\nRNA", "donor\njoint", "donor\nRNA", "disease\njoint", "disease\nRNA"]
COL = [C["cmc"]] + [C["base"], C["pos"]] * 3    # top=blue; joint=orange, RNA-only=green
x = np.arange(len(ORDER))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.4)); plt.subplots_adjust(wspace=0.28, bottom=0.19, top=0.78)
fig.suptitle("Conditioning depletes the learned SCM (joint and RNA-only) without orienting it", fontsize=10.5, y=0.975)

# A: GRN self-amplification
amp = [S[m]["arr_amp"] * 100 for m in ORDER]
axA.bar(x, amp, 0.66, color=COL)
axA.axhline(amp[0], ls=":", lw=0.9, c=C["muted"])
for xi, v in zip(x, amp): axA.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=6.6)
axA.set_xticks(x); axA.set_xticklabels(LBL, fontsize=7.2)
axA.set_ylabel(r"GRN self-amplification (%)"); axA.set_ylim(0, 88)
axA.set_title(r"$A_{rr}$ recurrence: $\|M_{rr}{-}I\|^2/\|M_{rr}\|^2$", fontsize=9.3); panel_label(axA, "A")

# B: cross-modal edge magnitude (log)
wrp = [S[m]["wrp_max"] for m in ORDER]
axB.bar(x, wrp, 0.66, color=COL)
axB.axhline(wrp[0], ls=":", lw=0.9, c=C["muted"])
axB.set_yscale("log"); axB.set_xticks(x); axB.set_xticklabels(LBL, fontsize=7.2)
axB.set_ylabel(r"translation edge  $\max|W_{rp}|$"); axB.set_ylim(3e-4, 5e-2)
axB.set_title("Cross-modal transmission collapses $\\sim$20$\\times$", fontsize=9.3); panel_label(axB, "B")

from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc=C["cmc"], label="top (unconditioned)"), Patch(fc=C["base"], label="cond-iVAE (joint $\\varepsilon$)"),
                    Patch(fc=C["pos"], label="cond-iVAE (RNA-only $\\varepsilon$)")],
           loc="upper center", ncol=3, fontsize=8.0, frameon=False, bbox_to_anchor=(0.5, 0.905))
save(fig, f"{R}/paper/figures/AppendixFigS9.pdf"); print("WROTE AppendixFigS9.pdf (SCM depletion)")
