"""Supp panel: conditional-prior (iVAE) directional signals vs shared-response controls, per seed.
a: forward-minus-reverse (learned A_rr minus A_rr^T) for raw RNA pearDE, centred pearDE and retrieval, per cohort.
b: RNA pearDE raw vs target-shuffled vs centred (per seed), Chen and Arce.
Input analyses/identifiability/ivae_summary.json."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, sys
import numpy as np
sys.path.insert(0, f"{_CMC_ROOT}/analysis/figures/original")
from style import C, save, panel_label, plt  # noqa

ROOT = f"{_CMC_RESULTS}/paper"
S = json.load(open(f"{ROOT}/analyses/identifiability/ivae_summary.json"))["aggregate"]
VAR = [v for v in ["base", "cp_celltype", "cp_donor", "cp_disease", "cp_disease_rna"] if v in S]
LAB = {"base": "final model", "cp_celltype": "cell type", "cp_donor": "donor", "cp_disease": "disease", "cp_disease_rna": "disease (RNA)"}
COH = ["frangieh", "iccite", "arce"]; CL = {"frangieh": "Frangieh", "iccite": "Chen", "arce": "Arce"}
MET = [("rr_pe", "raw pearDE", C["model"]), ("rr_pe_cent", "centred pearDE", C["pos"]), ("rr_retrieval", "retrieval", C["accent"])]
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4), sharey=True)
for ai, c in enumerate(COH):
    ax = axs[ai]
    for vi, v in enumerate(VAR):
        for mi, (k, lab, col) in enumerate(MET):
            d = S[v]["fwd_minus_rev"][f"{c}.{k}"]["per_seed"]
            x = vi + (mi - 1) * 0.22
            ax.scatter(np.full(len(d), x), d, s=8, color=col, label=lab if (ai == 0 and vi == 0) else None, zorder=3)
            if d: ax.plot([x - 0.08, x + 0.08], [np.mean(d)] * 2, color=col, lw=1.2)
    ax.axhline(0, color=C["muted"], lw=0.7); ax.set_xticks(range(len(VAR))); ax.set_xticklabels([LAB[v] for v in VAR], rotation=35, ha="right")
    ax.set_title(CL[c])
    if ai == 0: ax.set_ylabel("Forward − reverse"); ax.legend(fontsize=6, loc="upper left")
    panel_label(ax, "abc"[ai])
fig.tight_layout(); save(fig, f"{ROOT}/figures/identifiability/SuppFig_ivae_forward_reverse_controls.pdf")

fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.4), sharey=True)
for ai, c in enumerate(["iccite", "arce"]):
    ax = axs[ai]
    for vi, v in enumerate(VAR):
        for mi, (k, lab, col) in enumerate([("rr_pe", "raw", C["model"]), ("rr_pe_shuf", "target-shuffled", C["shuf"]), ("rr_pe_cent", "centred", C["pos"])]):
            d = S[v]["metrics"][f"{c}.{k}"]["per_seed"]; x = vi + (mi - 1) * 0.22
            ax.scatter(np.full(len(d), x), d, s=8, color=col, label=lab if (ai == 0 and vi == 0) else None, zorder=3)
            if d: ax.plot([x - 0.08, x + 0.08], [np.mean(d)] * 2, color=col, lw=1.2)
    ax.axhline(0, color=C["muted"], lw=0.7); ax.set_xticks(range(len(VAR))); ax.set_xticklabels([LAB[v] for v in VAR], rotation=35, ha="right")
    ax.set_title(CL[c]); panel_label(ax, "ab"[ai])
    if ai == 0: ax.set_ylabel("RNA pearDE (all KOs)"); ax.legend(fontsize=6, loc="upper left")
fig.tight_layout(); save(fig, f"{ROOT}/figures/identifiability/SuppFig_ivae_pearDE_controls.pdf")
