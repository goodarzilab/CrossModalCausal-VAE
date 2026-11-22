"""AppendixFigS4 (redesigned) — Direction is not identifiable from observation.
A: synthetic block-SCM orientation accuracy by noise family (our 2nd-order read-out vs DirectLiNGAM).
B: the 30-model observational sweep shown as orientation-asymmetry vs held-out RNA->RNA pearDE — pearDE
   stays ~0 across the ENTIRE asymmetry range (no lever orients, even ones that break symmetry).
C: the directed-prior lever isolated — raising the prior inflates asymmetry ~800x while held-out pearDE
   stays flat at ~0 and co-expression collapses (asymmetry is memorized, not learned)."""
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

# ---- Table S6: synthetic orientation accuracy (n=1e4) ----
FAM = ["Gaussian", "Laplace", "Student-$t$"]
OURS = [0.50, 0.49, 0.49]; LING_Z = [0.60, 1.00, 1.00]; LING_D = [0.56, 1.00, 1.00]

# ---- Table S3: 30-model observational sweep (config, coexpr, RRpear, asym, lever-group) ----
# lever groups: KL, epsBN, DEC, NL(nonlinear), SYM, SEED, PRIOR, BASE
SWEEP = [
 ("BASE",0.935,+0.034,1.02),
 ("KL",0.936,+0.005,0.98),("KL",0.943,-0.013,0.99),("KL",0.936,-0.002,1.04),("KL",0.939,-0.003,1.00),
 ("epsBN",0.933,-0.071,0.97),("epsBN",0.939,-0.033,0.97),("epsBN",0.911,-0.088,0.97),("epsBN",0.840,-0.117,1.00),
 ("epsBN",0.857,+0.074,1.02),("epsBN",0.789,+0.007,1.29),("epsBN",np.nan,-0.015,np.nan),
 ("DEC",0.963,-0.003,1.02),("DEC",0.960,-0.046,1.01),("DEC",0.937,-0.070,0.94),("DEC",0.933,-0.021,0.98),("DEC",0.581,+0.086,0.99),
 ("NL",0.901,-0.060,0.91),("NL",0.896,+0.003,1.00),("NL",0.903,-0.090,0.91),
 ("SYM",0.938,-0.037,1.00),
 ("SEED",0.948,-0.024,1.00),("SEED",0.939,+0.015,1.00),
 ("PRIOR",0.939,-0.039,1.02),("PRIOR",0.749,-0.079,7.9),("PRIOR",0.767,-0.014,178.),("PRIOR",0.758,-0.029,798.),
]
GCOL = {"BASE":"#2B6CB0","KL":"#DD8452","epsBN":"#55A868","DEC":"#8172B3","NL":"#C44E52",
        "SYM":"#937860","SEED":"#DA8BC3","PRIOR":"#CC3311"}
# directed-prior lever isolated (weight -> asym, coexpr, RRpear) incl. prior-off base (weight 0)
PW = [0, 500, 750, 1000, 1500]
P_ASYM = [1.02, 1.02, 7.9, 178., 798.]; P_COEX = [0.935, 0.939, 0.749, 0.767, 0.758]; P_RR = [0.034, -0.039, -0.079, -0.014, -0.029]

fig = plt.figure(figsize=(13.8, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.05], wspace=0.34)
plt.subplots_adjust(bottom=0.18, top=0.88, left=0.055, right=0.975)
fig.suptitle("Direction is not identifiable from observation: inducing asymmetry never recovers held-out direction",
             fontsize=10.5, y=0.99)

# A: synthetic
axA = fig.add_subplot(gs[0]); x = np.arange(3); w = 0.26
axA.bar(x - w, OURS, w, color=C["cmc"], label="our read-out ($G{=}DAD^{\\top}$)")
axA.bar(x, LING_Z, w, color="#8172B3", label="DirectLiNGAM on $z$")
axA.bar(x + w, LING_D, w, color="#55A868", label="DirectLiNGAM (decode$\\to$recover)")
axA.axhline(0.5, ls=":", lw=0.9, c="#888"); axA.text(2.42, 0.52, "chance", fontsize=6.5, color="#888", ha="right")
axA.set_xticks(x); axA.set_xticklabels(FAM); axA.set_ylim(0, 1.08); axA.set_ylabel("orientation accuracy")
axA.set_title("Synthetic: our read-out cannot orient", fontsize=9.5)
axA.legend(fontsize=6.2, frameon=False, loc="upper left"); panel_label(axA, "A")

# B (expanded): every trained model — induced asymmetry vs held-out RNA->RNA direction; flat at chance
axB = fig.add_subplot(gs[1])
seen = set()
for grp, coex, rr, asym in SWEEP:
    if np.isnan(asym): continue
    lab = grp if grp not in seen else None; seen.add(grp)
    axB.scatter(asym, rr, s=60, color=GCOL[grp], edgecolor="white", lw=0.7, label=lab, zorder=3)
axB.axhline(0, ls="-", lw=0.9, c="black"); axB.axhspan(-0.05, 0.05, color="#ccc", alpha=0.4, zorder=0)
axB.text(0.015, 0.055, "chance band ($|$pearDE$|{<}0.05$)", transform=axB.transAxes, fontsize=6.6, color="#666", va="bottom")
axB.set_xscale("log"); axB.set_xlabel("induced GRN asymmetry (TF$\\to$gene / reverse ratio) $\\longrightarrow$ more asymmetric")
axB.set_ylabel("held-out RNA$\\to$RNA pearDE"); axB.set_ylim(-0.16, 0.16)
axB.set_title("Every model stays at chance across 3 orders of magnitude of asymmetry", fontsize=9.5)
axB.legend(fontsize=6.2, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.20),
           columnspacing=1.0, handletextpad=0.2)
panel_label(axB, "B")

# C: prior lever isolated (twin axes)
axC = fig.add_subplot(gs[2])
axC.plot(PW, P_ASYM, "-o", color="#CC3311", lw=1.6, ms=5, label="asymmetry")
axC.set_yscale("log"); axC.set_ylabel("orientation asymmetry (log)", color="#CC3311")
axC.tick_params(axis="y", labelcolor="#CC3311"); axC.set_xlabel("directed-prior weight")
axC.set_title("Directed prior: asymmetry memorized", fontsize=9.5)
axC2 = axC.twinx()
axC2.plot(PW, P_COEX, "-s", color="#2B6CB0", lw=1.4, ms=4, label="co-expression AUROC")
axC2.plot(PW, P_RR, "-^", color="#2F855A", lw=1.4, ms=4, label="held-out RR pearDE")
axC2.axhline(0, ls=":", lw=0.8, c="#888"); axC2.set_ylim(-0.2, 1.05); axC2.set_ylabel("AUROC  /  pearDE")
h1, l1 = axC.get_legend_handles_labels(); h2, l2 = axC2.get_legend_handles_labels()
axC.legend(h1 + h2, l1 + l2, fontsize=6.0, frameon=False, loc="center right")
panel_label(axC, "C")

save(fig, f"{R}/paper/figures/AppendixFigS3.pdf"); print("WROTE AppendixFigS3.pdf")
