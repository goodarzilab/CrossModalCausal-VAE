"""AppendixFigS6 -- Feedback block (protein->RNA, G_pr): structured, but inherited from translation.
Trimmed to the two informative panels. (A) transpose diagnostic: G_pr is ~the transpose of the learned
translation map G_rp (Pearson 0.71; ~50% of feedback energy transpose-explained) -- structured, but
not independent post-receptor signalling. (B) on the transpose-removed residual, no receptor's target
set beats protein-gene co-expression -- there is no independent directed feedback signal to read.
Panel A recomputed from the two projected-block CSVs; panel B read from results/feedback_biology.json."""
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
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = _CMC_ROOT
RES = os.path.join(REPO, "results"); sys.path.insert(0, HERE)
from style import plt, C, panel_label, save  # noqa: E402

rd = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(RES, "sweep_sparse", "sp_dec_gl1000")
out = json.load(open(os.path.join(RES, "feedback_biology.json")))

# ---- panel-A scatter arrays from the projected CSVs ----
Gpr = pd.read_csv(f"{rd}/Regulation_Prot_to_RNA.csv", index_col=0).values.astype(np.float64)      # [P, G]
GrpT = pd.read_csv(f"{rd}/Transcription_RNA_to_Prot.csv", index_col=0).values.T.astype(np.float64)  # [P, G]
a = Gpr.ravel(); b = GrpT.ravel()
td = out["transpose_diagnostic"]; beta = td["ls_beta"]; var_explained = td["variance_explained_by_transpose"]
pr = td["pearson_vec"]; sr_corr = td["spearman_vec"]
enrich = out["receptor_enrichment"]
INTENDED = [("CD14", "TNFA_NFKB"), ("CD3-1", "TCR"), ("CD3-1", "IEG"),
            ("CD25", "JAK_STAT"), ("CD122", "JAK_STAT"), ("CD127", "JAK_STAT")]
has_coexpr = bool(out.get("coexpr_baseline", {}).get("computed", False))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.0)); plt.subplots_adjust(wspace=0.30, bottom=0.20, top=0.86)

# ---- A: transpose diagnostic ----
rng = np.random.RandomState(0); ss = rng.choice(len(a), size=min(8000, len(a)), replace=False)
sc_a = a[ss] * 1e3; sc_b = b[ss] * 1e3
axA.scatter(sc_b, sc_a, s=2, c=C["muted"], alpha=0.25, rasterized=True, edgecolors="none")
lim = np.percentile(np.abs(np.concatenate([sc_a, sc_b])), 99.5); xs = np.linspace(-lim, lim, 50)
axA.plot(xs, beta * xs, c=C["neg"], lw=1.4, label=f"LS fit $\\beta={beta:.2f}$ ({var_explained*100:.0f}% var)")
axA.set_xlim(-lim, lim); axA.set_ylim(-lim, lim)
axA.set_xlabel("translation transpose $G_{rp}^{\\top}$ ($\\times10^{-3}$)")
axA.set_ylabel("feedback $G_{pr}$ ($\\times10^{-3}$)")
axA.set_title(f"Feedback $\\approx$ transpose of translation ($r={pr:.2f}$)", fontsize=9.8)
axA.legend(loc="lower right", fontsize=7.2, frameon=False); panel_label(axA, "A")

# ---- B: receptor->target enrichment (raw vs residual vs coexpr) ----
labs = [f"{r}\n$\\rightarrow${t.replace('_','/')}" for r, t in INTENDED]
x = np.arange(len(INTENDED)); w = 0.26
def _v(series): return [(enrich[f"{r}->{t}"].get(series) or {}).get("auroc", np.nan) for r, t in INTENDED]
axB.bar(x - w, _v("raw_Gpr"), w, color=C["shuf"], label="raw $|G_{pr}|$")
axB.bar(x,     _v("residual_R"), w, color=C["model"], label="transpose-removed residual $|R|$")
axB.bar(x + w, (_v("coexpr") if has_coexpr else [np.nan]*len(INTENDED)), w, color=C["base"], label="protein-gene co-expr")
for xi, (r, t) in zip(x, INTENDED):
    rr = enrich[f"{r}->{t}"]["residual_R"]
    if rr and rr["perm_null_p"] < 0.05:
        axB.text(xi, rr["auroc"] + 0.02, "*", ha="center", va="bottom", fontsize=11, color=C["model"])
axB.axhline(0.5, ls="--", lw=0.8, c=C["muted"]); axB.text(len(INTENDED)-0.5, 0.515, "chance", fontsize=6.6, ha="right", color=C["muted"])
axB.set_xticks(x); axB.set_xticklabels(labs, fontsize=6.6)
axB.set_ylabel("receptor$\\to$target within-row AUROC"); axB.set_ylim(0.2, 1.0)
axB.set_title("Transpose-removed residual rarely exceeds co-expression", fontsize=9.8)
axB.legend(loc="upper right", fontsize=6.6, frameon=False); panel_label(axB, "B")

save(fig, os.path.join(REPO, "paper", "figures", "AppendixFigS6.pdf"))
print(f"WROTE AppendixFigS6.pdf | Gpr~Grp^T r={pr:.2f} beta={beta:.2f} var={var_explained*100:.0f}%")
