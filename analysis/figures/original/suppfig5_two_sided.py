"""Supplementary Fig. 5 redrawn with TWO-SIDED paired t-tests (Benjamini–Hochberg within panel), replacing the one-sided
(forward > reverse) stars of the original panel. Adapted from paper/figscripts/figS_orientation_battery.py (copy in
orig_figscripts/); same data: results/orientation_battery.json and per-perturbation results/<model>/edge_reversal_<cohort>.json
(earlier cell-level DE definition, seed-42 models, as in the original panel). Stars shown for BH q < 0.05 where |gap| >= 0.05 (display rule of the original panel; all q in the source JSON).
Output: figures/final/_parts/SupplementaryFig5_two_sided.{pdf,png} + source data JSON."""

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, C, DS, panel_label
from scipy import stats as st
R = _CMC_ROOT; OUT = f"{_CMC_RESULTS}/paper/figures/final/_parts"
B = json.load(open(f"{R}/results/orientation_battery.json"))
COH = ["frangieh", "iccite", "arce"]; CN = ["Frangieh", "Chen", "Arce"]; CCOL = [DS[c] for c in COH]
ORDER = ["Baseline", "cond-iVAE celltype", "cond-iVAE donor", "cond-iVAE disease", "non-Gaussian Laplace", "resid-indep TC", "resid-indep HSIC", "reversal-loss"]
XL = ["CMC-VAE\n(baseline)", "iVAE:\ncell type", "iVAE:\ndonor", "iVAE:\ndisease", "Laplace\nprior", "TC\npenalty", "HSIC\npenalty", "Reversal\nloss"]
def bh(p):
    p = np.asarray(p, float); ok = np.isfinite(p); q = np.full_like(p, np.nan); v = p[ok]; n = len(v)
    if n:
        o = np.argsort(v); r = v[o] * n / np.arange(1, n + 1); r = np.minimum.accumulate(r[::-1])[::-1]; t = np.empty(n); t[o] = np.minimum(r, 1); q[ok] = t
    return q
def collect(block):
    G = np.full((8, 3), np.nan); S = np.zeros((8, 3)); P = np.full((8, 3), np.nan); N = np.zeros((8, 3), int)
    for mi, name in enumerate(ORDER):
        for ci, c in enumerate(COH):
            j = json.load(open(f"{R}/{B[name]['dir']}/edge_reversal_{c}.json")); pp = j["per_perturbation"]
            a = np.array([p.get(f"{block}_pearDE_true", np.nan) for p in pp], float); b = np.array([p.get(f"{block}_pearDE_rev", np.nan) for p in pp], float)
            m = np.isfinite(a) & np.isfinite(b); dlt = (a - b)[m]; N[mi, ci] = m.sum()
            if m.sum() >= 2: G[mi, ci] = dlt.mean(); S[mi, ci] = dlt.std(ddof=1) / np.sqrt(m.sum())
            if m.sum() >= 3 and not np.allclose(dlt, 0): P[mi, ci] = st.ttest_rel(a[m], b[m]).pvalue
    Q = bh(P.ravel()).reshape(P.shape)
    return G, S, P, Q, N
RR = collect("rr"); RP = collect("rp")
sym = np.array([B[n]["corr_G_GT"] for n in ORDER])
def lab(q): return "***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else ""
fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.9)); plt.subplots_adjust(hspace=0.45, top=0.97, bottom=0.10, left=0.11, right=0.98)
x = np.arange(8); w = 0.26
ax = axes[0]; ax.axvspan(0.5, 3.5, color=C["accent"], alpha=0.09, zorder=0)
ax.bar(x, sym, 0.62, color=[C["accent"] if "iVAE" in n else C["muted"] for n in ORDER], zorder=3); ax.axhline(1, ls=":", lw=0.8, c=C["ink"])
for xi, v in zip(x, sym): ax.text(xi, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=6.4)
ax.set_ylim(0.4, 1.1); ax.set_ylabel(r"corr$(G_{rr},G_{rr}^{\top})$" + "\n(1 = undirected)"); ax.set_xticks(x); ax.set_xticklabels([])
ax.text(2.0, 1.075, "iVAE auxiliary priors", ha="center", va="top", fontsize=6.8, color=C["accent"], style="italic")
def gp(ax, D, ylab):
    G, S, P, Q, N = D; ax.axvspan(0.5, 3.5, color=C["accent"], alpha=0.09, zorder=0)
    for ci in range(3):
        xs = x + (ci - 1) * w
        ax.bar(xs, G[:, ci], w, color=CCOL[ci], label=CN[ci], yerr=S[:, ci], error_kw=dict(lw=0.6, capsize=1.5, ecolor="#555"))
        for xi, v, s_, q in zip(xs, G[:, ci], S[:, ci], Q[:, ci]):
            if np.isfinite(v) and np.isfinite(q) and lab(q) and abs(v) >= 0.05:   # display rule as in the original panel
                ax.text(xi, v + s_ + 0.01 if v >= 0 else v - s_ - 0.01, lab(q), ha="center", va="bottom" if v >= 0 else "top", fontsize=6.0)
    ax.axhline(0, lw=0.9, c=C["ink"]); ax.set_ylabel(ylab, fontsize=8.5); ax.set_ylim(-0.62, 0.62); ax.set_xticks(x)
gp(axes[1], RR, r"$A_{rr}$ forward $-$ reverse" + "\n(signed pearDE)"); axes[1].set_xticklabels([])
gp(axes[2], RP, r"$W_{rp}$ forward $-$ reverse" + "\n(signed pearDE)"); axes[2].set_xticklabels(XL, fontsize=7.4)
axes[2].legend(loc="lower right", ncol=3, fontsize=7.2)
for i, a in enumerate(axes): panel_label(a, "abc"[i], x=-0.075)
fig.savefig(f"{OUT}/SupplementaryFig5_two_sided.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(f"{OUT}/SupplementaryFig5_two_sided.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
out = {blk: {f"{ORDER[mi]}|{COH[ci]}": dict(gap=float(D[0][mi, ci]), sem=float(D[1][mi, ci]), p_two=float(D[2][mi, ci]), q_bh_panel=float(D[3][mi, ci]), n=int(D[4][mi, ci]))
             for mi in range(8) for ci in range(3)} for blk, D in (("rr", RR), ("rp", RP))}
json.dump(out, open(f"{OUT}/SupplementaryFig5_two_sided_source.json", "w"), indent=1)
for blk, D in (("A_rr", RR), ("W_rp", RP)):
    print(blk); [print(f"  {ORDER[mi]:22s}", " ".join(f"{D[0][mi,ci]:+.3f}(q={D[3][mi,ci]:.2g},n={D[4][mi,ci]})" for ci in range(3))) for mi in range(8)]
