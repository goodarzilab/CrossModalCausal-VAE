"""AppendixFigS7 — observational orientation battery. Eight models trained on the identical two-cohort
corpus, each an identifiability-theoretic lever for orienting the latent SCM: the headline baseline,
three conditional-iVAE auxiliary priors (celltype/donor/disease), a non-Gaussian (Laplace) exogenous
prior, two residual-independence penalties (TC / HSIC), and an explicit edge-reversal loss. (A) induced
GRN asymmetry corr(G,G^T) [1=undirected]; (B) A_rr and (C) W_rp forward-minus-reverse held-out
perturbation gap in signed pearDE per cohort (bars = mean gap, error bars = SEM over the held-out
perturbations, stars = paired one-sided t (fwd>rev), |gap|>=0.05). Reads orientation_battery.json +
per-cohort edge_reversal_*.json."""
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
sys.path.insert(0, os.path.dirname(__file__))
from style import plt, C, DS, panel_label, save, sig_label
from scipy import stats as _st
R = f"{_CMC_ROOT}"
B = json.load(open(f"{_CMC_RESULTS}/orientation_battery.json"))
COH = ["frangieh", "iccite", "arce"]; CN = ["Frangieh", "Chen", "Arce"]
CCOL = [DS["frangieh"], DS["iccite"], DS["arce"]]
ORDER = ["Baseline", "cond-iVAE celltype", "cond-iVAE donor", "cond-iVAE disease",
         "non-Gaussian Laplace", "resid-indep TC", "resid-indep HSIC", "reversal-loss"]
XL = ["Baseline", "cond-iVAE\ncelltype", "cond-iVAE\ndonor", "cond-iVAE\ndisease",
      "non-Gauss.\nLaplace", "resid-indep\nTC", "resid-indep\nHSIC", "reversal\nloss"]
EFF_FLOOR = 0.05

def paired(pp, kr, kc):
    a = np.array([p.get(kr, np.nan) for p in pp], float); b = np.array([p.get(kc, np.nan) for p in pp], float)
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) < 3 or np.allclose(a, b): return np.nan
    return float(_st.ttest_rel(a, b, alternative="greater").pvalue)

def collect(block):
    gaps = np.full((len(ORDER), 3), np.nan); ps = np.full((len(ORDER), 3), np.nan); sems = np.zeros((len(ORDER), 3))
    for mi, name in enumerate(ORDER):
        d = B[name]["dir"]
        for ci, c in enumerate(COH):
            f = f"{R}/{d}/edge_reversal_{c}.json"
            if not os.path.exists(f): continue
            j = json.load(open(f)); s = j["summary"]; pp = j["per_perturbation"]
            t = s.get(f"mean_{block}_pearDE_true"); r = s.get(f"mean_{block}_pearDE_rev")
            if t is not None and r is not None and t == t and r == r: gaps[mi, ci] = t - r
            a = np.array([p.get(f"{block}_pearDE_true", np.nan) for p in pp], float)
            b = np.array([p.get(f"{block}_pearDE_rev", np.nan) for p in pp], float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= 2: sems[mi, ci] = float(np.std((a - b)[m], ddof=1) / np.sqrt(m.sum()))
            ps[mi, ci] = paired(pp, f"{block}_pearDE_true", f"{block}_pearDE_rev")
    return gaps, ps, sems

RR, RRp, RRs = collect("rr"); RP, RPp, RPs = collect("rp")
sym = np.array([B[n]["corr_G_GT"] for n in ORDER])

fig, axes = plt.subplots(3, 1, figsize=(7.6, 8.4))
plt.subplots_adjust(hspace=0.58, top=0.905, bottom=0.10, left=0.11, right=0.98)
fig.suptitle("No identifiability-theoretic lever recovers a coherent GRN orientation", fontsize=10.5, y=0.965)
fig.text(0.5, 0.928, r"$^{*}\,p<0.05$      $^{**}\,p<0.01$      $^{***}\,p<0.001$", ha="center", fontsize=8.0, color="#333333")
x = np.arange(len(ORDER)); w = 0.26

# ---- Panel A: induced asymmetry ----
axA = axes[0]
cols = [C["accent"] if "cond-iVAE" in n else C["muted"] for n in ORDER]
axA.axvspan(0.5, 3.5, color=C["accent"], alpha=0.09, zorder=0)
axA.bar(x, sym, 0.62, color=cols, zorder=3)
axA.axhline(1.0, ls=":", lw=0.8, c=C["ink"])
axA.set_ylim(0.4, 1.10); axA.set_ylabel(r"corr$(G_{rr},G_{rr}^{\top})$")
axA.text(2.0, 1.075, "iVAE priors", ha="center", va="top", fontsize=6.8, color=C["accent"], style="italic")
axA.set_title("Only iVAE auxiliary priors break GRN symmetry (1 = undirected)", fontsize=9.2)
for xi, v in zip(x, sym): axA.text(xi, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=6.4, zorder=4)
axA.set_xticks(x); axA.set_xticklabels([])

def gappanel(ax, G, S, P, title, ytop):
    ax.axvspan(0.5, 3.5, color=C["accent"], alpha=0.09, zorder=0)
    for ci in range(3):
        xs = x + (ci - 1) * w
        ax.bar(xs, G[:, ci], w, color=CCOL[ci], label=CN[ci],
               yerr=S[:, ci], error_kw=dict(lw=0.6, capsize=1.5, ecolor="#555"))
        for xi, v, p in zip(xs, G[:, ci], P[:, ci]):
            if not np.isfinite(v): continue
            lab = sig_label(p, show_ns=False) if (np.isfinite(p) and abs(v) >= EFF_FLOOR) else ""
            if lab:
                ax.text(xi, v + (0.014 if v >= 0 else -0.03), lab, ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=6.0, color=C["ink"])
    ax.axhline(0, lw=0.9, c=C["ink"])
    ax.set_ylabel("forward $-$ reverse\n(signed pearDE)", fontsize=8.5)
    ax.set_title(title, fontsize=9.2); ax.set_ylim(-0.62, ytop); ax.set_xticks(x)

gappanel(axes[1], RR, RRs, RRp, r"$A_{rr}$ (GRN): forward $-$ reverse gap sign-flips across cohorts", 0.44)
axes[1].set_xticklabels([])
gappanel(axes[2], RP, RPs, RPp, r"$W_{rp}$ (translation): oriented at baseline, broken by disease/donor priors", 0.40)
axes[2].set_xticklabels(XL, fontsize=7.4)
axes[2].legend(loc="lower right", ncol=3, fontsize=7.2, columnspacing=1.0, handletextpad=0.4)
for i, ax in enumerate(axes): panel_label(ax, "ABC"[i], x=-0.075)
save(fig, f"{R}/paper/figures/AppendixFigS7.pdf"); print("WROTE AppendixFigS7.pdf (orientation battery)")
