"""Shared publication style for CMC-VAE manuscript figures.
Limited, consistent palette: model=blue, baseline=orange, nulls=grays, signed=green/red,
one purple accent used sparingly. Vector PDF with editable (Type-42) text."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False, "axes.grid": False, "axes.titlepad": 4,
})

# ---- unified limited palette ----
C = {
    "model":  "#2b6cb0",   # CMC-VAE / primary
    "base":   "#dd6b20",   # baseline / comparison (corr, totalVI, GENIE3)
    "shuf":   "#a0aec0",   # shuffled null
    "zero":   "#cbd5e0",   # zeroed null
    "pos":    "#2f855a",   # signed positive
    "neg":    "#c53030",   # signed negative
    "accent": "#6b46c1",   # single purple accent (use sparingly)
    "ink":    "#1a202c",   # text/dark
    "muted":  "#718096",   # secondary text / chance lines
}
# datasets share one consistent color mapping when shown together
DS = {"frangieh": "#2b6cb0", "iccite": "#4a90c2", "arce": "#7fb0d6"}

# ---- aliases into the unified palette (semantic, limited to ~6 hues) ----
C.update({
    "cmc": C["model"], "cmc2": "#63b3ed", "totalvi": C["base"], "genie3": "#319795",
    "neutral": C["muted"], "gray": C["shuf"], "dark": C["ink"],
    "matched": C["base"], "unmatched": "#319795",
    # sweep lever groups map onto the same limited palette
    "kl": C["model"], "ebn": C["accent"], "dec": C["pos"], "nl": C["base"], "pri": C["neg"],
})

def panel_label(ax, s, x=-0.16, y=1.06, size=12):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="bottom", ha="right")

def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("WROTE", path)

# ---- significance helpers ---------------------------------------------------
import numpy as _np  # noqa: E402
from scipy import stats as _st  # noqa: E402

def stars(p):
    """*/**/*** for p<0.05/0.01/0.001; '' otherwise."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""

def p_vs_null(mean, sem, n, null=0.0):
    """Two-sided one-sample t of a per-item mean (with its SEM over n items) vs a null value."""
    if not sem or sem <= 0 or not n or n < 2:
        return 1.0
    t = (mean - null) / sem
    return float(2 * _st.t.sf(abs(t), n - 1))

def p_paired_greater(a, b):
    """One-sided Wilcoxon signed-rank: is paired a > b (per-item arrays)?"""
    a = _np.asarray(a, float); b = _np.asarray(b, float)
    m = _np.isfinite(a) & _np.isfinite(b); a, b = a[m], b[m]
    if len(a) < 3 or _np.allclose(a, b):
        return 1.0
    try:
        return float(_st.wilcoxon(a, b, alternative="greater").pvalue)
    except Exception:
        return 1.0

def p_two_mean(m1, s1, m2, s2):
    """Two-sided p for difference of two means given their SEMs (conservative independent-sample z,
    valid since same-perturbation SEMs are positively correlated -> this over-estimates the SE)."""
    se = ((s1 or 0.0) ** 2 + (s2 or 0.0) ** 2) ** 0.5
    if se <= 0 or not _np.isfinite(m1) or not _np.isfinite(m2):
        return 1.0
    return float(2 * _st.norm.sf(abs(m1 - m2) / se))

def p_two_mean_greater(m1, s1, m2, s2):
    """One-sided p: is mean1 > mean2 (given SEMs; conservative independent z)."""
    se = ((s1 or 0.0) ** 2 + (s2 or 0.0) ** 2) ** 0.5
    if se <= 0 or not _np.isfinite(m1) or not _np.isfinite(m2):
        return 1.0
    return float(_st.norm.sf((m1 - m2) / se))

def sig_label(p, show_ns=True):
    """'***'/'**'/'*' for p<.001/.01/.05, else 'n.s.' (or '')."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ("n.s." if show_ns else "")

def sig_bracket(ax, x1, x2, y, label, h=0.02, lw=0.8, fs=6.0, color="#333333"):
    """Draw a significance bracket spanning [x1,x2] at height y with `label` centred above it.
    h is bracket-tick height in data units. Use ax.set_ylim headroom before calling.
    Non-significant / empty labels draw nothing (only significant comparisons are annotated)."""
    if not label or label == "n.s.":
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c=color, clip_on=False)
    va = "bottom"; ax.text((x1 + x2) / 2.0, y + h, label, ha="center", va=va,
                           fontsize=(fs - 1.0 if label == "n.s." else fs), color=color, clip_on=False)
