"""Shared style for the final figures: identical to the original manuscript figures
(Carlito font, same palette as analysis/figures/original/style.py and rev_fig3_parts.py)."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
_fd = os.path.expanduser("~/.fonts")
for f in (os.listdir(_fd) if os.path.isdir(_fd) else []):   # Carlito (metric-compatible with Calibri); falls back to default font if absent
    if f.startswith("Carlito"):
        fm.fontManager.addfont(os.path.join(_fd, f))
plt.rcParams.update({
    "font.family": "Carlito", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "legend.frameon": False, "axes.titlepad": 4,
    "mathtext.fontset": "custom", "mathtext.rm": "Carlito", "mathtext.it": "Carlito:italic",
    "mathtext.bf": "Carlito:bold"})

R = f"{_CMC_ROOT}"
NR = f"{_CMC_RESULTS}/paper"
OUT = f"{NR}/FINAL_FIGURES"
COL = {"CMC-VAE": "#2b6cb0", "totalVI": "#dd6b20", "Correlation": "#a0aec0",
       "GRNBoost2": "#319795", "DeepSEM": "#6b46c1",
       "Learned": "#2b6cb0", "Shuffled": "#a0aec0", "Zeroed": "#cbd5e0",
       "obs": "#a0aec0", "ft": "#2b6cb0", "ftnce": "#1a365d", "shuf": "#cbd5e0"}
SCREENS = ["frangieh", "iccite", "arce"]
SLBL = {"frangieh": "Frangieh", "iccite": "Chen", "arce": "Arce"}
EK = dict(lw=0.7, capsize=1.6, ecolor="#333333")


def lab(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def panel(ax, s, x=-0.13, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def bracket(ax, x1, x2, y, txt, h=None, fs=7.5):
    lo, hi = ax.get_ylim(); h = h if h is not None else 0.015 * (hi - lo)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.7, c="k", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, txt, ha="center", va="bottom", fontsize=fs)


def chance(ax, y=0.5):
    ax.axhline(y, ls=":", lw=0.9, c="#4a5568", zorder=0)
