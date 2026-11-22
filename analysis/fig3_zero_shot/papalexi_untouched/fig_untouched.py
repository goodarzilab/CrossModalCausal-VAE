#!/usr/bin/env python
"""Figure panels for the untouched-screen (Papalexi THP-1 ECCITE-seq) evaluation.
Usage: python fig_untouched.py [tag=papalexi]"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import sys, json, os
import numpy as np, pandas as pd
sys.path.insert(0, f"{_CMC_ROOT}/analysis/figures/original")
from style import plt, C, panel_label, save
UNT = f"{_CMC_RESULTS}/paper/analyses/untouched"
FIG = f"{_CMC_RESULTS}/paper/figures/untouched"; os.makedirs(FIG, exist_ok=True)
TAG = sys.argv[1] if len(sys.argv) > 1 else "papalexi"
S = json.load(open(f"{UNT}/stats_{TAG}.json"))
Z = pd.read_csv(f"{UNT}/source_data_{TAG}_zeroshot_per_unit.csv")
T = pd.read_csv(f"{UNT}/source_data_{TAG}_transfer_per_unit.csv")
V = pd.read_csv(f"{UNT}/source_data_{TAG}_cv_per_unit.csv")
GRAY = C["muted"]; PUR = C["accent"]; TEAL = "#319795"

def dotpanel(ax, df, key, methods, labels, colors, metric, chance, xlabel, title, hatch_note=None):
    """horizontal: per-unit points + mean with 95% bootstrap CI over units."""
    rng = np.random.default_rng(0); rows = []
    for i, m in enumerate(methods):
        v = df.loc[df[key] == m, metric].astype(float).dropna().values
        y = len(methods) - 1 - i
        if len(v) == 0:
            ax.text(chance, y, "undefined", va="center", ha="center", fontsize=6.5, color=GRAY); continue
        ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=9, color=colors[i], alpha=0.45, lw=0, zorder=2)
        bs = rng.choice(v, (5000, len(v))).mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.plot([lo, hi], [y, y], color=C["ink"], lw=1.2, zorder=3)
        ax.scatter([v.mean()], [y], s=34, color=colors[i], edgecolor=C["ink"], lw=0.7, zorder=4)
        rows.append(dict(panel=title, method=labels[i], metric=metric, n=len(v), mean=v.mean(), ci_lo=lo, ci_hi=hi))
    ax.axvline(chance, color=GRAY, lw=0.8, ls="--", zorder=1)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(labels[::-1])
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left")
    return rows

ZS = ["CMC-VAE (seed mean)", "corr", "totalvi", "deepsem", "zs_ctrl_expr", "zs_ctrl_var", "zs_decoder_rownorm", "mean_response_ref", "dev_copy"]
ZL = ["CMC-VAE", "Correlation", "totalVI", "DeepSEM", "Ctrl expression", "Ctrl variance", "Decoder-row norm", "Mean response (ref.)", "Same gene, dev. screen*"]
ZC = [C["model"], C["base"], C["base"], C["base"], GRAY, GRAY, GRAY, GRAY, PUR]
# ridge transfer rows join the zero-shot primary panels (they use no Papalexi perturbation data)
TR = T[T.arm.isin(["ridge_decoder", "ridge_coexpr"])].rename(columns={"arm": "method"})
ZP = pd.concat([Z, TR], ignore_index=True)
PM = ["CMC-VAE (seed mean)", "corr", "totalvi", "deepsem", "ridge_decoder", "ridge_coexpr", "dev_copy"]
PL = ["CMC-VAE", "Correlation", "totalVI", "DeepSEM", "Ridge (decoder), dev-trained", "Ridge (co-expr.), dev-trained", "Same gene, dev. screen*"]
PC = [C["model"], C["base"], C["base"], C["base"], TEAL, TEAL, PUR]

fig, axs = plt.subplots(2, 3, figsize=(10.2, 6.2), gridspec_kw=dict(wspace=0.95, hspace=0.55))
src = []
src += dotpanel(axs[0, 0], ZP, "method", PM, PL, PC, "rr_pe_cent", 0, "Mean-centred RNA pearDE", "Primary 1: centred pearDE")
src += dotpanel(axs[0, 1], ZP, "method", PM, PL, PC, "rr_retrieval", 0.5, "Held-out retrieval (chance 0.5)", "Primary 2: retrieval")
src += dotpanel(axs[0, 2], Z, "method", ZS, ZL, ZC, "rr_auroc", 0.5, "RNA-response AUROC", "Secondary: AUROC")
TA = ["obs", "ft_standard", "ft_labelshuffle_mean", "ft_meantarget"]
TL = ["Observational", "Fine-tuned (dev screens)", "Label-shuffled FT", "Screen-mean FT"]
TCc = [C["model"], "#63b3ed", C["shuf"], C["zero"]]
src += dotpanel(axs[1, 0], T, "arm", TA, TL, TCc, "rr_pe_cent", 0, "Mean-centred RNA pearDE", "S2: cross-screen transfer")
src += dotpanel(axs[1, 1], T, "arm", TA, TL, TCc, "rr_retrieval", 0.5, "Held-out retrieval", "S2: cross-screen transfer")
VA = ["obs", "ft_real", "ft_shuffle_mean", "ft_mean", "ridge_decoder", "ridge_coexpr"]
VL = ["Observational", "Fine-tuned (Papalexi)", "Label-shuffled FT", "Screen-mean FT", "Ridge (decoder)", "Ridge (co-expr.)"]
VC = [C["model"], "#63b3ed", C["shuf"], C["zero"], TEAL, TEAL]
src += dotpanel(axs[1, 2], V, "arm", VA, VL, VC, "rr_pe_cent", 0, "Mean-centred RNA pearDE (held-out fold)", "S1: within-screen 3-fold CV")
for ax, l in zip(axs.flat, "abcdef"): panel_label(ax, l, x=-0.02, y=1.04)
n = S["units"]["n"]
fig.text(0.01, -0.02, f"Papalexi THP-1 ECCITE-seq (IFN-γ), untouched screen; n = {n} evaluable knockouts (points). Mean ± 95% bootstrap CI over knockouts; "
         "CMC-VAE = per-knockout mean over seeds 42/1/2.\n*Same-gene response measured in Frangieh/Chen/Arce, available for the 7 evaluable targets shared with a development screen. "
         "Knockout-invariant rankings have undefined centred pearDE and chance retrieval by construction.", fontsize=6.5, color=C["ink"])
save(fig, f"{FIG}/untouched_{TAG}.pdf")
pd.DataFrame(src).to_csv(f"{FIG}/untouched_{TAG}_source_data_summary.csv", index=False)
for nm, df in [("zeroshot", Z), ("transfer", T), ("cv", V)]: df.to_csv(f"{FIG}/untouched_{TAG}_{nm}_per_unit.csv", index=False)
