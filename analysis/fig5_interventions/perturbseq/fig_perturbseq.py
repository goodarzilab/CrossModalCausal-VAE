"""Revised Fig. 5e-g + new panels (h: ridge vs RRR-span(D) vs SCM; i: cross-context) from
paper_nov_revisions/analyses/perturbseq/summary.json. Style copied from paper/figscripts/rev_fig5_parts.py.
Outputs PDF+PNG + source-data JSON in paper_nov_revisions/figures/perturbseq/."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fd = os.path.expanduser("~/.fonts")
if os.path.isdir(fd):
    for f in os.listdir(fd):
        if f.startswith("Carlito"): fm.fontManager.addfont(f"{fd}/{f}")
plt.rcParams.update({"font.family": "Carlito", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False,
    "mathtext.fontset": "custom", "mathtext.rm": "Carlito", "mathtext.it": "Carlito:italic"})
R = f"{_CMC_ROOT}"; AN = f"{_CMC_RESULTS}/paper/analyses/perturbseq"; OUT = f"{_CMC_RESULTS}/paper/figures/perturbseq"
os.makedirs(OUT, exist_ok=True)
S = json.load(open(f"{AN}/summary.json"))
OBS, FT, DS, DSL, SH, NCE = "#a0aec0", "#2b6cb0", "#6b46c1", "#b7a6e0", "#e2e8f0", "#1a365d"
BASE, BASE2, TEAL = "#dd6b20", "#f6ad55", "#319795"
EK = dict(lw=0.7, capsize=2, ecolor="#333")
def stars(p): return "ns" if p is None else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
def bracket(ax, x1, x2, y, s, h=0.004):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.7, c="k"); ax.text((x1 + x2) / 2, y + h, s, ha="center", va="bottom", fontsize=6.5)
def lab(ax, s, x=-0.2, y=1.06): ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
def save(fig, name, data):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.01); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig); json.dump(data, open(f"{OUT}/{name}_data.json", "w"), indent=1, default=str); print("wrote", name)
LINES = [("jurkat", "Jurkat"), ("k562", "K562")]
PRES = [("obs", "controls + training perts"), ("ctrl", "controls only")]

# ---------------- e: centred pearDE + retrieval, both pretrainings, seed-level ----------------
fig, axes = plt.subplots(2, 1, figsize=(7.05, 4.4), gridspec_kw=dict(hspace=0.45)); edata = {}
arms = [("obs", "Observational", OBS, ""), ("real", "Fine-tuned", FT, ""), ("shuffle_avg", "Label-shuffled", SH, "////"),
        ("infonce", "Fine-tuned + InfoNCE", NCE, ""), ("infonce_shuffle_avg", "InfoNCE, label-shuffled", SH, "xxxx")]
for ax, (M, ylab, ylim) in zip(axes, [("rr_pe_cent", "Held-out centred pearDE", (-0.03, 0.115)), ("rr_retrieval", "Held-out gene retrieval", (0.45, 0.68))]):
    groups = [(l, p) for l, _ in LINES for p in ["obs", "ctrl", "ctrl_ep1000"] if f"{l}_{p}" in S["ft"]]
    w = 0.15
    for j, (l, p) in enumerate(groups):
        T = S["ft"][f"{l}_{p}"]["arms"]; C = S["ft_contrasts"][f"{l}_{p}"]
        for i, (a, name, col, hat) in enumerate(arms):
            v = [x for x in T[a][M]["per_seed"] if x is not None]; x0 = j + (i - 2) * w
            ax.bar(x0, np.mean(v), w, color=col, hatch=hat, edgecolor="#718096" if hat else col, lw=0.4, label=name if j == 0 else None)
            ax.scatter([x0] * len(v), v, s=3, c="k", zorder=3, lw=0); edata[f"{l}|{p}|{a}|{M}"] = v
        for a, b, i1, i2, dy in [("real", "shuffle_avg", 1, 2, 0), ("infonce", "infonce_shuffle_avg", 3, 4, 1)]:
            c = C.get(f"{a}_vs_{b}|{M}")
            if not c: continue
            top = max(max(T[a][M]["per_seed"]), max(T[b][M]["per_seed"]))
            bracket(ax, j + (i1 - 2) * w, j + (i2 - 2) * w, top + 0.004 + dy * 0.012 * (1 if M == "rr_pe_cent" else 2), stars(c.get("q_bh")), h=0.002 if M == "rr_pe_cent" else 0.004)
    ax.axhline(0 if M == "rr_pe_cent" else 0.5, ls=":", lw=0.8, c="#4a5568")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{dict(LINES)[l]}\n" + {"obs": "orig.", "ctrl": "ctrl-only\n120 ep", "ctrl_ep1000": "ctrl-only\n1000 ep"}[p] for l, p in groups], fontsize=6)
    ax.set_ylabel(ylab); ax.set_ylim(*ylim)
axes[0].legend(loc="upper left", fontsize=5.8, handlelength=1.2, labelspacing=0.2, borderaxespad=0.1, ncol=2)
axes[0].set_title("Perturb-seq, held-out genes (3 models per bar; dots = models)"); lab(axes[0], "e", x=-0.13)
save(fig, "fig5e_rev", edata)

# ---------------- f: learning curve, SCM vs ridge vs RRR (both lines, both pretrainings) ----------------
ks = ["5", "25", "100", "400", "all"]
fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.3), gridspec_kw=dict(wspace=0.25)); fdata = {}
series = [("real", "SCM fine-tuned", FT, "-"), ("shuffle", "SCM label-shuffled", OBS, "--"), ("infonce", "SCM + InfoNCE", NCE, "-"),
          ("rrr_spanD_r192", "RRR, span(D)", TEAL, "-"), ("ridge_footprint", "Ridge, decoder loadings", BASE, "-"), ("ridge_ntcorr", "Ridge, co-expression", BASE2, "-")]
for ax, (l, ln) in zip(axes, LINES):
    for p, mk, alpha in [("obs", "o", 1.0), ("ctrl", "s", 0.55)]:
        lc = S["lc"].get(f"{l}_{p}")
        if not lc: continue
        for v, name, col, ls in series:
            xs = [i for i, k in enumerate(ks) if v in lc.get(k, {}) and "rr_pe_cent" in lc[k][v]]
            if not xs: continue
            y = [lc[ks[i]][v]["rr_pe_cent"]["mean"] for i in xs]; e = [lc[ks[i]][v]["rr_pe_cent"]["sd"] for i in xs]
            ax.errorbar(xs, y, yerr=e, fmt=ls + mk, ms=2.8, lw=1.0, capsize=1.2, color=col, alpha=alpha,
                        label=f"{name}" if p == "obs" else None); fdata[f"{l}|{p}|{v}"] = dict(x=[ks[i] for i in xs], y=y, sd=e)
    ax.axhline(0, ls=":", lw=0.8, c="#4a5568"); ax.set_xticks(range(5)); ax.set_xticklabels(ks)
    ax.set_xlabel("Number of training perturbations"); ax.set_ylabel("Held-out centred pearDE"); ax.set_title(f"{ln} (circles: orig. pretraining; squares, faded: control-only)", fontsize=7)
axes[0].legend(loc="upper left", fontsize=5.8, labelspacing=0.2, borderaxespad=0.1); lab(axes[0], "f", x=-0.13)
save(fig, "fig5f_rev", fdata)

# ---------------- g: CMC-VAE vs DeepSEM trained on the same cells: specific gain (FT - label-shuffled FT) ----------------
#   rows: centred pearDE (canonical scale), retrieval; columns: pretraining. Stars above each DeepSEM bar = BH q of the
#   seed-level difference-in-differences (CMC gain - DeepSEM gain; two-sided Wilcoxon on seed-averaged per-KO values).
fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.3), gridspec_kw=dict(wspace=0.22, hspace=0.55)); gdata = {}
VAR = [("top73728", "DeepSEM, top-73,728 entries", DS), ("top36672", "DeepSEM, top-36,672 entries", DSL), ("lr12", "DeepSEM, rank-12 update, all genes", "#9f7aea")]
ARMS_G = [("real", "shuffle_avg", "std."), ("infonce", "infonce_shuffle_avg", "InfoNCE"), ("centered", "centered_shuffle_avg", "centred")]
for r, (M, ylab) in enumerate([("rr_pe_cent_canon", "Specific gain, centred pearDE\n(canonical scale)"), ("rr_retrieval", "Specific gain, retrieval")]):
    for cidx, (p, pn) in enumerate(PRES):
        ax = axes[r, cidx]; xt = []; xi = 0; ymax = 0
        for l, ln in LINES:
            dsk = S["ds"].get(f"{l}_{p}"); ft = S["ft"].get(f"{l}_{p}")
            if not dsk or not ft: continue
            for a, sh, an in ARMS_G:
                cm = np.array(ft["arms"][a][M]["per_seed"], float) - np.array(ft["arms"][sh][M]["per_seed"], float)
                bars = [("CMC-VAE", FT, cm, None)]
                for mode, name, col in VAR:
                    vt = dsk["variants"].get(mode)
                    c = vt["contrasts"].get(f"DiD_CMCgain_minus_DSgain:{a}|{M}") if vt else None
                    if c is None or len(c["seed_diffs"]) != len(cm): continue
                    bars.append((name, col, cm - np.array(c["seed_diffs"]), c))
                w = 0.8 / 4
                for i, (name, col, v, c) in enumerate(bars):
                    x0 = xi + (i - 1.5) * w
                    ax.bar(x0, np.mean(v), w, color=col, label=name if (xi == 0 and r == 0) else None); ax.scatter([x0] * len(v), v, s=2.5, c="k", zorder=3, lw=0)
                    top = max(np.max(v), 0) + (0.0025 if r == 0 else 0.004); ymax = max(ymax, top)
                    if c: ax.text(x0, top, stars(c.get("q_bh")), ha="center", va="bottom", fontsize=4.8, rotation=90 if stars(c.get("q_bh")) == "ns" else 0)
                    gdata[f"{l}|{p}|{a}|{M}|{name}"] = list(map(float, v))
                xt.append((xi, f"{ln}\n{an}")); xi += 1
        ax.axhline(0, ls=":", lw=0.8, c="#4a5568"); ax.set_xticks([x for x, _ in xt]); ax.set_xticklabels([t for _, t in xt], fontsize=5.8)
        ax.set_ylim(top=ymax * 1.18 + 1e-3); ax.set_ylabel(ylab, fontsize=6.5)
        if r == 0: ax.set_title(f"Pretrained on {pn}")
h_, l_ = axes[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 1.035), ncol=4, fontsize=6); lab(axes[0, 0], "g", x=-0.15, y=1.12)
save(fig, "fig5g_rev", gdata)

# ---------------- h: C7 control -- SCM vs RRR restricted to span(D) vs ridge ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.4), gridspec_kw=dict(wspace=0.25)); hdata = {}
PR = [("SCM:real", "SCM fine-tuned", FT), ("SCM:infonce", "SCM + InfoNCE", NCE), ("rrr_spanD_r192", "RRR, span(D), rank 192", TEAL),
      ("rrr_spanD_r8", "RRR, span(D), rank 8", "#81e6d9"), ("ridge_footprint", "Ridge, decoder loadings", BASE), ("ridge_ntcorr", "Ridge, co-expression", BASE2),
      ("knn10_ntcorr", "kNN-10, co-expression", "#fbd38d")]
for ax, (M, ylab, ylim) in zip(axes, [("rr_pe_cent", "Held-out centred pearDE", (-0.01, 0.27)), ("rr_retrieval", "Held-out gene retrieval", (0.45, 0.7))]):
    xt = []; xi = 0
    for l, ln in LINES:
        for p, pn in PRES:
            nb = S["nb"].get(f"{l}_{p}")
            if not nb: continue
            w = 0.8 / len(PR)
            for i, (k, name, col) in enumerate(PR):
                v = [x for x in nb["table"][k][M]["per_seed"] if x is not None]
                if not v: continue
                x0 = xi + (i - (len(PR) - 1) / 2) * w
                ax.bar(x0, np.mean(v), w, color=col, label=name if xi == 0 else None); ax.scatter([x0] * len(v), v, s=2.5, c="k", zorder=3, lw=0)
                hdata[f"{l}|{p}|{k}|{M}"] = v
            xt.append((xi, f"{ln}\n{'orig.' if p == 'obs' else 'ctrl-only'}")); xi += 1
    ax.axhline(0 if M == "rr_pe_cent" else 0.5, ls=":", lw=0.8, c="#4a5568"); ax.set_xticks([x for x, _ in xt]); ax.set_xticklabels([t for _, t in xt], fontsize=6.5)
    ax.set_ylabel(ylab); ax.set_ylim(*ylim)
axes[0].legend(loc="upper left", fontsize=5.6, labelspacing=0.15, borderaxespad=0.1, ncol=2); axes[0].set_title("Same frozen decoder loadings: SCM vs reduced-rank vs ridge")
lab(axes[0], "h", x=-0.13)
save(fig, "fig5h_ridge_rrr_scm", hdata)

# ---------------- i: cross-context (no target-line perturbation data) ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.5), gridspec_kw=dict(wspace=0.25)); idata = {}
XA = [("obs", "Observational", OBS), ("scm_real", "SCM FT on source", FT), ("scm_shuffle_avg", "SCM FT, shuffled", SH),
      ("rrr_spanD_T", "RRR span(D), target D", TEAL), ("ridge_footprint_T", "Ridge, target D", BASE), ("ridge_coexpr_T", "Ridge, target co-expr.", BASE2),
      ("ridge_coexpr_S_to_T", "Ridge fit in source, applied to target co-expr.", "#fbd38d"), ("copy_source_response", "Copy source response", "#2f855a"),
      ("within_scm_real", "Within-target SCM FT (uses target data)", "#90cdf4")]
for ax, (reg, title) in zip(axes, [("R1_prev_perturbed", "Context transfer of previously perturbed genes"), ("R2_never_perturbed", "Genes never perturbed in either line")]):
    xt = []; xi = 0
    for l, ln in LINES:
        X = S["xctx2"].get(f"{l}_obs", {}).get(reg)
        if not X: continue
        arms_ = [a for a in XA if a[0] in X["table"]]; w = 0.85 / len(arms_)
        for i, (k, name, col) in enumerate(arms_):
            m_, s_ = X["table"][k]["rr_retrieval"]["mean"], X["table"][k]["rr_retrieval"]["sd"]
            x0 = xi + (i - (len(arms_) - 1) / 2) * w
            ax.bar(x0, m_, w, color=col, yerr=s_, error_kw=dict(lw=0.5, capsize=1, ecolor="#333"), label=name if xi == 0 else None)
            idata[f"{l}|{reg}|{k}"] = dict(mean=m_, sd=s_)
        src = {"jurkat": "K562", "k562": "Jurkat"}[l]
        xt.append((xi, f"{src} → {ln}\n(n = {X['n_eval']})")); xi += 1
    ax.axhline(0.5, ls=":", lw=0.8, c="#4a5568"); ax.set_xticks([x for x, _ in xt]); ax.set_xticklabels([t for _, t in xt], fontsize=6.5)
    ax.set_ylim(0.45, 1.0 if reg.startswith("R1") else 0.76); ax.set_ylabel("Held-out gene retrieval (target line)"); ax.set_title(title, fontsize=7.5)
    ax.legend(loc="upper left", fontsize=5.3, labelspacing=0.15, borderaxespad=0.1, ncol=2)
lab(axes[0], "i", x=-0.13)
save(fig, "fig5i_cross_context", idata)
