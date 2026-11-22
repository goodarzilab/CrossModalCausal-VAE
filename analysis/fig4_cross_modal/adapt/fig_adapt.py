"""adapt workstream figures (PDF+PNG) + source data. Reads nov_adapt/summary.json (aggregate.py).
FigAD_A: endpoint (a) control-cell imputation per screen/variant vs ridge.
FigAD_B: perturbation endpoints: translation, CITE transfer (protein), RNA held-out, GWCD4i->SCITO M1, GWCD4i LC retrieval."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
O = f"{_CMC_SCRATCH}/nov_adapt"; F = f"{_CMC_RESULTS}/paper/figures/adapt"
os.makedirs(F, exist_ok=True)
S = json.load(open(f"{O}/summary.json"))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
VAR = [("none", "unadapted"), ("iii", "(iii) affine"), ("ii", "(ii) enc/dec"), ("i", "(i) enc/dec+A"), ("i_xm1", "(i) λxm=1")]
VC = {"none": "#9a9893", "iii": C[3], "ii": C[2], "i": C[0], "i_xm1": C[4]}
SRC = {}
def save(fig, name):
    fig.savefig(f"{F}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{F}/{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- A: imputation
imp = S["imputation"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
SRC["A_imputation"] = imp
for ax, dom, title in zip(axs, ["iccite", "arce", "frangieh"], ["Chen (78 prot.)", "Arce (84)", "Frangieh (7)"]):
    rows = {r["variant"]: r for r in imp if r["base"] == "sp" and r["domain"] == dom}
    x = 0
    for v, lab in VAR:
        if v not in rows: continue
        r = rows[v]; ax.bar(x, r["rna_only"], 0.7, color=VC[v], edgecolor="#fcfcfb", linewidth=1)
        ax.plot([x, x], r["rna_only_range"], color=INK, lw=1)
        x += 1
    xm = [r for r in imp if r["base"] == "xm" and r["domain"] == dom and r["variant"] == "i"]
    if xm: ax.bar(x, xm[0]["rna_only"], 0.7, color=C[6], edgecolor="#fcfcfb"); ax.plot([x, x], xm[0]["rna_only_range"], color=INK, lw=1); x += 1
    rg = rows["none"]["ridge"]; ax.axhline(rg, color=INK, ls="--", lw=1); ax.text(x - 0.5, rg, " ctrl-cell ridge", va="bottom", ha="right", color=MUTED, fontsize=7)
    ax.set_xticks(range(x)); ax.set_xticklabels([l for v, l in VAR if v in rows] + (["xm (i)"] if xm else []), rotation=45, ha="right")
    ax.set_title(title, fontsize=8, color=INK); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel("protein-from-RNA r\n(held-out control cells, median over proteins)")
fig.suptitle("a  Control-cell adaptation restores within-screen RNA→protein imputation (3 seeds; bar = mean, line = range)", fontsize=8, x=0.02, ha="left", y=1.06)
save(fig, "FigAD_A_imputation")

# ---------------------------------------------------------------- B: perturbation endpoints
fig, axs = plt.subplots(1, 5, figsize=(12, 2.8))
# (b1) translation retrieval / centred
ts = S.get("translate_summary", {})
ax = axs[0]; names = []; vals = []; cols = []
for lab, nm, c in [("sp_none", "unadapted", VC["none"]), ("sp_iii", "(iii)", VC["iii"]), ("sp_ii", "(ii)", VC["ii"]), ("sp_i", "(i)", VC["i"]), ("sp_i_xm1", "(i) λ1", VC["i_xm1"]), ("xm_i", "xm (i)", C[6])]:
    if lab in ts: names.append(nm); vals.append(ts[lab]["CMC-enc"]["overall"]["retrieval"]); cols.append(c)
for mth, nm in [("corr", "corpus corr"), ("ctrlridge90", "ctrl ridge")]:
    if "sp_none" in ts: names.append(nm); vals.append(ts["sp_none"][mth]["overall"]["retrieval"]); cols.append("#3d3c39")
ax.bar(range(len(vals)), vals, color=cols, edgecolor="#fcfcfb"); ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=50, ha="right"); ax.set_ylim(0.4, 1.0)
ax.set_ylabel("held-out retrieval"); ax.set_title("b  measured RNA → protein\n(translation, 204 KOs)", fontsize=8)
SRC["B_translation_retrieval"] = dict(zip(names, vals))
# (b2) transfer protein retrieval (ftR vs ftR_shuf vs baselines)
tf = S.get("transfer_summary", {}); ax = axs[1]; names, vals, cols = [], [], []
for key, arm, nm, c in [("sp_none|clamp_zero", "ftR", "unadapt. ftR", VC["none"]), ("sp_i|clamp_zero", "ftR_shuf", "(i) ftR shuf", "#c3c2b7"),
                        ("sp_i|clamp_zero", "ftR", "(i) ftR", VC["i"]), ("sp_ii|clamp_zero", "ftR", "(ii) ftR", VC["ii"]), ("sp_iii|clamp_zero", "ftR", "(iii) ftR", VC["iii"]),
                        ("sp_none|clamp_zero", "ridgeR_ctrlcx+ctrl", "ridge(ctrl)", "#3d3c39"), ("sp_none|clamp_zero", "measRNA+ctrl", "meas RNA+ctrl", "#7d7c78")]:
    v = tf.get(key, {}).get(arm, {}).get("rp_retrieval")
    if v: names.append(nm); vals.append(v["mean"]); cols.append(c)
ax.bar(range(len(vals)), vals, color=cols, edgecolor="#fcfcfb"); ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=50, ha="right"); ax.set_ylim(0.3, 1.0)
ax.set_title("c  RNA-label FT → held-out\nprotein (clamp)", fontsize=8); SRC["C_transfer_rp_retrieval"] = dict(zip(names, vals))
# (b3) RNA held-out centred pearDE
ax = axs[2]; names, vals, cols = [], [], []
for key, arm, nm, c in [("sp_none|clamp_zero", "obs", "unadapt. obs", "#c3c2b7"), ("sp_none|clamp_zero", "ftR", "unadapt. ftR", VC["none"]),
                        ("sp_i|clamp_zero", "obs", "(i) obs", "#9cc0ec"), ("sp_i|clamp_zero", "ftR", "(i) ftR", VC["i"]),
                        ("sp_none|clamp_zero", "ridgeP_ctrlcx", "ridge(ctrl cx)", "#3d3c39"), ("sp_none|clamp_zero", "ridgeP_cx", "ridge(corpus cx)", "#7d7c78")]:
    v = tf.get(key, {}).get(arm, {}).get("rr_retrieval")
    if v: names.append(nm); vals.append(v["mean"]); cols.append(c)
ax.bar(range(len(vals)), vals, color=cols, edgecolor="#fcfcfb"); ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=50, ha="right"); ax.set_ylim(0.3, 0.8)
ax.set_title("d  held-out RNA retrieval\n(CITE-seq, clamp)", fontsize=8); SRC["D_rna_retrieval"] = dict(zip(names, vals))
# (b4) GWCD4i->SCITO M1
gs = S.get("gwscito_summary", {}); ax = axs[3]; names, vals, cols, cis = [], [], [], []
for lab, nm, c in [("sp_none", "unadapted", VC["none"]), ("sp_iii", "(iii)", VC["iii"]), ("sp_ii", "(ii)", VC["ii"]), ("sp_i", "(i)", VC["i"]), ("xm_i", "xm (i)", C[6])]:
    r = gs.get(f"{lab}|Stim48hr|ftR")
    if r: names.append(nm); vals.append(r["pair_auroc"]); cols.append(c); cis.append(r.get("pair_auroc_ci") or [r["pair_auroc"]] * 2)
for i, (v, ci) in enumerate(zip(vals, cis)):
    ax.bar(i, v, color=cols[i], edgecolor="#fcfcfb"); ax.plot([i, i], ci, color=INK, lw=1)
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":"); ax.axhline(0.643, color=INK, lw=1, ls="--"); ax.text(len(vals) - 0.5, 0.645, "protein-label ridge", ha="right", va="bottom", fontsize=7, color=MUTED)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=50, ha="right"); ax.set_ylim(0.4, 0.7)
ax.set_ylabel("pair AUROC (M1)"); ax.set_title("e  GWCD4i → SCITO, FT\n(pre-registered, Stim48hr)", fontsize=8); SRC["E_gwscito_M1"] = dict(zip(names, vals))
# (b5) LC retrieval
lc = S.get("lc_summary", {}); ax = axs[4]; sizes = ["5", "10", "25", "50", "100", "200", "all"]; xs = [5, 10, 25, 50, 100, 200, 335]
lines = [("sp_none", "base_shift", "zero-shot graph, unadapted", VC["none"], "-"), ("sp_i", "base_shift", "zero-shot graph, (i)", VC["i"], "-"),
         ("sp_i", "F_sym_shift_short_pm", "SCM FT, (i)", C[6], "-"), ("sp_none", "ridge_ntcorr", "ridge (NTC co-expr.)", "#3d3c39", "--"),
         ("sp_none", "linear_ae_ctrlpca", "linear (NTC PCA)", C[1], "--")]
SRC["F_lc"] = {}
for lab, meth, nm, c, ls in lines:
    y = [lc.get(f"{lab}|{s}", {}).get(meth, {}).get("rr_retrieval", np.nan) for s in sizes]
    if np.isfinite(y).any():
        ax.plot(xs, y, ls, color=c, lw=2, marker="o", ms=4, label=nm); SRC["F_lc"][nm] = dict(zip(sizes, y))
ax.set_xscale("log"); ax.axhline(0.5, color=MUTED, lw=0.8, ls=":"); ax.set_xlabel("training perturbations"); ax.set_ylabel("held-out retrieval")
ax.set_title("f  GWCD4i Stim8hr learning curve", fontsize=8); ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=1)
for a in axs: a.yaxis.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
fig.tight_layout(); save(fig, "FigAD_B_perturbation")
json.dump(SRC, open(f"{F}/source_data_adapt.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else o)
print("figures written")
