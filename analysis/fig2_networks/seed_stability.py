#!/usr/bin/env python
"""Seed-stability of the LEARNED STRUCTURE for CMC-VAE (headline sp_dec_gl1000).

tab:seeds already reports scalar-metric mean+-SD across seeds 42/1/2. What was
missing is EDGE-LEVEL reproducibility: is the recovered graph (not just its
summary scores) stable across training seeds? This script reads the on-disk CSV
outputs of the three seed runs and quantifies, over the 3 seed pairs
(42v1, 42v2, 1v2):

  1. Projected GRN G_rr (gene x gene) off-diagonal edge-weight reproducibility:
     Pearson + Spearman of the signed weights, and top-1%% |edge| Jaccard.
  2. Cognate/translation map G_rp (gene x protein) reproducibility: Pearson +
     Spearman of the weights, plus rank stability of the cognate edges (do the
     same cognate lineage markers recur as top edges each seed?).
  3. Metric distribution: the same scalars as tab:seeds, per seed, as a strip.

All three runs share an identical gene/protein vocabulary (same training data);
we still align on the intersection defensively and report the overlap size.

Outputs
  results/seed_stability.json                 (all correlations / Jaccards + n_shared)
  paper/figures/_new_seed_stability.pdf/.png  (A: G_rr edge scatter+summary,
                                               B: G_rp cognate rank stability,
                                               C: metric distribution)

Rerunnable, CSV-only, no GPU.  Usage:  python analysis/fig2_networks/seed_stability.py
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, re
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = _CMC_ROOT
os.chdir(ROOT)
sys.path.insert(0, "analysis/figures/original")
from style import plt, C, panel_label, save  # noqa: E402

RUNS = {
    "42": "results/sweep_sparse/sp_dec_gl1000",
    "1":  "results/sweep_sparse/sp_dec_gl1000_s1",
    "2":  "results/sweep_sparse/sp_dec_gl1000_s2",
}
PAIRS = [("42", "1"), ("42", "2"), ("1", "2")]
SEED_ORDER = ["42", "1", "2"]
TOPK_FRAC = 0.01  # top 1% strongest |edges| for Jaccard


# ----------------------------------------------------------------------------- helpers
def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    # Spearman == Pearson on ranks; rankdata is fast and memory-light on ~7M elts.
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def topk_jaccard(a_abs, b_abs, frac):
    """Jaccard overlap of the top-`frac` strongest-|weight| edge sets."""
    k = max(1, int(round(frac * a_abs.size)))
    ia = np.argpartition(a_abs, -k)[-k:]
    ib = np.argpartition(b_abs, -k)[-k:]
    sa, sb = set(ia.tolist()), set(ib.tolist())
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union), k


def mean_range(vals):
    v = np.asarray(vals, float)
    return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                sd=float(v.std(ddof=1)) if v.size > 1 else 0.0)


def align_frame(df, genes, cols):
    """Reindex a DataFrame onto the shared row/col vocabulary."""
    return df.reindex(index=genes, columns=cols)


# ----------------------------------------------------------------------------- load G_rr
print("[load] projected GRN G_rr (gene x gene) for 3 seeds ...", flush=True)
grr = {}
grr_index, grr_cols = None, None
for s, d in RUNS.items():
    df = pd.read_csv(f"{d}/GRN_inferred.csv", index_col=0)
    grr[s] = df
    if grr_index is None:
        grr_index, grr_cols = df.index, df.columns
    print(f"  seed {s}: {df.shape}", flush=True)

# shared vocabulary (intersection preserving seed-42 order)
shared_rows = [g for g in grr_index if all(g in grr[s].index for s in RUNS)]
shared_cols = [g for g in grr_cols if all(g in grr[s].columns for s in RUNS)]
n_genes = len(shared_rows)
assert shared_rows == shared_cols, "G_rr is expected to be square gene x gene"
print(f"[align] G_rr shared genes: {n_genes} "
      f"(seed sizes: {[grr[s].shape[0] for s in RUNS]})", flush=True)

# numpy off-diagonal edge vectors on the shared vocabulary
offdiag = ~np.eye(n_genes, dtype=bool)
grr_vec, grr_absvec = {}, {}
for s in RUNS:
    M = align_frame(grr[s], shared_rows, shared_cols).values.astype(np.float64)
    grr_vec[s] = M[offdiag]
    grr_absvec[s] = np.abs(grr_vec[s])
del grr  # free the DataFrames

grr_pair = {}
for a, b in PAIRS:
    r = pearson(grr_vec[a], grr_vec[b])
    rho = spearman(grr_vec[a], grr_vec[b])
    jac, k = topk_jaccard(grr_absvec[a], grr_absvec[b], TOPK_FRAC)
    grr_pair[f"{a}v{b}"] = dict(pearson=r, spearman=rho, top1pct_jaccard=jac)
    print(f"  G_rr {a}v{b}: pearson={r:.3f} spearman={rho:.3f} "
          f"top{TOPK_FRAC*100:.0f}%_jaccard={jac:.3f} (k={k})", flush=True)

grr_summary = dict(
    n_offdiag_edges=int(grr_vec[SEED_ORDER[0]].size),
    topk_frac=TOPK_FRAC, topk_edges=k,
    pearson=mean_range([grr_pair[f"{a}v{b}"]["pearson"] for a, b in PAIRS]),
    spearman=mean_range([grr_pair[f"{a}v{b}"]["spearman"] for a, b in PAIRS]),
    top1pct_jaccard=mean_range([grr_pair[f"{a}v{b}"]["top1pct_jaccard"] for a, b in PAIRS]),
    per_pair=grr_pair,
)

# ----------------------------------------------------------------------------- load G_rp
print("[load] translation map G_rp (gene x protein) for 3 seeds ...", flush=True)
grp = {s: pd.read_csv(f"{RUNS[s]}/Transcription_RNA_to_Prot.csv", index_col=0) for s in RUNS}
gp_rows = [g for g in grp["42"].index if all(g in grp[s].index for s in RUNS)]
gp_cols = [p for p in grp["42"].columns if all(p in grp[s].columns for s in RUNS)]
n_gp_genes, n_prot = len(gp_rows), len(gp_cols)
print(f"[align] G_rp shared: {n_gp_genes} genes x {n_prot} proteins", flush=True)

grp_np = {s: align_frame(grp[s], gp_rows, gp_cols).values.astype(np.float64) for s in RUNS}
grp_vec = {s: grp_np[s].ravel() for s in RUNS}

grp_pair = {}
for a, b in PAIRS:
    r = pearson(grp_vec[a], grp_vec[b])
    rho = spearman(grp_vec[a], grp_vec[b])
    grp_pair[f"{a}v{b}"] = dict(pearson=r, spearman=rho)
    print(f"  G_rp {a}v{b}: pearson={r:.3f} spearman={rho:.3f}", flush=True)

# ---- cognate pairs (same rule as analysis/fig2_networks/cognate_nulls_csv.py, operating on the CSV) ----
def _norm(x):
    return str(x).upper().replace("-", "").replace("_", "").replace(".", "")

gnorm = [_norm(g) for g in gp_rows]
pnorm = [_norm(p) for p in gp_cols]
gup = {g: i for i, g in enumerate(gnorm)}
cognate = []  # (gene_idx, prot_idx, gene_name, prot_name)
for j, pr in enumerate(pnorm):
    for cand in [pr, pr.replace("CD", ""), pr.split("/")[0]]:
        if cand in gup:
            gi = gup[cand]
            cognate.append((gi, j, gp_rows[gi], gp_cols[j]))
            break
n_cog = len(cognate)
print(f"[cognate] {n_cog} cognate (protein<-gene) pairs on shared vocab", flush=True)

# |weight| of each cognate edge per seed, plus its fractional rank within its
# own protein column (0 == the strongest gene for that protein).
cog_w = {s: np.array([abs(grp_np[s][gi, j]) for gi, j, _, _ in cognate]) for s in RUNS}
cog_rankfrac = {}
for s in RUNS:
    rf = []
    for gi, j, _, _ in cognate:
        col = np.abs(grp_np[s][:, j])
        rf.append(float((col > col[gi]).sum()) / len(col))
    cog_rankfrac[s] = np.array(rf)

# rank stability of the cognate edges across seeds (do the same markers stay on top?)
cog_pair = {}
for a, b in PAIRS:
    rho_w = spearman(cog_w[a], cog_w[b])            # magnitude ordering of cognate edges
    r_w = pearson(cog_w[a], cog_w[b])
    # overlap of the top-5 / top-10 strongest cognate edges
    def topset(w, kk):
        return set(np.argsort(-w)[:kk].tolist())
    j5 = len(topset(cog_w[a], 5) & topset(cog_w[b], 5)) / len(topset(cog_w[a], 5) | topset(cog_w[b], 5))
    j10 = len(topset(cog_w[a], 10) & topset(cog_w[b], 10)) / len(topset(cog_w[a], 10) | topset(cog_w[b], 10))
    cog_pair[f"{a}v{b}"] = dict(spearman_weight=rho_w, pearson_weight=r_w,
                                top5_jaccard=float(j5), top10_jaccard=float(j10))
    print(f"  cognate {a}v{b}: spearman|w|={rho_w:.3f} top5_jac={j5:.3f} top10_jac={j10:.3f}",
          flush=True)

# median fractional rank of cognate edges per seed (how "top" the cognate is in its column)
cog_median_rankfrac = {s: float(np.median(cog_rankfrac[s])) for s in RUNS}
cog_frac_in_top10pct = {s: float(np.mean(cog_rankfrac[s] < 0.1)) for s in RUNS}

# per-marker table: mean |weight| across seeds + per-seed values (for the figure + JSON)
cog_names = [pn for _, _, _, pn in cognate]
cog_mean_w = np.mean([cog_w[s] for s in SEED_ORDER], axis=0)
order = np.argsort(-cog_mean_w)  # strongest cognate first
cog_table = []
for oi in order:
    cog_table.append(dict(
        protein=cog_names[oi], gene=cognate[oi][2],
        weight={s: float(cog_w[s][oi]) for s in SEED_ORDER},
        rankfrac={s: float(cog_rankfrac[s][oi]) for s in SEED_ORDER},
        mean_weight=float(cog_mean_w[oi])))

grp_summary = dict(
    n_shared_genes=n_gp_genes, n_shared_proteins=n_prot, n_cognate_pairs=n_cog,
    pearson=mean_range([grp_pair[f"{a}v{b}"]["pearson"] for a, b in PAIRS]),
    spearman=mean_range([grp_pair[f"{a}v{b}"]["spearman"] for a, b in PAIRS]),
    per_pair=grp_pair,
    cognate=dict(
        spearman_weight=mean_range([cog_pair[f"{a}v{b}"]["spearman_weight"] for a, b in PAIRS]),
        top5_jaccard=mean_range([cog_pair[f"{a}v{b}"]["top5_jaccard"] for a, b in PAIRS]),
        top10_jaccard=mean_range([cog_pair[f"{a}v{b}"]["top10_jaccard"] for a, b in PAIRS]),
        median_rankfrac_per_seed=cog_median_rankfrac,
        frac_in_top10pct_per_seed=cog_frac_in_top10pct,
        per_pair=cog_pair,
        table=cog_table,
    ),
)

# ----------------------------------------------------------------------------- metrics
print("[metrics] pulling per-seed sweep_metrics.json + cognate fold ...", flush=True)
# label -> (json path within sweep_metrics, higher-is-better?), matched to tab:seeds
MET = [
    ("co-expr AUROC",  lambda m: m["recovery"]["coexpr"]),
    ("STRING AUROC",   lambda m: m["recovery"]["STRING"]),
    ("DoRothEA AUROC", lambda m: m["recovery"]["DoRothEA"]),
    ("CollecTRI AUROC", lambda m: m["recovery"]["CollecTRI"]),
    ("RNA->RNA AUROC",  lambda m: m["perturb_mean"]["rr_auroc"]),
    ("RNA->prot AUROC", lambda m: m["perturb_mean"]["rp_auroc"]),
    ("RNA->RNA pearDE", lambda m: m["perturb_mean"]["rr_pearDE"]),
    ("RNA->prot pearDE", lambda m: m["perturb_mean"]["rp_pearDE"]),
]
metrics_per_seed = {}
for s in RUNS:
    m = json.load(open(f"{RUNS[s]}/sweep_metrics.json"))
    metrics_per_seed[s] = {name: float(fn(m)) for name, fn in MET}

# cognate translation fold per seed, computed from the CSV exactly as
# analysis/fig2_networks/cognate_nulls_csv.py does (median|matched|/median|non-cognate in cognate cols|),
# so all three seeds get a value (only seed 42 has the cached json on disk).
cbp = {}
for gi, j, _, _ in cognate:
    cbp.setdefault(j, set()).add(gi)
def _fold_of(M):
    matched = np.array([abs(M[gi, j]) for gi, j, _, _ in cognate])
    bg = np.concatenate([np.delete(np.abs(M[:, j]), list(cbp[j])) for gi, j, _, _ in cognate])
    return float(np.median(matched) / (np.median(bg) + 1e-12))
fold_per_seed = {s: round(_fold_of(grp_np[s]), 2) for s in RUNS}
print(f"[cognate] translation fold per seed: {fold_per_seed}", flush=True)

metrics_summary = {}
for name, _ in MET:
    vals = [metrics_per_seed[s][name] for s in SEED_ORDER]
    metrics_summary[name] = dict(per_seed={s: metrics_per_seed[s][name] for s in SEED_ORDER},
                                 **mean_range(vals))
folds = [fold_per_seed[s] for s in SEED_ORDER if fold_per_seed[s] is not None]
metrics_summary["cognate fold"] = dict(per_seed=fold_per_seed, **mean_range(folds)) if folds else {}

# ----------------------------------------------------------------------------- write JSON
out = dict(
    runs=RUNS, pairs=[f"{a}v{b}" for a, b in PAIRS], seeds=SEED_ORDER,
    n_shared=dict(grr_genes=n_genes, grp_genes=n_gp_genes, proteins=n_prot,
                  vocab_identical_across_seeds=True),
    G_rr=grr_summary, G_rp=grp_summary, metrics=metrics_summary,
)
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/seed_stability.json", "w"), indent=2)
print("WROTE results/seed_stability.json", flush=True)

# ============================================================================= FIGURE
SEED_C = {"42": C["model"], "1": C["accent"], "2": C["genie3"]}
fig = plt.figure(figsize=(7.2, 2.55))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.15], wspace=0.55)

# ---- Panel A: G_rr off-diagonal edge scatter (representative pair) + summary ----
axA = fig.add_subplot(gs[0, 0])
a, b = "42", "1"
xa, xb = grr_vec[a], grr_vec[b]
# density hist2d on a symmetric window covering the bulk of the edge weights
lim = float(np.percentile(np.abs(np.concatenate([xa, xb])), 99.9))
h = axA.hist2d(xa, xb, bins=140, range=[[-lim, lim], [-lim, lim]],
               cmap="Blues", norm=__import__("matplotlib").colors.LogNorm())
axA.plot([-lim, lim], [-lim, lim], "-", color=C["muted"], lw=0.7, zorder=3)
axA.set_xlim(-lim, lim); axA.set_ylim(-lim, lim)
axA.set_xlabel("$G_{rr}$ edge weight (seed 42)")
axA.set_ylabel("$G_{rr}$ edge weight (seed 1)")
axA.set_title("Projected GRN edges", fontweight="bold")
pr_m = grr_summary["pearson"]; sp_m = grr_summary["spearman"]; jc_m = grr_summary["top1pct_jaccard"]
axA.text(0.04, 0.965,
         f"3-pair mean\nPearson {pr_m['mean']:.2f} [{pr_m['min']:.2f}-{pr_m['max']:.2f}]\n"
         f"Spearman {sp_m['mean']:.2f} [{sp_m['min']:.2f}-{sp_m['max']:.2f}]\n"
         f"top-1% Jaccard {jc_m['mean']:.2f}",
         transform=axA.transAxes, va="top", ha="left", fontsize=6.2, color=C["ink"],
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["shuf"], lw=0.5, alpha=0.9))
panel_label(axA, "A")

# ---- Panel B: cognate translation-edge rank stability across seeds ----
axB = fig.add_subplot(gs[0, 1])
TOPN = 14
tab = cog_table[:TOPN]
ypos = np.arange(len(tab))[::-1]  # strongest at top
for s in SEED_ORDER:
    axB.scatter([t["weight"][s] for t in tab], ypos, s=16, color=SEED_C[s],
                edgecolor="white", linewidth=0.3, label=f"seed {s}", zorder=3)
for yi, t in zip(ypos, tab):
    ws = [t["weight"][s] for s in SEED_ORDER]
    axB.plot([min(ws), max(ws)], [yi, yi], "-", color=C["shuf"], lw=0.8, zorder=1)
axB.set_yticks(ypos); axB.set_yticklabels([t["protein"] for t in tab], fontsize=6.2)
axB.set_ylim(-0.7, len(tab) - 0.3)
axB.set_xlabel("$|G_{rp}|$ cognate edge weight")
axB.set_title("Cognate RNA$\\rightarrow$protein edges", fontweight="bold")
csp = grp_summary["cognate"]["spearman_weight"]; ct = grp_summary["cognate"]["top5_jaccard"]
axB.text(0.03, 0.97,
         f"cognate Spearman {csp['mean']:.2f}\n[{csp['min']:.2f}-{csp['max']:.2f}]\n"
         f"top-5 Jaccard {ct['mean']:.2f}",
         transform=axB.transAxes, va="top", ha="left", fontsize=6.2, color=C["ink"],
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["shuf"], lw=0.5, alpha=0.9))
axB.legend(loc="lower right", fontsize=6, handletextpad=0.2, borderaxespad=0.4)
panel_label(axB, "B")

# ---- Panel C: metric distribution across seeds (same scalars as tab:seeds) ----
axC = fig.add_subplot(gs[0, 2])
names = [n for n, _ in MET]
ypos = np.arange(len(names))[::-1]
for s in SEED_ORDER:
    axC.scatter([metrics_per_seed[s][n] for n in names], ypos, s=16, color=SEED_C[s],
                edgecolor="white", linewidth=0.3, zorder=3, label=f"seed {s}")
for yi, n in zip(ypos, names):
    vals = [metrics_per_seed[s][n] for s in SEED_ORDER]
    axC.plot([min(vals), max(vals)], [yi, yi], "-", color=C["shuf"], lw=0.8, zorder=1)
axC.axvline(0.5, color=C["muted"], ls=":", lw=0.7, zorder=0)
axC.set_yticks(ypos); axC.set_yticklabels(names, fontsize=6.2)
axC.yaxis.tick_right()  # labels on the right edge -> clean gap next to panel B
axC.set_ylim(-0.7, len(names) - 0.3)
axC.set_xlim(-0.12, 1.02)
axC.set_xlabel("metric value")
axC.set_title("Scalar metrics per seed", fontweight="bold")
ff = metrics_summary.get("cognate fold", {})
if ff:
    axC.text(0.02, 0.02, f"cognate fold {ff['mean']:.1f}$\\times$ [{ff['min']:.1f}-{ff['max']:.1f}]",
             transform=axC.transAxes, va="bottom", ha="left", fontsize=6.2, color=C["ink"],
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["shuf"], lw=0.5, alpha=0.9))
axC.legend(loc="upper left", fontsize=6, handletextpad=0.2, borderaxespad=0.4)
panel_label(axC, "C", x=-0.05)

save(fig, "paper/figures/AppendixFigS10.pdf")
print("done.", flush=True)
