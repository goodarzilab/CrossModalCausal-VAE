"""Feedback-block (protein->RNA, G_pr) BIOLOGY analysis for CMC-VAE  --  honesty-first.

The paper makes NO magnitude-based directionality claim (the translation/feedback magnitude ratio
is regularization-dependent -- here decoder-sparsity is 1000 on RNA vs 18 on protein). So this script:

  (1) TRANSPOSE DIAGNOSTIC  (run first, GATES everything).  How much of the feedback block
      G_pr = D_p W_pr D_r^T is just the transpose of the translation block G_rp = D_r W_rp D_p^T?
      Pearson/Spearman of vec(G_pr) vs vec(G_rp^T); per-receptor-row correlation; variance
      explained by the least-squares transpose; residual R = G_pr - beta*G_rp^T.
      If G_pr ~ G_rp^T, any "receptor->target" signal on raw G_pr is INHERITED from translation,
      not independent feedback biology.  We therefore gate all biology on the residual R.

  (2) RECEPTOR->TARGET enrichment  (within-row, MAGNITUDE-INVARIANT rankings).  For panel receptors
      with clean downstream sets, rank genes by |G_pr[receptor,:]| and test target-set membership
      (self cognate gene excluded) via within-row AUROC + MWU + within-row label-permutation null.
      Run on BOTH raw |G_pr| AND the transpose-residual |R|.  Two controls:
        (i)  DECOY specificity  -- receptor x pathway AUROC confusion matrix (does a receptor rank
             ITS OWN target set above other receptors' target sets?).
        (ii) COEXPR baseline    -- raw |corr(protein r, gene g)| across cells (feasible subsample).

  (3) MAGNITUDE / SPARSITY ASYMMETRY  (honest negative).  Gini + mean|.| of both projected blocks
      in observable space (both [P,G], directly comparable) + the latent-symmetric p128 cross-check.
      Presented as regularization-dependent, NOT biological directionality.

Inputs are the SAVED projected CSVs (same objects fig2/fig_crossmodal read):
  <run>/Regulation_Prot_to_RNA.csv     G_pr  [protein x gene]   (feedback)
  <run>/Transcription_RNA_to_Prot.csv  G_rp  [gene x protein]   (translation)

Outputs: results/feedback_biology.json  and  paper/figures/_new_feedback_biology.pdf
Usage:   python analysis/fig2_networks/feedback_biology.py [results/<run>]   (default: results/sweep_sparse/sp_dec_gl1000)
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
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _CMC_ROOT
RES = os.path.join(REPO, "results")
CACHE = os.path.join(HERE, "_pathway_genesets_cache.json")
N_CELLS = int(os.environ.get("FB_N_CELLS", "50000"))   # cells for the coexpr baseline
B_PERM = int(os.environ.get("FB_N_PERM", "5000"))      # within-row label permutations

rd = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(RES, "sweep_sparse", "sp_dec_gl1000")
run = os.path.basename(rd)
cfg = json.load(open(f"{rd}/config.json"))

# antibody -> HGNC map (EXACT copy from analysis/fig2_networks/ppi_feedback_eval.py) + canon fallback
import re
def canon_prot(x):
    x = re.sub(r'^AB_', '', x); x = re.sub(r'^(human|mouse)', '', x, flags=re.I)
    x = re.sub(r'-\d+$', '', x); return x.split('_')[0].upper()
CURATED_AB = {
 'CD25':'IL2RA','CD3-1':'CD3E','CD122':'IL2RB','CD127':'IL7R','CD14':'CD14',
}  # only the receptors we self-exclude here need to be right; others use canon_prot

def gini(v):
    v = np.sort(np.abs(np.asarray(v, dtype=np.float64)).ravel())
    if v.sum() == 0: return 0.0
    n = len(v); cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)

# --------------------------------------------------------------------------- load projected blocks
gpr_df = pd.read_csv(f"{rd}/Regulation_Prot_to_RNA.csv", index_col=0)    # [P prot, G gene]  feedback
grp_df = pd.read_csv(f"{rd}/Transcription_RNA_to_Prot.csv", index_col=0)  # [G gene, P prot]  translation
prots = list(map(str, gpr_df.index)); genes = list(map(str, gpr_df.columns))
assert prots == list(map(str, grp_df.columns)) and genes == list(map(str, grp_df.index)), "CSV axis mismatch"
Gpr = gpr_df.values.astype(np.float64)          # [P, G]
GrpT = grp_df.values.T.astype(np.float64)       # [P, G]  transpose of translation
P, G = Gpr.shape
gidx = {g: i for i, g in enumerate(genes)}
out = {"run": run, "csv_source": True, "n_proteins": P, "n_genes": G,
       "prior_off": cfg.get("prior_w_final", 0) == 0,
       "dec_sparse": {"rna": cfg.get("lambda_dec_sparse_rna"), "prot": cfg.get("lambda_dec_sparse_prot"),
                      "mode": cfg.get("dec_sparse_mode")},
       "latent_dims": {"rna": cfg.get("latent_dim_rna"), "prot": cfg.get("latent_dim_prot")}}

# ========================================================================= (1) TRANSPOSE DIAGNOSTIC
a = Gpr.ravel(); b = GrpT.ravel()
pr = pearsonr(a, b); sr = spearmanr(a, b)
beta = float((a @ b) / (b @ b))                 # LS: G_pr ~= beta * G_rp^T (through origin)
R = Gpr - beta * GrpT
var_explained = float(1.0 - (R.ravel() @ R.ravel()) / (a @ a))   # frac of ||G_pr||^2 removed by beta*G_rp^T
row_r = []
for i in range(P):
    x, y = Gpr[i], GrpT[i]
    if x.std() > 0 and y.std() > 0:
        row_r.append(float(np.corrcoef(x, y)[0, 1]))
row_r = np.array(row_r)
transpose = {
    "pearson_vec": float(pr[0]), "pearson_p": float(pr[1]),
    "spearman_vec": float(sr.correlation), "spearman_p": float(sr.pvalue),
    "pearson_abs": float(pearsonr(np.abs(a), np.abs(b))[0]),
    "ls_beta": beta, "variance_explained_by_transpose": var_explained,
    "residual_frob_frac": float(np.linalg.norm(R) / np.linalg.norm(Gpr)),
    "per_receptor_row_r_median": float(np.median(row_r)),
    "per_receptor_row_r_mean": float(row_r.mean()),
    "per_receptor_row_r_min": float(row_r.min()), "per_receptor_row_r_max": float(row_r.max()),
    "frac_rows_r_gt_0.5": float((np.abs(row_r) > 0.5).mean()),
    "frac_rows_r_gt_0.3": float((np.abs(row_r) > 0.3).mean()),
    "verdict": ("feedback block is SUBSTANTIALLY the transpose of translation "
                f"(vec r={pr[0]:.2f}, ~{100*var_explained:.0f}% of |G_pr|^2 explained); "
                "biology on raw G_pr is largely INHERITED, so gate biology on residual R"),
}
out["transpose_diagnostic"] = transpose
print("=" * 78)
print(f"(1) TRANSPOSE DIAGNOSTIC  [{run}]")
print(f"    Pearson vec(G_pr) vs vec(G_rp^T) = {pr[0]:.3f}   Spearman = {sr.correlation:.3f}")
print(f"    LS beta = {beta:.3f}   variance explained by transpose = {var_explained:.3f}")
print(f"    per-receptor-row r: median {np.median(row_r):.3f}  ({(np.abs(row_r)>0.5).mean()*100:.0f}% of rows |r|>0.5)")
print(f"    -> {transpose['verdict']}")

# ========================================================================= (2) RECEPTOR->TARGET
cache = json.load(open(CACHE))
vocab = set(genes)
TARGET_SETS = {   # name -> genes in vocab
    "TNFA_NFKB":  sorted(set(cache["HALLMARK_TNFA_SIGNALING_VIA_NFKB"]) & vocab),
    "TCR":        sorted(set(cache["REACTOME_TCR_SIGNALING"]) & vocab),
    "IEG":        sorted({"NR4A1","NR4A2","NR4A3","EGR1","FOS","JUN"} & vocab),
    "JAK_STAT":   sorted(set(cache["KEGG_JAK_STAT_SIGNALING_PATHWAY"]) & vocab),
}
# receptor -> (cognate self gene to exclude, its OWN target-set name)
RECEPTORS = {
    "CD14":  ("CD14",  "TNFA_NFKB"),
    "CD3-1": ("CD3E",  "TCR"),        # also tested against IEG (see intended pairs)
    "CD25":  ("IL2RA", "JAK_STAT"),
    "CD122": ("IL2RB", "JAK_STAT"),
    "CD127": ("IL7R",  "JAK_STAT"),
}
# intended (receptor, target-set) pairs to report as the primary enrichment
INTENDED = [("CD14","TNFA_NFKB"), ("CD3-1","TCR"), ("CD3-1","IEG"),
            ("CD25","JAK_STAT"), ("CD122","JAK_STAT"), ("CD127","JAK_STAT")]

def within_row_enrichment(score_row, exclude_gene, target_genes, B=B_PERM, seed=0):
    """score_row: [G] non-negative scores over genes (aligned to `genes`). Rank-based within-row test."""
    keep = np.ones(G, dtype=bool)
    if exclude_gene in gidx: keep[gidx[exclude_gene]] = False
    s = score_row[keep]
    lab = np.array([(g in target_genes) for g, k in zip(genes, keep) if k], dtype=bool)
    npos = int(lab.sum()); n = len(lab); nneg = n - npos
    if npos < 3: return None
    ranks = rankdata(s)                                   # AUROC == U/(npos*nneg) with avg ranks
    obs = float((ranks[lab].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))
    mwu = mannwhitneyu(s[lab], s[~lab], alternative="greater")
    rng = np.random.default_rng(seed); null = np.empty(B)
    for k in range(B):
        sel = rng.choice(n, size=npos, replace=False)
        null[k] = (ranks[sel].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    perm_p = float((np.sum(null >= obs) + 1) / (B + 1))
    med_rank_frac = float(np.median(1.0 - (ranks[lab] - 1) / (n - 1)))   # 0=top, small=enriched top
    return dict(auroc=obs, mwu_p=float(mwu.pvalue), perm_null_p=perm_p, n_target_in_vocab=npos,
                median_target_rank_frac=med_rank_frac)

# ---- coexpr baseline: |corr(protein r, gene g)| across a cell subsample (feasible) ----
coexpr_abs = None
try:
    import scanpy as sc, scipy.sparse as ssp
    ar = sc.read_h5ad(cfg["rna_path"], backed="r"); ap = sc.read_h5ad(cfg["prot_path"], backed="r")
    assert list(map(str, ar.var_names)) == genes and list(map(str, ap.var_names)) == prots
    rng = np.random.RandomState(42)
    sel = np.sort(rng.choice(ar.n_obs, size=min(N_CELLS, ar.n_obs), replace=False))
    Xg = ar.X[sel]; Xg = Xg.toarray() if ssp.issparse(Xg) else np.asarray(Xg); Xg = Xg.astype(np.float64)
    Xp = ap.X[sel]; Xp = Xp.toarray() if ssp.issparse(Xp) else np.asarray(Xp); Xp = Xp.astype(np.float64)
    Zg = (Xg - Xg.mean(0)) / (Xg.std(0) + 1e-8)
    Zp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)
    Cpg = (Zp.T @ Zg) / (len(sel) - 1)            # [P, G] protein-gene correlation
    coexpr_abs = np.abs(Cpg)
    out["coexpr_baseline"] = {"n_cells": int(len(sel)), "computed": True}
    print(f"(2) coexpr baseline computed on {len(sel)} matched cells")
except Exception as e:
    out["coexpr_baseline"] = {"computed": False, "err": str(e)[:200]}
    print(f"(2) coexpr baseline FAILED: {str(e)[:120]}")

# ---- run enrichment on raw |G_pr|, residual |R|, and coexpr ----
absGpr = np.abs(Gpr); absR = np.abs(R)
pidx = {p: i for i, p in enumerate(prots)}
enrich = {}
for rec, tgt in INTENDED:
    cog = RECEPTORS[rec][0]; ri = pidx[rec]; tset = set(TARGET_SETS[tgt]); key = f"{rec}->{tgt}"
    e = {"cognate_excluded": cog,
         "raw_Gpr":  within_row_enrichment(absGpr[ri], cog, tset),
         "residual_R": within_row_enrichment(absR[ri], cog, tset)}
    if coexpr_abs is not None:
        e["coexpr"] = within_row_enrichment(coexpr_abs[ri], cog, tset)
    enrich[key] = e
out["receptor_enrichment"] = enrich

# ---- DECOY specificity: receptor x target-set AUROC confusion matrix (on residual R) ----
decoy_recs = ["CD14", "CD3-1", "CD25", "CD122", "CD127"]
decoy_sets = ["TNFA_NFKB", "TCR", "IEG", "JAK_STAT"]
def _auroc_only(score_row, exclude_gene, target_genes):
    r = within_row_enrichment(score_row, exclude_gene, target_genes, B=1)   # B=1 (skip null) for speed
    return None if r is None else r["auroc"]
decoy_R = {rec: {ts: _auroc_only(absR[pidx[rec]], RECEPTORS[rec][0], set(TARGET_SETS[ts]))
                 for ts in decoy_sets} for rec in decoy_recs}
decoy_coex = None
if coexpr_abs is not None:
    decoy_coex = {rec: {ts: _auroc_only(coexpr_abs[pidx[rec]], RECEPTORS[rec][0], set(TARGET_SETS[ts]))
                        for ts in decoy_sets} for rec in decoy_recs}
# self vs best-other specificity summary (on residual R)
own = {"CD14": "TNFA_NFKB", "CD3-1": "TCR", "CD25": "JAK_STAT", "CD122": "JAK_STAT", "CD127": "JAK_STAT"}
spec = {}
for rec in decoy_recs:
    row = {k: v for k, v in decoy_R[rec].items() if v is not None}
    o = own[rec]; others = [v for k, v in row.items() if k != o]
    spec[rec] = {"own_set": o, "own_auroc": row.get(o),
                 "best_other_auroc": (max(others) if others else None),
                 "own_is_argmax": (o == max(row, key=row.get)) if row else None}
out["decoy"] = {"confusion_residual_R": decoy_R, "confusion_coexpr": decoy_coex,
                "specificity_residual_R": spec, "receptors": decoy_recs, "sets": decoy_sets}

print("    receptor->target AUROC (residual R):")
for rec, tgt in INTENDED:
    e = enrich[f"{rec}->{tgt}"]
    rr = e["residual_R"]; rw = e["raw_Gpr"]; cx = e.get("coexpr")
    print(f"      {rec:6s}->{tgt:10s}  raw={rw['auroc']:.2f}  R={rr['auroc']:.2f} (perm p={rr['perm_null_p']:.3f})"
          + (f"  coexpr={cx['auroc']:.2f}" if cx else "") + f"   [n_tgt={rr['n_target_in_vocab']}]")
print("    decoy specificity (residual R): own-set is argmax?")
for rec in decoy_recs:
    s = spec[rec]
    print(f"      {rec:6s} own={s['own_set']:10s} own_auroc={s['own_auroc']:.2f} "
          f"best_other={s['best_other_auroc'] if s['best_other_auroc'] is None else round(s['best_other_auroc'],2)} "
          f"argmax_is_own={s['own_is_argmax']}")

# ========================================================================= (3) MAGNITUDE / SPARSITY
mag = {
    "gini_G_pr": gini(Gpr), "gini_G_rp": gini(GrpT),
    "mean_abs_G_pr": float(np.abs(Gpr).mean()), "mean_abs_G_rp": float(np.abs(GrpT).mean()),
    "Grp_over_Gpr": float(np.abs(GrpT).mean() / (np.abs(Gpr).mean() + 1e-12)),
    "note": ("both blocks projected to observable space [P,G] so magnitudes ARE comparable here; "
             "ratio is regularization-dependent (decoder group-lasso lambda: "
             f"RNA={cfg.get('lambda_dec_sparse_rna')} vs protein={cfg.get('lambda_dec_sparse_prot')}), "
             "NOT a biological directionality claim."),
}
# latent-symmetric (128/128) p128 cross-check for the magnitude confound
P128 = os.path.join(RES, "linfb_2c_fb10_b20_v4_p128")
try:
    if os.path.exists(f"{P128}/final_model.pt"):
        import torch, scanpy as sc
        sys.path.insert(0, os.path.join(REPO, "src")); from model import CrossModalCausalLDVAE
        c2 = json.load(open(f"{P128}/config.json"))
        ar2 = sc.read_h5ad(c2["rna_path"], backed="r"); ap2 = sc.read_h5ad(c2["prot_path"], backed="r")
        ng2, npr2 = ar2.n_vars, ap2.n_vars
        m2 = CrossModalCausalLDVAE(ng2, npr2, 1, latent_dim_rna=c2["latent_dim_rna"], latent_dim_prot=c2["latent_dim_prot"],
            recon_likelihood="mse", scm_nonlinear=c2.get("scm_nonlinear", False), phi=c2.get("phi", "tanh"))
        m2.load_state_dict(torch.load(f"{P128}/final_model.pt", map_location="cpu"), strict=False); m2.eval()
        with torch.no_grad():
            _, _, w_rp2, w_pr2 = m2.scm.get_block_adjacencies(tau=0.5)
            Dr2 = m2._dec_r_w().detach().float(); Dp2 = m2.dec_prot.weight.detach().float()
            g_rp2 = (Dr2 @ w_rp2.float() @ Dp2.T).cpu().numpy(); g_pr2 = (Dp2 @ w_pr2.float() @ Dr2.T).cpu().numpy()
        mag["unconfounded_p128"] = {"model": "linfb_2c_fb10_b20_v4_p128 (symmetric 128/128 latent, different reg)",
            "mean_abs_G_rp": float(np.abs(g_rp2).mean()), "mean_abs_G_pr": float(np.abs(g_pr2).mean()),
            "Grp_over_Gpr": float(np.abs(g_rp2).mean() / (np.abs(g_pr2).mean() + 1e-12)),
            "gini_G_pr": gini(g_pr2), "gini_G_rp": gini(g_rp2)}
    else:
        mag["unconfounded_p128"] = "NOT AVAILABLE"
except Exception as e:
    mag["unconfounded_p128"] = f"ERR: {str(e)[:150]}"
out["magnitude_asymmetry"] = mag
print("(3) MAGNITUDE / GINI")
print(f"    Gini  G_pr={mag['gini_G_pr']:.3f}  G_rp={mag['gini_G_rp']:.3f}   (both sparse)")
print(f"    mean|G_rp|/mean|G_pr| = {mag['Grp_over_Gpr']:.3f}  (regularization-dependent; RNA-dec lambda "
      f"{cfg.get('lambda_dec_sparse_rna')} vs prot {cfg.get('lambda_dec_sparse_prot')})")
if isinstance(mag["unconfounded_p128"], dict):
    print(f"    p128 symmetric-latent cross-check: Grp/Gpr = {mag['unconfounded_p128']['Grp_over_Gpr']:.3f}")

# --------------------------------------------------------------------------- save json
outp = os.path.join(RES, "feedback_biology.json")
json.dump(out, open(outp, "w"), indent=2)
print(f"\nWROTE {outp}")

# ============================================================================================ FIGURE
sys.path.insert(0, os.path.join(_CMC_ROOT, "analysis", "figures", "original"))
from style import plt, C, panel_label, save
import matplotlib.patches as mpatches

fig = plt.figure(figsize=(9.2, 6.6))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

# ---- A: transpose diagnostic ----
rng = np.random.RandomState(0); ss = rng.choice(len(a), size=min(8000, len(a)), replace=False)
sc_a = a[ss] * 1e3; sc_b = b[ss] * 1e3   # display in 1e-3 units
axA.scatter(sc_b, sc_a, s=2, c=C["muted"], alpha=0.25, rasterized=True, edgecolors="none")
lim = np.percentile(np.abs(np.concatenate([sc_a, sc_b])), 99.5)
xs = np.linspace(-lim, lim, 50)
axA.plot(xs, beta * xs, c=C["neg"], lw=1.2, label=f"LS fit  $\\beta$={beta:.2f}")
axA.set_xlim(-lim, lim); axA.set_ylim(-lim, lim)
axA.set_xlabel("translation transpose  $G_{rp}^{T}$  ($\\times10^{-3}$)")
axA.set_ylabel("feedback  $G_{pr}$  ($\\times10^{-3}$)")
axA.set_title("Feedback $\\approx$ transpose of translation")
axA.text(0.04, 0.96, f"Pearson $r$={pr[0]:.2f}\nSpearman={sr.correlation:.2f}\nvar expl.={var_explained*100:.0f}%",
         transform=axA.transAxes, va="top", ha="left", fontsize=7.2,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["muted"], lw=0.6, alpha=0.9))
axA.legend(loc="lower right", fontsize=6.8)
panel_label(axA, "A")

# ---- B: receptor->target enrichment (raw vs residual vs coexpr) ----
pairs = INTENDED
labs = [f"{r}\n$\\rightarrow${t.replace('_','/')}" for r, t in pairs]
x = np.arange(len(pairs)); w = 0.26
def _v(series):
    return [ (enrich[f"{r}->{t}"].get(series) or {}).get("auroc", np.nan) for r, t in pairs ]
raw_v, res_v = _v("raw_Gpr"), _v("residual_R")
cox_v = _v("coexpr") if coexpr_abs is not None else [np.nan]*len(pairs)
axB.bar(x - w, raw_v, w, color=C["shuf"], label="raw $|G_{pr}|$")
axB.bar(x,     res_v, w, color=C["model"], label="residual $|R|$")
axB.bar(x + w, cox_v, w, color=C["base"], label="coexpr $|corr|$")
# star residual bars significant by permutation null (p<0.05)
for xi, (r, t) in zip(x, pairs):
    rr = enrich[f"{r}->{t}"]["residual_R"]
    if rr and rr["perm_null_p"] < 0.05:
        axB.text(xi, rr["auroc"] + 0.02, "*", ha="center", va="bottom", fontsize=11, color=C["model"])
axB.axhline(0.5, ls="--", lw=0.8, c=C["muted"]); axB.text(len(pairs)-0.5, 0.51, "chance", fontsize=6.5, ha="right", color=C["muted"])
axB.set_xticks(x); axB.set_xticklabels(labs, fontsize=6.3)
axB.set_ylabel("within-row AUROC"); axB.set_ylim(0.2, 1.0)
axB.set_title("Receptor $\\rightarrow$ target-set enrichment")
axB.legend(loc="upper right", fontsize=6.3, ncol=1)
panel_label(axB, "B")

# ---- C: decoy specificity confusion matrix (residual R) ----
M = np.array([[ (decoy_R[rec][ts] if decoy_R[rec][ts] is not None else np.nan) for ts in decoy_sets]
              for rec in decoy_recs])
im = axC.imshow(M, cmap="RdBu_r", vmin=0.3, vmax=0.7, aspect="auto")
axC.set_xticks(range(len(decoy_sets))); axC.set_xticklabels([s.replace("_","/") for s in decoy_sets], fontsize=6.5, rotation=30, ha="right")
axC.set_yticks(range(len(decoy_recs))); axC.set_yticklabels(decoy_recs, fontsize=6.8)
for i, rec in enumerate(decoy_recs):
    for j, ts in enumerate(decoy_sets):
        if not np.isnan(M[i, j]):
            axC.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.3,
                     color=("white" if abs(M[i,j]-0.5) > 0.13 else C["ink"]))
        if own[rec] == ts:   # box the intended (own) target set
            axC.add_patch(mpatches.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec=C["pos"], lw=1.8))
axC.set_title("Decoy specificity  (AUROC on residual $R$)")
cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6)
cb.set_label("AUROC", fontsize=6.5)
panel_label(axC, "C")

# ---- D: magnitude / Gini asymmetry (honest negative) ----
axD2 = axD.twinx()
gx = np.array([0, 1])
axD.bar(gx - 0.17, [mag["gini_G_pr"], mag["gini_G_rp"]], 0.34, color=[C["accent"], C["pos"]], alpha=0.9)
axD.set_xticks(gx); axD.set_xticklabels(["Gini\n$G_{pr}$", "Gini\n$G_{rp}$"], fontsize=6.8)
axD.set_ylabel("Gini (sparsity)"); axD.set_ylim(0, 1.0)
for xi, v in zip(gx - 0.17, [mag["gini_G_pr"], mag["gini_G_rp"]]):
    axD.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
# ratio annotation
ratio_txt = (f"mean$|G_{{rp}}|$/mean$|G_{{pr}}|$ = {mag['Grp_over_Gpr']:.2f}\n"
             f"(dec-$\\lambda$ RNA {cfg.get('lambda_dec_sparse_rna')} vs prot {cfg.get('lambda_dec_sparse_prot')}"
             " -> reg-dependent)")
if isinstance(mag["unconfounded_p128"], dict):
    ratio_txt += f"\np128 (sym latent): ratio={mag['unconfounded_p128']['Grp_over_Gpr']:.2f}"
axD.text(0.5, 0.97, ratio_txt, transform=axD.transAxes, va="top", ha="center", fontsize=6.2,
         bbox=dict(boxstyle="round,pad=0.3", fc="#f7fafc", ec=C["muted"], lw=0.6))
axD2.set_yticks([])
axD.set_title("Magnitude/sparsity asymmetry\n(regularization-dependent, not biology)")
panel_label(axD, "D")

save(fig, os.path.join(REPO, "paper", "figures", "_new_feedback_biology.pdf"))
plt.close(fig)
print("DONE")
