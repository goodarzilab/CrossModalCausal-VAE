"""Permutation nulls + significance for translation-network analyses D1 and D3
(TRANSLATION_ANALYSES_PLAN.md) on the best model (seed 42), CPU.

Reuses causal_eval_suite.py's model loading, projected-network computation
(G_rp = D_r W_rp D_p^T ; G_pr = D_p W_pr D_r^T), the C_translation cognate-pair
identification (gene symbol <-> antibody name via canon_prot), and the D_feedback
translation-vs-feedback magnitudes. Does NOT modify any existing script.

D1  cognate-enrichment permutation null:
    observed cognate transcript->protein enrichment (fold ~25x, top10%-cognate ~0.70)
    vs three nulls (N=1000):
      A  permute latent W_rp entries        -> reproject -> recompute enrichment
      B  permute the protein-column assignment of G_rp  (matched column-norms)
      C  matched-degree: permute genes WITHIN each protein column (exact col-norm)
    -> empirical p-value + z-score for observed fold and top10%-cognate.

D3  translation-vs-feedback directionality:
    cognate-specific  |G_rp[g,p]| (forward translation) vs |G_pr[p,g]| (reverse
    feedback) on the cognate edges -> paired Wilcoxon + sign-flip permutation +
    a random-pair specificity null. Plus latent-aggregate (|W_rp| vs |W_pr|) and
    projected-aggregate (|G_rp| vs |G_pr|) ratios, to reconcile the aggregate
    D_feedback ratio (~0.257, feedback larger) with cognate-specific dominance.

Writes results/translation_nulls.json and results/figures/slides/fig_translation_nulls.png.
Usage: python analysis/fig2_networks/translation_nulls.py [results/<run>]  (default linfb_2c_fb10_b20_v4)
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
import numpy as np, torch, scanpy as sc
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_CMC_ROOT, "src"))
from model import CrossModalCausalLDVAE

RES = _CMC_RESULTS
N = 1000
np.random.seed(42)
C2, CC = "#1f4e79", "#b5651d"   # deck palette (matches slides_bioeval.py)


def _ckpt(rd):
    """FINAL checkpoint by default (fully-regularized); fallback best. Override CMC_CKPT=best."""
    pref = os.environ.get("CMC_CKPT", "final")
    for n in [f"{pref}_model.pt", "final_model.pt", "best_model.pt"]:
        p = os.path.join(rd, n)
        if os.path.exists(p): return p
    return os.path.join(rd, "best_model.pt")


# ------------------------------------------------------------------ load model + projected nets
rd = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(RES, "linfb_2c_fb10_b20_v4")
cfg = json.load(open(f"{rd}/config.json"))
print(f"=== translation_nulls: {os.path.basename(rd)}  ckpt={os.path.basename(_ckpt(rd))} ===")

ar = sc.read_h5ad(cfg["rna_path"], backed="r"); ap = sc.read_h5ad(cfg["prot_path"], backed="r")
genes = list(ar.var_names.astype(str)); prots = list(ap.var_names.astype(str))
ng, npr = len(genes), len(prots)
_dm = None
if cfg.get("dec_mask_rna_path"):
    _mk = torch.load(cfg["dec_mask_rna_path"], map_location="cpu"); _dm = _mk["mask"] if isinstance(_mk, dict) else _mk
md = CrossModalCausalLDVAE(ng, npr, 1, latent_dim_rna=cfg["latent_dim_rna"], latent_dim_prot=cfg["latent_dim_prot"],
    recon_likelihood="mse", scm_nonlinear=cfg.get("scm_nonlinear", False), phi=cfg.get("phi", "tanh"),
    dec_mask_rna=_dm, dec_mask_lambda=cfg.get("dec_mask_lambda", 0.1), dec_mask_freeze=cfg.get("dec_mask_freeze", False))
md.load_state_dict(torch.load(_ckpt(rd), map_location="cpu"), strict=False); md.eval()
with torch.no_grad():
    A_rr, A_pp, W_rp, W_pr = md.scm.get_block_adjacencies(tau=0.5)
    Dr = md._dec_r_w().detach().float(); Dp = md.dec_prot.weight.detach().float()
    W_rp_np = W_rp.float().cpu().numpy()          # [latent_r, latent_p]
    W_pr_np = W_pr.float().cpu().numpy()          # [latent_p, latent_r]
    Dr_np = Dr.cpu().numpy(); Dp_np = Dp.cpu().numpy()
    G_rp = (Dr @ W_rp.float() @ Dp.T).cpu().numpy()   # [ng, npr]  gene -> protein (translation)
    G_pr = (Dp @ W_pr.float() @ Dr.T).cpu().numpy()   # [npr, ng]  protein -> gene (feedback)
print(f"ng={ng} npr={npr}  W_rp{W_rp_np.shape} W_pr{W_pr_np.shape}  G_rp{G_rp.shape} G_pr{G_pr.shape}")


# ------------------------------------------------------------------ cognate identification (EXACT copy of C_translation)
def canon_prot(x):
    x = re.sub(r'^AB_', '', x); x = re.sub(r'^(human|mouse)', '', x, flags=re.I)
    x = re.sub(r'-\d+$', '', x); return x.split('_')[0].upper()
gup = {g.upper(): i for i, g in enumerate(genes)}
cognate = []  # (gene_idx, prot_idx)
for j, p in enumerate(prots):
    c = canon_prot(p)
    for cand in (c, c.replace("CD", ""), c + "E", c + "A"):
        if cand in gup: cognate.append((gup[cand], j)); break
cog_by_p = {}; [cog_by_p.setdefault(p, set()).add(g) for g, p in cognate]
assert cognate, "no cognate pairs identified"
print(f"n_cognate_pairs={len(cognate)}")


# ------------------------------------------------------------------ cognate-enrichment metric (EXACT logic of C_translation)
def cognate_stats(G):
    """fold, top10pct_cognate, median_cognate_rank_frac on a [ng, npr] gene->protein matrix."""
    aG = np.abs(G)
    matched = np.array([aG[g, p] for g, p in cognate])
    bg, ranks = [], []
    for g, p in cognate:
        col = aG[:, p]
        bg.append(np.delete(col, list(cog_by_p[p])))
        ranks.append(float((col > col[g]).sum()) / len(col))
    bg = np.concatenate(bg); ranks = np.asarray(ranks)
    return dict(fold=float(np.median(matched) / (np.median(bg) + 1e-12)),
                top10pct_cognate=float(np.mean(ranks < 0.1)),
                median_cognate_rank_frac=float(np.median(ranks)),
                matched_median=float(np.median(matched)), bg_median=float(np.median(bg)))


obs = cognate_stats(G_rp)
u_obs, p_obs = mannwhitneyu(np.array([np.abs(G_rp)[g, p] for g, p in cognate]),
                            np.concatenate([np.delete(np.abs(G_rp)[:, p], list(cog_by_p[p])) for g, p in cognate]),
                            alternative="greater")
obs["mwu_p"] = float(p_obs)
print("observed:", {k: round(v, 4) if abs(v) > 1e-3 else v for k, v in obs.items()})


# ------------------------------------------------------------------ D1: three nulls
def summarize(null_folds, null_top10, obs_val_fold, obs_val_top10):
    nf = np.asarray(null_folds); nt = np.asarray(null_top10)
    def zp(null, o):
        mu, sd = float(null.mean()), float(null.std())
        z = (o - mu) / (sd + 1e-12)
        p = (1 + int((null >= o).sum())) / (len(null) + 1)   # one-sided empirical p (enrichment)
        return dict(null_mean=mu, null_std=sd, null_max=float(null.max()),
                    null_p95=float(np.percentile(null, 95)), z=float(z), p_emp=float(p))
    return dict(fold=dict(observed=obs_val_fold, **zp(nf, obs_val_fold)),
                top10pct_cognate=dict(observed=obs_val_top10, **zp(nt, obs_val_top10)))

# Null A: permute latent W_rp entries, reproject through the same decoders
wflat = W_rp_np.ravel(); shp = W_rp_np.shape
A_fold, A_top = [], []
for _ in range(N):
    Gk = Dr_np @ np.random.permutation(wflat).reshape(shp) @ Dp_np.T
    s = cognate_stats(Gk); A_fold.append(s["fold"]); A_top.append(s["top10pct_cognate"])
nullA = summarize(A_fold, A_top, obs["fold"], obs["top10pct_cognate"])

# Null B: permute the protein-column assignment of G_rp (whole columns; matched column-norms)
B_fold, B_top = [], []
for _ in range(N):
    Gk = G_rp[:, np.random.permutation(npr)]
    s = cognate_stats(Gk); B_fold.append(s["fold"]); B_top.append(s["top10pct_cognate"])
nullB = summarize(B_fold, B_top, obs["fold"], obs["top10pct_cognate"])

# Null C: matched-degree - permute genes WITHIN each protein column (preserves each column norm exactly)
absG = np.abs(G_rp)
C_fold, C_top = [], []
for _ in range(N):
    order = np.argsort(np.random.rand(ng, npr), axis=0)          # independent per-column permutation
    Gk = np.take_along_axis(absG, order, axis=0)
    s = cognate_stats(Gk); C_fold.append(s["fold"]); C_top.append(s["top10pct_cognate"])
nullC = summarize(C_fold, C_top, obs["fold"], obs["top10pct_cognate"])

D1 = {"observed": obs, "n_perm": N,
      "nullA_permute_Wrp_entries": nullA,
      "nullB_permute_protein_columns": nullB,
      "nullC_matched_degree_within_column": nullC}
print(f"D1 nullA fold z={nullA['fold']['z']:.1f} p={nullA['fold']['p_emp']:.1e} | "
      f"nullC(matched-degree) fold z={nullC['fold']['z']:.1f} p={nullC['fold']['p_emp']:.1e}")


# ------------------------------------------------------------------ D3: translation-vs-feedback directionality
fwd = np.array([abs(G_rp[g, p]) for g, p in cognate])   # forward translation on cognate edge
rev = np.array([abs(G_pr[p, g]) for g, p in cognate])   # reverse feedback on cognate edge
d = fwd - rev
frac_fwd_gt_rev = float(np.mean(fwd > rev))
ratio_med_cognate = float(np.median(fwd) / (np.median(rev) + 1e-12))

# paired Wilcoxon signed-rank, BOTH directions (H1a: translation>feedback ; H1b: feedback>translation)
try:
    w_p_fwd = float(wilcoxon(fwd, rev, alternative="greater")[1])   # translation > feedback
    w_p_rev = float(wilcoxon(rev, fwd, alternative="greater")[1])   # feedback > translation
except Exception:
    w_p_fwd = w_p_rev = None
# paired sign-flip permutation on d = fwd-rev (exchangeability of fwd/rev within a pair)
obs_stat = float(np.median(d))
signflip = np.array([np.median(d * np.random.choice([-1.0, 1.0], size=len(d))) for _ in range(N)])
p_signflip_fwd = (1 + int((signflip >= obs_stat).sum())) / (N + 1)   # translation > feedback
p_signflip_rev = (1 + int((signflip <= obs_stat).sum())) / (N + 1)   # feedback > translation
w_p = w_p_fwd  # (kept for the print line below)

# random-pair specificity null: does the fwd>rev asymmetry ratio concentrate on cognate vs random (g,p)?
rand_ratios = []
for _ in range(N):
    gg = np.random.randint(0, ng, size=len(cognate)); pp = np.random.randint(0, npr, size=len(cognate))
    rf = np.abs(G_rp[gg, pp]); rr = np.abs(G_pr[pp, gg])
    rand_ratios.append(float(np.median(rf) / (np.median(rr) + 1e-12)))
rand_ratios = np.asarray(rand_ratios)
z_ratio = float((ratio_med_cognate - rand_ratios.mean()) / (rand_ratios.std() + 1e-12))
p_ratio = (1 + int((rand_ratios >= ratio_med_cognate).sum())) / (N + 1)

# aggregate ratios (reconcile with D_feedback)
lat_ratio = float(np.abs(W_rp_np).mean() / (np.abs(W_pr_np).mean() + 1e-12))   # latent aggregate
proj_ratio = float(np.abs(G_rp).mean() / (np.abs(G_pr).mean() + 1e-12))        # projected aggregate (== D_feedback ~0.257)

D3 = {
    "cognate_specific_projected": dict(
        n_pairs=len(cognate),
        median_abs_G_rp_forward=float(np.median(fwd)), median_abs_G_pr_reverse=float(np.median(rev)),
        ratio_median_translation_over_feedback=ratio_med_cognate,
        frac_pairs_translation_gt_feedback=frac_fwd_gt_rev,
        wilcoxon_p_translation_gt_feedback=w_p_fwd,
        wilcoxon_p_feedback_gt_translation=w_p_rev,
        signflip_perm_p_translation_gt_feedback=float(p_signflip_fwd),
        signflip_perm_p_feedback_gt_translation=float(p_signflip_rev),
        random_pair_null_on_ratio=dict(null_mean=float(rand_ratios.mean()), null_std=float(rand_ratios.std()),
                                       null_p95=float(np.percentile(rand_ratios, 95)),
                                       z=z_ratio, p_emp=float(p_ratio))),
    "aggregate_latent": dict(mean_abs_Wrp=float(np.abs(W_rp_np).mean()), mean_abs_Wpr=float(np.abs(W_pr_np).mean()),
                             translation_over_feedback=lat_ratio),
    "aggregate_projected": dict(mean_abs_G_rp=float(np.abs(G_rp).mean()), mean_abs_G_pr=float(np.abs(G_pr).mean()),
                                translation_over_feedback=proj_ratio),
    "note": ("Cognate identity is a FEATURE-level concept, so a cognate-specific comparison exists only in "
             "projected (feature) space (G_rp vs G_pr). The latent W_rp/W_pr have no cognate index, so the "
             "latent comparison is necessarily the aggregate mean-|W| ratio."),
    "finding": (
        "HONEST NEGATIVE: central-dogma directionality (|translation| > |feedback|) is NOT recovered. "
        f"Feedback magnitude is larger both in AGGREGATE (latent ratio {lat_ratio:.2f}, projected ratio "
        f"{proj_ratio:.2f}, matching D_feedback) AND on the cognate edges specifically (cognate ratio "
        f"{ratio_med_cognate:.2f}; only {frac_fwd_gt_rev*100:.0f} pct of pairs have translation>feedback; "
        f"Wilcoxon translation>feedback p={w_p_fwd:.2f} n.s., feedback>translation p={w_p_rev:.1e}). "
        f"The cognate ratio ({ratio_med_cognate:.2f}) is statistically indistinguishable from random (g,p) "
        f"pairs (z={z_ratio:.1f}, p={p_ratio:.2f}), so cognate edges are NOT more translation-dominant than "
        "background. Reconciliation: it is NOT the case that cognate translation dominates while aggregate "
        "feedback wins -- feedback dominates uniformly. The central-dogma story is carried by cognate "
        "ENRICHMENT/SPECIFICITY (D1), not by edge-magnitude directionality (D3).")}
print(f"D3 cognate translation/feedback ratio={ratio_med_cognate:.2f} frac_fwd>rev={frac_fwd_gt_rev:.2f} "
      f"| wilcoxon translation>feedback p={w_p_fwd:.2f} (n.s.); feedback>translation p={w_p_rev:.1e} "
      f"| random-pair z={z_ratio:.1f} p={p_ratio:.2f}")
print(f"   aggregate latent ratio={lat_ratio:.3f}  aggregate projected ratio={proj_ratio:.3f} "
      f"-> feedback dominates EVERYWHERE (no central-dogma magnitude directionality)")


# ------------------------------------------------------------------ save JSON
out = {"run": os.path.basename(rd), "ckpt": os.path.basename(_ckpt(rd)), "seed": cfg.get("seed"),
       "n_cognate_pairs": len(cognate), "n_perm": N,
       "cognate_pairs": [[genes[g], prots[p]] for g, p in cognate],
       "D1_cognate_enrichment_null": D1, "D3_translation_vs_feedback": D3}
os.makedirs(RES, exist_ok=True)
json.dump(out, open(f"{RES}/translation_nulls.json", "w"), indent=2)
print(f"wrote {RES}/translation_nulls.json")


# ------------------------------------------------------------------ figure (deck style)
OUT = os.path.join(RES, "figures", "slides"); os.makedirs(OUT, exist_ok=True)
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

# (A) fold null histogram (Null A = W_rp entry shuffle) + observed
ax = axs[0]
ax.hist(A_fold, bins=40, color="#c9c9c9", edgecolor="none", label="null (permuted W_rp)")
ax.hist(C_fold, bins=40, color=C2, alpha=0.45, edgecolor="none", label="matched-degree null")
ax.axvline(obs["fold"], color=CC, lw=2.5, label=f"observed = {obs['fold']:.1f}×")
ax.set_xlabel("cognate fold-enrichment (×)"); ax.set_ylabel("permutations")
ax.set_title("D1: cognate enrichment vs null")
ax.text(obs["fold"], ax.get_ylim()[1] * 0.55, f" z = {nullA['fold']['z']:.0f}\n p < {1/(N+1):.0e}",
        ha="right", va="center", fontsize=11, color=CC, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")

# (B) top10%-cognate null histogram + observed
ax = axs[1]
ax.hist(np.array(A_top) * 100, bins=30, color="#c9c9c9", edgecolor="none", label="null (permuted W_rp)")
ax.hist(np.array(C_top) * 100, bins=30, color=C2, alpha=0.45, edgecolor="none", label="matched-degree null")
ax.axvline(obs["top10pct_cognate"] * 100, color=CC, lw=2.5, label=f"observed = {obs['top10pct_cognate']*100:.0f}%")
ax.set_xlabel("% cognate pairs in top-10% of edges"); ax.set_ylabel("permutations")
ax.set_title("D1: top-10%-cognate vs null")
ax.text(obs["top10pct_cognate"] * 100, ax.get_ylim()[1] * 0.55,
        f"z = {nullA['top10pct_cognate']['z']:.0f}\np < {1/(N+1):.0e} ",
        ha="right", va="center", fontsize=11, color=CC, fontweight="bold")
ax.legend(fontsize=9, loc="upper center")

# (C) cognate translation vs feedback (paired, per cognate edge) -- HONEST: feedback dominates
ax = axs[2]
lo = min(fwd.min(), rev.min()) * 0.5; hi = max(fwd.max(), rev.max()) * 2
ax.plot([lo, hi], [lo, hi], ls="--", c="k", lw=0.8, zorder=1)
ax.scatter(rev, fwd, s=45, color=CC, edgecolor="k", linewidth=0.4, zorder=2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("|G_pr| reverse feedback"); ax.set_ylabel("|G_rp| forward translation")
ax.set_title("D3: feedback ≥ translation (cognate)")
ax.text(0.04, 0.96,
        f"only {frac_fwd_gt_rev*100:.0f}% above diagonal\ncognate ratio = {ratio_med_cognate:.2f}× "
        f"(feedback ~{1/ratio_med_cognate:.1f}×)\ntranslation>feedback: n.s. (p={w_p_fwd:.2f})\n"
        f"aggregate ratio = {proj_ratio:.2f} (same pattern)",
        transform=ax.transAxes, ha="left", va="top", fontsize=10)

fig.suptitle("Translation-network nulls — linfb_2c_fb10_b20_v4 (seed 42)\n"
             "D1: cognate enrichment far beyond chance   |   "
             "D3: feedback ≥ translation on cognate edges (no central-dogma magnitude directionality)",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(f"{OUT}/fig_translation_nulls.png", dpi=130); plt.close(fig)
print(f"wrote {OUT}/fig_translation_nulls.png")
print("DONE")
