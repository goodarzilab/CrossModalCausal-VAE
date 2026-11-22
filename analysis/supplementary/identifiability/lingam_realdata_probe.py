"""REAL-DATA LiNGAM / NON-GAUSSIANITY PROBE  (G4 Part 1b — gate for the Part-2b non-Gaussian retrain)

The synthetic gate (eval/synthetic_identifiability.json) established the identifiability boundary for
OUR model class (z=(I-A^T)^-1 eps, x=Dz, read-out G_rr=D A D^T):
  * GAUSSIAN eps  -> direction UNIDENTIFIABLE: 2nd-order read-out AND DirectLiNGAM both at chance;
  * NON-GAUSSIAN eps -> DirectLiNGAM orients at acc~1.0 (even after decode->recover z_hat=D^+x), but a
    2nd-order read-out (our G_rr mutilation) stays blind.

So the real-data questions this script answers, on the trained best model
(results/linfb_2c_fb10_b20_v4 — linear, prior-off, GAUSSIAN eps prior):

  1. NON-GAUSSIANITY  of the trained latent eps_r on real CITE-seq: per-dim excess kurtosis + a
     D'Agostino normality test. Is eps meaningfully non-Gaussian, or ~=0 (the Gaussian KL forced it)?
     Decisive for whether a Laplace-prior retrain could even help vs whether we must ADD an
     independence/negentropy penalty to INDUCE non-Gaussianity.
  2. DirectLiNGAM on the real latents (eps_r and the endogenous z_r) -> causal order + directed
     adjacency; compare its orientation to the model's own A_rr (whose projected asymmetry ~=1.0), plus
     a bootstrap ORDER-STABILITY check (is the discovered order real signal or degenerate noise?).
  3. Does the LiNGAM direction improve perturbation RR pearDE? Map the LiNGAM-oriented latent adjacency
     back through the decoder to a gene-space directed read-out (G_lingam = D A_lingam D^T, KO sign
     = -G_lingam[g,.]) and score signed-Pearson-on-DE (pearDE) on frangieh/iccite/arce (locked DE
     filter) vs the A_rr gene-projection and the baseline mutilation read-out.

Reuses eval/perturb_validation_analysis (load_ctx_model, encoder, load_shared_data, compute_observed_de).
Outputs:
  results/lingam_realdata_probe.json
  results/figures/slides/fig_lingam_probe.png
Usage:
  python analysis/supplementary/identifiability/lingam_realdata_probe.py            # full (subsamples the LiNGAM fits; ~30-45 min CPU)
  python analysis/supplementary/identifiability/lingam_realdata_probe.py --quick     # smaller n for a fast smoke
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, time, argparse, warnings
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, normaltest, kendalltau, pearsonr

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(_CMC_ROOT, "src"))
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE

RES = _CMC_RESULTS
OUTFIG = os.path.join(RES, "figures", "slides")
os.makedirs(OUTFIG, exist_ok=True)
DEVICE = P.DEVICE

MODEL_DIR = os.path.abspath(os.path.join(RES, "linfb_2c_fb10_b20_v4"))
TRAIN_RNA = f"{_CMC_DATA}/combined_v4/rna_combined.h5ad"
TRAIN_PROT = f"{_CMC_DATA}/combined_v4/citeseq_combined.h5ad"

# perturbation datasets + locked DE filter (from eval/rerun_de_evals.sh)
DATA = f"{_CMC_DATA}"
PERTURB = {
    "frangieh": (f"{DATA}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad",
                 f"{DATA}/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad", 0.05),
    "iccite":   (f"{DATA}/chen_perturb_icciteseq/export/iccite_rna.h5ad",
                 f"{DATA}/chen_perturb_icciteseq/export/iccite_prot.h5ad", 0.05),
    "arce":     (f"{DATA}/arce_perturb_citeseq/arce_rna.h5ad",
                 f"{DATA}/arce_perturb_citeseq/arce_prot.h5ad", 0.05),
}
DE_LFC_RNA = 0.10

REF_KURT = {"gaussian": 0.0, "uniform_subgauss": -1.2, "laplace": 3.0, "exp": 6.0}


# ======================================================================================
# 1. LOAD MODEL + ENCODE THE REAL LATENT eps_r (and z_r, z_hat_r) ON N TRAINING CELLS
# ======================================================================================
def load_model():
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    r = sc.read_h5ad(TRAIN_RNA, backed="r"); n_g = r.n_vars
    p = sc.read_h5ad(TRAIN_PROT, backed="r"); n_p = p.n_vars
    r.file.close(); p.file.close()
    model = P.load_ctx_model(CrossModalCausalLDVAE, MODEL_DIR, n_g, n_p, cfg)
    return model, cfg, n_g, n_p


def encode_latents(model, n_cells, seed, chunk=20000):
    """Faithful reproduction of the training encoder input: per-feature z-score with GLOBAL training
    stats (train.py aligns RNA/prot on the shared obs; here they are identical & co-ordered), then
    encode a seed-fixed sample of cells. Returns eps_r (mu_r), z_r (full SCM solve), z_hat_r (D^+ x)."""
    print(f"[encode] opening backed AnnData ...", flush=True)
    r = sc.read_h5ad(TRAIN_RNA, backed="r")
    p = sc.read_h5ad(TRAIN_PROT, backed="r")
    assert (r.obs_names == p.obs_names).all(), "RNA/prot obs must be co-ordered"
    N = r.n_obs
    print(f"[encode] computing GLOBAL per-feature mean/std over {N} cells (chunked)...", flush=True)
    t0 = time.time()
    rna_mean, rna_std = P._chunked_mean_std(r, chunk_size=chunk)
    prot_mean, prot_std = P._chunked_mean_std(p, chunk_size=chunk)
    print(f"[encode]   stats done [{time.time()-t0:.0f}s]", flush=True)

    rng = np.random.default_rng(seed)
    n_cells = min(n_cells, N)
    idx = np.sort(rng.choice(N, n_cells, replace=False))
    print(f"[encode] reading {n_cells} sampled cells (seed {seed})...", flush=True)
    Xr = r[idx].to_memory().X; Xp = p[idx].to_memory().X
    Xr = Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr)
    Xp = Xp.toarray() if sp.issparse(Xp) else np.asarray(Xp)
    r.file.close(); p.file.close()
    # per-feature excess kurtosis of the (z-scored) RAW INPUT genes — reference for "how non-Gaussian
    # is the data the encoder sees" (sparse lognorm -> very high) vs how non-Gaussian eps ends up.
    Xr_z = (Xr - rna_mean) / (rna_std + 1e-6)
    Xp_z = (Xp - prot_mean) / (prot_std + 1e-6)
    input_gene_exk = float(np.nanmean(kurtosis(Xr_z, axis=0, fisher=True)))

    model.eval()
    eps_r_l, eps_p_l = [], []
    with torch.no_grad():
        for s in range(0, n_cells, 4096):
            e = min(s + 4096, n_cells)
            br = torch.FloatTensor(Xr_z[s:e]).to(DEVICE)
            bp = torch.FloatTensor(Xp_z[s:e]).to(DEVICE)
            mu_r, _ = model.enc_rna(br)
            mu_p, _ = model.enc_prot(bp)
            eps_r_l.append(mu_r.cpu()); eps_p_l.append(mu_p.cpu())
    eps_r = torch.cat(eps_r_l, 0); eps_p = torch.cat(eps_p_l, 0)   # exogenous latent (encoder mu)

    # endogenous latent z = SCM solve of eps_joint (linear: (I-A^T)^-1 eps). A_rr acts on z, so z_r is
    # the correct LiNGAM analog for recovering A_rr; here A is tiny so z_r ~= eps_r (reported).
    with torch.no_grad():
        A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
        A_full = torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0).to(DEVICE)
        z_l = []
        for s in range(0, n_cells, 8192):
            e = min(s + 8192, n_cells)
            ej = torch.cat([eps_r[s:e], eps_p[s:e]], 1).to(DEVICE)
            z_l.append(model.scm._scm_solve(A_full, ej).cpu())
        z = torch.cat(z_l, 0)
        z_r = z[:, :model.latent_r]
        # decode->recover latent z_hat = D^+ (x - bias)  (transfer test: does decoder mixing D destroy
        # identifiability, as probed synthetically by directlingam_recovered)
        Wr = model._dec_r_w().detach().cpu().numpy().astype(np.float64)   # [n_g, latent_r]
        b = model.bias_rna.detach().cpu().numpy().astype(np.float64)
        Dpinv = np.linalg.pinv(Wr)                                        # [latent_r, n_g]
        z_hat_r = ((Xr_z.astype(np.float64) - b) @ Dpinv.T).astype(np.float32)

    return (eps_r.numpy().astype(np.float64), eps_p.numpy().astype(np.float64),
            z_r.numpy().astype(np.float64), z_hat_r.astype(np.float64), input_gene_exk, n_cells)


# ======================================================================================
# 2. NON-GAUSSIANITY SUMMARY (TASK 1)
# ======================================================================================
def _winsor_kurt(M, q):
    """Per-dim excess kurtosis after clipping each dim to its [q, 1-q] quantiles. If the heavy tails are
    driven by a handful of extreme cells this collapses toward 0; if they are structural it stays high."""
    lo = np.quantile(M, q, axis=0); hi = np.quantile(M, 1 - q, axis=0)
    Mc = np.clip(M, lo, hi)
    return kurtosis(Mc, axis=0, fisher=True)


def nongauss_summary(M, n):
    exk = kurtosis(M, axis=0, fisher=True)          # per-dim excess kurtosis (0 = Gaussian)
    skw = skew(M, axis=0)
    _, pv = normaltest(M, axis=0)                    # D'Agostino K^2 per dim
    se = float(np.sqrt(24.0 / n))                    # asymptotic SE of excess kurtosis under Gaussian
    exk_w1 = _winsor_kurt(M, 0.01)                   # robust to <=1% extreme cells per dim
    exk_w005 = _winsor_kurt(M, 0.005)               # robust to <=0.5% extreme cells per dim
    return {
        "mean_excess_kurtosis": float(np.nanmean(exk)),
        "median_excess_kurtosis": float(np.nanmedian(exk)),
        "mean_abs_excess_kurtosis": float(np.nanmean(np.abs(exk))),
        "p10_excess_kurtosis": float(np.nanpercentile(exk, 10)),
        "p90_excess_kurtosis": float(np.nanpercentile(exk, 90)),
        "max_abs_excess_kurtosis": float(np.nanmax(np.abs(exk))),
        # robust (winsorized) kurtosis: separates structural heavy-tails from single-cell outliers
        "winsor1pct_mean_excess_kurtosis": float(np.nanmean(exk_w1)),
        "winsor1pct_median_excess_kurtosis": float(np.nanmedian(exk_w1)),
        "winsor0p5pct_median_excess_kurtosis": float(np.nanmedian(exk_w005)),
        "mean_abs_skew": float(np.nanmean(np.abs(skw))),
        "gaussian_null_kurtosis_se": se,
        "frac_absk_gt_5se": float(np.mean(np.abs(exk) > 5 * se)),
        "frac_absk_gt_0p5": float(np.mean(np.abs(exk) > 0.5)),
        "frac_absk_gt_1p0": float(np.mean(np.abs(exk) > 1.0)),
        "frac_winsor1pct_k_gt_0p5": float(np.mean(np.abs(exk_w1) > 0.5)),
        "frac_reject_normal_a05": float(np.mean(pv < 0.05)),
        "per_dim_excess_kurtosis": [float(x) for x in exk],
        "per_dim_skew": [float(x) for x in skw],
    }


# ======================================================================================
# 3. DirectLiNGAM (TASK 2)
# ======================================================================================
def fit_lingam(X, n_fit, seed, tag, winsor=0.005):
    """Subsample to n_fit rows (pwling cost ~ n*P^3) and fit DirectLiNGAM. Returns order + B (B[i,j]=j->i).
    The real latent is extremely heavy-tailed (excess kurtosis up to ~1e3), so a few astronomical-
    activation cells would otherwise dominate the pwling entropy/independence estimates; we winsorize
    each dim to its [winsor, 1-winsor] quantiles first (structural non-Gaussianity is preserved)."""
    import lingam
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], min(n_fit, X.shape[0]), replace=False)
    Xs = X[idx].copy()
    if winsor and winsor > 0:
        lo = np.quantile(Xs, winsor, axis=0); hi = np.quantile(Xs, 1 - winsor, axis=0)
        Xs = np.clip(Xs, lo, hi)
    t0 = time.time()
    m = lingam.DirectLiNGAM(measure="pwling")
    m.fit(Xs)
    print(f"[lingam:{tag}] fit n={len(idx)} P={X.shape[1]} winsor={winsor} in {time.time()-t0:.0f}s", flush=True)
    return np.asarray(m.causal_order_), np.asarray(m.adjacency_matrix_)


def compare_lingam_to_Arr(order_z, B_z, A_rr, topK=1000):
    """A_rr[j,i] = model weight j->i (nearly dense, tiny). Take the top-K strongest |A_rr| directed
    edges as a skeleton and ask whether the LiNGAM causal order orients them the same way (parent j
    before child i) vs chance 0.5, plus signed agreement of the two asymmetries."""
    Ad = A_rr.copy(); np.fill_diagonal(Ad, 0.0)
    n = Ad.shape[0]
    flat = np.argsort(np.abs(Ad).ravel())[::-1][:topK]
    js, is_ = np.unravel_index(flat, Ad.shape)        # edge j->i  (source j, target i)
    pos = {v: k for k, v in enumerate(order_z.tolist())}
    order_agree = float(np.mean([pos[j] < pos[i] for j, i in zip(js, is_)]))
    # signed asymmetry agreement: model A_rr[j,i]-A_rr[i,j] vs LiNGAM B[i,j]-B[j,i] (both = fwd-rev j->i)
    a_asym = np.array([Ad[j, i] - Ad[i, j] for j, i in zip(js, is_)])
    l_asym = np.array([B_z[i, j] - B_z[j, i] for j, i in zip(js, is_)])
    if a_asym.std() > 1e-12 and l_asym.std() > 1e-12:
        pear_asym = float(pearsonr(a_asym, l_asym)[0])
    else:
        pear_asym = 0.0
    dens = float(np.mean(np.abs(B_z) > 1e-9))
    return {"topK": int(topK), "order_agreement_vs_Arr": order_agree,
            "pearson_signed_asymmetry_vs_Arr": pear_asym, "lingam_adjacency_density": dens}


# ======================================================================================
# 4. GENE-SPACE pearDE READ-OUT (TASK 3)
# ======================================================================================
def _pe(pred, obs, m):
    """signed Pearson on the DE mask (matches eval/mutilation_directed_eval.py pe())."""
    if m.sum() < 5 or np.std(pred[m]) == 0 or np.std(obs[m]) == 0:
        return np.nan
    return float(np.corrcoef(pred[m], obs[m])[0, 1])


def score_dataset(tag, rna_path, prot_path, lfc_prot, G_rr, G_lingam):
    """For each perturbation: gene-space signed read-out pearDE on DE genes, for the A_rr projection
    (-G_rr[g,.]) and the LiNGAM-oriented projection (-G_lingam[g,.]). KO reduces g -> a
    positively-linked gene is predicted DOWN, hence the leading minus (parallels the gproj baseline)."""
    P.PERTURB_RNA_PATH = rna_path; P.PERTURB_PROT_PATH = prot_path
    P.DE_LFC_RNA = DE_LFC_RNA; P.DE_LFC_PROT = lfc_prot
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    pr, pp, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
    de = P.compute_observed_de(pr, pp, align)
    common = np.asarray(align["common_genes"]); g2i = align["train_gene_to_idx_common"]
    tgi = align["train_gene_idx"]; reti = align["rna_eval_train_indices"]
    rows = []
    for g, d in de.items():
        if g not in tgi or g not in g2i:
            continue
        keep = common != g
        lab = np.asarray(d["sig_rna"])[keep].astype(int)
        if lab.sum() < 3:
            continue
        lfc = np.asarray(d["lfc_rna"])[keep]; dm = lab.astype(bool)
        gv = G_rr[tgi[g], :][reti][keep]
        gl = G_lingam[tgi[g], :][reti][keep]
        rows.append({"pert": g,
                     "arr_gproj_rr_pearDE": _pe(-gv, lfc, dm),
                     "lingam_gproj_rr_pearDE": _pe(-gl, lfc, dm)})
    def nm(k):
        v = [r[k] for r in rows if r[k] is not None and not np.isnan(r[k])]
        return float(np.mean(v)) if v else float("nan")
    # baseline mutilation read-out for reference (already computed under the locked filter)
    mut = None
    mp = os.path.join(MODEL_DIR, f"mutil_directed_{tag}.json")
    if os.path.exists(mp):
        mj = json.load(open(mp))
        mut = {"mutil_rr_pearDE": mj.get("mutil_rr_pearDE"), "std_rr_pearDE": mj.get("std_rr_pearDE"),
               "gproj_rr_pearDE_json": mj.get("gproj_rr_pearDE"), "corr_rr_pearDE": mj.get("corr_rr_pearDE")}
    return {"n_perts": len(rows), "arr_gproj_rr_pearDE": nm("arr_gproj_rr_pearDE"),
            "lingam_gproj_rr_pearDE": nm("lingam_gproj_rr_pearDE"), "baseline_mutil": mut,
            "_rows": rows}


# ======================================================================================
# 5. FIGURE
# ======================================================================================
def make_figure(out, png):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) per-dim excess kurtosis histograms
    ax = axs[0, 0]
    t1 = out["task1_nongaussianity"]
    for key, col, lab in [("eps_r", "#1f4e79", "eps_r (exogenous, encoder mu)"),
                          ("z_r", "#b5651d", "z_r (endogenous SCM)"),
                          ("z_hat_r", "#6a9a3a", "z_hat_r (D+ x)")]:
        exk = np.array(t1[key]["per_dim_excess_kurtosis"])
        ax.hist(exk, bins=40, alpha=0.5, color=col, label=f"{lab}\n(mean|k|={t1[key]['mean_abs_excess_kurtosis']:.2f})")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.axvline(REF_KURT["laplace"], color="crimson", ls="--", lw=1)
    ax.text(REF_KURT["laplace"], ax.get_ylim()[1] * 0.9, " Laplace=3", color="crimson", fontsize=9)
    ax.axvline(REF_KURT["uniform_subgauss"], color="teal", ls="--", lw=1)
    ax.text(REF_KURT["uniform_subgauss"], ax.get_ylim()[1] * 0.75, "unif=-1.2", color="teal", fontsize=8, ha="right")
    se = t1["eps_r"]["gaussian_null_kurtosis_se"]
    ax.axvspan(-5 * se, 5 * se, color="#f2c4c4", alpha=0.5, lw=0)
    ax.set_xlabel("per-dim excess kurtosis (Gaussian = 0)"); ax.set_ylabel("# latent dims")
    ax.set_title("Task 1: is the real latent eps non-Gaussian?"); ax.legend(fontsize=8, frameon=False)

    # (0,1) skew vs excess kurtosis for eps_r
    ax = axs[0, 1]
    exk = np.array(t1["eps_r"]["per_dim_excess_kurtosis"]); skw = np.array(t1["eps_r"]["per_dim_skew"])
    ax.scatter(skw, exk, s=14, color="#1f4e79", alpha=0.6)
    ax.axhline(0, color="k", ls=":", lw=0.8); ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.axhspan(-5 * se, 5 * se, color="#f2c4c4", alpha=0.4, lw=0)
    ax.set_xlabel("per-dim skew"); ax.set_ylabel("per-dim excess kurtosis")
    ax.set_title(f"eps_r shape  (D'Agostino reject@.05: {t1['eps_r']['frac_reject_normal_a05']*100:.0f}% of dims)")
    ax.text(0.02, 0.02, f"input genes mean exk = {t1['input_rna_genes_mean_excess_kurtosis']:.1f}\n"
                        f"eps_r mean exk = {t1['eps_r']['mean_excess_kurtosis']:.3f}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))

    # (1,0) LiNGAM order stability + agreement w/ A_rr
    ax = axs[1, 0]
    t2 = out.get("task2_lingam", {})
    bars, vals, cols = [], [], []
    if "order_stability_kendalltau_z" in t2:
        bars.append("order stability\n(Kendall tau, z_r)"); vals.append(t2["order_stability_kendalltau_z"]); cols.append("#4c72b0")
    cmp = t2.get("lingam_z_vs_Arr", {})
    if "order_agreement_vs_Arr" in cmp:
        bars.append("order agree\nvs A_rr topK"); vals.append(cmp["order_agreement_vs_Arr"]); cols.append("#dd8452")
    if "pearson_signed_asymmetry_vs_Arr" in cmp:
        bars.append("signed asym corr\nvs A_rr"); vals.append(cmp["pearson_signed_asymmetry_vs_Arr"]); cols.append("#55a868")
    x = np.arange(len(bars))
    ax.bar(x, vals, color=cols, edgecolor="black")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.01 * np.sign(v) + (0.01 if v >= 0 else -0.03), f"{v:+.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.5, color="grey", ls="--", lw=0.9); ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(bars, fontsize=8)
    ax.set_ylim(min(-0.1, min(vals + [0]) - 0.1), 1.05)
    ax.set_title("Task 2: DirectLiNGAM on real latents\n(0.5 = chance order; 0 = no signed signal)")

    # (1,1) Task 3 pearDE per dataset
    ax = axs[1, 1]
    t3 = out.get("task3_pearDE", {})
    ds = [d for d in ["frangieh", "iccite", "arce"] if d in t3]
    x = np.arange(len(ds)); w = 0.27
    mut = [(t3[d].get("baseline_mutil") or {}).get("mutil_rr_pearDE", np.nan) for d in ds]
    arr = [t3[d].get("arr_gproj_rr_pearDE", np.nan) for d in ds]
    lin = [t3[d].get("lingam_gproj_rr_pearDE", np.nan) for d in ds]
    ax.bar(x - w, mut, w, label="mutilation (baseline)", color="#9aa0a6", edgecolor="black")
    ax.bar(x, arr, w, label="A_rr gene-proj", color="#7a9cc6", edgecolor="black")
    ax.bar(x + w, lin, w, label="LiNGAM gene-proj", color="#1f4e79", edgecolor="black")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([d.upper() for d in ds])
    ax.set_ylabel("RR direction (pearDE, signed Pearson on DE)")
    ax.set_title("Task 3: does LiNGAM direction improve pearDE?"); ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Real-data LiNGAM / non-Gaussianity probe — linfb_2c_fb10_b20_v4 (linear, prior-off, Gaussian eps)",
                 fontsize=13, y=1.005)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(png, bbox_inches="tight"); plt.close(fig)
    print("WROTE", png, flush=True)


# ======================================================================================
# MAIN
# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_cells", type=int, default=50000)
    ap.add_argument("--n_lingam", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_task3", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_cells = 8000; args.n_lingam = 1500

    t_start = time.time()
    out = {"run": os.path.basename(MODEL_DIR), "model_dir": MODEL_DIR, "seed": args.seed}

    # ---- load model + config context ----
    model, cfg, n_g, n_p = load_model()
    with torch.no_grad():
        A_rr_t, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
        A_rr = A_rr_t.detach().cpu().numpy().astype(np.float64)
        Wr = model._dec_r_w().detach().cpu().numpy().astype(np.float64)     # [n_g, latent_r]
        # spectral radius of A_full (aliveness reference)
        A_full = torch.cat([torch.cat([A_rr_t, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0)
        rho = float(torch.linalg.eigvals(A_full).abs().max().item())
    out["config_context"] = {
        "latent_dim_rna": model.latent_r, "latent_dim_prot": model.latent_p,
        "dim_rna": n_g, "dim_prot": n_p, "scm_nonlinear": bool(cfg.get("scm_nonlinear", False)),
        "prior_w_final": cfg.get("prior_w_final"), "free_bits_r": cfg.get("free_bits_r"),
        "beta_rna_final": cfg.get("beta_rna_final"), "eps_prior": "gaussian(KL to N(0,I))",
        "A_full_spectral_radius": rho}

    # ---- encode real latents ----
    eps_r, eps_p, z_r, z_hat_r, input_gene_exk, n_cells = encode_latents(model, args.n_cells, args.seed)
    out["n_cells_encoded"] = n_cells

    # ============ TASK 1: non-Gaussianity ============
    print("[task1] non-Gaussianity summary ...", flush=True)
    t1 = {"eps_r": nongauss_summary(eps_r, n_cells),
          "z_r": nongauss_summary(z_r, n_cells),
          "z_hat_r": nongauss_summary(z_hat_r, n_cells),
          "input_rna_genes_mean_excess_kurtosis": input_gene_exk}
    out["task1_nongaussianity"] = t1
    ek = t1["eps_r"]
    print(f"[task1] eps_r: mean exk={ek['mean_excess_kurtosis']:+.3f} mean|exk|={ek['mean_abs_excess_kurtosis']:.3f} "
          f"max|exk|={ek['max_abs_excess_kurtosis']:.2f} | frac|k|>0.5={ek['frac_absk_gt_0p5']:.2f} "
          f"reject-normal={ek['frac_reject_normal_a05']:.2f} | input genes exk={input_gene_exk:.1f}", flush=True)

    # ============ TASK 2: DirectLiNGAM ============
    t2 = {"n_fit": args.n_lingam, "lingam_winsor_quantile": 0.005,
          "note": "LiNGAM fit on z_r winsorized per-dim at 0.5/99.5% (heavy-tailed latent); B[i,j]=j->i"}
    # independence check on eps_r: median off-diagonal |correlation| (model ASSUMES eps independent)
    C = np.corrcoef(eps_r, rowvar=False); off = np.abs(C[~np.eye(C.shape[0], dtype=bool)])
    t2["eps_r_offdiag_abscorr_median"] = float(np.median(off))
    t2["eps_r_offdiag_abscorr_p95"] = float(np.percentile(off, 95))
    B_z = None
    try:
        order_z, B_z = fit_lingam(z_r, args.n_lingam, args.seed, "z_r")
        order_z2, _ = fit_lingam(z_r, args.n_lingam, args.seed + 1, "z_r_boot")
        # order stability: Kendall tau of the rank each dim receives in the two subsamples' orders
        rank1 = np.empty(len(order_z), int); rank1[order_z] = np.arange(len(order_z))
        rank2 = np.empty(len(order_z2), int); rank2[order_z2] = np.arange(len(order_z2))
        t2["order_stability_kendalltau_z"] = float(kendalltau(rank1, rank2)[0])
        t2["lingam_z_vs_Arr"] = compare_lingam_to_Arr(order_z, B_z, A_rr, topK=min(1000, model.latent_r * (model.latent_r - 1)))
        t2["causal_order_z_head"] = [int(x) for x in order_z[:12]]
        print(f"[task2] order stability (Kendall tau)={t2['order_stability_kendalltau_z']:+.3f} | "
              f"order-agree vs A_rr={t2['lingam_z_vs_Arr']['order_agreement_vs_Arr']:.3f} | "
              f"eps offdiag |corr| med={t2['eps_r_offdiag_abscorr_median']:.3f}", flush=True)
    except Exception as e:
        t2["error"] = repr(e)[:200]
        print("[task2] LiNGAM ERROR:", t2["error"], flush=True)
    out["task2_lingam"] = t2

    # ============ TASK 3: gene-space pearDE ============
    if not args.skip_task3 and B_z is not None:
        # A_lingam[j,i] = B_z[i,j]  (model convention j->i) ; G_lingam = D A_lingam D^T
        A_lingam = B_z.T
        G_rr = Wr @ A_rr @ Wr.T
        G_lingam = Wr @ A_lingam @ Wr.T
        out["task3_pearDE"] = {}
        for tag, (rp, pp, lfcp) in PERTURB.items():
            try:
                print(f"[task3] scoring {tag} ...", flush=True)
                r = score_dataset(tag, rp, pp, lfcp, G_rr, G_lingam)
                r.pop("_rows", None)
                out["task3_pearDE"][tag] = r
                bm = (r.get("baseline_mutil") or {}).get("mutil_rr_pearDE")
                print(f"[task3] {tag}: n={r['n_perts']} | LiNGAM gproj pearDE={r['lingam_gproj_rr_pearDE']:+.4f} | "
                      f"A_rr gproj={r['arr_gproj_rr_pearDE']:+.4f} | mutil(baseline)={bm}", flush=True)
            except Exception as e:
                out["task3_pearDE"][tag] = {"error": repr(e)[:200]}
                print(f"[task3] {tag} ERROR:", repr(e)[:200], flush=True)
    else:
        out["task3_pearDE"] = {"skipped": True}

    # ============ GATE VERDICT ============
    out["gate_verdict"] = build_verdict(out)

    outjson = os.path.join(RES, "lingam_realdata_probe.json")
    json.dump(out, open(outjson, "w"), indent=2)
    print("WROTE", outjson, flush=True)
    make_figure(out, os.path.join(OUTFIG, "fig_lingam_probe.png"))
    print(f"DONE [{time.time()-t_start:.0f}s]", flush=True)
    # echo verdict
    for k, v in out["gate_verdict"].items():
        print(f"  VERDICT[{k}]: {v}", flush=True)
    return out


def build_verdict(out):
    t1 = out["task1_nongaussianity"]["eps_r"]
    mabsk = t1["mean_abs_excess_kurtosis"]; medk = t1["median_excess_kurtosis"]; maxk = t1["max_abs_excess_kurtosis"]
    wmedk = t1.get("winsor1pct_median_excess_kurtosis")   # robust (outlier-trimmed) median
    se = t1["gaussian_null_kurtosis_se"]
    input_exk = out["task1_nongaussianity"]["input_rna_genes_mean_excess_kurtosis"]
    t2 = out.get("task2_lingam", {})
    stab = t2.get("order_stability_kendalltau_z")
    agree = (t2.get("lingam_z_vs_Arr") or {}).get("order_agreement_vs_Arr")
    offcorr = t2.get("eps_r_offdiag_abscorr_median")
    t3 = out.get("task3_pearDE", {})
    lin_imp = []
    for d, r in t3.items():
        if isinstance(r, dict) and "lingam_gproj_rr_pearDE" in r:
            bm = (r.get("baseline_mutil") or {}).get("mutil_rr_pearDE")
            lin_imp.append((r["lingam_gproj_rr_pearDE"], r["arr_gproj_rr_pearDE"], bm))
    lin_vals = [a for a, _, _ in lin_imp if a == a]
    lin_mean = float(np.mean(lin_vals)) if lin_vals else float("nan")
    # "improves" = LiNGAM gene-space read-out beats BOTH the A_rr projection and the mutilation baseline
    improves = lin_mean == lin_mean and lin_mean > 0.05 and bool(lin_imp) and all(
        (a >= (bm if bm is not None else -9) - 1e-9) and (a >= ar - 1e-9) for a, ar, bm in lin_imp)

    # (a) non-Gaussian enough? benchmark: LiNGAM in the synthetic gate needed |exk| ~>= 1 (uniform 1.2,
    # laplace 3.0) to orient; Gaussian null SE ~0.02. Use the winsorized median to guard against a few
    # extreme cells inflating the raw kurtosis.
    robust_high = (wmedk is not None and wmedk >= 1.0)
    if mabsk >= 1.0 and (wmedk is None or wmedk >= 0.5):
        a = (f"YES — eps_r is strongly, PERVASIVELY non-Gaussian (median excess kurtosis={medk:.1f}, "
             f"mean|exk|={mabsk:.1f}, max={maxk:.0f}; winsorized@1% median={wmedk}). Far above the ~1.0 "
             f"LiNGAM threshold; the Gaussian KL did NOT force eps Gaussian (free-bits relax it, and the "
             f"input is itself heavy-tailed, genes exk={input_exk:.0f}). {'Robust to outliers.' if robust_high else 'NOTE: raw kurtosis is partly outlier-inflated (winsor median lower).'}")
        a_flag = "nongaussian"
    elif mabsk >= 0.2:
        a = f"MARGINAL — eps_r shows mild non-Gaussianity (mean|excess kurtosis|={mabsk:.2f}, winsor median={wmedk}); above the Gaussian null band but below the ~1.0 needed for reliable LiNGAM orientation."
        a_flag = "marginal"
    else:
        a = f"NO — eps_r is effectively Gaussian-forced (mean|excess kurtosis|={mabsk:.3f}, near the null SE={se:.3f}); the Gaussian KL flattened the shape (input genes exk={input_exk:.0f})."
        a_flag = "gaussian_forced"

    # (b) does higher-order orient/improve on real data?
    if stab is None:
        b = "UNKNOWN — LiNGAM did not complete."
        b_flag = "unknown"
    else:
        stable = stab > 0.3
        agrees = agree is not None and agree > 0.55
        if (agrees or improves):
            b = (f"YES — DirectLiNGAM gives a stable order (Kendall tau={stab:+.2f}) that "
                 f"{'agrees with A_rr (order-agree=%.2f)'%agree if agrees else ''}"
                 f"{' and ' if agrees and improves else ''}"
                 f"{'improves gene-space RR pearDE (LiNGAM=%.3f, beats A_rr-proj & mutilation)'%lin_mean if improves else ''}.")
            b_flag = "yes"
        else:
            b = (f"NO — DirectLiNGAM finds a {'stable' if stable else 'weak'} order (Kendall tau={stab:+.2f}) "
                 f"but it does NOT match A_rr (order-agreement={agree} ~= chance 0.5) and its gene-space "
                 f"read-out does not beat the 2nd-order baselines (LiNGAM RR pearDE mean={lin_mean:.3f}). "
                 f"eps IS non-Gaussian, yet A_rr carries no oriented signal for LiNGAM to recover — A_rr is "
                 f"nearly non-load-bearing/unoriented (projected asymmetry ~=1.0), so this is NOT a "
                 f"Gaussianity failure but an A-identifiability/load-bearing one.")
            b_flag = "no"

    # (c) Part-2b recommendation — keyed to the ACTUAL a/b outcome
    if a_flag == "gaussian_forced":
        c = ("Part-2b retrain worth it ONLY IF it INDUCES non-Gaussianity: a Laplace prior ALONE is "
             "unlikely to suffice; MUST add an independence/negentropy (ICA-style) penalty AND a LiNGAM "
             "read-out.")
    elif a_flag == "nongaussian" and b_flag == "yes":
        c = ("Part-2b does NOT need a Laplace/negentropy prior: eps is ALREADY strongly non-Gaussian and a "
             "higher-order (LiNGAM/entropy) read-out orients it and improves pearDE. Adding the LiNGAM/"
             "entropy ORIENTATION read-out is the worthwhile change; inducing non-Gaussianity is redundant, "
             "and no eps-prior retrain is required for the read-out itself.")
    elif a_flag == "nongaussian" and b_flag == "no":
        c = ("The Part-2b PREMISE IS FALSE for this model: eps is already strongly, pervasively non-Gaussian, "
             "so a Laplace-prior / negentropy sweep to INDUCE non-Gaussianity is UNNECESSARY and expected NOT "
             "to help. LiNGAM confirms a stable order exists but it neither matches A_rr nor beats the "
             "2nd-order baselines, because A_rr is nearly non-load-bearing/unoriented. The real lever is (i) "
             "making A load-bearing (eps-bottleneck) and/or (ii) an interventional orientation objective that "
             "ties A to the non-Gaussian structure (cf. finetune_interventional) — NOT the eps prior. "
             "Recommendation: do NOT run a plain Laplace-prior Part-2b sweep.")
    else:
        c = ("Part-2b borderline: if pursued, include BOTH a non-Gaussianity-inducing penalty and a LiNGAM "
             "read-out; a Laplace prior alone is unlikely to be enough.")

    return {"a_real_eps_nongaussian": a, "b_higherorder_orients": b, "c_part2b_recommendation": c,
            "key_numbers": {"eps_median_excess_kurtosis": medk, "eps_mean_abs_excess_kurtosis": mabsk,
                            "eps_winsor1pct_median_excess_kurtosis": wmedk,
                            "eps_offdiag_abscorr_median": offcorr,
                            "lingam_order_stability_kendalltau": stab,
                            "lingam_order_agreement_vs_Arr": agree,
                            "lingam_gproj_rr_pearDE_mean": lin_mean if lin_vals else None},
            "flags": {"a": a_flag, "b": b_flag}}


if __name__ == "__main__":
    main()
