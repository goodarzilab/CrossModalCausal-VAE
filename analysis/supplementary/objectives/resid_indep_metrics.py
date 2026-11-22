#!/usr/bin/env python
"""Residual-independence experiment metrics for ONE trained model dir.

Answers two questions for the resid-indep ablation vs the sp_dec_gl1000 Gaussian baseline:
  (A) ORIENTATION: did the penalty break the (D,A)-factorization symmetry / orient A_rr?
      - corr(A_rr, A_rr^T) latent + antisym-energy fraction        (baseline 0.9945 / 0.28%)
      - corr(G_rr, G_rr^T) projected (G_rr=D_r A_rr D_r^T) + antisym (baseline 0.9997 / 0.017%)
  (B) ACHIEVED RESIDUAL INDEPENDENCE: did the penalty actually reduce the dependence of the joint
      exogenous residual eps=[eps_r;eps_p]?
      - beta-TCVAE Total Correlation (the TC training objective; averaged over B=256 minibatches)
      - mean pairwise HSIC (RBF, median-heuristic; the more honest, dynamic-range-having measure)
      - mean |Pearson corr| over eps off-diagonal dim-pairs (simple/interpretable)
      plus per-dim excess kurtosis (marginal non-Gaussianity, for continuity with the eps-prior work).

Metrics are computed on the ENCODER-MEAN residual eps_mu (deterministic) AND on the reparameterized
draw eps_sample (what the training penalty actually saw). Writes <results_dir>/resid_indep_metrics.json.
Usage: python analysis/supplementary/objectives/resid_indep_metrics.py <results_dir> [n_cells]
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
import os, sys, json, warnings, time
import numpy as np, torch, scanpy as sc
import scipy.sparse as sp
from scipy.stats import kurtosis
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(_CMC_ROOT, "src"))
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE
from train import total_correlation_mss, pairwise_hsic
DEVICE = P.DEVICE
TRAIN_RNA = f"{_CMC_DATA}/combined_v4/rna_combined.h5ad"
TRAIN_PROT = f"{_CMC_DATA}/combined_v4/citeseq_combined.h5ad"

rd = os.path.abspath(sys.argv[1]); run = os.path.basename(rd.rstrip("/"))
n_cells = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
cfg = json.load(open(os.path.join(rd, "config.json")))


def offdiag_corr(M):
    off = ~np.eye(M.shape[0], dtype=bool)
    return float(np.corrcoef(M[off], M.T[off])[0, 1])


def antisym_energy_frac(M):
    Mo = M.copy(); np.fill_diagonal(Mo, 0.0)
    asym = 0.5 * (Mo - Mo.T)
    denom = float((Mo * Mo).sum())
    return float((asym * asym).sum() / denom) if denom > 0 else float("nan")


def mean_abs_offdiag_corr(E):
    """mean |Pearson corr| over off-diagonal dim-pairs of the residual matrix E [N, D]."""
    C = np.corrcoef(E.T)
    off = ~np.eye(C.shape[0], dtype=bool)
    return float(np.nanmean(np.abs(C[off])))


def achieved_tc(mu, lv, sample, dataset_size, B=256, n_batches=60, seed=0):
    """Average the beta-TCVAE TC-MSS estimator over n_batches random B-cell minibatches (matches the
    B=256 training regime). mu/lv/sample are [N, D] numpy joint-eps arrays."""
    N = mu.shape[0]; rng = np.random.default_rng(seed); vals = []
    for _ in range(n_batches):
        idx = rng.choice(N, min(B, N), replace=False)
        m = torch.tensor(mu[idx], dtype=torch.float32)
        l = torch.tensor(lv[idx], dtype=torch.float32)
        s = torch.tensor(sample[idx], dtype=torch.float32)
        vals.append(float(total_correlation_mss(s, m, l, dataset_size)))
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_batches": n_batches, "batch_size": B}


def achieved_hsic(E, B=512, n_pairs=1200, chunk=200, seed=0):
    """Mean pairwise HSIC over n_pairs random dim-pairs, evaluated on a B-cell subsample (chunked to
    bound memory). E [N, D] numpy joint-eps."""
    N, D = E.shape; rng = np.random.default_rng(seed)
    idx = rng.choice(N, min(B, N), replace=False)
    Z = torch.tensor(E[idx], dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    vals = []
    done = 0
    while done < n_pairs:
        k = min(chunk, n_pairs - done)
        vals.append(float(pairwise_hsic(Z, k, generator=g)))
        done += k
    return {"mean": float(np.mean(vals)), "n_pairs": n_pairs, "n_cells": int(min(B, N))}


def kurt_summary(M):
    exk = kurtosis(M, axis=0, fisher=True, nan_policy="omit")
    return {"mean_excess_kurtosis": float(np.nanmean(exk)),
            "median_excess_kurtosis": float(np.nanmedian(exk)),
            "pooled_excess_kurtosis": float(kurtosis(M.reshape(-1), fisher=True, nan_policy="omit"))}


# ---- load model ----
r = sc.read_h5ad(TRAIN_RNA, backed="r"); n_g = r.n_vars
p = sc.read_h5ad(TRAIN_PROT, backed="r"); n_p = p.n_vars
r.file.close(); p.file.close()
model = P.load_ctx_model(CrossModalCausalLDVAE, rd, n_g, n_p, cfg); model.eval()
model.eps_prior = cfg.get("eps_prior", "gaussian"); model.eps_df = float(cfg.get("eps_df", 3.0))

# ---- (A) latent + projected adjacency symmetry / orientation ----
with torch.no_grad():
    A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
    A_rr = A_rr.float().cpu().numpy(); A_pp = A_pp.float().cpu().numpy()
    Dr = model._dec_r_w().detach().float().cpu().numpy()
    G = Dr @ A_rr @ Dr.T
out = {
    "run": run, "resid_indep": cfg.get("resid_indep"), "lambda_resid_indep": cfg.get("lambda_resid_indep", 0.0),
    "A_rr_latent": {"corr_A_AT": offdiag_corr(A_rr), "antisym_energy_frac": antisym_energy_frac(A_rr),
                    "fro_norm": float(np.linalg.norm(A_rr))},
    "A_pp_latent": {"corr_A_AT": offdiag_corr(A_pp), "antisym_energy_frac": antisym_energy_frac(A_pp)},
    "G_rr_projected": {"corr_G_GT": offdiag_corr(G), "antisym_energy_frac": antisym_energy_frac(G)},
}
print(f"[{run}] A_rr corr(A,A^T)={out['A_rr_latent']['corr_A_AT']:.4f} "
      f"antisym={out['A_rr_latent']['antisym_energy_frac']:.4f} | "
      f"G_rr corr={out['G_rr_projected']['corr_G_GT']:.4f} antisym={out['G_rr_projected']['antisym_energy_frac']:.4f}",
      flush=True)

# ---- encode a seed-fixed sample of training cells: eps_mu (residual) + eps_sample (posterior draw) ----
rr = sc.read_h5ad(TRAIN_RNA, backed="r"); pp = sc.read_h5ad(TRAIN_PROT, backed="r")
assert (rr.obs_names == pp.obs_names).all(), "RNA/prot obs must be co-ordered"
N = rr.n_obs
t0 = time.time()
rna_mean, rna_std = P._chunked_mean_std(rr); prot_mean, prot_std = P._chunked_mean_std(pp)
print(f"[{run}] global stats over {N} cells done [{time.time()-t0:.0f}s]", flush=True)
rng = np.random.default_rng(42); n_cells = min(n_cells, N)
idx = np.sort(rng.choice(N, n_cells, replace=False))
Xr = rr[idx].to_memory().X; Xp = pp[idx].to_memory().X
Xr = Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr)
Xp = Xp.toarray() if sp.issparse(Xp) else np.asarray(Xp)
rr.file.close(); pp.file.close()
Xr_z = (Xr - rna_mean) / (rna_std + 1e-6); Xp_z = (Xp - prot_mean) / (prot_std + 1e-6)
mur_l, lvr_l, mup_l, lvp_l, sr_l, spd_l = [], [], [], [], [], []
torch.manual_seed(0)
with torch.no_grad():
    for s in range(0, n_cells, 4096):
        e = min(s + 4096, n_cells)
        br = torch.FloatTensor(Xr_z[s:e]).to(DEVICE); bp = torch.FloatTensor(Xp_z[s:e]).to(DEVICE)
        mu_r, lv_r = model.enc_rna(br); mu_p, lv_p = model.enc_prot(bp)
        mur_l.append(mu_r.cpu().numpy()); lvr_l.append(lv_r.cpu().numpy())
        mup_l.append(mu_p.cpu().numpy()); lvp_l.append(lv_p.cpu().numpy())
        sr_l.append(model.reparameterize(mu_r, lv_r).cpu().numpy())
        spd_l.append(model.reparameterize(mu_p, lv_p).cpu().numpy())
mu_r = np.concatenate(mur_l, 0).astype(np.float64); lv_r = np.concatenate(lvr_l, 0).astype(np.float64)
mu_p = np.concatenate(mup_l, 0).astype(np.float64); lv_p = np.concatenate(lvp_l, 0).astype(np.float64)
s_r = np.concatenate(sr_l, 0).astype(np.float64); s_p = np.concatenate(spd_l, 0).astype(np.float64)
mu_j = np.concatenate([mu_r, mu_p], 1); lv_j = np.concatenate([lv_r, lv_p], 1)
s_j = np.concatenate([s_r, s_p], 1)
dataset_size = int(round(N * (1.0 - float(cfg.get("test_size", 0.2)))))   # train-split size (TC weighting)

# ---- (B) achieved residual independence ----
out["dataset_size_used"] = dataset_size
out["eps_mu_kurtosis"] = {"rna": kurt_summary(mu_r), "prot": kurt_summary(mu_p)}
out["achieved_TC_mss"] = {
    "eps_mu": achieved_tc(mu_j, np.full_like(mu_j, -6.0), mu_j, dataset_size),   # mu as a (delta-ish) draw
    "eps_sample": achieved_tc(mu_j, lv_j, s_j, dataset_size),                    # true posterior draw
}
out["achieved_mean_pairwise_HSIC"] = {
    "eps_mu": achieved_hsic(mu_j), "eps_sample": achieved_hsic(s_j),
}
out["achieved_mean_abs_offdiag_corr"] = {
    "eps_mu": mean_abs_offdiag_corr(mu_j), "eps_sample": mean_abs_offdiag_corr(s_j),
}
out["n_cells"] = int(n_cells)
json.dump(out, open(os.path.join(rd, "resid_indep_metrics.json"), "w"), indent=2)
print(f"[{run}] achieved  TC(sample)={out['achieved_TC_mss']['eps_sample']['mean']:.3f}  "
      f"meanHSIC(sample)={out['achieved_mean_pairwise_HSIC']['eps_sample']['mean']:.5f}  "
      f"mean|corr|(sample)={out['achieved_mean_abs_offdiag_corr']['eps_sample']:.4f}", flush=True)
print(f"WROTE {os.path.join(rd, 'resid_indep_metrics.json')}")
