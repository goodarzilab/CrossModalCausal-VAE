#!/usr/bin/env python
"""A_rr symmetry + antisymmetric-energy fraction (latent A_rr AND projected G_rr = D_r A_rr D_r^T),
plus exogenous-eps non-Gaussianity (per-dim excess kurtosis of the encoder mu on training cells).
Answers: did a non-Gaussian eps prior break the (D,A)-factorization symmetry / orient the graph, and
did it actually change the eps tail vs the Gaussian-prior baseline?  Writes <results_dir>/eps_prior_symkurt.json.
Usage: python analysis/supplementary/objectives/eps_prior_symkurt.py <results_dir> [n_cells]
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
DEVICE = P.DEVICE
TRAIN_RNA = f"{_CMC_DATA}/combined_v4/rna_combined.h5ad"
TRAIN_PROT = f"{_CMC_DATA}/combined_v4/citeseq_combined.h5ad"

rd = os.path.abspath(sys.argv[1]); run = os.path.basename(rd.rstrip("/"))
n_cells = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
cfg = json.load(open(os.path.join(rd, "config.json")))

def offdiag_corr(M):
    off = ~np.eye(M.shape[0], dtype=bool)
    return float(np.corrcoef(M[off], M.T[off])[0, 1])

def antisym_energy_frac(M):
    """||(M - M^T)/2||_F^2 / ||M||_F^2 (off-diagonal). 0 = perfectly symmetric, ->1 = anti-symmetric."""
    Mo = M.copy(); np.fill_diagonal(Mo, 0.0)
    asym = 0.5 * (Mo - Mo.T)
    denom = float((Mo * Mo).sum())
    return float((asym * asym).sum() / denom) if denom > 0 else float("nan")

# ---- load model ----
r = sc.read_h5ad(TRAIN_RNA, backed="r"); n_g = r.n_vars
p = sc.read_h5ad(TRAIN_PROT, backed="r"); n_p = p.n_vars
r.file.close(); p.file.close()
model = P.load_ctx_model(CrossModalCausalLDVAE, rd, n_g, n_p, cfg); model.eval()
# load_ctx_model does not pass eps_prior; set it from cfg so reparameterize() draws from the TRAINED
# posterior family (affects only the secondary 'sampled eps' kurtosis; encoder-mu + adjacency metrics
# are prior-independent).
model.eps_prior = cfg.get("eps_prior", "gaussian"); model.eps_df = float(cfg.get("eps_df", 3.0))

# ---- latent + projected adjacency symmetry ----
with torch.no_grad():
    A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
    A_rr = A_rr.float().cpu().numpy(); A_pp = A_pp.float().cpu().numpy()
    Dr = model._dec_r_w().detach().float().cpu().numpy()
    G = Dr @ A_rr @ Dr.T
out = {
    "run": run, "eps_prior": cfg.get("eps_prior", "gaussian"),
    "A_rr_latent": {"corr_A_AT": offdiag_corr(A_rr), "antisym_energy_frac": antisym_energy_frac(A_rr),
                    "fro_norm": float(np.linalg.norm(A_rr))},
    "A_pp_latent": {"corr_A_AT": offdiag_corr(A_pp), "antisym_energy_frac": antisym_energy_frac(A_pp)},
    "G_rr_projected": {"corr_G_GT": offdiag_corr(G), "antisym_energy_frac": antisym_energy_frac(G)},
}
print(f"[{run}] A_rr corr(A,A^T)={out['A_rr_latent']['corr_A_AT']:.4f} "
      f"antisym={out['A_rr_latent']['antisym_energy_frac']:.4f} | "
      f"G_rr corr={out['G_rr_projected']['corr_G_GT']:.4f} antisym={out['G_rr_projected']['antisym_energy_frac']:.4f}",
      flush=True)

# ---- eps non-Gaussianity: encode a seed-fixed sample of training cells (encoder mu = exogenous latent) ----
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
eps_r_l, eps_p_l, seps_r_l = [], [], []
torch.manual_seed(0)
with torch.no_grad():
    for s in range(0, n_cells, 4096):
        e = min(s + 4096, n_cells)
        br = torch.FloatTensor(Xr_z[s:e]).to(DEVICE); bp = torch.FloatTensor(Xp_z[s:e]).to(DEVICE)
        mu_r, lv_r = model.enc_rna(br); mu_p, _ = model.enc_prot(bp)
        eps_r_l.append(mu_r.cpu().numpy()); eps_p_l.append(mu_p.cpu().numpy())
        seps_r_l.append(model.reparameterize(mu_r, lv_r).cpu().numpy())   # actual posterior draw
eps_r = np.concatenate(eps_r_l, 0).astype(np.float64)
eps_p = np.concatenate(eps_p_l, 0).astype(np.float64)
seps_r = np.concatenate(seps_r_l, 0).astype(np.float64)

def kurt_summary(M):
    exk = kurtosis(M, axis=0, fisher=True, nan_policy="omit")   # per-dim excess kurtosis (0=Gaussian)
    return {"mean_excess_kurtosis": float(np.nanmean(exk)),
            "median_excess_kurtosis": float(np.nanmedian(exk)),
            "p90_excess_kurtosis": float(np.nanpercentile(exk, 90)),
            "pooled_excess_kurtosis": float(kurtosis(M.reshape(-1), fisher=True, nan_policy="omit"))}
out["eps_r_mu_kurtosis"] = kurt_summary(eps_r)      # matches lingam_realdata_probe (encoder mu)
out["eps_p_mu_kurtosis"] = kurt_summary(eps_p)
out["eps_r_sampled_kurtosis"] = kurt_summary(seps_r)  # reparameterized draw (reflects posterior family)
out["n_cells"] = int(n_cells)
out["REF_kurtosis"] = {"gaussian": 0.0, "laplace": 3.0}
json.dump(out, open(os.path.join(rd, "eps_prior_symkurt.json"), "w"), indent=2)
print(f"[{run}] eps_r mu mean-exk={out['eps_r_mu_kurtosis']['mean_excess_kurtosis']:.2f} "
      f"median={out['eps_r_mu_kurtosis']['median_excess_kurtosis']:.2f} | "
      f"eps_r SAMPLED mean-exk={out['eps_r_sampled_kurtosis']['mean_excess_kurtosis']:.2f}", flush=True)
print(f"WROTE {os.path.join(rd, 'eps_prior_symkurt.json')}")
