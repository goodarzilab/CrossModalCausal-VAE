#!/usr/bin/env python
"""Is the exogenous eps elliptical / rotation-invariant?  ICA/LiNGAM orientation needs non-Gaussianity
that is COORDINATE-ALIGNED (non-rotation-invariant). An isotropic-elliptical eps (e.g. isotropic
Student-t) has heavy-tailed marginals yet is rotation-invariant -> Q.eps ~ eps for any rotation Q, so
NO ICA/LiNGAM method can orient it, regardless of kurtosis.
Test: whiten eps, then compare marginal excess kurtosis along COORDINATE AXES vs along Haar-RANDOM
directions.  ratio = median(k_random)/median(k_axis):
  ratio ~ 1              => rotation-invariant / elliptical  => non-Gaussianity is causally useless
  ratio ~ 3/(d+2)        => coordinate-aligned independent   => ICA-identifiable in principle
Calibrated against synthetic independent-Laplace, isotropic-Student-t, and Gaussian at matched (n,d).
Writes results/eps_elliptical_test.json.  Usage: python analysis/supplementary/identifiability/eps_elliptical_test.py [results_dir] [n_cells]
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

rd = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
     f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000"
n_cells = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
cfg = json.load(open(os.path.join(rd, "config.json")))
rng = np.random.default_rng(0)

def whiten(X):
    Xc = X - X.mean(0)
    C = np.cov(Xc, rowvar=False)
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-8, None)
    return Xc @ (V @ np.diag(1.0 / np.sqrt(w)) @ V.T)

def dir_kurt(Xw, dirs):
    return kurtosis(Xw @ dirs.T, axis=0, fisher=True, nan_policy="omit")

def rand_dirs(d, k):
    G = rng.standard_normal((k, d)); return G / np.linalg.norm(G, axis=1, keepdims=True)

def report(X, name, K=512):
    Xw = whiten(np.asarray(X, float)); d = Xw.shape[1]
    kax = kurtosis(Xw, axis=0, fisher=True, nan_policy="omit")
    krand = dir_kurt(Xw, rand_dirs(d, K))
    ma, mr = float(np.nanmedian(kax)), float(np.nanmedian(krand))
    ratio = mr / ma if ma > 1e-6 else float("nan")
    theo = 3.0 / (d + 2)     # coordinate-aligned independent expectation
    o = {"name": name, "d": int(d), "n": int(Xw.shape[0]),
         "median_axis_kurt": ma, "median_random_kurt": mr,
         "p90_axis_kurt": float(np.nanpercentile(kax, 90)),
         "max_axis_kurt": float(np.nanmax(kax)),
         "ratio_random_over_axis": ratio, "coord_aligned_expected_ratio": theo}
    print(f"  {name:26s} d={d:3d}  axis-kurt(med)={ma:7.3f}  rand-kurt(med)={mr:7.3f}  "
          f"ratio={ratio:6.3f}  [coord-aligned~{theo:.3f}, elliptical~1.0]", flush=True)
    return o

# ---------- synthetic calibration ----------
print("=== synthetic calibration (n=40000) ===", flush=True)
N = 40000
cal = {}
for d in (128, 192):
    lap = rng.laplace(size=(N, d))                                   # independent heavy-tailed (coord-aligned)
    nu = 4.0
    z = rng.standard_normal((N, d)); g = rng.chisquare(nu, size=(N, 1)) / nu
    ist = z / np.sqrt(g)                                             # isotropic multivariate Student-t (elliptical)
    gau = rng.standard_normal((N, d))
    cal[f"laplace_indep_d{d}"] = report(lap, f"Laplace-indep (d={d})")
    cal[f"isotropic_t_d{d}"]   = report(ist, f"isotropic-t nu4 (d={d})")
    cal[f"gaussian_d{d}"]      = report(gau, f"Gaussian (d={d})")

# ---------- real eps from the headline model ----------
print("\n=== real eps (encode training cells) ===", flush=True)
r = sc.read_h5ad(TRAIN_RNA, backed="r"); n_g = r.n_vars
p = sc.read_h5ad(TRAIN_PROT, backed="r"); n_p = p.n_vars
r.file.close(); p.file.close()
model = P.load_ctx_model(CrossModalCausalLDVAE, rd, n_g, n_p, cfg); model.eval()
model.eps_prior = cfg.get("eps_prior", "gaussian"); model.eps_df = float(cfg.get("eps_df", 3.0))
rr = sc.read_h5ad(TRAIN_RNA, backed="r"); pp = sc.read_h5ad(TRAIN_PROT, backed="r")
assert (rr.obs_names == pp.obs_names).all()
Nc = rr.n_obs
rna_mean, rna_std = P._chunked_mean_std(rr); prot_mean, prot_std = P._chunked_mean_std(pp)
n_cells = min(n_cells, Nc)
srng = np.random.default_rng(42); idx = np.sort(srng.choice(Nc, n_cells, replace=False))
Xr = rr[idx].to_memory().X; Xp = pp[idx].to_memory().X
Xr = Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr)
Xp = Xp.toarray() if sp.issparse(Xp) else np.asarray(Xp)
rr.file.close(); pp.file.close()
Xr_z = (Xr - rna_mean) / (rna_std + 1e-6); Xp_z = (Xp - prot_mean) / (prot_std + 1e-6)
er, ep = [], []
torch.manual_seed(0)
with torch.no_grad():
    for s in range(0, n_cells, 4096):
        e = min(s + 4096, n_cells)
        br = torch.FloatTensor(Xr_z[s:e]).to(DEVICE); bp = torch.FloatTensor(Xp_z[s:e]).to(DEVICE)
        mu_r, _ = model.enc_rna(br); mu_p, _ = model.enc_prot(bp)
        er.append(mu_r.cpu().numpy()); ep.append(mu_p.cpu().numpy())
eps_r = np.concatenate(er, 0).astype(np.float64); eps_p = np.concatenate(ep, 0).astype(np.float64)
eps_full = np.concatenate([eps_r, eps_p], 1)
real = {"eps_r": report(eps_r, "REAL eps_r (RNA)"),
        "eps_full": report(eps_full, "REAL eps_full [r;p]"),
        "eps_p": report(eps_p, "REAL eps_p (prot)")}

out = {"run": os.path.basename(rd.rstrip("/")), "n_cells": int(n_cells),
       "synthetic_calibration": cal, "real": real,
       "interpretation": "ratio~1 => elliptical/rotation-invariant (ICA cannot orient); "
                          "ratio~3/(d+2) => coordinate-aligned independent (ICA-identifiable in principle)"}
op = f"{_CMC_RESULTS}/eps_elliptical_test.json"
json.dump(out, open(op, "w"), indent=2)
print(f"\nWROTE {op}", flush=True)
