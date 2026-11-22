"""Quick structure comparison of exogenous eps vs endogenous latent z (z = (I-Aᵀ)⁻¹ eps) in the top model.
Metrics: (1) per-dim 1-Wasserstein distance to a standard normal (after per-dim standardization -> pure
SHAPE/non-Gaussianity; lower = closer to Gaussian), (2) cell-type silhouette (structure/clustering;
higher = more cell-type structure). Question: is eps LESS structured (more noise-like, closer to normal,
lower silhouette) than z, as the SCM (z propagates eps through A) would predict?"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "analysis/common")
import torch, scanpy as sc
from scipy.stats import wasserstein_distance
from sklearn.metrics import silhouette_score
from model import CrossModalCausalLDVAE
import perturb_validation_analysis as P

RD = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "results/linfb_2c_fb10_b20_v4"; DEV = P.DEVICE
DATA = f"{_CMC_DATA}/combined_v4"
N = int(os.environ.get("N_CELLS", "40000")); rng = np.random.RandomState(42)
cfg = json.load(open(f"{RD}/config.json"))

ar = sc.read_h5ad(f"{DATA}/rna_combined.h5ad"); ap = sc.read_h5ad(f"{DATA}/citeseq_combined.h5ad")
idx = np.sort(rng.choice(ar.n_obs, min(N, ar.n_obs), replace=False))
Xr = ar.X[idx]; Xr = Xr.toarray() if hasattr(Xr, "toarray") else np.asarray(Xr)
Xp = ap.X[idx]; Xp = Xp.toarray() if hasattr(Xp, "toarray") else np.asarray(Xp)
ct = ar.obs["celltype"].astype(str).values[idx]
Zr = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-6); Zp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-6)

sd = torch.load(P.resolve_ckpt(RD), map_location="cpu")
NG, NP = sd["dec_rna.weight"].shape[0], sd["dec_prot.weight"].shape[0]; del sd
model = P.load_ctx_model(CrossModalCausalLDVAE, RD, NG, NP, cfg); model.eval()
lr_, lp_ = model.latent_r, model.latent_p
with torch.no_grad():
    er = model.enc_rna(torch.FloatTensor(Zr).to(DEV))[0]
    ep = model.enc_prot(torch.FloatTensor(Zp).to(DEV))[0]
    eps = torch.cat([er, ep], 1)
    A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
    A_full = torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0).to(DEV)
    z = model.scm._scm_solve(A_full, eps)
eps_r = eps[:, :lr_].cpu().numpy(); z_r = z[:, :lr_].cpu().numpy()
eps_j = eps.cpu().numpy(); z_j = z.cpu().numpy()

norm_ref = rng.randn(20000)
def w1_to_normal(M):
    # per-dim: standardize (robust: use median/MAD to resist the heavy-tail dims), W1 to N(0,1)
    out = []
    for j in range(M.shape[1]):
        x = M[:, j]; s = x.std()
        if s < 1e-9: continue
        xs = (x - x.mean()) / s
        out.append(wasserstein_distance(xs, norm_ref))
    return np.array(out)

def sil(M, labels, k=6000):
    Ms = (M - M.mean(0)) / (M.std(0) + 1e-9)   # z-score dims so silhouette compares STRUCTURE not scale
    return float(silhouette_score(Ms, labels, sample_size=min(k, len(labels)), random_state=0))

res = {"run": os.path.basename(RD), "n_cells": int(len(idx)), "latent_r": lr_, "latent_p": lp_, "n_celltypes": int(len(set(ct)))}
for name, E, Z in [("RNA (eps_r vs z_r)", eps_r, z_r), ("JOINT (eps vs z)", eps_j, z_j)]:
    we, wz = w1_to_normal(E), w1_to_normal(Z)
    se, sz = sil(E, ct), sil(Z, ct)
    res[name] = dict(w1_normal_eps_mean=float(we.mean()), w1_normal_eps_median=float(np.median(we)),
                     w1_normal_z_mean=float(wz.mean()), w1_normal_z_median=float(np.median(wz)),
                     silhouette_eps=se, silhouette_z=sz)
    print(f"\n[{name}]")
    print(f"  W1-to-normal (lower=more Gaussian):  eps mean {we.mean():.3f} median {np.median(we):.3f}  |  z mean {wz.mean():.3f} median {np.median(wz):.3f}")
    print(f"  cell-type silhouette (higher=more structure):  eps {se:+.3f}  |  z {sz:+.3f}")
OUT = os.environ.get("CMC_EPS_OUT") or f"{RD}/eps_vs_z_structure.json"
json.dump(res, open(OUT, "w"), indent=2)
print(f"\nWROTE {OUT}")
