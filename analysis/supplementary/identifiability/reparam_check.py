"""Numerical check of Proposition 1 on the fitted headline model (sp_dec_gl1000, seed 42, final_model.pt).

Block-diagonal reparametrisation: R = diag(R_r, I), D' = D R^{-1}, I - A'^T = (I - A^T) R^{-1}.
For a target RNA block A~ (zero diagonal) choose R_r = (I - A~^T)^{-1} (I - A_rr^T); then A'_rr = A~,
W'_rp = R_r^{-T} W_rp, W'_pr = W_pr, A'_pp = A_pp, D'_r = D_r R_r^{-1}.
Encoders (hence q(eps|x), the KL term and any prior on eps) are untouched; the reconstruction D' z' must equal
D z for every eps. Cases: A~ = A_rr^T (reversed latent GRN) and A~ = 0 (latent GRN absorbed into the decoder).
Also reports what is NOT invariant: decoder-orthogonality penalty, spectral norm, G_rr, and the decoder-guided
do-operator prediction (so interventional data can discriminate between likelihood-equivalent parameters).
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
import json, os, sys
import numpy as np
import torch
REPO = f"{_CMC_ROOT}"
sys.path.insert(0, f"{REPO}/src")
from model import CrossModalCausalLDVAE  # noqa
torch.set_default_dtype(torch.float64)

RD = f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000"
OUT = sys.argv[1]
sd = torch.load(f"{RD}/final_model.pt", map_location="cpu")
cfg = json.load(open(f"{RD}/config.json"))
m = CrossModalCausalLDVAE(sd["dec_rna.weight"].shape[0], sd["dec_prot.weight"].shape[0], 1,
                          latent_dim_rna=cfg["latent_dim_rna"], latent_dim_prot=cfg["latent_dim_prot"])
m.load_state_dict(sd, strict=False); m.eval(); m.double()
lr_, lp_ = m.latent_r, m.latent_p; d = lr_ + lp_
with torch.no_grad():
    A_rr, A_pp, W_rp, W_pr = m.scm.get_block_adjacencies(tau=0.5)
    A = torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0)
    Dr = m._dec_r_w().clone(); Dp = m.dec_prot.weight.clone()
I = torch.eye(d); Ir = torch.eye(lr_)
g = torch.Generator().manual_seed(0)
eps = torch.randn(4000, d, generator=g)


def recon(Amat, Dr_, Dp_, e):
    z = torch.linalg.solve(I - Amat.T, e.T).T
    return torch.cat([z[:, :lr_] @ Dr_.T, z[:, lr_:] @ Dp_.T], 1)


def ortho(W):
    G = W.T @ W; return float(torch.norm(G - torch.eye(G.shape[0])) ** 2 / G.numel())


def do_pred(Amat, Dr_, Dp_, e, gidx):
    """paper operator (eq. 4, shock to the training mean): mutilate + shock along the decoder row of gene g."""
    rec = recon(Amat, Dr_, Dp_, e); rr = rec[:, :Dr_.shape[0]]
    dx = torch.zeros_like(rr); dx[:, gidx] = -rr[:, gidx]
    deps = torch.cat([dx @ Dr_, torch.zeros(e.shape[0], lp_)], 1)
    s = (Dr_[gidx].abs() / Dr_[gidx].abs().max()).clamp(0, 1)
    Am = Amat.clone(); Am[:, :lr_] = Am[:, :lr_] * (1 - s)
    return (recon(Am, Dr_, Dp_, e + deps) - rec).mean(0)


def sym(M):
    iu = np.triu_indices(M.shape[0], 1); M = M.numpy(); return float(np.corrcoef(M[iu], M.T[iu])[0, 1])


X0 = recon(A, Dr, Dp, eps)
G0 = Dr @ A_rr @ Dr.T
res = dict(model="sp_dec_gl1000 seed42 final_model.pt", n_eps=eps.shape[0],
           original=dict(ortho_rna=ortho(Dr), spectral_norm=float(torch.linalg.svdvals(A)[0]),
                         spectral_radius=float(torch.linalg.eigvals(A).abs().max()), corr_Grr_GrrT=sym(G0),
                         corr_Arr_ArrT=float(np.corrcoef(A_rr.numpy()[~np.eye(lr_, dtype=bool)], A_rr.T.numpy()[~np.eye(lr_, dtype=bool)])[0, 1])))
genes = [int(i) for i in torch.argsort(Dr.norm(dim=1), descending=True)[:20]]
P0 = [do_pred(A, Dr, Dp, eps[:1000], gi) for gi in genes]
_U = torch.triu(A_rr + A_rr.T, 1)   # DAG with every edge oriented low->high index (same skeleton/weights magnitude)
for name, At in [("reversed_Arr", A_rr.T.clone()), ("zero_Arr", torch.zeros_like(A_rr)),
                 ("DAG_forward_order", _U.clone()), ("DAG_reverse_order", _U.T.clone())]:
    Rr = torch.linalg.solve(Ir - At.T, Ir - A_rr.T)          # R_r = (I - A~^T)^{-1} (I - A_rr^T)
    Rr_inv = torch.linalg.inv(Rr)
    Dr2 = Dr @ Rr_inv
    S2 = (I - A.T).clone(); S2[:, :lr_] = S2[:, :lr_] @ Rr_inv  # (I - A^T) R^{-1}, R = diag(R_r, I)
    A2 = (I - S2).T
    X2 = recon(A2, Dr2, Dp, eps)
    G2 = Dr2 @ A2[:lr_, :lr_] @ Dr2.T
    P2 = [do_pred(A2, Dr2, Dp, eps[:1000], gi) for gi in genes]
    off = ~np.eye(lr_, dtype=bool)
    res[name] = dict(
        max_abs_recon_diff=float((X2 - X0).abs().max()), rel_recon_diff=float((X2 - X0).norm() / X0.norm()),
        max_abs_diag_A2_rr=float(A2[:lr_, :lr_].diag().abs().max()), max_abs_diag_A2_pp=float(A2[lr_:, lr_:].diag().abs().max()),
        A2_rr_equals_target=float((A2[:lr_, :lr_] - At).abs().max()),
        W2_pr_unchanged=float((A2[lr_:, :lr_] - W_pr).abs().max()), A2_pp_unchanged=float((A2[lr_:, lr_:] - A_pp).abs().max()),
        rank_Wrp=[int(torch.linalg.matrix_rank(W_rp)), int(torch.linalg.matrix_rank(A2[:lr_, lr_:]))],
        ortho_rna=ortho(Dr2), spectral_norm=float(torch.linalg.svdvals(A2)[0]),
        spectral_radius=float(torch.linalg.eigvals(A2).abs().max()),
        corr_Grr_GrrT=sym(G2) if At.abs().sum() > 0 else None,
        corr_G2_G0=float(np.corrcoef(G2.numpy()[np.triu_indices(G2.shape[0], 1)], G0.numpy()[np.triu_indices(G0.shape[0], 1)])[0, 1]) if At.abs().sum() > 0 else None,
        corr_G2_G0T=float(np.corrcoef(G2.numpy().ravel(), G0.T.numpy().ravel())[0, 1]) if At.abs().sum() > 0 else None,
        doop_pred_corr_vs_original_median=float(np.median([np.corrcoef(a.numpy(), b.numpy())[0, 1] for a, b in zip(P0, P2)])),
        doop_pred_corr_vs_original_range=[float(np.min([np.corrcoef(a.numpy(), b.numpy())[0, 1] for a, b in zip(P0, P2)])),
                                          float(np.max([np.corrcoef(a.numpy(), b.numpy())[0, 1] for a, b in zip(P0, P2)]))],
        cond_Rr=float(torch.linalg.cond(Rr)))
    print(name, json.dumps(res[name], indent=1))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res["original"], indent=1))
