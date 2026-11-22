"""Fit the ACTUAL CMC-VAE model class (src/model.py CrossModalCausalLDVAE + src/train.py loss_fn, headline
sp_dec_gl1000 loss weights and schedules) on data generated from a PLANTED latent block SCM + linear decoder,
then (i) measure observational orientation recovery and (ii) test whether interventional fine-tuning of the
latent graph with KNOWN targets (encoders/decoders frozen, as in the paper) recovers orientation.

Ground truth (source-by-target convention, A[j,i] = weight j->i, z = (I - A^T)^{-1} eps):
  * RNA latents dr, protein latents dp. A_rr = random DAG (<= ppn parents / node), W_rp = each protein latent
    has 1-2 RNA parents (RNA -> protein only, W_pr = 0), A_pp = sparse DAG. |w| ~ U(0.5, 0.9), random sign.
  * eps iid, unit variance: gaussian | laplace | t (df 8).
  * decoder: each gene/protein has ONE primary latent (marker), loading U(0.8,1.2)*sign, plus dense N(0, 0.15^2)
    loadings on all other latents of its modality. x = D z + N(0, (0.1 sd)^2); features standardized.
Fitted model: same latent dims as the truth; encoder 512/128 hidden (model default); 120 epochs, batch 256,
Adam 1e-3, warmup 20 / alpha ramp 30 / tau switch 30, beta 1e-4 -> 2, free bits 1.0/0.5, rec_weight 100,
lambda_stable 20 (rho<0.9), lambda_ortho 5, decoder group lasso (rna 1000, prot 18, x dec_sparse_w 0.05), no
latent sparsity (as in the final model).

Metrics (chance 0.5 unless stated):
  A_perm   : orientation of learned A on TRUE edges after Hungarian matching of learned to true latents
             (|corr(z_hat, z_true)| within modality): |A[pi(p),pi(c)]| > |A[pi(c),pi(p)]|.
  G_marker : orientation in feature space, G = D A D^T, on marker-gene pairs of true edges p->c:
             |G[a_p, a_c]| > |G[a_c, a_p]| (RR) and |G_rp[a_p, b_c]| > |G_pr[b_c, a_p]| (RNA->protein).
  LiNGAM_z : DirectLiNGAM (lingam package) causal order on the learned RNA latent z_hat (encoder means pushed
             through the SCM), order agreement on true RR edges after Hungarian matching.
  symmetry : corr(G_rr, G_rr^T) over off-diagonal pairs.
Interventional fine-tuning (known targets): KO of marker gene a_k = perfect intervention do(z_k = -2 sd) in
the truth (2000 cells / KO). Model operator = paper's decoder-guided operator (eq. 4): mutilate incoming edges
of RNA latents weighted by |D_r[a_k,:]| and shock the target gene to the knock-down value (-2 sd, the RC_SHOCK
'zero' analogue), solve, decode. Only the SCM (A gates/weights) is trained, loss = -Pearson + MSE on all
features. Held-out: KOs of 25% of RNA latents (never targeted in training). Read-outs before/after FT plus
held-out forward-vs-reverse (A_rr transposed) response correlation.
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
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import kurtosis

REPO = _CMC_ROOT
sys.path.insert(0, os.path.join(REPO, "src"))
from model import CrossModalCausalLDVAE          # noqa: E402
from train import loss_fn, ExperimentConfig      # noqa: E402
import lingam                                     # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- ground truth
def make_truth(dr, dp, gr, gp, ppn, rng):
    d = dr + dp
    A = np.zeros((d, d))
    order = rng.permutation(dr)
    for k, c in enumerate(order):
        if k == 0:
            continue
        pa = rng.choice(order[:k], size=min(ppn, k), replace=False)
        A[pa, c] = rng.uniform(0.5, 0.9, len(pa)) * rng.choice([-1, 1], len(pa))
    porder = rng.permutation(dp)
    for k, c in enumerate(porder):
        rp = rng.choice(dr, size=rng.integers(1, 3), replace=False)
        A[rp, dr + c] = rng.uniform(0.5, 0.9, len(rp)) * rng.choice([-1, 1], len(rp))
        if k > 0 and rng.random() < 0.5:
            pp = rng.choice(porder[:k])
            A[dr + pp, dr + c] = rng.uniform(0.5, 0.9) * rng.choice([-1, 1])

    def dec(G, L):
        D = rng.normal(0, 0.15, (G, L))
        prim = np.arange(G) % L
        rng.shuffle(prim)
        D[np.arange(G), prim] = rng.uniform(0.8, 1.2, G) * rng.choice([-1, 1], G)
        return D, prim
    Dr, prim_r = dec(gr, dr)
    Dp, prim_p = dec(gp, dp)
    return A, Dr, Dp, prim_r, prim_p


def gen_eps(fam, n, d, rng):
    if fam == "gaussian":
        return rng.standard_normal((n, d))
    if fam == "laplace":
        return rng.laplace(0, 1 / np.sqrt(2), (n, d))
    if fam == "t8":
        return rng.standard_t(8, (n, d)) / np.sqrt(8 / 6)
    raise ValueError(fam)


def simulate(A, Dr, Dp, eps, rng, noise=0.1, intervene=None):
    """intervene=(k, value): perfect intervention on latent k (incoming edges removed, z_k := value)."""
    d = A.shape[0]
    Ai = A.copy(); e = eps.copy()
    if intervene is not None:
        k, val = intervene
        Ai[:, k] = 0.0; e[:, k] = val
    Z = e @ np.linalg.inv(np.eye(d) - Ai.T).T          # row form of z = (I - A^T)^{-1} eps
    dr = Dr.shape[1]
    Xr = Z[:, :dr] @ Dr.T; Xp = Z[:, dr:] @ Dp.T
    return Z, Xr, Xp


def add_noise(X, sd_ref, rng, noise=0.1):
    return X + rng.standard_normal(X.shape) * noise * sd_ref


# ----------------------------------------------------------------------------- model fit
def headline_cfg(dr, dp, epochs):
    c = ExperimentConfig()
    c.latent_dim_rna, c.latent_dim_prot = dr, dp
    c.epochs, c.warmup_epochs, c.alpha_ramp_epochs, c.tau_switch_epoch = epochs, 20, 30, 30
    c.rec_weight, c.lambda_stable, c.stable_rho_bound = 100.0, 20.0, 0.9
    c.lambda_sparse_intra = c.lambda_sparse_inter = 0.01; c.scm_sparse_target = None
    c.lambda_ortho, c.dec_sparse_mode = 5.0, "group_lasso"
    c.lambda_dec_sparse, c.lambda_dec_sparse_rna, c.lambda_dec_sparse_prot = 0.001, 1000, 18
    c.dec_sparse_w_init = c.dec_sparse_w_final = 0.05
    c.n_decoder_groups = max(1, dr // 2)
    c.free_bits_r, c.free_bits_p = 1.0, 0.5
    c.beta_rna_init = c.beta_protein_init = 1e-4; c.beta_rna_final = c.beta_protein_final = 2.0
    c.dag_w_final = 0.0; c.prior_w_final = 0.0; c.tau_high, c.tau_low = 1.0, 0.5
    c.baseline = False
    return c


def fit_cmcvae(Xr, Xp, dr, dp, epochs, seed, log):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = headline_cfg(dr, dp, epochs)
    m = CrossModalCausalLDVAE(Xr.shape[1], Xp.shape[1], num_batches=1, latent_dim_rna=dr, latent_dim_prot=dp,
                              gate_mode="gumbel", rna_feature_dropout=0.1).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    xr = torch.as_tensor(Xr, dtype=torch.float32, device=DEV); xp = torch.as_tensor(Xp, dtype=torch.float32, device=DEV)
    n = xr.shape[0]; bs = 256
    for ep in range(epochs):
        params, alpha, tau = cfg.get_params(ep)
        m.train(); perm = torch.randperm(n, device=DEV); tot = 0.0
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            out = m(xr[idx], xp[idx], batch_idx=None, current_tau=tau, alpha=alpha)
            loss, _ = loss_fn(out, (xr[idx], xp[idx]), None, params)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += float(loss)
        if ep % 20 == 0 or ep == epochs - 1:
            log(f"   ep {ep:3d} loss {tot / (n // bs):.3f} alpha {alpha:.2f}")
    m.eval()
    return m


def assemble(m):
    A_rr, A_pp, W_rp, W_pr = m.scm.get_block_adjacencies(tau=0.5)
    return torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0)


@torch.no_grad()
def encode(m, Xr, Xp):
    xr = torch.as_tensor(Xr, dtype=torch.float32, device=DEV); xp = torch.as_tensor(Xp, dtype=torch.float32, device=DEV)
    mu_r, _ = m.enc_rna(xr); mu_p, _ = m.enc_prot(xp)
    e = torch.cat([mu_r, mu_p], 1)
    z = m.scm._scm_solve(assemble(m), e)
    zr, zp = torch.split(z, [m.latent_r, m.latent_p], 1)
    rec_r = zr @ m._dec_r_w().T + m.bias_rna; rec_p = m.dec_prot(zp) + m.bias_prot
    r2 = 1 - (((rec_r - xr) ** 2).mean() + ((rec_p - xp) ** 2).mean()) / (xr.var(0).mean() + xp.var(0).mean())
    return e.cpu().numpy(), z.cpu().numpy(), float(r2)


# ----------------------------------------------------------------------------- metrics
def hungarian(Zhat, Ztrue):
    C = np.abs(np.corrcoef(Zhat.T, Ztrue.T)[:Zhat.shape[1], Zhat.shape[1]:])
    C = np.nan_to_num(C)
    r, c = linear_sum_assignment(-C)
    pi = np.empty(Ztrue.shape[1], int); pi[c] = r            # pi[true] = learned
    return pi, float(C[r, c].mean())


def acc_of(fwd, rev):
    fwd, rev = np.asarray(fwd), np.asarray(rev)
    if len(fwd) == 0:
        return float("nan")
    return float(np.mean(np.where(fwd > rev, 1.0, np.where(fwd < rev, 0.0, 0.5))))


def edges_of(A, rows, cols):
    pp, cc = np.where(np.abs(A[np.ix_(rows, cols)]) > 0)
    return [(rows[p], cols[c]) for p, c in zip(pp, cc)]


def markers(D, prim, k, top=2):
    """top-`top` most specific features of latent k (largest |loading_k| / row-norm, primary == k)."""
    idx = np.where(prim == k)[0]
    spec = np.abs(D[idx, k]) / np.linalg.norm(D[idx], axis=1)
    return idx[np.argsort(-spec)[:top]]


def structure_metrics(Ahat, Drh, Dph, A, Dr, Dp, prim_r, prim_p, pi_r, pi_p, dr, dp, covered=None):
    out = {}
    pi = np.concatenate([pi_r, dr + pi_p])
    rr = edges_of(A, list(range(dr)), list(range(dr)))
    rp = edges_of(A, list(range(dr)), list(range(dr, dr + dp)))
    out["A_perm_rr"] = acc_of([abs(Ahat[pi[p], pi[c]]) for p, c in rr], [abs(Ahat[pi[c], pi[p]]) for p, c in rr])
    out["A_perm_rp"] = acc_of([abs(Ahat[pi[p], pi[c]]) for p, c in rp], [abs(Ahat[pi[c], pi[p]]) for p, c in rp])
    Arr, Wrp, Wpr = Ahat[:dr, :dr], Ahat[:dr, dr:], Ahat[dr:, :dr]
    Grr = Drh @ Arr @ Drh.T; Grp = Drh @ Wrp @ Dph.T; Gpr = Dph @ Wpr @ Drh.T
    f, r = [], []
    for p, c in rr:
        for a in markers(Dr, prim_r, p):
            for b in markers(Dr, prim_r, c):
                f.append(abs(Grr[a, b])); r.append(abs(Grr[b, a]))
    out["G_marker_rr"] = acc_of(f, r)
    if covered is not None:   # edges whose SOURCE latent was a training intervention target vs not
        for tag, sel in (("covered", True), ("uncovered", False)):
            ff, rr_ = [], []
            for p, c in rr:
                if (p in covered) != sel:
                    continue
                for a in markers(Dr, prim_r, p):
                    for b in markers(Dr, prim_r, c):
                        ff.append(abs(Grr[a, b])); rr_.append(abs(Grr[b, a]))
            out[f"G_marker_rr_{tag}"] = acc_of(ff, rr_)
            sel_e = [(p, c) for p, c in rr if (p in covered) == sel]
            out[f"A_perm_rr_{tag}"] = acc_of([abs(Ahat[pi[p], pi[c]]) for p, c in sel_e], [abs(Ahat[pi[c], pi[p]]) for p, c in sel_e])
    f, r = [], []
    for p, c in rp:
        for a in markers(Dr, prim_r, p):
            for b in markers(Dp, prim_p, c - dr):
                f.append(abs(Grp[a, b])); r.append(abs(Gpr[b, a]))
    out["G_marker_rp"] = acc_of(f, r)
    iu = np.triu_indices(Grr.shape[0], 1)
    out["corr_Grr_GrrT"] = float(np.corrcoef(Grr[iu], Grr.T[iu])[0, 1])
    off = ~np.eye(dr, dtype=bool)
    out["corr_Arr_ArrT"] = float(np.corrcoef(Arr[off], Arr.T[off])[0, 1])
    out["n_edges_rr"], out["n_edges_rp"] = len(rr), len(rp)
    return out


def lingam_order_acc(Z, edges, pi=None):
    mdl = lingam.DirectLiNGAM(); mdl.fit(Z)
    pos = {v: i for i, v in enumerate(mdl.causal_order_)}
    if pi is None:
        pi = np.arange(Z.shape[1])
    ok = [pos[pi[p]] < pos[pi[c]] for p, c in edges]
    return float(np.mean(ok)) if ok else float("nan")


# ----------------------------------------------------------------------------- interventional fine-tuning
def ko_truth(A, Dr, Dp, fam, k, n, rng, mu_r, sd_r, mu_p, sd_p, sdz, Xr_ctrl, Xp_ctrl, nsd_r, nsd_p):
    e = gen_eps(fam, n, A.shape[0], rng)
    _, Xr, Xp = simulate(A, Dr, Dp, e, rng, intervene=(k, -2.0 * sdz[k]))
    Xr = add_noise(Xr, nsd_r, rng); Xp = add_noise(Xp, nsd_p, rng)
    Xr = (Xr - mu_r) / sd_r; Xp = (Xp - mu_p) / sd_p
    return np.concatenate([Xr.mean(0) - Xr_ctrl.mean(0), Xp.mean(0) - Xp_ctrl.mean(0)])


def predict_ko(m, A_full, e, gidx, val, rec_c=None):
    lr_, lp_ = m.latent_r, m.latent_p
    Wr = m._dec_r_w()
    if rec_c is None:
        z = m.scm._scm_solve(A_full, e); zr, zp = torch.split(z, [lr_, lp_], 1)
        rec_c = (zr @ Wr.T + m.bias_rna, m.dec_prot(zp) + m.bias_prot)
    rr, rp = rec_c
    dx = torch.zeros_like(rr); dx[:, gidx] = val - rr[:, gidx]
    deps = torch.cat([dx @ Wr, torch.zeros(e.shape[0], lp_, device=e.device)], 1)
    s = (Wr[gidx].abs() / (Wr[gidx].abs().max() + 1e-8)).clamp(0, 1)
    Am = A_full.clone(); Am[:, :lr_] = Am[:, :lr_] * (1 - s).unsqueeze(0)
    z2 = m.scm._scm_solve(Am, e + deps); zr2, zp2 = torch.split(z2, [lr_, lp_], 1)
    return torch.cat([((zr2 @ Wr.T + m.bias_rna) - rr).mean(0), ((m.dec_prot(zp2) + m.bias_prot) - rp).mean(0)])


def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


def finetune(m, e, kos, steps, lr, log):
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.scm.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in m.scm.parameters()], lr=lr, weight_decay=1e-4)
    for st in range(steps):
        A = assemble(m); L = 0.0
        for _, gidx, val, obs, _ in kos:
            pr = predict_ko(m, A, e, gidx, val)
            L = L + (-pear(pr, obs) + ((pr - obs) ** 2).mean())
        L = L / len(kos)
        rho = torch.linalg.svdvals(A)[0]
        L = L + 10.0 * torch.relu(rho - 0.9) ** 2
        opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(m.scm.parameters(), 1.0); opt.step()
        if st % 100 == 0 or st == steps - 1:
            log(f"   ft step {st} loss {float(L):.4f} rho {float(rho):.3f}")


@torch.no_grad()
def ko_preds(m, e, kos, transpose_rr=False):
    A = assemble(m)
    if transpose_rr:
        lr_ = m.latent_r; A = A.clone(); A[:lr_, :lr_] = A[:lr_, :lr_].T
    return np.array([predict_ko(m, A, e, g, v).cpu().numpy() for _, g, v, _, _ in kos])


@torch.no_grad()
def ko_scores(m, e, kos, ref, transpose_rr=False):
    """kos/ref: (latent k, gene idx, knock-down value, observed response, downstream mask); ref = TRAINING KOs.
    pearDE        : all features (includes the target latent's own genes = trivial self-effect)
    pearDE_down   : only features NOT primarily loaded on the target latent (descendant effects: direction matters)
    pearDE_cent   : downstream features after subtracting the TRAINING-KO mean response from prediction and
                    observation (shared-response control, as in the paper)
    meanresp      : model-free reference: training-KO mean observed response vs observed (downstream features)
    retrieval     : latent-level, centred, downstream features common to all KOs: own observed response vs
                    responses of KOs of OTHER latents (chance 0.5)"""
    P = ko_preds(m, e, kos, transpose_rr); Pref = ko_preds(m, e, ref, transpose_rr)
    O = np.array([o.cpu().numpy() for _, _, _, o, _ in kos]); Oref = np.array([o.cpu().numpy() for _, _, _, o, _ in ref])
    K = np.array([k for k, *_ in kos]); M = np.array([mk for *_, mk in kos])
    mp, mo = Pref.mean(0), Oref.mean(0)
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    pe = [c(p, o) for p, o in zip(P, O)]
    pd = [c(p[mk], o[mk]) for p, o, mk in zip(P, O, M)]
    pc = [c((p - mp)[mk], (o - mo)[mk]) for p, o, mk in zip(P, O, M)]
    mr = [c(mo[mk], o[mk]) for o, mk in zip(O, M)]
    common = M.all(0); ret = []
    for i in range(len(P)):
        own = c((P[i] - mp)[common], (O[i] - mo)[common])
        oth = [c((P[i] - mp)[common], (O[j] - mo)[common]) for j in range(len(P)) if K[j] != K[i]]
        if oth:
            ret.append(float(np.mean([1.0 if own > x else 0.5 if own == x else 0.0 for x in oth])))
    nm = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
    return dict(pearDE=nm(pe), pearDE_down=nm(pd), pearDE_cent=nm(pc), meanresp=nm(mr), retrieval=nm(ret), n=len(P))


# ----------------------------------------------------------------------------- main
def run(fam, seed, dr, dp, gr, gp, n, epochs, ft_steps, log):
    rng = np.random.default_rng(seed)
    A, Dr, Dp, prim_r, prim_p = make_truth(dr, dp, gr, gp, 2, rng)
    eps = gen_eps(fam, n, dr + dp, rng)
    Z, Xr0, Xp0 = simulate(A, Dr, Dp, eps, rng)
    sd_r0, sd_p0 = Xr0.std(0), Xp0.std(0)
    Xr0 = add_noise(Xr0, sd_r0, rng); Xp0 = add_noise(Xp0, sd_p0, rng)
    mu_r, sd_r, mu_p, sd_p = Xr0.mean(0), Xr0.std(0), Xp0.mean(0), Xp0.std(0)
    Xr = (Xr0 - mu_r) / sd_r; Xp = (Xp0 - mu_p) / sd_p
    rec = dict(family=fam, seed=seed, dr=dr, dp=dp, genes=gr, proteins=gp, n=n, epochs=epochs,
               eps_excess_kurtosis=float(np.mean(kurtosis(eps, axis=0))))
    rr_edges = edges_of(A, list(range(dr)), list(range(dr)))
    # oracle references on the same data
    rec["oracle_lingam_true_z"] = lingam_order_acc(Z[:, :dr], rr_edges)
    Drs = Dr / sd_r[:, None]
    rec["oracle_lingam_true_decoder"] = lingam_order_acc(Xr @ np.linalg.pinv(Drs).T, rr_edges)
    U, S, Vt = np.linalg.svd(Xr - Xr.mean(0), full_matrices=False)
    Zpca = U[:, :dr] * S[:dr]
    pi_pca, _ = hungarian(Zpca, Z[:, :dr])
    rec["lingam_pca_latent"] = lingam_order_acc(Zpca, rr_edges, pi_pca)

    t0 = time.time()
    m = fit_cmcvae(Xr, Xp, dr, dp, epochs, seed, log)
    rec["fit_seconds"] = time.time() - t0
    e_hat, z_hat, r2 = encode(m, Xr, Xp)
    rec["recon_r2"] = r2
    pi_r, mc_r = hungarian(z_hat[:, :dr], Z[:, :dr]); pi_p, mc_p = hungarian(z_hat[:, dr:], Z[:, dr:])
    rec["latent_match_absr_rna"], rec["latent_match_absr_prot"] = mc_r, mc_p
    Ahat = assemble(m).detach().cpu().numpy()
    Drh = m._dec_r_w().detach().cpu().numpy(); Dph = m.dec_prot.weight.detach().cpu().numpy()
    held_lat = set(rng.choice(dr, max(2, dr // 4), replace=False).tolist())
    covered = set(range(dr)) - held_lat
    rec["obs"] = structure_metrics(Ahat, Drh, Dph, A, Dr / sd_r[:, None], Dp / sd_p[:, None], prim_r, prim_p,
                                   pi_r, pi_p, dr, dp, covered)
    rec["obs"]["lingam_learned_z"] = lingam_order_acc(z_hat[:, :dr], rr_edges, pi_r)
    rec["obs"]["lingam_learned_eps"] = lingam_order_acc(e_hat[:, :dr], rr_edges, pi_r)
    log(f" [{fam} s{seed} dr{dr}] R2 {r2:.3f} match {mc_r:.2f} | A_perm_rr {rec['obs']['A_perm_rr']:.2f} "
        f"G_rr {rec['obs']['G_marker_rr']:.2f} G_rp {rec['obs']['G_marker_rp']:.2f} sym {rec['obs']['corr_Grr_GrrT']:.2f} "
        f"LiNGAM(z_hat) {rec['obs']['lingam_learned_z']:.2f} | oracle z {rec['oracle_lingam_true_z']:.2f} "
        f"dec {rec['oracle_lingam_true_decoder']:.2f} pca {rec['lingam_pca_latent']:.2f}")

    # ---------------- interventional fine-tuning with known targets
    sdz = Z.std(0)
    ctrl_idx = rng.choice(n, 2000, replace=False)
    Xr_c, Xp_c = Xr[ctrl_idx], Xp[ctrl_idx]
    kos_all = []
    for k in range(dr):
        for g in markers(Dr, prim_r, k):
            obs = ko_truth(A, Dr, Dp, fam, k, 2000, rng, mu_r, sd_r, mu_p, sd_p, sdz, Xr_c, Xp_c, sd_r0, sd_p0)
            val = float(Xr_c[:, g].mean() + obs[g])     # known knock-down level of the target gene
            mask = np.concatenate([prim_r != k, np.ones(gp, bool)])
            kos_all.append((k, int(g), val, torch.as_tensor(obs, dtype=torch.float32, device=DEV), mask))
    e_c = torch.as_tensor(e_hat[ctrl_idx], dtype=torch.float32, device=DEV)
    SCM0 = {k: v.detach().clone() for k, v in m.scm.state_dict().items()}
    # regime 'partial': KOs of 75% of RNA latents (both marker genes); test = KOs of the held-out 25% latents
    # regime 'all'    : one marker-gene KO for EVERY RNA latent (the one-intervention-per-node condition of
    #                   Squires et al. 2023); test = the other marker gene of each latent (unseen genes)
    regimes = {"partial": ([x for x in kos_all if x[0] not in held_lat], [x for x in kos_all if x[0] in held_lat], covered),
               "all": (kos_all[0::2], kos_all[1::2], set(range(dr)))}
    rec["ft"] = {}
    for rg, (tr, te, cov) in regimes.items():
        m.scm.load_state_dict(SCM0)
        s = dict(n_train_kos=len(tr), n_test_kos=len(te), covered_latents=sorted(cov))
        s["before_test"] = ko_scores(m, e_c, te, tr); s["before_test_rev"] = ko_scores(m, e_c, te, tr, True)
        s["before_train"] = ko_scores(m, e_c, tr, tr)
        s["before_structure"] = structure_metrics(Ahat, Drh, Dph, A, Dr / sd_r[:, None], Dp / sd_p[:, None],
                                                  prim_r, prim_p, pi_r, pi_p, dr, dp, cov)
        finetune(m, e_c, tr, ft_steps, 3e-3, log)
        Aft = assemble(m).detach().cpu().numpy()
        s["after_structure"] = structure_metrics(Aft, Drh, Dph, A, Dr / sd_r[:, None], Dp / sd_p[:, None],
                                                 prim_r, prim_p, pi_r, pi_p, dr, dp, cov)
        with torch.no_grad():
            z_ft = m.scm._scm_solve(assemble(m), torch.as_tensor(e_hat, dtype=torch.float32, device=DEV)).cpu().numpy()
        s["after_structure"]["lingam_learned_z"] = lingam_order_acc(z_ft[:, :dr], rr_edges, pi_r)
        s["after_test"] = ko_scores(m, e_c, te, tr); s["after_test_rev"] = ko_scores(m, e_c, te, tr, True)
        s["after_train"] = ko_scores(m, e_c, tr, tr); s["after_train_rev"] = ko_scores(m, e_c, tr, tr, True)
        rec["ft"][rg] = s
        log(f" [{fam} s{seed} {rg}] FT: G_rr {s['before_structure']['G_marker_rr']:.2f}->{s['after_structure']['G_marker_rr']:.2f} "
            f"G_rp {s['before_structure']['G_marker_rp']:.2f}->{s['after_structure']['G_marker_rp']:.2f} "
            f"A_perm {s['before_structure']['A_perm_rr']:.2f}->{s['after_structure']['A_perm_rr']:.2f} | test down "
            f"{s['before_test']['pearDE_down']:.2f}->{s['after_test']['pearDE_down']:.2f} (rev {s['after_test_rev']['pearDE_down']:.2f}) "
            f"cent {s['before_test']['pearDE_cent']:.2f}->{s['after_test']['pearDE_cent']:.2f} meanresp {s['after_test']['meanresp']:.2f} "
            f"retr {s['before_test']['retrieval']:.2f}->{s['after_test']['retrieval']:.2f}")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="gaussian,laplace,t8")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--dr", type=int, default=8); ap.add_argument("--dp", type=int, default=4)
    ap.add_argument("--genes", type=int, default=200); ap.add_argument("--proteins", type=int, default=40)
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ft_steps", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T0 = time.time()
    def log(*x): print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)
    log("device", DEV)
    recs = []
    for fam in a.families.split(","):
        for s in [int(x) for x in a.seeds.split(",")]:
            recs.append(run(fam, s, a.dr, a.dp, a.genes, a.proteins, a.n, a.epochs, a.ft_steps, log))
            json.dump(recs, open(a.out, "w"), indent=1)
    log("done", a.out)
