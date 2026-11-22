"""adapt workstream: control-cell (unperturbed, unlabelled) domain adaptation of a PBMC-trained CMC-VAE to each screen.

Domains
  frangieh / iccite (Chen) / arce : the screen's control cells from the pseudobulk-DE object (pb_<t>.pkl -> ct.cells_r/p;
                                    identical cells to those used by the control-cell ridge in code/multimodal/mm_translate.py).
                                    RNA and protein inputs standardised with the TRAINING-corpus mean/sd (as every harness);
                                    vocabulary features not measured in the screen = 0 (training mean) and masked from the loss.
  gw8 / gw48                      : RNA side = GWCD4i NTC pseudobulks (keep_min_cells) of Stim8hr / Stim48hr (log-normalised,
                                    training-standardised; 419 unmeasured vocab genes = 0, masked) ; protein side = SCITO NTC
                                    cells (20,000 random 'non-targeting' cells). SCITO values are the authors' per-batch deviance
                                    residuals z-scored per protein (NOT CLR): they are used AS the standardised protein input
                                    (both are unit-variance per protein; per-protein location/scale offsets are absorbed by the
                                    adapted decoder bias/encoder). Unpaired -> RNA loss on GWCD4i batches, protein loss on SCITO
                                    batches, no routing loss (no cell has both modalities).
Split: 10% of the domain's control units held out (RandomState(12345), seed-independent) for early stopping and diagnostics;
the control-cell ridge baselines are fitted on the same 90%.

Variants (pre-registered, PREREGISTRATION.md)
  i      encoders + decoders + A (all SCM parameters); objective = final-epoch observational objective + routing loss
  ii     encoders + decoders, A frozen
  iii    minimal: feature-wise affine only = BN affine (gamma, beta) of both encoders (hidden-unit scale-shift; running
         statistics frozen) + per-feature decoder scale & bias (scale folded into the decoder rows at save time)
  i_xm1  sensitivity: variant i with lambda_xm = 1.0
Unmeasured-feature decoder rows / biases never change (gradient masked).
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
import os, sys, json, copy, glob, pickle, time, hashlib
import numpy as np, torch
REPO = f"{_CMC_ROOT}"
sys.path.insert(0, os.path.join(_CMC_ROOT, "analysis", "common")); sys.path.insert(0, os.path.join(REPO, "src"))
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE

OUT = f"{_CMC_SCRATCH}/nov_adapt"
DATA = f"{_CMC_DATA}"
XM = f"{_CMC_SCRATCH}/nov_xmodal/models"
BASES = {"sp": {"s42": f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000", "s1": f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000_s1",
                "s2": f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000_s2"},
         "xm": {"s42": f"{XM}/xm_l0.03", "s1": f"{XM}/xm_l0.03_s1", "s2": f"{XM}/xm_l0.03_s2"}}
SEEDNUM = {"s42": 42, "s1": 1, "s2": 2}
CITE = ["frangieh", "iccite", "arce"]
HP = dict(lr=1e-4, batch=256, max_steps=4000, eval_every=50, patience=10, lam_xm=0.1, rna_feature_dropout=0.1,
          beta_r=2.0, beta_p=2.0, free_bits_r=1.0, free_bits_p=0.5, rec_weight=100.0, lambda_stable=20.0, rho_bound=0.9,
          lambda_ortho=5.0, lambda_gl_rna=1000 * 0.05, lambda_gl_prot=18 * 0.05, n_groups=16, grad_clip=1.0, val_frac=0.1,
          scito_ntc_cells=20000)
VARIANTS = {"i": dict(enc=True, dec=True, A=True, lam_xm=HP["lam_xm"]),
            "ii": dict(enc=True, dec=True, A=False, lam_xm=HP["lam_xm"]),
            "iii": dict(minimal=True, lam_xm=HP["lam_xm"], lr=1e-3),   # few parameters -> 10x learning rate
            "i_xm1": dict(enc=True, dec=True, A=True, lam_xm=1.0)}
DEV = P.DEVICE
if os.environ.get("AD_SMOKE") == "1":   # mechanics test only (outputs to OUT/smoke, deleted afterwards)
    HP.update(max_steps=60, eval_every=20); OUT = OUT + "/smoke"; os.makedirs(OUT + "/diag", exist_ok=True); os.makedirs(OUT + "/models", exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

def align_of(tag):
    f = sorted(glob.glob(f"{_CMC_RESULTS}/_ftcache/rc_{tag}_sp_dec_gl1000_3a5d257c9323_0.1_0.05.pkl"))[0]
    return pickle.load(open(f, "rb"))["align"]

def split_idx(n):
    prm = np.random.RandomState(12345).permutation(n); nv = int(round(HP["val_frac"] * n))
    return np.sort(prm[nv:]), np.sort(prm[:nv])

# ------------------------------------------------------------------ domain data
def screen_data(tag):
    pb = pickle.load(open(f"{_CMC_RESULTS}/paper/analyses/stats/de/pb_{tag}.pkl", "rb")); al = align_of(tag)
    assert list(pb["common_genes"]) == list(al["common_genes"])
    reti = np.asarray(al["rna_eval_train_indices"]); preti = np.asarray(al["prot_eval_train_indices"])
    NG, NP = len(al["train_gene_names"]), len(al["train_prot_names"])
    mr_, sr_ = np.asarray(al["train_rna_mean"], float), np.clip(np.asarray(al["train_rna_std"], float), 1e-8, None)
    mp_, sp_ = np.asarray(al["train_prot_mean"], float), np.clip(np.asarray(al["train_prot_std"], float), 1e-8, None)
    Xr, Xp = pb["ct"]["cells_r"].astype(np.float32), pb["ct"]["cells_p"].astype(np.float32)
    xr = np.zeros((len(Xr), NG), np.float32); xr[:, reti] = (Xr - mr_[reti]) / sr_[reti]
    xp = np.zeros((len(Xr), NP), np.float32); xp[:, preti] = (Xp - mp_[preti]) / sp_[preti]
    mr = np.zeros(NG, bool); mr[reti] = True; mp = np.zeros(NP, bool); mp[preti] = True
    tr, va = split_idx(len(xr))
    src = dict(name=tag, xr=xr, xp=xp, mr=mr, mp=mp, tr=tr, va=va, paired=True)
    return dict(tag=tag, sources=[src], raw_r=Xr, raw_p=Xp, reti=reti, preti=preti, NG=NG, NP=NP)

def gw_data(tag):
    import anndata as ad, scipy.sparse as sps
    cond = {"gw8": "Stim8hr", "gw48": "Stim48hr"}[tag]; al = align_of("arce")
    VG = list(al["train_gene_names"]); VGI = {g: i for i, g in enumerate(VG)}; VP = list(al["train_prot_names"]); VPI = {p: i for i, p in enumerate(VP)}
    NG, NP = len(VG), len(VP)
    mr_, sr_ = np.asarray(al["train_rna_mean"], float), np.clip(np.asarray(al["train_rna_std"], float), 1e-8, None)
    pb = ad.read_h5ad(f"{DATA}/gwcd4i_zhu2026_perturbseq/gwcd4i_pb_rna.h5ad", backed="r"); o = pb.obs
    mk = ((o.target.astype(str) == "non-targeting") & (o.condition.astype(str) == cond) & o.keep_min_cells.astype(bool)).values
    idx = np.where(mk)[0]; X = pb.X[idx]; X = np.asarray(X.toarray() if sps.issparse(X) else X, np.float32)
    pci = np.array([VGI[g] for g in pb.var_names.astype(str)]); pb.file.close()
    xr = np.zeros((len(X), NG), np.float32); xr[:, pci] = (X - mr_[pci]) / sr_[pci]
    mr = np.zeros(NG, bool); mr[pci] = True
    tr, va = split_idx(len(xr))
    s1 = dict(name=f"gwcd4i_{cond}_ntc_pb", xr=xr, xp=np.zeros((len(xr), NP), np.float32), mr=mr, mp=np.zeros(NP, bool), tr=tr, va=va, paired=False)
    sc = ad.read_h5ad(f"{DATA}/scito_perturb_wang2026/scito_prot.h5ad", backed="r")
    ntc = np.where((sc.obs.target.astype(str) == "non-targeting").values)[0]
    sel = np.sort(np.random.RandomState(0).choice(ntc, min(HP["scito_ntc_cells"], len(ntc)), replace=False))
    pm = sc.var.in_cmcvae_panel.astype(bool).values; pn = list(sc.var_names[pm])
    Y = sc.X[sel]; Y = np.asarray(Y.toarray() if sps.issparse(Y) else Y, np.float32)[:, pm]; sc.file.close()
    pidx = np.array([VPI[p] for p in pn])
    xp = np.zeros((len(Y), NP), np.float32); xp[:, pidx] = Y
    mp = np.zeros(NP, bool); mp[pidx] = True
    tr2, va2 = split_idx(len(xp))
    s2 = dict(name="scito_ntc", xr=np.zeros((len(xp), NG), np.float32), xp=xp, mr=np.zeros(NG, bool), mp=mp, tr=tr2, va=va2, paired=False)
    log(f"{tag}: GWCD4i {cond} NTC pseudobulks {len(xr)} ({mr.sum()} genes); SCITO NTC cells {len(xp)} of {len(ntc)} ({mp.sum()} proteins)")
    return dict(tag=tag, sources=[s1, s2], NG=NG, NP=NP)

def domain_data(tag): return screen_data(tag) if tag in CITE else gw_data(tag)

# ------------------------------------------------------------------ model
def load_model(rd):
    cfg = json.load(open(f"{rd}/config.json"))
    sd0 = torch.load(P.resolve_ckpt(rd), map_location="cpu"); ng, np_ = sd0["dec_rna.weight"].shape[0], sd0["dec_prot.weight"].shape[0]; del sd0
    m = P.load_ctx_model(CrossModalCausalLDVAE, rd, ng, np_, cfg); m.eval()
    assert m.dec_r_mask is None and not m.scm.scm_nonlinear and m.context_adapt is None and not m.eps_bottleneck
    return m, cfg

def assemble(m):
    A_rr, A_pp, W_rp, W_pr = m.scm.get_block_adjacencies(tau=0.5)
    return torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0)

def ortho(W):
    g = W.T @ W; return torch.norm(g - torch.eye(g.shape[0], device=W.device), p="fro") ** 2 / g.numel()
def glasso(W, ng):
    G, L = W.shape; k = max(1, L // ng); return W[:, :k * (L // k)].reshape(G, L // k, k).norm(dim=2).sum() / G

class Adapter:
    """holds the trainable parameterisation for one variant; forward() reproduces the model forward at the final
    training epoch (alpha = 1, deterministic tau = 0.5 gates as in every eval harness)."""
    def __init__(self, m, variant, mr_all, mp_all):
        self.m, self.v = m, VARIANTS[variant]; self.name = variant
        for p in m.parameters(): p.requires_grad_(False)
        self.params = []; self.s_r = self.s_p = None
        mr_t = torch.as_tensor(mr_all, device=DEV); mp_t = torch.as_tensor(mp_all, device=DEV)
        self.mr_t, self.mp_t = mr_t, mp_t
        if self.v.get("minimal"):
            for enc in (m.enc_rna, m.enc_prot):
                for mod in enc.net:
                    if isinstance(mod, torch.nn.BatchNorm1d): self.params += [mod.weight, mod.bias]
            self.s_r = torch.zeros(m.dim_rna, device=DEV, requires_grad=True); self.s_p = torch.zeros(m.dim_prot, device=DEV, requires_grad=True)
            self.params += [self.s_r, self.s_p, m.bias_rna, m.bias_prot]
        else:
            if self.v["enc"]: self.params += list(m.enc_rna.parameters()) + list(m.enc_prot.parameters())
            if self.v["dec"]: self.params += [m.dec_rna.weight, m.dec_prot.weight, m.bias_rna, m.bias_prot]
            if self.v["A"]: self.params += [p for p in m.scm.parameters()]
        for p in self.params: p.requires_grad_(True)
        # unmeasured features never move: mask decoder-row / bias / scale gradients
        def rowmask(t_, msk):
            if not t_.requires_grad: return None
            return t_.register_hook(lambda g: g * (msk.view(-1, *([1] * (g.dim() - 1))).to(g.dtype)))
        self.hooks = [rowmask(m.dec_rna.weight, mr_t), rowmask(m.dec_prot.weight, mp_t), rowmask(m.bias_rna, mr_t), rowmask(m.bias_prot, mp_t)]
        if self.s_r is not None: self.hooks += [rowmask(self.s_r, mr_t), rowmask(self.s_p, mp_t)]
        self.hooks = [h for h in self.hooks if h is not None]
        self.lam_xm = self.v["lam_xm"]

    def dec_w(self):
        Wr, Wp = self.m._dec_r_w(), self.m.dec_prot.weight
        if self.s_r is not None: Wr = Wr * torch.exp(self.s_r)[:, None]; Wp = Wp * torch.exp(self.s_p)[:, None]
        return Wr, Wp

    def modes(self, train):
        # BatchNorm running statistics are FROZEN in every variant (BN modules always in eval mode): batch statistics of
        # single-domain / single-modality minibatches (e.g. all-zero RNA input on SCITO batches) corrupt them and create a
        # train/eval mismatch (smoke test). Dropout follows train/eval.
        self.m.enc_rna.train(train); self.m.enc_prot.train(train); self.m.scm.eval()
        for enc in (self.m.enc_rna, self.m.enc_prot):
            for mod in enc.net:
                if isinstance(mod, torch.nn.BatchNorm1d): mod.eval()

    def forward(self, xr, xp, train):
        m = self.m; self.modes(train)
        xin = xr * (torch.rand_like(xr) > HP["rna_feature_dropout"]) if train else xr
        mu_r, lv_r = m.enc_rna(xin); mu_p, lv_p = m.enc_prot(xp)
        if train: er = mu_r + torch.randn_like(mu_r) * torch.exp(0.5 * lv_r); ep = mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * lv_p)
        else: er, ep = mu_r, mu_p
        A = assemble(m); lr_ = m.latent_r
        z = m.scm._scm_solve(A, torch.cat([er, ep], 1)); zx = m.scm._scm_solve(A, torch.cat([er, torch.zeros_like(ep)], 1))
        Wr, Wp = self.dec_w()
        rec_r = z[:, :lr_] @ Wr.T + m.bias_rna; rec_p = z[:, lr_:] @ Wp.T + m.bias_prot; xm_p = zx[:, lr_:] @ Wp.T + m.bias_prot
        return dict(rec_r=rec_r, rec_p=rec_p, xm_p=xm_p, mu_r=mu_r, lv_r=lv_r, mu_p=mu_p, lv_p=lv_p, A=A, Wr=Wr, Wp=Wp)

    def data_loss(self, o, xr, xp, mr, mp, paired):
        """masked MSE (per measured element), routing loss (paired sources only), free-bits KL."""
        B = xr.shape[0]; rw = HP["rec_weight"]; out = {}
        mse_r = (((o["rec_r"] - xr) ** 2) * mr).sum() / (mr.sum() * B) if mr.sum() > 0 else torch.zeros((), device=DEV)
        mse_p = (((o["rec_p"] - xp) ** 2) * mp).sum() / (mp.sum() * B) if mp.sum() > 0 else torch.zeros((), device=DEV)
        mse_x = (((o["xm_p"] - xp) ** 2) * mp).sum() / (mp.sum() * B) if (paired and mp.sum() > 0) else torch.zeros((), device=DEV)
        kl_r = (-0.5 * torch.mean(1 + o["lv_r"] - o["mu_r"].pow(2) - o["lv_r"].exp(), 0)).clamp_min(HP["free_bits_r"]).mean()
        kl_p = (-0.5 * torch.mean(1 + o["lv_p"] - o["mu_p"].pow(2) - o["lv_p"].exp(), 0)).clamp_min(HP["free_bits_p"]).mean()
        tot = rw * (mse_r + mse_p) + rw * self.lam_xm * mse_x + HP["beta_r"] * kl_r + HP["beta_p"] * kl_p
        return tot, dict(mse_r=float(mse_r), mse_p=float(mse_p), mse_xm=float(mse_x), kl_r=float(kl_r), kl_p=float(kl_p))

    def reg_loss(self, o):
        r = torch.zeros((), device=DEV)
        if not self.v.get("minimal") and self.v["A"]:
            r = r + HP["lambda_stable"] * torch.relu(torch.linalg.svdvals(o["A"])[0] - HP["rho_bound"])
        if not self.v.get("minimal") and self.v["dec"]:
            r = r + HP["lambda_ortho"] * (ortho(o["Wr"]) + ortho(o["Wp"])) + HP["lambda_gl_rna"] * glasso(o["Wr"], HP["n_groups"]) \
                + HP["lambda_gl_prot"] * glasso(o["Wp"], HP["n_groups"])
        return r

    def fold(self):
        """fold the minimal variant's per-feature decoder scale into the decoder rows (checkpoint stays loadable)."""
        for h in self.hooks: h.remove()
        if self.s_r is not None:
            with torch.no_grad():
                self.m.dec_rna.weight.mul_(torch.exp(self.s_r)[:, None]); self.m.dec_prot.weight.mul_(torch.exp(self.s_p)[:, None])
            self.s_r = self.s_p = None

def tens(a): return torch.as_tensor(a, device=DEV)

@torch.no_grad()
def val_crit(ad_, D):
    tot, parts = 0.0, {}
    for s in D["sources"]:
        mr, mp = tens(s["mr"]).float(), tens(s["mp"]).float(); n = 0; acc = 0.0; pa = {}
        for st in range(0, len(s["va"]), 2048):
            ii = s["va"][st:st + 2048]; xr, xp = tens(s["xr"][ii]), tens(s["xp"][ii])
            o = ad_.forward(xr, xp, False); l, d = ad_.data_loss(o, xr, xp, mr, mp, s["paired"])
            acc += float(l) * len(ii); n += len(ii)
            for k, v in d.items(): pa[k] = pa.get(k, 0.0) + v * len(ii)
        tot += acc / n; parts[s["name"]] = {k: v / n for k, v in pa.items()}
    return tot, parts

def run_adapt(m, D, variant, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mr_all = np.any([s["mr"] for s in D["sources"]], 0); mp_all = np.any([s["mp"] for s in D["sources"]], 0)
    ad_ = Adapter(m, variant, mr_all, mp_all)
    opt = torch.optim.Adam(ad_.params, lr=VARIANTS[variant].get("lr", HP["lr"]))
    hist = []; c0, p0 = val_crit(ad_, D); hist.append(dict(step=0, crit=c0, parts=p0))
    snap = lambda: (copy.deepcopy(m.state_dict()), None if ad_.s_r is None else (ad_.s_r.detach().clone(), ad_.s_p.detach().clone()))
    best, best_state, best_step, since = c0, snap(), 0, 0
    rng = np.random.RandomState(seed)
    for step in range(1, HP["max_steps"] + 1):
        opt.zero_grad(); loss = torch.zeros((), device=DEV); o = None
        for s in D["sources"]:
            ii = s["tr"][rng.randint(0, len(s["tr"]), HP["batch"])]; xr, xp = tens(s["xr"][ii]), tens(s["xp"][ii])
            o = ad_.forward(xr, xp, True); l, _ = ad_.data_loss(o, xr, xp, tens(s["mr"]).float(), tens(s["mp"]).float(), s["paired"])
            loss = loss + l
        loss = loss + ad_.reg_loss(o)
        loss.backward(); torch.nn.utils.clip_grad_norm_(ad_.params, HP["grad_clip"]); opt.step()
        if step % HP["eval_every"] == 0:
            c, pz = val_crit(ad_, D); hist.append(dict(step=step, crit=c, parts=pz, train_loss=float(loss)))
            if c < best - 1e-6: best, best_state, best_step, since = c, snap(), step, 0
            else: since += 1
            if since >= HP["patience"]: break
    m.load_state_dict(best_state[0])
    if best_state[1] is not None:
        with torch.no_grad(): ad_.s_r.copy_(best_state[1][0]); ad_.s_p.copy_(best_state[1][1])
    ad_.fold(); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return dict(best_step=best_step, crit0=c0, crit_best=best, n_steps_run=hist[-1]["step"], hist=hist)

# ------------------------------------------------------------------ diagnostics (endpoint a)
def pe(a, b):
    if np.std(a) == 0 or np.std(b) == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

@torch.no_grad()
def diag_screen(m, D):
    """held-out control cells: per-protein r of protein imputed from RNA ONLY (protein input = training mean = 0) and
    with protein input; share of decoded protein variance via the RNA path; RNA reconstruction r (measured genes)."""
    s = D["sources"][0]; va = s["va"]; preti = D["preti"]; reti = D["reti"]; m.eval()
    xr, xp = tens(s["xr"][va]), tens(s["xp"][va]); A = assemble(m); lr_ = m.latent_r
    N = torch.linalg.inv(torch.eye(A.shape[0], device=DEV) - A)
    er, _ = m.enc_rna(xr); ep, _ = m.enc_prot(xp); ep0, _ = m.enc_prot(torch.zeros_like(xp))
    out = {}
    for lab, e2 in [("with_protein", ep), ("rna_only", ep0)]:
        z = torch.cat([er, e2], 1) @ N; rp = (m.dec_prot(z[:, lr_:]) + m.bias_prot).cpu().numpy()[:, preti]
        r = [pe(rp[:, j], s["xp"][va][:, preti[j]]) for j in range(len(preti))]; out[lab] = float(np.nanmedian(r)); out[lab + "_perprot"] = r
    z = torch.cat([er, ep], 1) @ N; rr = (z[:, :lr_] @ m._dec_r_w().T + m.bias_rna).cpu().numpy()[:, reti]
    out["rna_recon_r_median"] = float(np.nanmedian([pe(rr[:, j], s["xr"][va][:, reti[j]]) for j in range(len(reti))]))
    dr = (er @ N[:lr_, lr_:]) @ m.dec_prot.weight.T; dp = (ep @ N[lr_:, lr_:]) @ m.dec_prot.weight.T
    dr, dp = dr.cpu().numpy()[:, preti], dp.cpu().numpy()[:, preti]
    out["var_share_rna_path"] = float(dr.var(0).sum() / (dr.var(0).sum() + dp.var(0).sum()))
    out["fro"] = {k: float(v.norm()) for k, v in zip(["A_rr", "A_pp", "W_rp", "W_pr"], m.scm.get_block_adjacencies(tau=0.5))}
    out["n_val_cells"] = int(len(va)); out["n_prot"] = int(len(preti))
    return out

def ridge_screen(tag, D):
    """control-cell ridge RNA->protein on the SAME 90% / 10% split (raw log-norm RNA -> CLR protein, as mm_translate)."""
    f = f"{OUT}/diag/ridge_{tag}.json"
    if os.path.exists(f):
        try: return json.load(open(f))
        except Exception: pass
    from sklearn.linear_model import RidgeCV
    s = D["sources"][0]; Xc, Yc = D["raw_r"].astype(np.float64), D["raw_p"].astype(np.float64); tr, va = s["tr"], s["va"]
    mx, my = Xc[tr].mean(0), Yc[tr].mean(0)
    rc = RidgeCV(alphas=np.logspace(0, 8, 17), alpha_per_target=True).fit(Xc[tr] - mx, Yc[tr] - my)
    ho = rc.predict(Xc[va] - mx) + my
    r = [pe(ho[:, j], Yc[va, j]) for j in range(Yc.shape[1])]
    res = dict(perprot=r, median=float(np.nanmedian(r)), alpha=np.atleast_1d(rc.alpha_).tolist(), n_train=int(len(tr)), n_val=int(len(va)))
    tmp = f"{f}.{os.getpid()}.tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, f); return res

def model_dir(base, variant, dom, sk): return f"{OUT}/models/{base}_{variant}_{dom}_{sk}"

def save_model(m, cfg, rd, info):
    os.makedirs(rd, exist_ok=True)
    torch.save(m.state_dict(), f"{rd}/final_model.pt")
    c = dict(cfg); c["name"] = os.path.basename(rd); c["results_dir"] = rd; c["adapt"] = info
    json.dump(c, open(f"{rd}/config.json", "w"), indent=1)
