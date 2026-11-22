"""Operator workstream (Nov-2026 revision, review M3/M4/C2/C8).

Adapted copy of analysis/common/reviewer_controls.py (data loading, DE ground truth, by-gene split, scorer, InfoNCE and the
fine-tuning loss are copied verbatim, lines 132-252 and 266-416 of the original). The CMC-VAE latent SCM and decoders
are LINEAR and the operator acts on the encoder posterior means of control cells, so the cell-averaged predicted
response equals the response computed from the mean control eps (checked at start-up against the per-cell harness).

Operators (z / standardized decoder space; RNA latent block = first l_r dims):
  edgecut        paper (Methods eq. 4): A[:, :l_r] *= (1-s), s=|D_r[g,:]|/max|D_r[g,:]|; d_eps = -x_hat_g * D_r[g,:]
  edgecut_noshift   same without d_eps;  shift_only   d_eps only (no edge cut)
  clamp          signed minimum-norm RNA-eps shift whose propagated, decoded effect on g equals delta_g:
                 d_eps = delta_g u_g/|u_g|^2, u_g = [(I-A)^-1 D^T]_{1:l_r, g}  (== eval/replogle/ft_eval.py "shift")
  cutclamp       edge cut + minimum-norm clamp in the mutilated system so that the TOTAL on-target change = delta_g
  gdo            hard do(x_g) in the gene-level linear SCM implied by the latent model, G = P A D^T with
                 P = D (D^T D + lam I)^-1: dx = delta_g T[g,:]/T[g,g], T[g,:] = 1_g + p_g ((I-A)^-1 - I) D^T
  delta_g modes: zero (drive g to zero log-normalized expression: Z0_g - xbar_hat_g, the Perturb-seq choice),
                 meas (measured on-target LFC of g / sd_g), unit (-1 s.d.); sign flipped for overexpression.

Usage: python op_controls.py RUN_DIR OUT_JSON      env OP_PARTS=diag,fig3,fig4,fig5  OP_FT_OPS=clamp_zero,cutclamp_zero,edgecut
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
import os, sys, json, gc, hashlib, pickle, time
os.environ.setdefault("CMC_DE_LFC_RNA", "0.10")
os.environ.setdefault("CMC_DE_LFC_PROT", "0.05")
import numpy as np
import torch
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, ttest_1samp, spearmanr, pearsonr, binomtest
from sklearn.metrics import roc_auc_score

REPO = f"{_CMC_ROOT}"
HERE = os.path.join(_CMC_ROOT, "analysis", "common")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "src"))
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

P.MAX_CONTROL_CELLS = 4000
RD = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CACHE_DIR = os.path.join(REPO, "results", "_ftcache")      # read-only use of the existing per-model caches (md5-keyed)
LFCR, LFCP = float(os.environ["CMC_DE_LFC_RNA"]), float(os.environ["CMC_DE_LFC_PROT"])
DEVICE = P.DEVICE
SEED = 42
QUICK = os.environ.get("OP_QUICK") == "1"
torch.manual_seed(SEED); np.random.seed(SEED)
DATA = f"{_CMC_DATA}"
DATASETS = {
    "frangieh": (f"{DATA}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad",
                 f"{DATA}/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad"),
    "iccite": (f"{DATA}/chen_perturb_icciteseq/export/iccite_rna.h5ad",
               f"{DATA}/chen_perturb_icciteseq/export/iccite_prot.h5ad"),
    "arce": (f"{DATA}/arce_perturb_citeseq/arce_rna.h5ad",
             f"{DATA}/arce_perturb_citeseq/arce_prot.h5ad"),
}
TAGS = ["frangieh", "iccite", "arce"]
STEPS = 20 if QUICK else int(os.environ.get("CMC_FT_STEPS", "150"))
PATIENCE, WD = 30, 1e-4
N_TRAIN_CELLS = 3000 if DEVICE.type == "cuda" else 1500
MC = dict(lr=5e-4, rho_guard=0.92, w_dir=1.0, w_mag=1.0, w_rp=0.5, w_rho=0.1, rho_pen=0.9)
STD_FLOOR = 0.05
log(f"run={os.path.basename(RD)} device={DEVICE} steps={STEPS} quick={QUICK}")

cfg = json.load(open(f"{RD}/config.json"))
_sd0 = torch.load(P.resolve_ckpt(RD), map_location="cpu")
N_G, N_P = _sd0["dec_rna.weight"].shape[0], _sd0["dec_prot.weight"].shape[0]; del _sd0
model = P.load_ctx_model(CrossModalCausalLDVAE, RD, N_G, N_P, cfg); model.eval()
assert not getattr(model, "eps_bottleneck", False) and not cfg.get("scm_nonlinear", False)
for p in model.parameters(): p.requires_grad_(False)
for p in model.scm.parameters(): p.requires_grad_(True)
SCM0 = {k: v.detach().clone() for k, v in model.scm.state_dict().items()}
lr_, lp_ = model.latent_r, model.latent_p; DL = lr_ + lp_
Wr = model._dec_r_w().detach().to(DEVICE); Wp = model.dec_prot.weight.detach().to(DEVICE)
bias_rna = model.bias_rna.detach().to(DEVICE); bias_prot = model.bias_prot.detach().to(DEVICE)
solve = lambda Amat, e: model.scm._scm_solve(Amat, e)
I_D = torch.eye(DL, device=DEVICE)
Z0 = None

def assemble():
    A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
    return torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0).to(DEVICE)

def control_side(A_full, e):
    z_c = solve(A_full, e); zr_c, zp_c = torch.split(z_c, [lr_, lp_], 1)
    return model.dec_rna(zr_c) + bias_rna, model.dec_prot(zp_c) + bias_prot

def predict_target_percell(A_full, rec_r_c, rec_p_c, e, gidx, mutilate=True):
    """verbatim reviewer_controls.predict_target (SHOCK_MODE='orig', no eps gate) -- reference for the check."""
    B = e.shape[0]
    dx = torch.zeros_like(rec_r_c); dx[:, gidx] = -rec_r_c[:, gidx]
    deps = torch.cat([dx @ Wr, torch.zeros(B, lp_, device=DEVICE)], 1)
    Amat = A_full
    if mutilate:
        s = (Wr[gidx].abs() / (Wr[gidx].abs().max() + 1e-8)).clamp(0, 1)
        Amat = A_full.clone(); Amat[:, :lr_] = Amat[:, :lr_] * (1 - s).unsqueeze(0)
    z_p = solve(Amat, e + deps); zr_p, zp_p = torch.split(z_p, [lr_, lp_], 1)
    return (((model.dec_rna(zr_p) + bias_rna) - rec_r_c).mean(0), ((model.dec_prot(zp_p) + bias_prot) - rec_p_c).mean(0))

def pearson_t(pred, obs):
    pc = pred - pred.mean(); oc = obs - obs.mean()
    return (pc * oc).sum() / (pc.norm() * oc.norm() + 1e-8)

_sn = [torch.randn(DL, 1, device=DEVICE), torch.randn(DL, 1, device=DEVICE)]
def spectral_norm(A):
    with torch.no_grad():
        u, v = _sn
        for _ in range(12):
            u = A @ v; u = u / (u.norm() + 1e-12); v = A.t() @ u; v = v / (v.norm() + 1e-12)
        _sn[0], _sn[1] = u.detach(), v.detach()
    return (_sn[0] * (A @ _sn[1])).sum()

def spectral_radius(A):
    with torch.no_grad(): return float(torch.linalg.eigvals(A.detach().cpu()).abs().max())

def pe(pred, obs, m):
    m = np.asarray(m, bool)
    if m.sum() < 5: return np.nan
    p, o = np.asarray(pred, float)[m], np.asarray(obs, float)[m]
    if not (np.isfinite(p).all() and np.isfinite(o).all()) or np.std(p) == 0 or np.std(o) == 0: return np.nan
    return float(np.corrcoef(p, o)[0, 1])

def au(lab, sc):
    lab = np.asarray(lab).astype(int); sc = np.asarray(sc, float)
    if lab.sum() < 3 or lab.sum() >= len(lab) or not np.isfinite(sc).all(): return np.nan
    return float(roc_auc_score(lab, sc))

def bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0: return p
    o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q

# ------------------------------------------------------------------ data (shared cache, keyed on model content)
def load_dataset(tag):
    _ck = P.resolve_ckpt(RD); h = hashlib.md5(open(_ck, "rb").read()).hexdigest()[:12]
    cache = f"{CACHE_DIR}/rc_{tag}_{os.path.basename(RD)}_{h}_{LFCR}_{LFCP}.pkl"
    if os.path.exists(cache):
        log(f"cache HIT {tag}"); b = pickle.load(open(cache, "rb")); return b["align"], b["de"], b["eps"]
    log(f"loading {tag} ...")
    P.PERTURB_RNA_PATH, P.PERTURB_PROT_PATH = DATASETS[tag]
    pr, pp, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
    de = P.compute_observed_de(pr, pp, align)
    _er, _ep, *_ = P.encode_control_cells(model, pr, pp, align)
    eps = torch.cat([_er, _ep], 1).numpy()[:4000]
    del pr, pp; gc.collect()
    pickle.dump({"align": align, "de": de, "eps": eps}, open(cache, "wb"))
    return align, de, eps

COG = json.load(open(f"{_CMC_RESULTS}/baselines/_cmcvae_cognate.json"))
_cog_map = {p: g for p, g in zip(COG["protein_names"], COG["cognate_gene"]) if g}

RAW = {t: load_dataset(t) for t in TAGS}
# ---- optional: condition-matched pseudobulk DE from the stats workstream (OP_DE=pb_bh), same as analysis/fig3_zero_shot/stats/rc_stats.py
OP_DE = os.environ.get("OP_DE", "orig")
if OP_DE == "pb_bh":
    ST_DIR = os.path.join(_CMC_RESULTS, "paper", "analyses", "stats")
    for _t in TAGS:
        _al, _de, _eps = RAW[_t]; pb = pickle.load(open(os.path.join(ST_DIR, "de", f"pb_{_t}.pkl"), "rb"))
        assert list(pb["common_genes"]) == list(_al["common_genes"]), "feature order mismatch"
        out = {}
        for g, r in pb["de"].items():
            lr = np.nan_to_num(r["lfc_rna"]); lp = np.nan_to_num(r["lfc_prot"])
            out[g] = dict(lfc_rna=lr, lfc_prot=lp, pvals_rna=r["pvals_rna"], pvals_prot=r["pvals_prot"],
                          sig_rna=(r["q_rna"] < 0.05) & (np.abs(lr) >= LFCR),
                          sig_prot=((r["q_prot"] < 0.05) & (np.abs(lp) >= LFCP)) if len(lp) else np.zeros(0, bool), n_cells=r["n_cells"])
        RAW[_t] = (_al, out, _eps); log(f"[pb] {_t}: pseudobulk DE units {len(out)}")
_al0 = RAW["frangieh"][0]
Z0 = torch.as_tensor(-np.asarray(_al0["train_rna_mean"], float) / np.clip(np.asarray(_al0["train_rna_std"], float), 1e-8, None),
                     dtype=torch.float32, device=DEVICE)
if os.environ.get("RC_CACHE_ONLY") == "1":
    log("cache built; exiting"); sys.exit(0)

def build_ctx(tag, de_mode="raw"):
    align, de, eps = RAW[tag]
    common = np.array(align["common_genes"]); reti = np.asarray(align["rna_eval_train_indices"])
    preti = np.asarray(align["prot_eval_train_indices"]); tgi = align["train_gene_idx"]; g2i = align["train_gene_to_idx_common"]
    sd_r = np.clip(np.asarray(align["train_rna_std"], float)[reti], 1e-8, None)
    sd_p = np.clip(np.asarray(align["train_prot_std"], float)[preti], 1e-8, None) if len(preti) else np.zeros(0)
    tr_std = np.asarray(align["train_rna_std"]); tp_std = np.asarray(align["train_prot_std"])
    pnames = [align["train_prot_names"][i] for i in preti]
    INFO = {}
    for g, d in de.items():
        if g not in tgi or g not in g2i: continue
        lfc_full = np.asarray(d["lfc_rna"], float)
        if de_mode == "bh":
            sig = (bh(d["pvals_rna"]) < 0.05) & (np.abs(lfc_full) >= LFCR)
            lfcp_full = np.asarray(d["lfc_prot"], float)
            sigp = ((bh(d["pvals_prot"]) < 0.05) & (np.abs(lfcp_full) >= LFCP)) if len(d["pvals_prot"]) else np.zeros(0, bool)
        else:
            sig = np.asarray(d["sig_rna"], bool); lfcp_full = np.asarray(d["lfc_prot"], float)
            sigp = np.asarray(d["sig_prot"], bool) if len(d["sig_prot"]) else np.zeros(0, bool)
        keep = common != g
        if sig[keep].sum() < 3: continue
        has_p = len(sigp) > 0 and sigp.sum() >= 3
        de_pos = np.where(keep & sig)[0]; tr_idx = reti[de_pos]; obs_raw = lfc_full[de_pos]
        obs_z = obs_raw / np.clip(tr_std[tr_idx], STD_FLOOR, None)
        cog_mask = np.array([_cog_map.get(pn) == g for pn in pnames], bool)
        info = dict(g=g, gidx=tgi[g], keep=keep, sig=sig, lfc=lfc_full, has_p=bool(has_p), sigp=sigp, lfcp=lfcp_full,
                    cog_mask=cog_mask, de_pos=de_pos,
                    sel_r=torch.as_tensor(tr_idx, dtype=torch.long, device=DEVICE),
                    obs_r_raw=torch.as_tensor(obs_raw, dtype=torch.float32, device=DEVICE),
                    obs_r_z=torch.as_tensor(obs_z, dtype=torch.float32, device=DEVICE), has_p_loss=False)
        if has_p:
            dp_pos = np.where(sigp)[0]; tp_idx = preti[dp_pos]; op_raw = lfcp_full[dp_pos]
            op_z = op_raw / np.clip(tp_std[tp_idx], STD_FLOOR, None)
            info.update(dp_pos=dp_pos, sel_p=torch.as_tensor(tp_idx, dtype=torch.long, device=DEVICE),
                        obs_p_raw=torch.as_tensor(op_raw, dtype=torch.float32, device=DEVICE),
                        obs_p_z=torch.as_tensor(op_z, dtype=torch.float32, device=DEVICE), has_p_loss=len(dp_pos) >= 3)
        INFO[g] = info
    return dict(tag=tag, eps=torch.as_tensor(eps, dtype=torch.float32, device=DEVICE)[:4000], reti=reti, preti=preti,
                common=common, sd_r=sd_r, sd_p=sd_p, INFO=INFO, evaluable=sorted(INFO), align=align, g2i=g2i)

def make_split(CTX):
    test, train = set(), set()
    for t in TAGS:
        gt = CTX[t]["evaluable"]; already = [g for g in gt if g in test]
        und = [g for g in gt if g not in test and g not in train]
        n_more = max(0, round(0.30 * len(gt)) - len(already))
        perm = np.random.RandomState(SEED).permutation(len(und)); pick = set(und[i] for i in perm[:n_more])
        for g in und: (test if g in pick else train).add(g)
    return sorted(train), sorted(test)

CTX = {t: build_ctx(t) for t in TAGS}
TRAIN_G, TEST_G = make_split(CTX)
if OP_DE != "orig":   # paper split restricted to evaluable units (as analysis/fig3_zero_shot/stats/rc_stats.py)
    _r0 = json.load(open(f"{_CMC_RESULTS}/finetune_split_genes.json")); _ev = set().union(*[set(CTX[t]["evaluable"]) for t in TAGS])
    TRAIN_G = sorted(set(_r0["train_genes"]) & _ev); TEST_G = sorted(set(_r0["test_genes"]) & _ev)
# sanity: the split must reproduce finetune_split_genes.json (same code, same seed)
_ref = json.load(open(f"{_CMC_RESULTS}/finetune_split_genes.json"))
SPLIT_OK = (_ref["test_genes"] == TEST_G and _ref["train_genes"] == TRAIN_G)
TRAIN = {t: [g for g in CTX[t]["evaluable"] if g in set(TRAIN_G)] for t in TAGS}
_rng_in = np.random.RandomState(7); _tg_in = np.array(TRAIN_G); _prm_in = _rng_in.permutation(len(_tg_in))
INNER_VAL = set(_tg_in[_prm_in[:round(0.3 * len(_tg_in))]].tolist())
IN_TR = {t: [g for g in TRAIN[t] if g not in INNER_VAL] for t in TAGS}; IN_VA = {t: [g for g in TRAIN[t] if g in INNER_VAL] for t in TAGS}
NCE_GRID = [float(x) for x in os.environ.get("RC_NCE_GRID", "0.3,1,3").split(",")]; NCE_TAU = float(os.environ.get("RC_NCE_TAU", "0.1"))
TEST = {t: [g for g in CTX[t]["evaluable"] if g in set(TEST_G)] for t in TAGS}
def set_S(ctx_d, train_units):
    """genes DE in >=1 training KO of the cohort, minus every knocked-out target gene (no own-gene shortcut);
    proteins: all panel proteins minus cognates of any target. Used by InfoNCE and the retrieval metric."""
    for t in TAGS:
        c = ctx_d[t]; tg = set(c["evaluable"])
        u = np.zeros(len(c["common"]), bool)
        for g in train_units[t]: u |= c["INFO"][g]["sig"]
        u &= ~np.isin(c["common"], list(tg)); c["S"] = np.where(u)[0]
        c["S_tr"] = torch.as_tensor(c["reti"][c["S"]], dtype=torch.long, device=DEVICE)
        cg = np.zeros(len(c["preti"]), bool)
        for g in tg: cg |= c["INFO"][g]["cog_mask"]
        c["Sp"] = np.where(~cg)[0]
set_S(CTX, {t: [g for g in CTX[t]["evaluable"] if g in set(TRAIN_G)] for t in TAGS})
log("units:", {t: len(CTX[t]["evaluable"]) for t in TAGS}, "S sizes", {t: len(CTX[t]["S"]) for t in TAGS}, "train", {t: len(TRAIN[t]) for t in TAGS},
    "test", {t: len(TEST[t]) for t in TAGS}, "split_matches_paper:", SPLIT_OK)

def ref_means(ctx_d, ref_units, preds):
    """cohort mean observed and predicted response over a reference KO set (own-gene entry excluded)."""
    M = {}
    for t, genes in ref_units.items():
        c = ctx_d[t]
        if not genes: continue
        O = np.stack([np.where(c["INFO"][g]["keep"], c["INFO"][g]["lfc"], np.nan) for g in genes])
        Pd = np.stack([np.where(c["INFO"][g]["keep"], preds[(t, g)][0], np.nan) for g in genes])
        Op = np.stack([c["INFO"][g]["lfcp"] for g in genes]) if c["INFO"][genes[0]]["lfcp"].size else None
        Pp = np.stack([preds[(t, g)][1] for g in genes])
        M[t] = dict(o_r=O, p_r=Pd, o_p=Op, p_p=Pp, genes=list(genes))
    return M

def _loo_mean(arr, genes, g):
    idx = [i for i, h in enumerate(genes) if h != g]
    return np.nanmean(arr[idx], 0) if idx else np.full(arr.shape[1], np.nan)

def score(ctx_d, eval_units, preds, refs, shuffle_pool=None):
    """Per-KO record with every scoring variant. refs: ref_means() for centring / mean-response."""
    rows = []
    RET = {}
    for t, genes in eval_units.items():
        c = ctx_d[t]
        if "S" not in c or len(genes) < 3 or t not in refs: continue
        R = refs[t]; S = c["S"]; sd = c["sd_r"][S]
        def _z(v): v = v - v.mean(); n = np.linalg.norm(v); return v / n if n > 0 else v
        Pm = np.stack([_z((preds[(t, g)][0] - _loo_mean(R["p_r"], R["genes"], g))[S]) for g in genes])
        Om = np.stack([_z(((c["INFO"][g]["lfc"] - _loo_mean(R["o_r"], R["genes"], g)) / c["sd_r"])[S]) for g in genes])
        C = np.nan_to_num(Pm @ Om.T)
        for i, g in enumerate(genes):
            oth = [j for j in range(len(genes)) if j != i]
            RET[(t, g, "rr")] = float(np.mean(C[i, i] > C[i, oth]) + 0.5 * np.mean(C[i, i] == C[i, oth]))
        gp = [g for g in genes if c["INFO"][g]["has_p"]]
        if len(c["preti"]) and len(gp) >= 3 and R["o_p"] is not None and len(c["Sp"]) >= 3:
            Sp = c["Sp"]
            Pp = np.stack([_z((preds[(t, g)][1] - _loo_mean(R["p_p"], R["genes"], g))[Sp]) for g in gp])
            Op = np.stack([_z(((c["INFO"][g]["lfcp"] - _loo_mean(R["o_p"], R["genes"], g)) / c["sd_p"])[Sp]) for g in gp])
            Cp = np.nan_to_num(Pp @ Op.T)
            if np.isfinite(Pp).all():
                for i, g in enumerate(gp):
                    oth = [j for j in range(len(gp)) if j != i]
                    RET[(t, g, "rp")] = float(np.mean(Cp[i, i] > Cp[i, oth]) + 0.5 * np.mean(Cp[i, i] == Cp[i, oth]))
    ALPHA = {}
    for t in eval_units:
        c = ctx_d[t]; R = refs.get(t)
        if R is None: continue
        num = den = nump = denp = 0.0
        for i, h in enumerate(R["genes"]):
            J = c["INFO"][h]; mh = J["sig"] & J["keep"]
            mo_ = _loo_mean(R["o_r"], R["genes"], h); mp_ = _loo_mean(R["p_r"], R["genes"], h)
            dv = ((R["p_r"][i] - mp_) * c["sd_r"])[mh]; tg_ = (R["o_r"][i] - mo_)[mh]; ok = np.isfinite(dv) & np.isfinite(tg_)
            num += float(dv[ok] @ tg_[ok]); den += float(dv[ok] @ dv[ok])
            if J["has_p"] and R["o_p"] is not None and np.isfinite(R["p_p"][i]).all():
                mq_ = _loo_mean(R["p_p"], R["genes"], h); mo2 = _loo_mean(R["o_p"], R["genes"], h); mpp = J["sigp"]
                dvp = ((R["p_p"][i] - mq_) * c["sd_p"])[mpp]; tgp = (R["o_p"][i] - mo2)[mpp]
                nump += float(dvp @ tgp); denp += float(dvp @ dvp)
        ALPHA[t] = (num / den if den > 0 else 0.0, nump / denp if denp > 0 else 0.0)
    for t, genes in eval_units.items():
        c = ctx_d[t]; sd_r, sd_p = c["sd_r"], c["sd_p"]; R = refs.get(t)
        pool = (shuffle_pool or eval_units)[t]
        a_r, a_p = ALPHA.get(t, (0.0, 0.0))
        for g in genes:
            I = c["INFO"][g]; pr, pp = preds[(t, g)]; k = I["keep"]; m = I["sig"] & k
            mu_o = _loo_mean(R["o_r"], R["genes"], g); mu_p = _loo_mean(R["p_r"], R["genes"], g)
            r = dict(cohort=t, gene=g, n_de_rna=int(m.sum()), rr_retrieval=RET.get((t, g, "rr"), np.nan), rp_retrieval=RET.get((t, g, "rp"), np.nan))
            r["rr_auroc"] = au(I["sig"][k], np.abs(pr[k]))
            r["rr_auroc_log"] = au(I["sig"][k], np.abs(pr[k] * sd_r[k]))
            r["rr_pe"] = pe(pr, I["lfc"], m)
            r["rr_pe_z"] = pe(pr, I["lfc"] / sd_r, m)
            r["rr_pe_log"] = pe(pr * sd_r, I["lfc"], m)
            r["rr_pe_cent"] = pe(pr * sd_r - mu_p * sd_r, I["lfc"] - mu_o, m)        # centred, common (log) scale
            r["rr_pe_cent_z"] = pe(pr - mu_p, (I["lfc"] - mu_o) / sd_r, m)          # centred, z scale
            r["rr_pe_meanresp"] = pe(mu_o, I["lfc"], m)                              # model-free mean response
            r["rr_auroc_meanresp"] = au(I["sig"][k], np.abs(np.nan_to_num(mu_o[k])))           # model-free: rank by |mean response|
            r["rr_pe_meanpred"] = pe(mu_p * sd_r, I["lfc"], m)                       # model's average prediction
            r["rr_pe_combo"] = pe(mu_o + (pr - mu_p) * sd_r, I["lfc"], m)              # mean response + model-specific deviation
            r["rr_pe_combofit"] = pe(mu_o + a_r * (pr - mu_p) * sd_r, I["lfc"], m)     # alpha fitted on reference (training) KOs
            r["alpha_rr"] = a_r
            sh, shl = [], []
            for h in pool:
                if h == g: continue
                J = c["INFO"][h]; mh = J["sig"] & J["keep"] & k
                sh.append(pe(pr, J["lfc"], mh)); shl.append(pe(pr * sd_r, J["lfc"], mh))
            r["rr_pe_shuf"] = float(np.nanmean(sh)) if sh else np.nan
            r["rr_pe_shuf_log"] = float(np.nanmean(shl)) if shl else np.nan
            if I["has_p"]:
                mp = I["sigp"]; r["n_de_prot"] = int(mp.sum())
                r["rp_auroc"] = au(mp, np.abs(pp)); r["rp_auroc_log"] = au(mp, np.abs(pp * sd_p))
                r["rp_pe"] = pe(pp, I["lfcp"], mp); r["rp_pe_z"] = pe(pp, I["lfcp"] / sd_p, mp)
                r["rp_pe_log"] = pe(pp * sd_p, I["lfcp"], mp)
                r["rp_pe_nocog"] = pe(pp, I["lfcp"], mp & ~I["cog_mask"])
                r["rp_has_cog_de"] = bool((mp & I["cog_mask"]).any())
                if R["o_p"] is not None:
                    mo = _loo_mean(R["o_p"], R["genes"], g); mq = _loo_mean(R["p_p"], R["genes"], g)
                    r["rp_pe_cent"] = pe(pp * sd_p - mq * sd_p, I["lfcp"] - mo, mp)
                    r["rp_pe_meanresp"] = pe(mo, I["lfcp"], mp)
                    r["rp_auroc_meanresp"] = au(mp, np.abs(np.nan_to_num(mo)))
                    r["rp_pe_combo"] = pe(mo + (pp - mq) * sd_p, I["lfcp"], mp)
                    r["rp_pe_combofit"] = pe(mo + a_p * (pp - mq) * sd_p, I["lfcp"], mp); r["alpha_rp"] = a_p
                shp = [pe(pp, c["INFO"][h]["lfcp"], c["INFO"][h]["sigp"]) for h in pool if h != g and c["INFO"][h]["has_p"]]
                r["rp_pe_shuf"] = float(np.nanmean(shp)) if shp else np.nan
            rows.append(r)
    return rows

METRICS = ["rr_auroc", "rr_auroc_log", "rr_pe", "rr_pe_z", "rr_pe_log", "rr_pe_cent", "rr_pe_cent_z", "rr_pe_meanresp",
           "rr_pe_meanpred", "rr_pe_shuf", "rr_pe_shuf_log", "rp_auroc", "rp_auroc_log", "rp_pe", "rp_pe_z", "rp_pe_log",
           "rp_pe_nocog", "rp_pe_cent", "rp_pe_meanresp", "rp_pe_shuf", "rr_pe_combo", "rp_pe_combo", "rr_retrieval", "rp_retrieval", "rr_auroc_meanresp", "rp_auroc_meanresp", "rr_pe_combofit", "rp_pe_combofit"]

def summarize(rows):
    out = {}
    for scope in ["overall"] + TAGS:
        rs = rows if scope == "overall" else [r for r in rows if r["cohort"] == scope]
        s = {}
        for mtr in METRICS:
            v = np.array([r.get(mtr, np.nan) for r in rs], float); f = np.isfinite(v)
            s[mtr] = float(np.mean(v[f])) if f.any() else float("nan"); s["n_" + mtr] = int(f.sum())
            s["sem_" + mtr] = float(np.std(v[f], ddof=1) / np.sqrt(f.sum())) if f.sum() > 1 else float("nan")
        out[scope] = s
    return out

def paired(rows_a, rows_b, mtr, alt="two-sided"):
    """paired test b vs a on the same KOs (matched by cohort,gene)."""
    A = {(r["cohort"], r["gene"]): r.get(mtr, np.nan) for r in rows_a}
    res = {}
    for scope in ["overall"] + TAGS:
        pairs = [(A[(r["cohort"], r["gene"])], r.get(mtr, np.nan)) for r in rows_b
                 if (r["cohort"], r["gene"]) in A and (scope == "overall" or r["cohort"] == scope)]
        pairs = np.array([p for p in pairs if np.isfinite(p[0]) and np.isfinite(p[1])], float).reshape(-1, 2)
        n = len(pairs)
        if n < 3 or np.allclose(pairs[:, 0], pairs[:, 1]):
            res[scope] = dict(n=n); continue
        dlt = pairs[:, 1] - pairs[:, 0]
        res[scope] = dict(n=n, mean_a=float(pairs[:, 0].mean()), mean_b=float(pairs[:, 1].mean()), mean_diff=float(dlt.mean()),
                          ci95=[float(dlt.mean() - 1.96 * dlt.std(ddof=1) / np.sqrt(n)), float(dlt.mean() + 1.96 * dlt.std(ddof=1) / np.sqrt(n))],
                          p_t=float(ttest_rel(pairs[:, 1], pairs[:, 0], alternative=alt).pvalue),
                          p_wilcoxon=float(wilcoxon(dlt, alternative=alt).pvalue) if np.any(dlt != 0) else 1.0)
    return res

# ------------------------------------------------------------------ fine-tuning (identical loss; optional target transforms)
def info_nce(P, O, tau):
    """P,O [K,F]: predicted / observed (centred, z) responses of K training KOs. CE of matching own response."""
    P = P - P.mean(0, keepdim=True); O = O - O.mean(0, keepdim=True)
    P = (P - P.mean(1, keepdim=True)); P = P / (P.norm(dim=1, keepdim=True) + 1e-8)
    O = (O - O.mean(1, keepdim=True)); O = O / (O.norm(dim=1, keepdim=True) + 1e-8)
    logits = (P @ O.T) / tau
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(P), device=P.device))

def nce_targets(ctx_d, t, genes):
    c = ctx_d[t]; S = c["S"]
    O = np.stack([(c["INFO"][g]["lfc"] / c["sd_r"])[S] for g in genes])
    return torch.as_tensor(O, dtype=torch.float32, device=DEVICE)
def similarity(ctx_d, units, preds):
    """mean pairwise Pearson between response vectors of different KOs in a cohort (pred vs observed)."""
    def mpc(M):
        M = M[:, np.isfinite(M).all(0)]
        if M.shape[0] < 2 or M.shape[1] < 3: return float("nan")
        C = np.corrcoef(M); iu = np.triu_indices(len(M), 1); return float(np.nanmean(C[iu]))
    out = {}
    for t, genes in units.items():
        c = ctx_d[t]; genes = [g for g in genes if (t, g) in preds]
        if len(genes) < 2: continue
        o = dict(pred_rna=mpc(np.stack([preds[(t, g)][0] for g in genes])),
                 obs_rna=mpc(np.stack([np.where(c["INFO"][g]["keep"], c["INFO"][g]["lfc"], np.nan) for g in genes])))
        gp = [g for g in genes if c["INFO"][g]["has_p"]]
        if len(c["preti"]) and len(gp) > 1:
            o.update(pred_prot=mpc(np.stack([preds[(t, g)][1] for g in gp])), obs_prot=mpc(np.stack([c["INFO"][g]["lfcp"] for g in gp])))
        out[t] = o
    return out

# ======================================================================================== operator workstream code
PARTS = os.environ.get("OP_PARTS", "diag,fig3,fig4,fig5").split(",")
FT_OPS = os.environ.get("OP_FT_OPS", "clamp_zero,cutclamp_zero,edgecut").split(",")
CTRLM = {}
for t in TAGS:
    f = f"{_CMC_RESULTS}/paper/analyses/operator/ctrl_means_{t}.npz"
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True); CTRLM[t] = {str(g): (float(a), float(b), float(c)) for g, a, b, c in zip(z["common"], z["mean_log"], z["mean_z"], z["frac_expr"])}
for t in TAGS:   # measured on-target change of each target (log scale) and its z-scaled version
    c = CTX[t]; com = list(c["common"]); trs = np.asarray(c["align"]["train_rna_std"], float)
    pn = [c["align"]["train_prot_names"][i] for i in c["preti"]]
    for g, I in c["INFO"].items():
        j = com.index(g); I["tgt_lfc"] = float(I["lfc"][j]); I["tgt_p"] = float(RAW[t][1][g]["pvals_rna"][j])
        I["tgt_z"] = float(I["tgt_lfc"] / max(trs[I["gidx"]], STD_FLOOR))
        I["tgt_ncells"] = int(RAW[t][1][g]["n_cells"])
        cg = [k for k, p_ in enumerate(pn) if _cog_map.get(p_) == g]
        I["tgt_prot_lfc"] = float(I["lfcp"][cg[0]]) if (cg and I["lfcp"].size) else None

# gene-level SCM implied by the latent model: z = x P  (ridge pseudo-inverse of the decoder), G = P A D^T
_WtW = Wr.T @ Wr; GDO_LAM = float(os.environ.get("OP_GDO_LAM", "0.01")) * float(torch.trace(_WtW)) / lr_
Pr = Wr @ torch.linalg.inv(_WtW + GDO_LAM * torch.eye(lr_, device=DEVICE))

def Nmat(A): return torch.linalg.inv(I_D - A + 1e-6 * I_D)      # z = eps @ N   (== model.scm._scm_solve)

def _dec(dz):
    return dz[:, :lr_] @ Wr.T, dz[:, lr_:] @ Wp.T

def _mut_solve(A, keep, V):
    """z_k = V_k (I - A_k)^-1 with A_k = A with columns :l_r scaled by keep_k. V [K,d] -> [K,d]."""
    K = keep.shape[0]
    Am = A.unsqueeze(0).expand(K, DL, DL).clone(); Am[:, :, :lr_] = Am[:, :, :lr_] * keep.unsqueeze(1)
    M = (I_D + 1e-6 * I_D).unsqueeze(0) - Am
    if DEVICE.type == "cpu":   # batched LAPACK solve hangs on some CPU nodes (MKL): per-unit solves
        return torch.stack([torch.linalg.solve(M[k].T, V[k]) for k in range(K)]), M
    return torch.linalg.solve(M.transpose(1, 2), V.unsqueeze(-1)).squeeze(-1), M

def predict_op(A, ebar, gidx, op, dmode="zero", dvec=None, shift_scale=1.0, eps_mode="full", sign=1.0):
    """mean predicted response [K,G_train], [K,P_train] and the delta used. op in edgecut|edgecut_noshift|shift_only|
    clamp|cutclamp|gdo. sign=-1 -> overexpression (shift / delta negated)."""
    K = len(gidx); N = Nmat(A)
    e = ebar if eps_mode == "full" else (torch.zeros_like(ebar) if eps_mode == "zero" else -ebar)
    zc = e @ N; xr_c = model.dec_rna(zc[:lr_].unsqueeze(0))[0] + bias_rna
    if op in ("clamp", "cutclamp", "gdo"):
        if dmode == "zero": delta = Z0[gidx] - xr_c[gidx]
        elif dmode == "meas": delta = dvec
        elif dmode == "unit": delta = -torch.ones(K, device=DEVICE)
        delta = sign * delta
    else: delta = None
    if op in ("edgecut", "edgecut_noshift", "shift_only"):
        deps = torch.zeros(K, DL, device=DEVICE)
        if op != "edgecut_noshift": deps[:, :lr_] = -sign * shift_scale * xr_c[gidx].unsqueeze(1) * Wr[gidx]
        if op == "shift_only": dz = deps @ N
        else:
            Wg = Wr[gidx].abs(); s = (Wg / (Wg.max(1, keepdim=True).values + 1e-8)).clamp(0, 1)
            zp, _ = _mut_solve(A, 1 - s, e.unsqueeze(0) + deps); dz = zp - zc.unsqueeze(0)
        dr, dp = _dec(dz)
    elif op == "clamp":
        u = (N[:lr_, :lr_] @ Wr.T[:, gidx]).T                  # [K,l_r] decoded on-target effect of a unit RNA-eps shift
        deps = torch.zeros(K, DL, device=DEVICE); deps[:, :lr_] = delta.unsqueeze(1) * u / (u.pow(2).sum(1, keepdim=True) + 1e-12)
        dr, dp = _dec(deps @ N)
    elif op == "cutclamp":
        Wg = Wr[gidx].abs(); s = (Wg / (Wg.max(1, keepdim=True).values + 1e-8)).clamp(0, 1)
        zp, M = _mut_solve(A, 1 - s, e.unsqueeze(0).expand(K, DL).contiguous()); dz0 = zp - zc.unsqueeze(0)
        c0 = (dz0[:, :lr_] * Wr[gidx]).sum(1)                   # on-target change from the cut alone
        Nm = torch.stack([torch.linalg.inv(M[k]) for k in range(K)]) if DEVICE.type == "cpu" else torch.linalg.inv(M)   # mutilated (I-A_k)^-1
        u = torch.einsum("kij,kj->ki", Nm[:, :lr_, :lr_], Wr[gidx])
        deps = torch.zeros(K, DL, device=DEVICE); deps[:, :lr_] = (delta - c0).unsqueeze(1) * u / (u.pow(2).sum(1, keepdim=True) + 1e-12)
        dz = dz0 + torch.einsum("ki,kij->kj", deps, Nm); dr, dp = _dec(dz)
    elif op == "gdo":
        pg = torch.zeros(K, DL, device=DEVICE); pg[:, :lr_] = Pr[gidx]
        v = pg @ (N - I_D); dr, dp = _dec(v)
        ar = torch.arange(K, device=DEVICE); dr = dr.clone(); dr[ar, gidx] = dr[ar, gidx] + 1.0
        tgg = dr[ar, gidx]; sc_ = (delta / tgg).unsqueeze(1); dr, dp = dr * sc_, dp * sc_
    return dr, dp, delta

def parse_op(name):
    """'clamp_zero' -> ('clamp','zero'); 'edgecut' -> ('edgecut','zero')"""
    if "_" in name and name.split("_")[-1] in ("zero", "meas", "unit"): return name.rsplit("_", 1)[0], name.rsplit("_", 1)[1]
    return name, "zero"

def ebar_of(c, n=None): return c["eps"][: (n or c["eps"].shape[0])].mean(0)

A_OVR = None
@torch.no_grad()
def predict_units_op(ctx_d, units, opname, **kw):
    op, dm = parse_op(opname); A = assemble() if A_OVR is None else A_OVR; out, ont = {}, {}
    for t, genes in units.items():
        if not genes: continue
        c = ctx_d[t]; gi = torch.as_tensor([c["INFO"][g]["gidx"] for g in genes], device=DEVICE)
        dv = torch.as_tensor([c["INFO"][g]["tgt_z"] for g in genes], dtype=torch.float32, device=DEVICE)
        dr, dp, _ = predict_op(A, ebar_of(c), gi, op, dm, dvec=dv, **kw)
        drn, dpn = dr.cpu().numpy().astype(float), dp.cpu().numpy().astype(float)
        for k, g in enumerate(genes):
            out[(t, g)] = (drn[k][c["reti"]], dpn[k][c["preti"]]); ont[(t, g)] = float(drn[k][c["INFO"][g]["gidx"]])
    return out, ont

def check_equivalence():
    """batched mean-eps edgecut vs the per-cell reviewer_controls operator (max relative error)."""
    A = assemble().detach(); res = {}
    with torch.no_grad():
        for t in TAGS:
            c = CTX[t]; rr, rp = control_side(A, c["eps"])
            for g in c["evaluable"][:3]:
                gi = c["INFO"][g]["gidx"]; ref = predict_target_percell(A, rr, rp, c["eps"], gi)
                b = predict_op(A, ebar_of(c), torch.tensor([gi], device=DEVICE), "edgecut")
                res[f"{t}:{g}"] = [float((ref[0] - b[0][0]).abs().max() / (ref[0].abs().max() + 1e-12)),
                                   float((ref[1] - b[1][0]).abs().max() / (ref[1].abs().max() + 1e-12))]
    return res

def corr_np(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 3 or np.std(a[f]) == 0 or np.std(b[f]) == 0: return np.nan
    return float(np.corrcoef(a[f], b[f])[0, 1])

def boot_ci(x, fn=np.mean, B=2000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return [np.nan, np.nan]
    r = np.random.RandomState(seed); v = [fn(x[r.randint(0, len(x), len(x))]) for _ in range(B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def boot_corr_ci(a, b, B=2000, seed=0, rank=False):
    a, b = np.asarray(a, float), np.asarray(b, float); f = np.isfinite(a) & np.isfinite(b); a, b = a[f], b[f]
    if len(a) < 4: return [np.nan, np.nan]
    r = np.random.RandomState(seed); v = []
    for _ in range(B):
        i = r.randint(0, len(a), len(a))
        if np.std(a[i]) > 0 and np.std(b[i]) > 0: v.append(spearmanr(a[i], b[i])[0] if rank else np.corrcoef(a[i], b[i])[0, 1])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

RES = dict(run=os.path.basename(RD), split_matches_paper=SPLIT_OK, steps=STEPS, de_floor=[LFCR, LFCP], gdo_lambda=GDO_LAM,
           n_units={t: len(CTX[t]["evaluable"]) for t in TAGS}, n_train_units={t: len(TRAIN[t]) for t in TAGS},
           n_test_units={t: len(TEST[t]) for t in TAGS})
def save():
    json.dump(RES, open(OUT, "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else (o.item() if hasattr(o, "item") else str(o)))
ALL = {t: CTX[t]["evaluable"] for t in TAGS}
model.scm.load_state_dict(SCM0)
RES["equivalence_check_relerr"] = check_equivalence(); log("per-cell vs mean-eps edgecut rel. err:", RES["equivalence_check_relerr"])
with torch.no_grad():
    A0 = assemble().detach().clone()
    RES["decoder"] = dict(rna_row_norm_median=float(Wr.norm(dim=1).median()), rna_row_norm_sq_median=float(Wr.pow(2).sum(1).median()),
                          rna_sv_max=float(torch.linalg.svdvals(Wr)[0]), rna_sv_min=float(torch.linalg.svdvals(Wr)[-1]),
                          spectral_radius_A=spectral_radius(A0))

def fig3_rows(opname, units=ALL, **kw):
    pr, ont = predict_units_op(CTX, units, opname, **kw)
    rows = score(CTX, units, pr, ref_means(CTX, units, pr))
    for r in rows: r["dx_target"] = ont[(r["cohort"], r["gene"])]
    return rows, pr, ont

KEYS3 = ["rr_auroc", "rr_auroc_meanresp", "rr_pe", "rr_pe_cent", "rr_pe_meanresp", "rr_pe_shuf", "rr_retrieval",
         "rp_auroc", "rp_auroc_meanresp", "rp_pe", "rp_pe_cent", "rp_pe_meanresp", "rp_pe_shuf", "rp_retrieval"]
def brief(s): return {k: round(s[k], 4) for k in KEYS3 if k in s and s[k] == s[k]}

# ------------------------------------------------------------------------------------------- (1) diagnostics
if "diag" in PARTS:
    D = {}
    # (a) ||eps_bar_ctrl|| per screen
    en = {}
    for t in TAGS:
        e = CTX[t]["eps"]; eb = e.mean(0)
        en[t] = dict(n_cells=int(e.shape[0]), norm_ebar=float(eb.norm()), norm_ebar_r=float(eb[:lr_].norm()), norm_ebar_p=float(eb[lr_:].norm()),
                     rms_cell_norm=float(e.norm(dim=1).pow(2).mean().sqrt()), sqrt_d=float(np.sqrt(DL)),
                     ratio_ebar_to_cell=float(eb.norm() / e.norm(dim=1).pow(2).mean().sqrt()),
                     norm_xbar_ctrl_r=float(control_side(A0, eb.unsqueeze(0))[0][0][CTX[t]["reti"]].norm()))
    D["eps_norms"] = en; log("||eps_bar||:", {t: round(v["norm_ebar"], 3) for t, v in en.items()})
    # (b) eps_bar zero / negated, edgecut
    base_rows, base_pr, base_ont = fig3_rows("edgecut")
    D["edgecut_full"] = dict(summary=summarize(base_rows))
    for mode in ["zero", "neg"]:
        rows, pr, ont = fig3_rows("edgecut", eps_mode=mode)
        cr = [corr_np(np.where(CTX[t]["INFO"][g]["keep"], pr[(t, g)][0], np.nan), np.where(CTX[t]["INFO"][g]["keep"], base_pr[(t, g)][0], np.nan)) for (t, g) in pr]
        cp = [corr_np(pr[k][1], base_pr[k][1]) for k in pr if len(pr[k][1])]
        D[f"edgecut_eps_{mode}"] = dict(summary=summarize(rows), corr_with_full_rna_median=float(np.nanmedian(cr)), corr_with_full_rna_q=[float(np.nanpercentile(cr, q)) for q in (5, 25, 75, 95)],
                                        corr_with_full_prot_median=float(np.nanmedian(cp)) if cp else None,
                                        per_cohort_corr_rna={t: float(np.nanmedian([cr[i] for i, k in enumerate(pr) if k[0] == t])) for t in TAGS},
                                        paired_vs_full={m: paired(base_rows, rows, m) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe", "rp_pe_cent"]})
        log(f"eps {mode}: corr with full (RNA) median {D[f'edgecut_eps_{mode}']['corr_with_full_rna_median']:+.3f}; overall {brief(summarize(rows)['overall'])}")
    # (c) shift contribution per DE feature, larger shifts and overexpression sign
    _, ns_pr, ns_ont = fig3_rows("edgecut_noshift")
    sc = {}
    for k_, sg in [(1, 1), (10, 1), (100, 1), (1000, 1), (10000, 1), (1, -1), (100, -1), (10000, -1)]:
        rows, pr, ont = fig3_rows("edgecut", shift_scale=float(k_), sign=float(sg))
        per = []
        for (t, g) in pr:
            I = CTX[t]["INFO"][g]; m = I["sig"] & I["keep"]
            f_, n_ = pr[(t, g)][0][m], ns_pr[(t, g)][0][m]; cont = f_ - n_
            rel = np.abs(cont) / (np.abs(f_) + 1e-12)
            per.append(dict(cohort=t, gene=g, n_de=int(m.sum()), rel_median=float(np.median(rel)), rel_p95=float(np.percentile(rel, 95)), rel_max=float(rel.max()),
                            frac_sign_change=float(np.mean(np.sign(f_) != np.sign(n_))), norm_ratio=float(np.linalg.norm(cont) / (np.linalg.norm(f_) + 1e-12)),
                            corr_full_noshift=corr_np(f_, n_), corr_with_base=corr_np(np.where(I["keep"], pr[(t, g)][0], np.nan), np.where(I["keep"], base_pr[(t, g)][0], np.nan)),
                            dx_target=ont[(t, g)], dx_target_noshift=ns_ont[(t, g)]))
        s_ = summarize(rows)
        key = f"scale{k_}_{'KD' if sg > 0 else 'OE'}"
        agg = {q: float(np.nanmedian([p[q] for p in per])) for q in ["rel_median", "rel_p95", "rel_max", "frac_sign_change", "norm_ratio", "corr_full_noshift", "corr_with_base"]}
        agg.update(frac_dx_target_neg=float(np.mean([p["dx_target"] < 0 for p in per])), max_rel_over_all_features=float(np.max([p["rel_max"] for p in per])),
                   frac_KO_any_DE_sign_change=float(np.mean([p["frac_sign_change"] > 0 for p in per])))
        sc[key] = dict(agg=agg, summary_overall=brief(s_["overall"]), per_cohort={t: brief(s_[t]) for t in TAGS},
                       paired_vs_base={m: paired(base_rows, rows, m) for m in ["rr_pe", "rr_auroc", "rp_pe"]}, per_ko=per if k_ in (1, 10000) else None)
        log(f"shift x{k_} {'KD' if sg > 0 else 'OE'}: {agg}")
    D["shift_contribution"] = sc
    # (d) on-target change under each operator + correlates; KD vs OE
    OPS_T = ["edgecut", "edgecut_noshift", "shift_only", "clamp_zero", "clamp_meas", "cutclamp_zero", "gdo_zero"]
    with torch.no_grad():
        XC = {t: control_side(A0, ebar_of(CTX[t]).unsqueeze(0))[0][0].cpu().numpy() for t in TAGS}
    ON = {}; tab = []
    for t in TAGS:
        for g in ALL[t]:
            I = CTX[t]["INFO"][g]; cm = CTRLM.get(t, {}).get(g, (np.nan, np.nan, np.nan))
            tab.append(dict(cohort=t, gene=g, ctrl_xhat_z=float(XC[t][I["gidx"]]), z0=float(Z0[I["gidx"]]), ctrl_mean_log=cm[0], ctrl_mean_z=cm[1], ctrl_frac_expr=cm[2],
                            meas_lfc=I["tgt_lfc"], meas_z=I["tgt_z"], meas_p=I["tgt_p"], meas_prot_lfc=I["tgt_prot_lfc"], n_cells=I["tgt_ncells"],
                            dec_row_norm=float(Wr[I["gidx"]].norm())))
    KDOE = {}
    for opn in OPS_T:
        _, prk, ontk = fig3_rows(opn); _, pro, onto = fig3_rows(opn, sign=-1.0)
        for r in tab: r[f"dx_{opn}"] = ontk[(r["cohort"], r["gene"])]; r[f"dxOE_{opn}"] = onto[(r["cohort"], r["gene"])]
        cc = {t: [corr_np(np.where(CTX[t]["INFO"][g]["keep"], prk[(t, g)][0], np.nan), np.where(CTX[t]["INFO"][g]["keep"], pro[(t, g)][0], np.nan)) for g in ALL[t]] for t in TAGS}
        KDOE[opn] = {t: dict(median_corr_KD_OE=float(np.nanmedian(v)), min=float(np.nanmin(v)), max=float(np.nanmax(v))) for t, v in cc.items()}
    for opn in OPS_T:
        o = {}
        for scope in ["overall"] + TAGS:
            rs = [r for r in tab if scope == "overall" or r["cohort"] == scope]
            dx = np.array([r[f"dx_{opn}"] for r in rs]); n = len(dx)
            fr = float(np.mean(dx < 0))
            o[scope] = dict(n=n, frac_neg=fr, frac_neg_ci=boot_ci((dx < 0).astype(float)), n_neg=int((dx < 0).sum()),
                            sign_test_p=float(binomtest(int((dx < 0).sum()), n, 0.5).pvalue) if n else None,
                            median_dx=float(np.median(dx)) if n else None,
                            median_dx_over_ctrl_dist_to_z0=float(np.median(dx / (np.array([r["z0"] - r["ctrl_xhat_z"] for r in rs]) + 1e-12))) if n else None,
                            r_ctrl_xhat_z=corr_np(dx, [r["ctrl_xhat_z"] for r in rs]), r_ctrl_xhat_z_ci=boot_corr_ci(dx, [r["ctrl_xhat_z"] for r in rs]),
                            r_ctrl_mean_log=corr_np(dx, [r["ctrl_mean_log"] for r in rs]), rho_ctrl_mean_log=float(spearmanr(dx, [r["ctrl_mean_log"] for r in rs], nan_policy="omit")[0]) if n > 3 else None,
                            r_meas_z=corr_np(dx, [r["meas_z"] for r in rs]), r_meas_z_ci=boot_corr_ci(dx, [r["meas_z"] for r in rs]),
                            rho_meas_z=float(spearmanr(dx, [r["meas_z"] for r in rs])[0]) if n > 3 else None,
                            frac_sign_agree_meas=float(np.mean(np.sign(dx) == np.sign([r["meas_z"] for r in rs]))) if n else None,
                            slope_meas_on_pred=float(np.polyfit(dx, [r["meas_z"] for r in rs], 1)[0]) if n > 3 and np.std(dx) > 0 else None)
        ON[opn] = o
        log(f"on-target {opn}: " + " | ".join(f"{s}: frac<0 {o[s]['frac_neg']:.2f} r(ctrl) {o[s]['r_ctrl_xhat_z']:+.2f} r(meas) {o[s]['r_meas_z']:+.2f}" for s in ["overall"] + TAGS))
    meas = {}
    for scope in ["overall"] + TAGS:
        rs = [r for r in tab if scope == "overall" or r["cohort"] == scope]; mz = np.array([r["meas_lfc"] for r in rs]); pv = np.array([r["meas_p"] for r in rs])
        meas[scope] = dict(n=len(rs), frac_meas_target_decreased=float(np.mean(mz < 0)), frac_sig_decreased=float(np.mean((mz < 0) & (pv < 0.05))),
                           median_meas_lfc=float(np.median(mz)), frac_ctrl_xhat_below_mean=float(np.mean(np.array([r["ctrl_xhat_z"] for r in rs]) < 0)))
    D["on_target"] = dict(by_operator=ON, measured=meas, kd_vs_oe_corr=KDOE, per_ko=tab)
    log("measured on-target:", meas); log("KD vs OE corr:", {k: {t: round(v[t]["median_corr_KD_OE"], 4) for t in TAGS} for k, v in KDOE.items()})
    RES["diag"] = D; save()

# ------------------------------------------------------------------------------------------- (2) Fig 3 re-scoring
OPS3 = ["edgecut", "edgecut_noshift", "shift_only", "clamp_zero", "clamp_meas", "clamp_unit", "cutclamp_zero", "gdo_zero", "gdo_meas"]
if "fig3" in PARTS:
    F3 = {}; ROWS3 = {}
    for opn in OPS3:
        rows, pr, ont = fig3_rows(opn); ROWS3[opn] = rows
        F3[opn] = dict(summary=summarize(rows), similarity=similarity(CTX, ALL, pr), per_ko=rows)
        log(f"Fig3 {opn}: {brief(F3[opn]['summary']['overall'])}")
    for opn in OPS3:
        if opn != "edgecut":
            F3[opn]["paired_vs_edgecut"] = {m: paired(ROWS3["edgecut"], ROWS3[opn], m) for m in KEYS3 if not m.endswith("meanresp")}
    RES["fig3"] = F3; save()

# ------------------------------------------------------------------------------------------- (3) Fig 4 block controls
if "fig4" in PARTS:
    g0 = torch.Generator().manual_seed(0)
    def _shuf_off(M):
        M = M.clone(); n = M.shape[0]; off = ~torch.eye(n, dtype=torch.bool, device=M.device)
        v = M[off]; M[off] = v[torch.randperm(v.numel(), generator=g0).to(M.device)]; return M
    def _shuf_all(M):
        f = M.flatten(); return f[torch.randperm(f.numel(), generator=g0).to(M.device)].reshape(M.shape)
    def put(rr=None, rp=None):
        A = A0.clone()
        if rr is not None: A[:lr_, :lr_] = rr
        if rp is not None: A[:lr_, lr_:] = rp
        return A
    Arr, Wrp, Wpr = A0[:lr_, :lr_], A0[:lr_, lr_:], A0[lr_:, :lr_]
    off = ~torch.eye(lr_, dtype=torch.bool, device=DEVICE)
    variants = {"learned": [A0], "Arr_transposed": [put(rr=Arr.T.contiguous())], "Arr_zeroed": [put(rr=torch.zeros_like(Arr))],
                "Arr_shuffled": [put(rr=_shuf_off(Arr)) for _ in range(3)], "Arr_antisym_part_only": [put(rr=0.5 * (Arr - Arr.T))],
                "Wrp_from_WprT": [put(rp=Wpr.T.contiguous())], "Wrp_zeroed": [put(rp=torch.zeros_like(Wrp))], "Wrp_shuffled": [put(rp=_shuf_all(Wrp)) for _ in range(3)]}
    F4 = dict(corr_Arr_ArrT=float(np.corrcoef(Arr[off].cpu().numpy(), Arr.T[off].cpu().numpy())[0, 1]))
    for opn in ["edgecut", "clamp_zero", "cutclamp_zero", "gdo_zero"]:
        F4[opn] = {}; base = None
        for vn, As in variants.items():
            rows_v = []
            for Av in As:
                A_OVR = Av; rows, pr, ont = fig3_rows(opn); A_OVR = None; rows_v.append(rows)
            keys = [k for k in rows_v[0][0] if k.startswith(("rr_", "rp_", "dx_"))]; avg = []
            for i, r in enumerate(rows_v[0]):
                a = dict(r)
                for k in keys:
                    vals = [rv[i].get(k, np.nan) for rv in rows_v]; vals = [v for v in vals if isinstance(v, float) and v == v]
                    a[k] = float(np.mean(vals)) if vals else np.nan
                avg.append(a)
            if vn == "learned": base = avg
            F4[opn][vn] = dict(summary=summarize(avg), per_ko=avg,
                               paired_variant_minus_learned={m: paired(base, avg, m) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval"]} if vn != "learned" else None)
            log(f"Fig4 {opn} {vn}: {brief(F4[opn][vn]['summary']['overall'])}")
    RES["fig4"] = F4; save()

# ------------------------------------------------------------------------------------------- (4) Fig 5 fine-tuning
def build_targets(ctx_d, train_units, target="real", perm_seed=0, with_nce=False):
    """verbatim reviewer_controls.finetune target construction (real | shuffle | mean)."""
    tg = {t: [g for g in gl if ctx_d[t]["INFO"][g]["obs_r_raw"].numel() >= 5] for t, gl in train_units.items()}
    active = [t for t in TAGS if tg.get(t)]; TGT = {}
    for t in active:
        c = ctx_d[t]; genes = tg[t]; src = list(genes)
        if target == "shuffle": src = list(np.array(genes)[np.random.RandomState(1000 + perm_seed).permutation(len(genes))])
        if target == "mean":
            meanvec = np.nanmean(np.stack([np.where(c["INFO"][h]["keep"], c["INFO"][h]["lfc"], np.nan) for h in genes]), 0)
            meanp = np.nanmean(np.stack([c["INFO"][h]["lfcp"] for h in genes]), 0) if c["INFO"][genes[0]]["lfcp"].size else None
        for g, h in zip(genes, src):
            I = c["INFO"][g]; J = c["INFO"][h]
            if target == "mean":
                raw = torch.as_tensor(meanvec[I["de_pos"]], dtype=torch.float32, device=DEVICE)
                zz = raw / torch.as_tensor(np.clip(np.asarray(c["align"]["train_rna_std"])[c["reti"][I["de_pos"]]], STD_FLOOR, None), device=DEVICE).float()
                e = dict(sel_r=I["sel_r"], r_raw=raw, r_z=zz, has_p=False)
                if I["has_p_loss"] and meanp is not None:
                    rp_raw = torch.as_tensor(meanp[I["dp_pos"]], dtype=torch.float32, device=DEVICE)
                    rp_z = rp_raw / torch.as_tensor(np.clip(np.asarray(c["align"]["train_prot_std"])[c["preti"][I["dp_pos"]]], STD_FLOOR, None), device=DEVICE).float()
                    e.update(has_p=True, sel_p=I["sel_p"], p_raw=rp_raw, p_z=rp_z)
            else:
                e = dict(sel_r=J["sel_r"], r_raw=J["obs_r_raw"], r_z=J["obs_r_z"], has_p=J["has_p_loss"])
                if J["has_p_loss"]: e.update(sel_p=J["sel_p"], p_raw=J["obs_p_raw"], p_z=J["obs_p_z"])
            TGT[(t, g)] = e
    return dict(tg=tg, active=active, TGT=TGT,
                GI={t: torch.as_tensor([ctx_d[t]["INFO"][g]["gidx"] for g in tg[t]], device=DEVICE) for t in active},
                DV={t: torch.as_tensor([ctx_d[t]["INFO"][g]["tgt_z"] for g in tg[t]], dtype=torch.float32, device=DEVICE) for t in active},
                EB={t: ebar_of(ctx_d[t], N_TRAIN_CELLS) for t in active},
                NCE_O={t: nce_targets(ctx_d, t, tg[t]) for t in active} if with_nce else {})

def objective(ctx_d, TG, opname, mc=MC, infonce=None, backward=False, dir_scale="raw"):
    """the reviewer_controls fine-tuning loss (per cohort mean, averaged over cohorts); optional backward per cohort."""
    op, dm = parse_op(opname); n_ds = max(1, len(TG["active"])); total = 0.0
    for t in TG["active"]:
        A = assemble(); DR, DP, _ = predict_op(A, TG["EB"][t], TG["GI"][t], op, dm, dvec=TG["DV"][t])
        L_rr, L_rp = [], []
        for k, g in enumerate(TG["tg"][t]):
            e = TG["TGT"][(t, g)]; pr = DR[k][e["sel_r"]]; l = torch.zeros((), device=DEVICE)
            tgt_dir = e["r_raw"] if dir_scale == "raw" else e["r_z"]
            if mc["w_dir"] > 0 and pr.numel() >= 5 and torch.std(pr) > 1e-12: l = l + mc["w_dir"] * (-pearson_t(pr, tgt_dir))
            if mc["w_mag"] > 0: l = l + mc["w_mag"] * ((pr - e["r_z"]) ** 2).mean()
            L_rr.append(l)
            if mc["w_rp"] > 0 and e["has_p"]:
                pq = DP[k][e["sel_p"]]; lp = torch.zeros((), device=DEVICE)
                tgt_p = e["p_raw"] if dir_scale == "raw" else e["p_z"]
                if mc["w_dir"] > 0 and pq.numel() >= 3 and torch.std(pq) > 1e-12: lp = lp + mc["w_dir"] * (-pearson_t(pq, tgt_p))
                if mc["w_mag"] > 0: lp = lp + mc["w_mag"] * ((pq - e["p_z"]) ** 2).mean()
                L_rp.append(lp)
        if not L_rr: continue
        ds = torch.stack(L_rr).mean() + (mc["w_rp"] * torch.stack(L_rp).mean() if L_rp else 0.0)
        if infonce is not None and len(TG["tg"][t]) >= 3:
            ds = ds + infonce[0] * info_nce(DR[:, ctx_d[t]["S_tr"]], TG["NCE_O"][t], infonce[1])
        if backward: (ds / n_ds).backward()
        total += float(ds.detach()) / n_ds
    return total

RHO_PROJ = 0.95
def _project():
    with torch.no_grad():
        r = spectral_radius(assemble())
        if r > RHO_PROJ:
            for nm in ["weights_rr", "weights_pp", "weights_rp", "weights_pr"]: getattr(model.scm, nm).mul_(RHO_PROJ / r)
            return 1
    return 0

def finetune_op(ctx_d, train_units, opname, mc=MC, target="real", perm_seed=0, infonce=None, protocol="guard", n_steps=None, val_units=None, val_infonce=None):
    """protocol 'guard'  : paper CITE-seq protocol (reviewer_controls.finetune): stop when rho(A) > 0.92 or 30 steps w/o
                           improvement, keep the best-training-loss state.
       protocol 'project': Perturb-seq protocol (eval/replogle/ft_eval.py): rescale the SCM weights so rho(A) <= 0.95
                           after every step, run exactly n_steps; if val_units given, record the real-target validation
                           loss after every update (index = #updates)."""
    model.scm.load_state_dict(SCM0)
    opt = torch.optim.Adam([p for p in model.scm.parameters() if p.requires_grad], lr=mc["lr"], weight_decay=WD)
    TG = build_targets(ctx_d, train_units, target, perm_seed, with_nce=infonce is not None)
    TV = build_targets(ctx_d, val_units, "real", 0, with_nce=val_infonce is not None) if val_units is not None else None
    crit = []; nproj = 0
    if TV is not None:
        with torch.no_grad(): crit.append(objective(ctx_d, TV, opname, mc, val_infonce))
    best, best_state, since, best_step = float("inf"), None, 0, -1
    for step in range(n_steps or STEPS):
        opt.zero_grad(); total = objective(ctx_d, TG, opname, mc, infonce, backward=True)
        if mc["w_rho"] > 0:
            rl = mc["w_rho"] * torch.relu(spectral_norm(assemble()) - mc["rho_pen"]) ** 2; rl.backward(); total += float(rl.detach())
        torch.nn.utils.clip_grad_norm_([p for p in model.scm.parameters() if p.requires_grad], 1.0); opt.step()
        if protocol == "project":
            nproj += _project()
            if TV is not None:
                with torch.no_grad(): crit.append(objective(ctx_d, TV, opname, mc, val_infonce))
            continue
        rho = spectral_radius(assemble())
        if not np.isfinite(total): break
        if total < best - 1e-4:
            best, since, best_step = total, 0, step; best_state = {k: v.detach().clone() for k, v in model.scm.state_dict().items()}
        else: since += 1
        if rho > mc["rho_guard"] or since >= PATIENCE: break
    if protocol == "project":
        return dict(n_steps=n_steps or STEPS, n_proj=nproj, rho=spectral_radius(assemble()), crit=crit)
    if best_state is not None: model.scm.load_state_dict(best_state)
    return dict(best_step=best_step, best_loss=best, rho=spectral_radius(assemble()))

PSEQ_STEPS = 30 if QUICK else int(os.environ.get("OP_PSEQ_STEPS", "300"))
def fit_pseq(opname, target="real", perm_seed=0, infonce=None):
    """Perturb-seq protocol: inner 70/30 split of training genes (IN_TR/IN_VA) -> #steps minimising the real-target
    validation loss -> refit on all training genes for that many steps."""
    inner = finetune_op(CTX, IN_TR, opname, target=target, perm_seed=perm_seed, infonce=infonce, protocol="project",
                        n_steps=PSEQ_STEPS, val_units=IN_VA, val_infonce=infonce)
    best = int(np.nanargmin(inner["crit"]))
    info = finetune_op(CTX, TRAIN, opname, target=target, perm_seed=perm_seed, infonce=infonce, protocol="project", n_steps=best) if best > 0 else dict(n_steps=0)
    if best == 0: model.scm.load_state_dict(SCM0)
    info.update(selected_steps=best, inner_crit_first=inner["crit"][0], inner_crit_best=float(np.nanmin(inner["crit"]))); info.pop("crit", None)
    return info

@torch.no_grad()
def structure():
    A_rr = model.scm.get_block_adjacencies(tau=0.5)[0]; a = A_rr.float().cpu().numpy(); offm = ~np.eye(a.shape[0], dtype=bool)
    return dict(corr_Arr_ArrT=float(np.corrcoef(a[offm], a.T[offm])[0, 1]), rho=spectral_radius(assemble()))

FT_KEYS = ["rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc", "rr_pe_meanresp", "rp_pe", "rp_pe_cent", "rp_retrieval", "rp_auroc", "rp_pe_meanresp", "rr_pe_shuf", "dx_target"]
PROTOCOLS = os.environ.get("OP_PROTOCOLS", "guard,project").split(",")
if "fig5" in PARTS:
    def eval_now(opname, train=TRAIN, test=TEST):
        pt, ont = predict_units_op(CTX, {t: train[t] + test[t] for t in TAGS}, opname)
        rows = score(CTX, test, pt, ref_means(CTX, train, pt))
        for r in rows: r["dx_target"] = ont[(r["cohort"], r["gene"])]
        return rows
    F5 = {}
    for opn in FT_OPS:
      for proto in PROTOCOLS:
        R = {}; model.scm.load_state_dict(SCM0); key = f"{opn}|{proto}"
        R["baseline"] = dict(per_ko=eval_now(opn)); R["baseline"]["summary"] = summarize(R["baseline"]["per_ko"]); R["baseline"]["structure"] = structure()
        log(f"FT {key} baseline: {brief(R['baseline']['summary']['overall'])}")
        def arm(name, **kw):
            info = finetune_op(CTX, TRAIN, opn, **kw) if proto == "guard" else fit_pseq(opn, **kw)
            rows = eval_now(opn)
            R[name] = dict(info=info, per_ko=rows, summary=summarize(rows), structure=structure())
            log(f"FT {key} {name} [{ {k: v for k, v in info.items() if k in ('best_step', 'selected_steps', 'rho', 'n_proj')} }]: {brief(R[name]['summary']['overall'])}")
            F5[key] = R; RES["fig5"] = F5; save()
        arm("real")
        for k in range(1 if QUICK else 3): arm(f"shuffle_{k}", target="shuffle", perm_seed=k)
        arm("mean", target="mean")
        grid = {}
        for lam in ([1.0] if QUICK else NCE_GRID):
            if proto == "guard": finetune_op(CTX, IN_TR, opn, infonce=(lam, NCE_TAU))
            else:
                inner = finetune_op(CTX, IN_TR, opn, infonce=(lam, NCE_TAU), protocol="project", n_steps=PSEQ_STEPS, val_units=IN_VA, val_infonce=(lam, NCE_TAU))
                bs = int(np.nanargmin(inner["crit"])); finetune_op(CTX, IN_TR, opn, infonce=(lam, NCE_TAU), protocol="project", n_steps=bs) if bs > 0 else model.scm.load_state_dict(SCM0)
            pt, _ = predict_units_op(CTX, {t: IN_TR[t] + IN_VA[t] for t in TAGS}, opn); sm = summarize(score(CTX, IN_VA, pt, ref_means(CTX, IN_TR, pt)))["overall"]
            grid[str(lam)] = dict(lam=lam, inner_retr=sm["rr_retrieval"], inner_cent=sm["rr_pe_cent"], inner_pe=sm["rr_pe"])
        best = max(grid.values(), key=lambda v: v["inner_retr"]); arm("infonce", infonce=(best["lam"], NCE_TAU)); R["infonce"]["inner_grid"] = grid
        sh = [R[f"shuffle_{k}"]["per_ko"] for k in range(1 if QUICK else 3)]; avg = []
        for i, r in enumerate(sh[0]):
            a = dict(r)
            for k in [k for k in r if k.startswith(("rr_", "rp_", "dx_"))]:
                v = [s_[i].get(k, np.nan) for s_ in sh]; v = [x for x in v if isinstance(x, float) and x == x]; a[k] = float(np.mean(v)) if v else np.nan
            avg.append(a)
        R["shuffle_avg"] = dict(per_ko=avg, summary=summarize(avg))
        R["tests"] = {f"{a}_vs_{b}": {m: paired(R[b]["per_ko"], R[a]["per_ko"], m) for m in FT_KEYS}
                      for a, b in [("real", "shuffle_avg"), ("real", "mean"), ("real", "baseline"), ("infonce", "shuffle_avg"), ("infonce", "baseline"), ("shuffle_avg", "baseline"), ("mean", "baseline")]}
        F5[key] = R; RES["fig5"] = F5; save()
    model.scm.load_state_dict(SCM0)

RES["elapsed_s"] = time.time() - T0; save(); log("DONE ->", OUT)
