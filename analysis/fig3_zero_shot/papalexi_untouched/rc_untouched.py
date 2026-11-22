"""Reviewer-requested controls for the perturbation results (Fig. 3 / Fig. 5).

Built on the exact machinery of analysis/fig5_interventions/citeseq_finetune/finetune_combined.py (same model load, soft-do operator,
DE ground truth, by-gene split, fine-tuning loss) so every number is directly comparable.

Addresses:
  R1.1  response scale   : score on a COMMON scale. pred is in the standardized (z) decoder-output space;
                           observed = difference of mean log-normalized (RNA) / CLR (protein) expression.
                           'z'   : pred_z      vs obs / train_sd      (observed put on the z scale)
                           'log' : pred_z*sd   vs obs                 (prediction back-transformed)
                           'orig': pred_z      vs obs                 (as in the manuscript)
  R1.6  BH-FDR DE        : DE = BH q<0.05 (per KO, across features) & |d|>=floor; re-run principal results.
  R1.2  nested selection : fine-tune (w_dir,w_mag) chosen by an inner split of the TRAIN genes only.
  R1.4  source data      : per-KO scores (baseline + tuned), paired tests, exact n.
  R2    shared-response controls on Fig. 5:
          - mean-response predictor (model-free: mean observed response of the training KOs in the cohort)
          - Systema-style mean-centred pearDE (subtract the cohort mean response from pred and obs)
          - target shuffle (pred for g scored against observed response of another KO h, same cohort)
          - negative-control fine-tunes: label-shuffled targets, sign-flipped targets, mean-response targets
          - structural read-outs before/after FT: corr(A_rr,A_rr^T), corr(G_rr,G_rr^T), CollecTRI fwd/rev
        + RNA->protein pearDE excluding the knocked-out gene's own cognate protein
  Fig.3 baselines rescored with KNOCKDOWN-consistent signs and per-KO gene IDs (paired tests).

Usage: python analysis/common/reviewer_controls.py results/sweep_sparse/sp_dec_gl1000  [OUT json]
Env:   RC_BASELINES=1 (score totalVI/correlation/DeepSEM on Fig.3 set), RC_LC=1 (learning curve),
       RC_BH=1, RC_NESTED=1, RC_NEG=1, RC_QUICK=1 (smoke test: few steps/sizes)
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
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, ttest_1samp
from sklearn.metrics import roc_auc_score

REPO = f"{_CMC_ROOT}"
HERE = os.path.join(_CMC_ROOT, "analysis", "common")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "src"))
UNT = os.path.join(_CMC_RESULTS, "paper", "analyses", "untouched")
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

P.MAX_CONTROL_CELLS = 4000
RD = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
P.SCRIPT_DIR = os.path.join(UNT, "cache", "sd")   # corr_cache -> UNT/cache/results/corr_cache (never results/)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CACHE_DIR = os.path.join(UNT, "cache"); os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(os.path.join(UNT, "cache", "results", "corr_cache"), exist_ok=True)
RO_CACHE = os.path.join(REPO, "results", "_ftcache")   # read-only: dev-screen caches built for the paper
LFCR, LFCP = float(os.environ["CMC_DE_LFC_RNA"]), float(os.environ["CMC_DE_LFC_PROT"])
DEVICE = P.DEVICE
SEED = 42
QUICK = os.environ.get("RC_QUICK") == "1"
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
PDIR = os.path.join(UNT, "data", "papalexi_proc")
DATASETS["papalexi"] = (f"{PDIR}/papalexi_rna.h5ad", f"{PDIR}/papalexi_prot.h5ad")
DATASETS["papalexi_mix"] = (f"{PDIR}/papalexi_mix_rna.h5ad", f"{PDIR}/papalexi_mix_prot.h5ad")
DATASETS["papalexi_clean"] = (f"{PDIR}/papalexi_clean_rna.h5ad", f"{PDIR}/papalexi_clean_prot.h5ad")   # labelled deviation
TAGS = ["frangieh", "iccite", "arce"]
EXT = [t for t in os.environ.get("UNT_EXT", "papalexi,papalexi_mix").split(",") if t]
STEPS = 20 if QUICK else int(os.environ.get("CMC_FT_STEPS", "150"))
PATIENCE, WD = 30, 1e-4
N_TRAIN_CELLS = 3000   # paper: 3,000 cells (the original CPU fallback of 1,500 is NOT used)
MC = dict(lr=5e-4, rho_guard=0.92, w_dir=1.0, w_mag=1.0, w_rp=0.5, w_rho=0.1, rho_pen=0.9)
STD_FLOOR = 0.05
log(f"run={os.path.basename(RD)} device={DEVICE} steps={STEPS} quick={QUICK}")

# ------------------------------------------------------------------ model (identical to finetune_combined)
cfg = json.load(open(f"{RD}/config.json"))
_sd0 = torch.load(P.resolve_ckpt(RD), map_location="cpu")
N_G, N_P = _sd0["dec_rna.weight"].shape[0], _sd0["dec_prot.weight"].shape[0]; del _sd0
model = P.load_ctx_model(CrossModalCausalLDVAE, RD, N_G, N_P, cfg); model.eval()
for p in model.parameters(): p.requires_grad_(False)
for p in model.scm.parameters(): p.requires_grad_(True)
SCM0 = {k: v.detach().clone() for k, v in model.scm.state_dict().items()}
lr_, lp_ = model.latent_r, model.latent_p
Wr = model._dec_r_w().detach().to(DEVICE)
bias_rna = model.bias_rna.detach().to(DEVICE); bias_prot = model.bias_prot.detach().to(DEVICE)
egate = model._eps_gate()[0].to(DEVICE) if getattr(model, "eps_bottleneck", False) else None
solve = lambda Amat, e: model.scm._scm_solve(Amat, e)
SHOCK_MODE = os.environ.get("RC_SHOCK", "orig")
Z0 = None   # set after data load: z-value of zero log-normalized expression, per train gene

def assemble():
    A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
    return torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0).to(DEVICE)

def control_side(A_full, e):
    z_c = solve(A_full, e); zr_c, zp_c = torch.split(z_c, [lr_, lp_], 1)
    return model.dec_rna(zr_c) + bias_rna, model.dec_prot(zp_c) + bias_prot

MUT_DEFAULT = os.environ.get("RC_MUT", "1") == "1"
def predict_target(A_full, rec_r_c, rec_p_c, e, gidx, mutilate=None):
    if mutilate is None: mutilate = MUT_DEFAULT
    B = e.shape[0]
    dx = torch.zeros_like(rec_r_c)
    dx[:, gidx] = (Z0[gidx] - rec_r_c[:, gidx]) if SHOCK_MODE == "zero" else -rec_r_c[:, gidx]
    deps = torch.cat([dx @ Wr, torch.zeros(B, lp_, device=DEVICE)], 1)
    if egate is not None: deps = deps * egate.unsqueeze(0)
    Amat = A_full
    if mutilate:
        s = (Wr[gidx].abs() / (Wr[gidx].abs().max() + 1e-8)).clamp(0, 1)
        Amat = A_full.clone(); Amat[:, :lr_] = Amat[:, :lr_] * (1 - s).unsqueeze(0)
    z_p = solve(Amat, e + deps); zr_p, zp_p = torch.split(z_p, [lr_, lp_], 1)
    return (((model.dec_rna(zr_p) + bias_rna) - rec_r_c).mean(0),
            ((model.dec_prot(zp_p) + bias_prot) - rec_p_c).mean(0))

_sn = [torch.randn(lr_ + lp_, 1, device=DEVICE), torch.randn(lr_ + lp_, 1, device=DEVICE)]
def spectral_norm(A):
    with torch.no_grad():
        u, v = _sn
        for _ in range(12):
            u = A @ v; u = u / (u.norm() + 1e-12); v = A.t() @ u; v = v / (v.norm() + 1e-12)
        _sn[0], _sn[1] = u.detach(), v.detach()
    return (_sn[0] * (A @ _sn[1])).sum()

def spectral_radius(A):
    with torch.no_grad(): return float(torch.linalg.eigvals(A.detach().cpu()).abs().max())

def pearson_t(pred, obs):
    pc = pred - pred.mean(); oc = obs - obs.mean()
    return (pc * oc).sum() / (pc.norm() * oc.norm() + 1e-8)

# ------------------------------------------------------------------ metrics
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
    ro = f"{RO_CACHE}/rc_{tag}_{os.path.basename(RD)}_{h}_{LFCR}_{LFCP}.pkl"
    if tag in ("frangieh", "iccite", "arce") and os.path.exists(ro): cache = ro
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

RAW = {t: load_dataset(t) for t in TAGS + EXT}
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
    for t in train_units:
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

# ------------------------------------------------------------------ predictions + scoring
A_OVERRIDE = None
@torch.no_grad()
def predict_units(ctx_d, units):
    """returns {(t,g): (pred_rna_common[np], pred_prot_eval[np])} for the CURRENT scm state."""
    A = assemble() if A_OVERRIDE is None else A_OVERRIDE; out = {}
    for t, genes in units.items():
        c = ctx_d[t]; rr, rp = control_side(A, c["eps"])
        for g in genes:
            d_r, d_p = predict_target(A, rr, rp, c["eps"], c["INFO"][g]["gidx"])
            out[(t, g)] = (d_r.cpu().numpy()[c["reti"]].astype(float), d_p.cpu().numpy()[c["preti"]].astype(float))
    return out

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
    for scope in ["overall"] + sorted({r["cohort"] for r in rows}):
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
    for scope in ["overall"] + sorted({r["cohort"] for r in rows_b}):
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

def finetune(ctx_d, train_units, mc=MC, target="real", dir_scale="raw", perm_seed=0, infonce=None):
    """target: real | shuffle (targets permuted among training KOs within cohort) | flip (negated) | mean (cohort mean)
       dir_scale: raw (paper: Pearson vs raw obs) | z (Pearson vs z-scaled obs, scale-consistent with the MSE term)"""
    model.scm.load_state_dict(SCM0)
    opt = torch.optim.Adam([p for p in model.scm.parameters() if p.requires_grad], lr=mc["lr"], weight_decay=WD)
    tg = {t: [g for g in gl if ctx_d[t]["INFO"][g]["obs_r_raw"].numel() >= 5] for t, gl in train_units.items()}
    active = [t for t in train_units if tg.get(t)]
    TGT = {}
    for t in active:
        c = ctx_d[t]; genes = tg[t]
        src = list(genes)
        if target == "shuffle":
            src = list(np.array(genes)[np.random.RandomState(1000 + perm_seed).permutation(len(genes))])
        meanvec = None
        if target == "centered":
            _O = np.stack([np.where(c["INFO"][h]["keep"], c["INFO"][h]["lfc"], np.nan) for h in genes])
            _Op = np.stack([c["INFO"][h]["lfcp"] for h in genes]) if c["INFO"][genes[0]]["lfcp"].size else None
        if target == "mean":
            meanvec = np.nanmean(np.stack([np.where(c["INFO"][h]["keep"], c["INFO"][h]["lfc"], np.nan) for h in genes]), 0)
            meanp = np.nanmean(np.stack([c["INFO"][h]["lfcp"] for h in genes]), 0) if c["INFO"][genes[0]]["lfcp"].size else None
        for g, h in zip(genes, src):
            I = c["INFO"][g]; J = c["INFO"][h]
            if target == "centered":   # g's own response minus the leave-one-out cohort-mean training response
                j = genes.index(g); oth = [i for i in range(len(genes)) if i != j]
                mu = np.nanmean(_O[oth], 0); raw_np = (I["lfc"] - mu)[I["de_pos"]]
                raw = torch.as_tensor(raw_np, dtype=torch.float32, device=DEVICE)
                zz = raw / torch.as_tensor(np.clip(np.asarray(c["align"]["train_rna_std"])[c["reti"][I["de_pos"]]], STD_FLOOR, None), device=DEVICE).float()
                e = dict(sel_r=I["sel_r"], r_raw=raw, r_z=zz, has_p=False)
                if I["has_p_loss"] and _Op is not None:
                    mup = np.nanmean(_Op[oth], 0); rp_raw = torch.as_tensor((I["lfcp"] - mup)[I["dp_pos"]], dtype=torch.float32, device=DEVICE)
                    rp_z = rp_raw / torch.as_tensor(np.clip(np.asarray(c["align"]["train_prot_std"])[c["preti"][I["dp_pos"]]], STD_FLOOR, None), device=DEVICE).float()
                    e.update(has_p=True, sel_p=I["sel_p"], p_raw=rp_raw, p_z=rp_z)
            elif target == "mean":   # g's own DE features, but the cohort-mean response as target
                raw = torch.as_tensor(meanvec[I["de_pos"]], dtype=torch.float32, device=DEVICE)
                zz = raw / torch.as_tensor(np.clip(np.asarray(c["align"]["train_rna_std"])[c["reti"][I["de_pos"]]], STD_FLOOR, None), device=DEVICE).float()
                e = dict(sel_r=I["sel_r"], r_raw=raw, r_z=zz, has_p=False)
                if I["has_p_loss"] and meanp is not None:
                    rp_raw = torch.as_tensor(meanp[I["dp_pos"]], dtype=torch.float32, device=DEVICE)
                    rp_z = rp_raw / torch.as_tensor(np.clip(np.asarray(c["align"]["train_prot_std"])[c["preti"][I["dp_pos"]]], STD_FLOOR, None), device=DEVICE).float()
                    e.update(has_p=True, sel_p=I["sel_p"], p_raw=rp_raw, p_z=rp_z)
            else:
                sgn = -1.0 if target == "flip" else 1.0
                e = dict(sel_r=J["sel_r"], r_raw=sgn * J["obs_r_raw"], r_z=sgn * J["obs_r_z"], has_p=J["has_p_loss"])
                if J["has_p_loss"]:
                    e.update(sel_p=J["sel_p"], p_raw=sgn * J["obs_p_raw"], p_z=sgn * J["obs_p_z"])
            TGT[(t, g)] = (I["gidx"], e)
    n_ds = max(1, len(active)); best, best_state, since = float("inf"), None, 0; best_step = -1
    NCE_O = {t: nce_targets(ctx_d, t, tg[t]) for t in active} if infonce is not None else {}
    for step in range(STEPS):
        opt.zero_grad(); total = 0.0
        for t in active:
            c = ctx_d[t]; e_tr = c["eps"][:N_TRAIN_CELLS]; A = assemble(); rr, rp = control_side(A, e_tr)
            L_rr, L_rp = [], []; NCE_P = []
            for g in tg[t]:
                gidx, e = TGT[(t, g)]
                d_r, d_p = predict_target(A, rr, rp, e_tr, gidx)
                if infonce is not None: NCE_P.append(d_r[ctx_d[t]["S_tr"]])
                pr = d_r[e["sel_r"]]; l = torch.zeros((), device=DEVICE)
                tgt_dir = e["r_raw"] if dir_scale == "raw" else e["r_z"]
                if mc["w_dir"] > 0 and pr.numel() >= 5 and torch.std(pr) > 1e-12: l = l + mc["w_dir"] * (-pearson_t(pr, tgt_dir))
                if mc["w_mag"] > 0: l = l + mc["w_mag"] * ((pr - e["r_z"]) ** 2).mean()
                L_rr.append(l)
                if mc["w_rp"] > 0 and e["has_p"]:
                    pq = d_p[e["sel_p"]]; lp = torch.zeros((), device=DEVICE)
                    tgt_p = e["p_raw"] if dir_scale == "raw" else e["p_z"]
                    if mc["w_dir"] > 0 and pq.numel() >= 3 and torch.std(pq) > 1e-12: lp = lp + mc["w_dir"] * (-pearson_t(pq, tgt_p))
                    if mc["w_mag"] > 0: lp = lp + mc["w_mag"] * ((pq - e["p_z"]) ** 2).mean()
                    L_rp.append(lp)
            if not L_rr: continue
            ds = torch.stack(L_rr).mean() + (mc["w_rp"] * torch.stack(L_rp).mean() if L_rp else 0.0)
            if infonce is not None and len(NCE_P) >= 3:
                ds = ds + infonce[0] * info_nce(torch.stack(NCE_P), NCE_O[t], infonce[1])
            (ds / n_ds).backward(); total += float(ds.detach()) / n_ds
        if mc["w_rho"] > 0:
            rl = mc["w_rho"] * torch.relu(spectral_norm(assemble()) - mc["rho_pen"]) ** 2; rl.backward(); total += float(rl.detach())
        torch.nn.utils.clip_grad_norm_([p for p in model.scm.parameters() if p.requires_grad], 1.0); opt.step()
        rho = spectral_radius(assemble())
        if total < best - 1e-4:
            best, since, best_step = total, 0, step; best_state = {k: v.detach().clone() for k, v in model.scm.state_dict().items()}
        else: since += 1
        if rho > mc["rho_guard"] or since >= PATIENCE: break
    if best_state is not None: model.scm.load_state_dict(best_state)
    return dict(best_step=best_step, best_loss=best, rho=spectral_radius(assemble()))

# ------------------------------------------------------------------ structural read-outs
_genes_train = list(map(str, CTX["frangieh"]["align"]["train_gene_names"]))
def _collectri():
    try:
        d = torch.load(cfg["prior_rna_path"], map_location="cpu"); m = d["prior_mask"]; m = m.numpy() if torch.is_tensor(m) else np.asarray(m)
        nm = list(map(str, d.get("gene_names", []))); gi = {g: i for i, g in enumerate(nm)}
        if nm != _genes_train:
            perm = [gi[g] for g in _genes_train]; m = m[np.ix_(perm, perm)]
        ii, jj = np.where(m.astype(bool)); k = ii != jj; return ii[k], jj[k]
    except Exception as e:
        log("collectri load failed", str(e)[:100]); return None
CTRI = _collectri()
Dr = model._dec_r_w().detach().float().cpu()

@torch.no_grad()
def structure():
    A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
    a = A_rr.float().cpu().numpy(); off = ~np.eye(a.shape[0], dtype=bool)
    G = (Dr @ A_rr.float().cpu() @ Dr.T).numpy(); np.fill_diagonal(G, 0.0)
    gi = np.triu_indices(G.shape[0], 1)
    out = dict(corr_Arr_ArrT=float(np.corrcoef(a[off], a.T[off])[0, 1]),
               corr_Grr_GrrT=float(np.corrcoef(G[gi], G.T[gi])[0, 1]),
               rho=spectral_radius(assemble()))
    if CTRI is not None:
        f, r = np.abs(G[CTRI[0], CTRI[1]]), np.abs(G[CTRI[1], CTRI[0]])
        out.update(collectri_n=int(len(f)), collectri_fwd_over_rev_median=float(np.median(f) / (np.median(r) + 1e-12)),
                   collectri_frac_fwd_gt_rev=float(np.mean(f > r)),
                   collectri_mwu_p_fwd_gt_rev=float(mannwhitneyu(f, r, alternative="greater").pvalue))
    return out

def a_change():
    A0 = None
    with torch.no_grad():
        cur = assemble().cpu().numpy()
    return cur

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

# ================================================================== RUN (untouched-screen evaluation)
from sklearn.linear_model import RidgeCV
import anndata as _ad
RES = dict(run=os.path.basename(RD), split_matches_paper=SPLIT_OK, steps=STEPS, n_train_cells=N_TRAIN_CELLS, de_floor=[LFCR, LFCP],
           device=str(DEVICE), ext=EXT)
def save():
    json.dump(RES, open(OUT, "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else str(o))
PRED_NPZ = {}   # every Papalexi prediction vector, for the external stats / bootstrap script
def keep_preds(name, preds):
    for (t, g), (r, p) in preds.items():
        if t in EXT: PRED_NPZ[f"{name}|{t}|{g}|r"] = np.asarray(r, np.float32); PRED_NPZ[f"{name}|{t}|{g}|p"] = np.asarray(p, np.float32)
def save_npz(): np.savez_compressed(OUT.replace(".json", "_preds.npz"), **PRED_NPZ)
def brief(s):
    return {k: (round(s[k], 4) if isinstance(s[k], float) and s[k] == s[k] else s[k]) for k in
            ["rr_auroc", "rr_pe", "rr_pe_log", "rr_pe_cent", "rr_retrieval", "rr_pe_meanresp", "rr_auroc_meanresp", "rr_pe_shuf",
             "rp_auroc", "rp_pe", "rp_retrieval", "n_rr_auroc", "n_rr_pe", "n_rr_pe_cent", "n_rr_retrieval", "n_rp_auroc"] if k in s}

ALL = {t: CTX[t]["evaluable"] for t in TAGS}
# ---- (0) dev-screen fidelity check (must reproduce results/reviewer_controls/rc_<run>.json fig3_obs)
model.scm.load_state_dict(SCM0)
P_obs_all = predict_units(CTX, ALL); rows_dev = score(CTX, ALL, P_obs_all, ref_means(CTX, ALL, P_obs_all))
RES["dev_fig3_obs_summary"] = summarize(rows_dev)
try:
    _rc = json.load(open(f"{_CMC_RESULTS}/reviewer_controls/rc_{os.path.basename(RD)}.json"))["fig3_obs"]["summary"]["overall"]
    RES["dev_fig3_fidelity"] = {k: [RES["dev_fig3_obs_summary"]["overall"][k], _rc[k]] for k in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rp_pe"]}
except Exception as e: RES["dev_fig3_fidelity"] = str(e)
log("dev fidelity [this run, paper json]:", RES["dev_fig3_fidelity"])

# ---- external contexts
CTXE = {t: build_ctx(t) for t in EXT}
for t in EXT:
    set_S(CTXE, {t: CTXE[t]["evaluable"]})
    c = CTXE[t]; al = c["align"]
    RES[f"{t}_units"] = dict(evaluable=c["evaluable"], n=len(c["evaluable"]), n_common_genes=int(len(c["common"])), proteins=list(al["prot_eval_names"]),
                             n_S=int(len(c["S"])), n_cells={g: int(RAW[t][1][g]["n_cells"]) for g in RAW[t][1]},
                             n_de_rna={g: int((np.asarray(RAW[t][1][g]["sig_rna"]) & (c["common"] != g)).sum()) for g in RAW[t][1]},
                             n_de_prot={g: int(np.asarray(RAW[t][1][g]["sig_prot"]).sum()) for g in RAW[t][1]}, n_eps=int(c["eps"].shape[0]))
    log(t, RES[f"{t}_units"])
save()

DEV_TARGETS = set(g for t in TAGS for g in RAW[t][1])
def tag_rows(rows):
    for r in rows: r["dev_target"] = r["gene"] in DEV_TARGETS
    return rows

# ---- baselines on the external contexts (knockdown sign)
tv = np.load(f"{_CMC_RESULTS}/baselines/totalvi_raw.npz", allow_pickle=True)
Grr_tv = np.asarray(tv["G_rr_tv"], float); Grp_tv = np.asarray(tv["G_rp_tv"], float); np.fill_diagonal(Grr_tv, 0.0)
tvg = {g: i for i, g in enumerate(np.asarray(tv["genes"]).astype(str))}
dsz = np.load(f"{_CMC_RESULTS}/baselines/deepsem_W_809k_clamp09.npz", allow_pickle=True)
Wd = np.asarray(dsz["W"], float); np.fill_diagonal(Wd, 0.0); dsg = {g: i for i, g in enumerate(np.asarray(dsz["genes"]).astype(str))}
Minv = np.linalg.inv(np.eye(Wd.shape[0]) - Wd.T)
Dr_np = Wr.detach().cpu().numpy(); Dp_np = model._dec_p_w().detach().cpu().numpy() if hasattr(model, "_dec_p_w") else model.dec_prot.weight.detach().cpu().numpy()
def ctrl_stats(t):
    """mean / variance of control cells on the stored (log-norm / CLR) layer, for the zero-shot rankings."""
    pr_path, pp_path = DATASETS[t]; c = CTXE[t]; al = c["align"]
    a = _ad.read_h5ad(pr_path); ctl = P.parse_sgRNA_to_gene(a.obs["sgRNA"]).values == "control"
    X = a[ctl][:, list(c["common"])].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    b = _ad.read_h5ad(pp_path); inv = {v: k for k, v in al["prot_map"].items()}
    Y = b[ctl][:, [inv[n] for n in al["prot_eval_names"]]].X; Y = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)
    return X.mean(0), X.var(0), Y.mean(0), Y.var(0)
def dev_copy_pred(t, g):
    """same gene's measured response in a development screen (Frangieh > Chen > Arce), mapped to this context; 0 where unmeasured."""
    c = CTXE[t]
    for d in TAGS:
        if g in RAW[d][1]:
            ald, de = RAW[d][0], RAW[d][1][g]; m = dict(zip(ald["common_genes"], np.asarray(de["lfc_rna"], float)))
            mp = dict(zip(ald["prot_eval_names"], np.asarray(de["lfc_prot"], float))) if len(de["lfc_prot"]) else {}
            return (np.array([m.get(x, 0.0) for x in c["common"]]), np.array([mp.get(x, 0.0) for x in c["align"]["prot_eval_names"]])), d
    return None, None
def baseline_preds(t):
    c = CTXE[t]; al = c["align"]; common = c["common"]; g2i = c["g2i"]; n_p = len(c["preti"])
    ctr, cgp = al.get("corr_train"), al.get("gene_prot_corr")
    tvcol = np.array([tvg.get(x, -1) for x in common]); dscol = np.array([dsg.get(x, -1) for x in common])
    mr, vr_, mpq, vpq = ctrl_stats(t)
    drn_r = np.linalg.norm(Dr_np[c["reti"]], axis=1); drn_p = np.linalg.norm(Dp_np[c["preti"]], axis=1) if n_p else np.zeros(0)
    out = {k: {} for k in ["corr", "totalvi", "deepsem", "zs_ctrl_expr", "zs_ctrl_var", "zs_decoder_rownorm", "dev_copy"]}; src = {}
    for g in c["evaluable"]:
        out["corr"][(t, g)] = (-np.asarray(ctr[g2i[g], :], float), -np.asarray(cgp[g2i[g], :], float) if (cgp is not None and n_p) else np.zeros(n_p))
        if g in tvg:
            vr = np.array([Grr_tv[tvg[g], j] if j >= 0 else 0.0 for j in tvcol]); vp = Grp_tv[tvg[g], c["preti"]] if n_p else np.zeros(0)
            out["totalvi"][(t, g)] = (-vr, -vp)
        if g in dsg:
            vr = np.array([Minv[j, dsg[g]] if j >= 0 else 0.0 for j in dscol]); out["deepsem"][(t, g)] = (-vr, np.full(n_p, np.nan))
        out["zs_ctrl_expr"][(t, g)] = (-mr, -mpq); out["zs_ctrl_var"][(t, g)] = (-vr_, -vpq)
        out["zs_decoder_rownorm"][(t, g)] = (-drn_r, -drn_p)
        cp, d = dev_copy_pred(t, g)
        if cp is not None: out["dev_copy"][(t, g)] = cp; src[g] = d
    RES[f"{t}_dev_copy_source"] = src
    return out

def eval_ext(t, preds, units=None, refs_units=None):
    c = CTXE[t]; units = units or c["evaluable"]; units = [g for g in units if (t, g) in preds]
    ru = refs_units or units
    rows = score(CTXE, {t: units}, preds, ref_means(CTXE, {t: ru}, preds))
    return tag_rows(rows)

# ---- (A) PRIMARY: frozen observational model, zero-shot on the untouched screen(s)
for t in EXT:
    model.scm.load_state_dict(SCM0)
    p_obs = predict_units(CTXE, {t: CTXE[t]["evaluable"]}); keep_preds("cmc_obs", p_obs)
    rows_obs = eval_ext(t, p_obs)
    with torch.no_grad():
        _A0 = assemble(); rr_, rp_ = control_side(_A0, CTXE[t]["eps"])
        dec = []
        for g in CTXE[t]["evaluable"]:
            gi_ = CTXE[t]["INFO"][g]["gidx"]; full = predict_target(_A0, rr_, rp_, CTXE[t]["eps"], gi_, mutilate=True)[0]
            sh = predict_target(_A0, rr_, rp_, CTXE[t]["eps"], gi_, mutilate=False)[0]
            ri = torch.as_tensor(CTXE[t]["reti"], device=DEVICE); f, a_ = full[ri], sh[ri]
            dec.append(dict(gene=g, shock_over_full=float(a_.norm() / (f.norm() + 1e-12))))
    blk = dict(cmc=dict(summary=summarize(rows_obs), per_ko=rows_obs, similarity=similarity(CTXE, {t: CTXE[t]["evaluable"]}, p_obs), doop=dec))
    log(f"[{t}] CMC obs:", brief(blk["cmc"]["summary"]["overall"]))
    for name, pb in baseline_preds(t).items():
        keep_preds(name, pb)
        if len([g for g in CTXE[t]["evaluable"] if (t, g) in pb]) < 3: blk[name] = dict(n=len(pb)); continue
        rows_b = eval_ext(t, pb)
        blk[name] = dict(summary=summarize(rows_b), per_ko=rows_b,
                         paired_cmc_minus_base={m: paired(rows_b, rows_obs, m) for m in ["rr_auroc", "rr_pe", "rr_pe_log", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe", "rp_retrieval"]})
        log(f"[{t}] {name}:", brief(blk[name]["summary"]["overall"]))
    RES[f"{t}_zeroshot"] = blk; save(); save_npz()

# ---- (B) S2 cross-screen transfer: paper's Fig-5 fine-tuning protocol on dev TRAIN genes -> apply to Papalexi
model.scm.load_state_dict(SCM0)
P_obs_tt = predict_units(CTX, {t: TRAIN[t] + TEST[t] for t in TAGS}); rows_base_dev = score(CTX, TEST, P_obs_tt, ref_means(CTX, TRAIN, P_obs_tt))
RES["dev_fig5_baseline"] = summarize(rows_base_dev)["overall"]
TRANSFER_ARMS = [("ft_standard", {}), ("ft_labelshuffle_0", dict(target="shuffle", perm_seed=0)), ("ft_labelshuffle_1", dict(target="shuffle", perm_seed=1)),
                 ("ft_labelshuffle_2", dict(target="shuffle", perm_seed=2)), ("ft_meantarget", dict(target="mean"))]
if os.environ.get("UNT_SKIP_TRANSFER") != "1":
    TR = {}
    for arm, kw in TRANSFER_ARMS:
        info = finetune(CTX, TRAIN, **kw)
        pt = predict_units(CTX, {t: TRAIN[t] + TEST[t] for t in TAGS}); rows_d = score(CTX, TEST, pt, ref_means(CTX, TRAIN, pt))
        ent = dict(info=info, dev_heldout=summarize(rows_d)["overall"], structure=structure())
        try: ent["dev_heldout_paper"] = {k: _rcj[arm]["summary"]["overall"][k] for k in ["rr_pe", "rr_pe_cent", "rr_retrieval"]}
        except Exception:
            try:
                _rcj = json.load(open(f"{_CMC_RESULTS}/reviewer_controls/rc_{os.path.basename(RD)}.json"))
                ent["dev_heldout_paper"] = {k: _rcj[arm]["summary"]["overall"][k] for k in ["rr_pe", "rr_pe_cent", "rr_retrieval"]}
            except Exception as e: ent["dev_heldout_paper"] = str(e)[:80]
        for t in EXT:
            pe_ = predict_units(CTXE, {t: CTXE[t]["evaluable"]}); keep_preds(f"transfer_{arm}", pe_)
            rows_t = eval_ext(t, pe_)
            ent[t] = dict(summary=summarize(rows_t), per_ko=rows_t,
                          paired_vs_obs={m: paired(RES[f"{t}_zeroshot"]["cmc"]["per_ko"], rows_t, m) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]})
        TR[arm] = ent
        log(f"transfer {arm}: dev held-out", {k: round(ent['dev_heldout'][k], 3) for k in ['rr_pe', 'rr_pe_cent', 'rr_retrieval']}, "paper", ent["dev_heldout_paper"],
            "| papalexi", brief(ent[EXT[0]]["summary"]["overall"]))
        RES["transfer"] = TR; save(); save_npz()
    model.scm.load_state_dict(SCM0)

# ---- ridge helpers (decoder loadings / co-expression profile -> response)
ALPHAS = np.logspace(-2, 4, 13)
_inter = sorted(set.intersection(*[set(RAW[t][0]["common_genes"]) for t in TAGS + EXT]))
def coexpr_X(t_src, g):
    al = RAW[t_src][0]; g2i = al["train_gene_to_idx_common"]; idx = [g2i[x] for x in _inter]
    return np.asarray(al["corr_train"][g2i[g], idx], float)
def ridge_fit_predict(Xtr, Ytr, Xte):
    mu = Ytr.mean(0); r = RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(Xtr, Ytr - mu); return r.predict(Xte) + mu, float(r.alpha_)

# ---- (C) S2 ridge transfer: fitted on ALL dev units (responses centred within screen), applied to Papalexi
if os.environ.get("UNT_SKIP_TRANSFER") != "1":
    RT = {}
    vocab = list(map(str, RAW["frangieh"][0]["train_gene_names"])); vi = {g: i for i, g in enumerate(vocab)}
    Ys, Xd, Xc = [], [], []
    for d in TAGS:
        al = RAW[d][0]; cidx = np.array([vi[x] for x in al["common_genes"]]); units = CTX[d]["evaluable"]
        Yd = np.full((len(units), len(vocab)), np.nan)
        for i, g in enumerate(units):
            l = np.asarray(RAW[d][1][g]["lfc_rna"], float).copy(); l[np.asarray(al["common_genes"]) == g] = np.nan; Yd[i, cidx] = l
        Yd = Yd - np.nanmean(Yd, 0, keepdims=True); Ys.append(np.nan_to_num(Yd))
        Xd += [Dr_np[al["train_gene_idx"][g]] for g in units]; Xc += [coexpr_X(d, g) for g in units]
    Ydev = np.vstack(Ys); Xd = np.vstack(Xd); Xc = np.vstack(Xc)
    for t in EXT:
        c = CTXE[t]; ev = c["evaluable"]; n_p = len(c["preti"])
        out = {}
        for nm, Xtr, Xte in [("ridge_decoder", Xd, np.vstack([Dr_np[c["INFO"][g]["gidx"]] for g in ev])),
                             ("ridge_coexpr", Xc, np.vstack([coexpr_X(t, g) for g in ev]))]:
            Yhat, a_ = ridge_fit_predict(Xtr, Ydev[:, c["reti"]], Xte)
            pb = {(t, g): (Yhat[i], np.full(n_p, np.nan)) for i, g in enumerate(ev)}; keep_preds(f"transfer_{nm}", pb)
            rows_r = eval_ext(t, pb)
            out[nm] = dict(alpha=a_, summary=summarize(rows_r), per_ko=rows_r,
                           paired_cmc_minus_base={m: paired(rows_r, RES[f"{t}_zeroshot"]["cmc"]["per_ko"], m) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]})
            log(f"[{t}] {nm} transfer:", brief(out[nm]["summary"]["overall"]))
        RT[t] = out
    RES["ridge_transfer"] = RT; save(); save_npz()

# ---- (D) S1 within-Papalexi 3-fold gene CV fine-tuning (frozen hyperparameters) + ridge
CV_ARMS = [("obs", None), ("ft_real", {}), ("ft_shuffle_0", dict(target="shuffle", perm_seed=0)), ("ft_shuffle_1", dict(target="shuffle", perm_seed=1)),
           ("ft_shuffle_2", dict(target="shuffle", perm_seed=2)), ("ft_mean", dict(target="mean"))]
if os.environ.get("UNT_SKIP_CV") != "1":
    CV = {}
    for t in EXT:
        c = CTXE[t]; ev = list(c["evaluable"]); n_p = len(c["preti"])
        perm = np.random.RandomState(SEED).permutation(len(ev)); folds = [sorted(ev[i] for i in perm[k::3]) for k in range(3)]
        rows_arm = {a: [] for a, _ in CV_ARMS}; rows_arm.update(ridge_decoder=[], ridge_coexpr=[]); finfo = []
        for k in range(3):
            te = folds[k]; tr = [g for g in ev if g not in te]
            set_S(CTXE, {t: tr})
            for arm, kw in CV_ARMS:
                if kw is None: model.scm.load_state_dict(SCM0); inf = None
                else: inf = finetune(CTXE, {t: tr}, **kw)
                pt = predict_units(CTXE, {t: ev}); keep_preds(f"cv{k}_{arm}", {kk: v for kk, v in pt.items() if kk[1] in te})
                rr = tag_rows(score(CTXE, {t: te}, pt, ref_means(CTXE, {t: tr}, pt)))
                for r in rr: r["fold"] = k
                rows_arm[arm] += rr; finfo.append(dict(fold=k, arm=arm, info=inf))
            for nm, fx in [("ridge_decoder", lambda g: Dr_np[c["INFO"][g]["gidx"]]), ("ridge_coexpr", lambda g: coexpr_X(t, g))]:
                Ytr = np.stack([np.where(c["INFO"][g]["keep"], c["INFO"][g]["lfc"], 0.0) for g in tr])
                Yhat, a_ = ridge_fit_predict(np.vstack([fx(g) for g in tr]), Ytr, np.vstack([fx(g) for g in ev]))
                pb = {(t, g): (Yhat[i], np.full(n_p, np.nan)) for i, g in enumerate(ev)}
                keep_preds(f"cv{k}_{nm}", {kk: v for kk, v in pb.items() if kk[1] in te})
                rr = tag_rows(score(CTXE, {t: te}, pb, ref_means(CTXE, {t: tr}, pb)))
                for r in rr: r["fold"] = k
                rows_arm[nm] += rr
            model.scm.load_state_dict(SCM0)
            log(f"[{t}] CV fold {k} done (train {len(tr)}, test {len(te)})")
        ent = dict(folds=folds, fit_info=finfo, arms={a: dict(summary=summarize(v), per_ko=v) for a, v in rows_arm.items()})
        MET = ["rr_auroc", "rr_pe", "rr_pe_log", "rr_pe_cent", "rr_retrieval", "rr_pe_meanresp"]
        ent["paired"] = {f"ft_real_vs_{b}": {m: paired(rows_arm[b], rows_arm["ft_real"], m) for m in MET}
                         for b in ["obs", "ft_shuffle_0", "ft_shuffle_1", "ft_shuffle_2", "ft_mean", "ridge_decoder", "ridge_coexpr"]}
        ent["paired"].update({f"obs_vs_{b}": {m: paired(rows_arm[b], rows_arm["obs"], m) for m in MET} for b in ["ridge_decoder", "ridge_coexpr"]})
        CV[t] = ent
        for a in rows_arm: log(f"[{t}] CV {a}:", brief(ent["arms"][a]["summary"]["overall"]))
        RES["cv"] = CV; save(); save_npz()
        set_S(CTXE, {t: CTXE[t]["evaluable"]})
RES["elapsed_s"] = time.time() - T0; save(); save_npz(); log("DONE")
