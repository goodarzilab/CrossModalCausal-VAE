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

HERE = f"{_CMC_ROOT}/analysis/common"  # copied from analysis/common/reviewer_controls.py (lines 1-403); unmodified logic
sys.path.insert(0, HERE)
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

P.MAX_CONTROL_CELLS = 4000
RD = os.path.abspath(sys.argv[1])
REPO = _CMC_ROOT
OUT = os.path.join(_CMC_RESULTS, "paper", "analyses", "baselines", f"zeroshot_{os.path.basename(RD)}.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CACHE_DIR = os.path.join(REPO, "results", "_ftcache"); os.makedirs(CACHE_DIR, exist_ok=True)
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
TAGS = ["frangieh", "iccite", "arce"]
STEPS = 20 if QUICK else int(os.environ.get("CMC_FT_STEPS", "150"))
PATIENCE, WD = 30, 1e-4
N_TRAIN_CELLS = 3000 if DEVICE.type == "cuda" else 1500
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

