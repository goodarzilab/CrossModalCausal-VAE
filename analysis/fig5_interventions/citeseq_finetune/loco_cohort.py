"""LOCO (leave-one-cohort-out) fine-tuning with shared-response controls. Definitions copied verbatim from analysis/common/reviewer_controls.py (lines 1-555); LOCO logic appended below.
Original header follows.

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

HERE = os.path.join(_CMC_ROOT, "analysis", "common")   # [public release] was eval/
sys.path.insert(0, HERE)
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

P.MAX_CONTROL_CELLS = 4000
RD = os.path.abspath(sys.argv[1])
REPO = _CMC_ROOT
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "results", "reviewer_controls", f"rc_{os.path.basename(RD)}.json")
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
    active = [t for t in TAGS if tg.get(t)]
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


# ================================================================== LOCO (leave-one-cohort-out)
# Env: LOCO_HOLD = held-out cohort; LOCO_MODE = orig (reference optimiser: 150 steps, rho>0.92 guard) |
#      relaxed (300 steps, project rho<=0.95 by rescaling SCM weights, early stop on inner 20% split of training KOs)
N_TRAIN_CELLS = int(os.environ.get("LOCO_NCELLS", N_TRAIN_CELLS))   # CPU runs: force the GPU value (3000)
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
HOLD = os.environ["LOCO_HOLD"]; MODE = os.environ.get("LOCO_MODE", "orig"); assert HOLD in TAGS and MODE in ("orig", "relaxed")
TRC = [t for t in TAGS if t != HOLD]
ALL = {t: CTX[t]["evaluable"] for t in TAGS}
TR_U = {t: list(ALL[t]) for t in TRC}
NTR_G = len(_al0["train_rna_std"]); NTR_P = len(_al0["train_prot_std"])
for t in TAGS:
    assert len(CTX[t]["align"]["train_rna_std"]) == NTR_G and len(CTX[t]["align"]["train_prot_std"]) == NTR_P
SH = sorted(g for g in ALL[HOLD] if any(g in set(ALL[t]) for t in TRC))      # genes perturbed in HOLD and >=1 training cohort
N_SHUF = int(os.environ.get("LOCO_NSHUF", "3"))
log(f"LOCO hold={HOLD} mode={MODE} train cohorts={TRC} n_train_KO={sum(len(v) for v in TR_U.values())} n_test_KO={len(ALL[HOLD])} shared={SH}")

def r2train(t, v):
    w = np.full(NTR_G, np.nan); w[CTX[t]["reti"]] = v; return w
def p2train(t, v):
    w = np.full(NTR_P, np.nan)
    if len(v): w[CTX[t]["preti"]] = v
    return w
def obs_r(t, g):
    I = CTX[t]["INFO"][g]; return np.where(I["keep"], I["lfc"], np.nan)

# ---- feature sets: training cohorts use their own KOs; HOLD uses training-cohort DE genes only (no test labels)
ALLTG = sorted(set().union(*[set(ALL[t]) for t in TAGS]))
def set_S_loco():
    utr = np.zeros(NTR_G, bool)
    for t in TRC:
        c = CTX[t]; u = np.zeros(len(c["common"]), bool)
        for g in ALL[t]: u |= c["INFO"][g]["sig"]
        utr[c["reti"][u]] = True
        u &= ~np.isin(c["common"], ALLTG); c["S"] = np.where(u)[0]
    c = CTX[HOLD]; c["S"] = np.where(utr[c["reti"]] & ~np.isin(c["common"], ALLTG))[0]
    for t in TAGS:
        c = CTX[t]; c["S_tr"] = torch.as_tensor(c["reti"][c["S"]], dtype=torch.long, device=DEVICE)
        cg = np.zeros(len(c["preti"]), bool)
        for g in ALL[t]: cg |= c["INFO"][g]["cog_mask"]
        c["Sp"] = np.where(~cg)[0]
set_S_loco()
log("S sizes", {t: len(CTX[t]["S"]) for t in TAGS})

# ---- model-free: cohort mean responses and how much they agree across cohorts
def _corr(a, b):
    f = np.isfinite(a) & np.isfinite(b)
    return (float(np.corrcoef(a[f], b[f])[0, 1]), int(f.sum())) if f.sum() > 5 else (float("nan"), int(f.sum()))
MU = {t: np.nanmean(np.stack([r2train(t, obs_r(t, g)) for g in ALL[t]]), 0) for t in TAGS}
MUP = {t: np.nanmean(np.stack([p2train(t, CTX[t]["INFO"][g]["lfcp"]) for g in ALL[t]]), 0) for t in TAGS}
DEF = {t: np.nanmean(np.stack([r2train(t, CTX[t]["INFO"][g]["sig"].astype(float)) for g in ALL[t]]), 0) for t in TAGS}  # DE frequency
def _ko_mat(t):
    return np.stack([r2train(t, obs_r(t, g)) for g in ALL[t]])
def mean_pairwise(Ma, Mb, same):
    f = np.isfinite(Ma).all(0) & np.isfinite(Mb).all(0)
    A = Ma[:, f]; B = Mb[:, f]; A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-12); B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-12)
    C = A @ B.T / A.shape[1]
    if same: C = C[~np.eye(len(C), dtype=bool)]
    return float(np.nanmean(C)), int(f.sum())
MEANCORR = {}
_M = {t: np.nan_to_num(_ko_mat(t), nan=np.nan) for t in TAGS}
for i, a in enumerate(TAGS):
    for b in TAGS[i:]:
        ga = np.isfinite(MU[a]) & np.isfinite(MU[b]); de = ga & ((DEF[a] >= 0.05) | (DEF[b] >= 0.05))
        r_all, n_all = _corr(MU[a], MU[b]); r_de, n_de = _corr(np.where(de, MU[a], np.nan), np.where(de, MU[b], np.nan))
        # per-KO similarity (each KO's own gene is NaN -> drop such columns per pair via all-finite mask on the de set)
        Ma = np.where(de, _M[a], np.nan); Mb = np.where(de, _M[b], np.nan)
        Ma = np.where(np.isfinite(Ma), Ma, 0.0)[:, de]; Mb = np.where(np.isfinite(Mb), Mb, 0.0)[:, de]
        pw, _ = mean_pairwise(Ma, Mb, same=(a == b))
        rp_, np_ = _corr(MUP[a], MUP[b])
        MEANCORR[f"{a}|{b}"] = dict(mean_resp_corr_allgenes=r_all, n_allgenes=n_all, mean_resp_corr_DEgenes=r_de, n_DEgenes=n_de,
                                   mean_pairwise_KO_corr_DEgenes=pw, mean_prot_resp_corr=rp_, n_prot=np_)
MU_TR = np.nanmean(np.stack([r2train(t, obs_r(t, g)) for t in TRC for g in ALL[t]]), 0)
MEANCORR[f"{HOLD}|pooled_train"] = dict(zip(["mean_resp_corr_allgenes", "n_allgenes"], _corr(MU[HOLD], MU_TR)))
log("cohort mean-response agreement:", json.dumps(MEANCORR))

# ---- fine-tuning (targets copied from reference finetune(); optimiser orig or relaxed)
def build_targets(ctx_d, train_units, target, perm_seed=0):
    tg = {t: [g for g in gl if ctx_d[t]["INFO"][g]["obs_r_raw"].numel() >= 5] for t, gl in train_units.items()}
    TGT = {}
    for t in [t for t in TAGS if tg.get(t)]:
        c = ctx_d[t]; genes = tg[t]; src = list(genes)
        if target == "shuffle":
            src = list(np.array(genes)[np.random.RandomState(1000 + perm_seed).permutation(len(genes))])
        if target == "centered":
            _O = np.stack([np.where(c["INFO"][h]["keep"], c["INFO"][h]["lfc"], np.nan) for h in genes])
            _Op = np.stack([c["INFO"][h]["lfcp"] for h in genes]) if c["INFO"][genes[0]]["lfcp"].size else None
        if target == "mean":
            meanvec = np.nanmean(np.stack([np.where(c["INFO"][h]["keep"], c["INFO"][h]["lfc"], np.nan) for h in genes]), 0)
            meanp = np.nanmean(np.stack([c["INFO"][h]["lfcp"] for h in genes]), 0) if c["INFO"][genes[0]]["lfcp"].size else None
        rstd = np.asarray(c["align"]["train_rna_std"]); pstd = np.asarray(c["align"]["train_prot_std"])
        for g, h in zip(genes, src):
            I = c["INFO"][g]; J = c["INFO"][h]
            if target in ("centered", "mean"):
                if target == "centered":
                    j = genes.index(g); oth = [i for i in range(len(genes)) if i != j]
                    raw_np = (I["lfc"] - np.nanmean(_O[oth], 0))[I["de_pos"]]
                else:
                    raw_np = meanvec[I["de_pos"]]
                raw = torch.as_tensor(raw_np, dtype=torch.float32, device=DEVICE)
                zz = raw / torch.as_tensor(np.clip(rstd[c["reti"][I["de_pos"]]], STD_FLOOR, None), device=DEVICE).float()
                e = dict(sel_r=I["sel_r"], r_raw=raw, r_z=zz, has_p=False)
                Pm = (_Op is not None) if target == "centered" else (meanp is not None)
                if I["has_p_loss"] and Pm:
                    pv = (I["lfcp"] - np.nanmean(_Op[oth], 0)) if target == "centered" else meanp
                    rp_raw = torch.as_tensor(pv[I["dp_pos"]], dtype=torch.float32, device=DEVICE)
                    rp_z = rp_raw / torch.as_tensor(np.clip(pstd[c["preti"][I["dp_pos"]]], STD_FLOOR, None), device=DEVICE).float()
                    e.update(has_p=True, sel_p=I["sel_p"], p_raw=rp_raw, p_z=rp_z)
            else:
                sgn = -1.0 if target == "flip" else 1.0
                e = dict(sel_r=J["sel_r"], r_raw=sgn * J["obs_r_raw"], r_z=sgn * J["obs_r_z"], has_p=J["has_p_loss"])
                if J["has_p_loss"]:
                    e.update(sel_p=J["sel_p"], p_raw=sgn * J["obs_p_raw"], p_z=sgn * J["obs_p_z"])
            TGT[(t, g)] = (I["gidx"], e)
    return tg, TGT

def cohort_loss(ctx_d, t, genes, TGT, mc, infonce=None, nce_O=None):
    c = ctx_d[t]; e_tr = c["eps"][:N_TRAIN_CELLS]; A = assemble(); rr, rp = control_side(A, e_tr)
    L_rr, L_rp, NCE_P = [], [], []
    for g in genes:
        gidx, e = TGT[(t, g)]
        d_r, d_p = predict_target(A, rr, rp, e_tr, gidx)
        if infonce is not None: NCE_P.append(d_r[c["S_tr"]])
        pr = d_r[e["sel_r"]]; l = torch.zeros((), device=DEVICE)
        if mc["w_dir"] > 0 and pr.numel() >= 5 and torch.std(pr) > 1e-12: l = l + mc["w_dir"] * (-pearson_t(pr, e["r_raw"]))
        if mc["w_mag"] > 0: l = l + mc["w_mag"] * ((pr - e["r_z"]) ** 2).mean()
        L_rr.append(l)
        if mc["w_rp"] > 0 and e["has_p"]:
            pq = d_p[e["sel_p"]]; lp = torch.zeros((), device=DEVICE)
            if mc["w_dir"] > 0 and pq.numel() >= 3 and torch.std(pq) > 1e-12: lp = lp + mc["w_dir"] * (-pearson_t(pq, e["p_raw"]))
            if mc["w_mag"] > 0: lp = lp + mc["w_mag"] * ((pq - e["p_z"]) ** 2).mean()
            L_rp.append(lp)
    if not L_rr: return None
    ds = torch.stack(L_rr).mean() + (mc["w_rp"] * torch.stack(L_rp).mean() if L_rp else 0.0)
    if infonce is not None and len(NCE_P) >= 3:
        ds = ds + infonce[0] * info_nce(torch.stack(NCE_P), nce_O, infonce[1])
    return ds

def _rho_pen():
    return MC["w_rho"] * torch.relu(spectral_norm(assemble()) - MC["rho_pen"]) ** 2

def finetune_orig(ctx_d, train_units, mc=MC, target="real", perm_seed=0, infonce=None):
    """identical to reference finetune() (dir_scale=raw)."""
    model.scm.load_state_dict(SCM0)
    opt = torch.optim.Adam([p for p in model.scm.parameters() if p.requires_grad], lr=mc["lr"], weight_decay=WD)
    tg, TGT = build_targets(ctx_d, train_units, target, perm_seed); active = [t for t in TAGS if tg.get(t)]
    n_ds = max(1, len(active)); best, best_state, since, best_step = float("inf"), None, 0, -1
    NCE_O = {t: nce_targets(ctx_d, t, tg[t]) for t in active} if infonce is not None else {}
    stop = "max_steps"; step = -1
    for step in range(STEPS):
        opt.zero_grad(); total = 0.0
        for t in active:
            ds = cohort_loss(ctx_d, t, tg[t], TGT, mc, infonce, NCE_O.get(t))
            if ds is None: continue
            (ds / n_ds).backward(); total += float(ds.detach()) / n_ds
        if mc["w_rho"] > 0:
            rl = _rho_pen(); rl.backward(); total += float(rl.detach())
        torch.nn.utils.clip_grad_norm_([p for p in model.scm.parameters() if p.requires_grad], 1.0); opt.step()
        rho = spectral_radius(assemble())
        if total < best - 1e-4:
            best, since, best_step = total, 0, step; best_state = {k: v.detach().clone() for k, v in model.scm.state_dict().items()}
        else: since += 1
        if rho > mc["rho_guard"]: stop = "rho_guard"; break
        if since >= PATIENCE: stop = "patience"; break
    if best_state is not None: model.scm.load_state_dict(best_state)
    return dict(best_step=best_step, best_loss=best, last_step=step, stop=stop, rho=spectral_radius(assemble()))

R_STEPS = int(os.environ.get("LOCO_RSTEPS", "300")); R_RHO = float(os.environ.get("LOCO_RHOMAX", "0.95"))
R_EVAL, R_PAT = 10, 5
def project_rho(rmax=R_RHO):
    rho = spectral_radius(assemble())
    if rho > rmax:
        cfac = rmax / rho * 0.999
        with torch.no_grad():
            for n, p in model.scm.named_parameters():
                if n.startswith("weights_"): p.mul_(cfac)
        return True
    return False

def finetune_relaxed(ctx_d, train_units, mc=MC, target="real", perm_seed=0, infonce=None):
    model.scm.load_state_dict(SCM0)
    opt = torch.optim.Adam([p for p in model.scm.parameters() if p.requires_grad], lr=mc["lr"], weight_decay=WD)
    tg, TGT = build_targets(ctx_d, train_units, target, perm_seed); active = [t for t in TAGS if tg.get(t)]
    n_ds = max(1, len(active)); tr_in, va_in = {}, {}
    for t in active:
        gl = tg[t]; prm = np.random.RandomState(7).permutation(len(gl)); nva = max(3, round(0.2 * len(gl)))
        va = set(np.array(gl)[prm[:nva]].tolist()); va_in[t] = [g for g in gl if g in va]; tr_in[t] = [g for g in gl if g not in va]
    NCE_TR = {t: nce_targets(ctx_d, t, tr_in[t]) for t in active} if infonce is not None else {}
    NCE_VA = {t: nce_targets(ctx_d, t, va_in[t]) for t in active} if infonce is not None else {}
    @torch.no_grad()
    def val():
        v = 0.0
        for t in active:
            ds = cohort_loss(ctx_d, t, va_in[t], TGT, mc, infonce, NCE_VA.get(t))
            if ds is not None: v += float(ds) / n_ds
        return v
    best = val(); v0 = best; best_state = {k: v.detach().clone() for k, v in model.scm.state_dict().items()}; best_step = -1; since = 0
    n_proj = 0; step = -1; stop = "max_steps"; hist = [(-1, v0)]
    for step in range(R_STEPS):
        opt.zero_grad()
        for t in active:
            ds = cohort_loss(ctx_d, t, tr_in[t], TGT, mc, infonce, NCE_TR.get(t))
            if ds is not None: (ds / n_ds).backward()
        if mc["w_rho"] > 0: _rho_pen().backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.scm.parameters() if p.requires_grad], 1.0); opt.step()
        n_proj += int(project_rho())
        if (step + 1) % R_EVAL == 0:
            v = val(); hist.append((step, v))
            if v < best - 1e-4:
                best, since, best_step = v, 0, step; best_state = {k: v_.detach().clone() for k, v_ in model.scm.state_dict().items()}
            else: since += 1
            if since >= R_PAT: stop = "val_patience"; break
    model.scm.load_state_dict(best_state)
    return dict(best_step=best_step, best_val=best, val_at_init=v0, last_step=step, stop=stop, n_projections=n_proj,
                val_hist=hist, rho=spectral_radius(assemble()))

FT = finetune_orig if MODE == "orig" else finetune_relaxed

# ---- prediction + references for the held-out cohort
REF_KO = [(t, h) for t in TRC for h in ALL[t]]
REF_GENES = sorted(set(h for _, h in REF_KO))
@torch.no_grad()
def predict_hold(genes):
    A = assemble(); c = CTX[HOLD]; rr, rp = control_side(A, c["eps"]); tgi = c["align"]["train_gene_idx"]; out = {}
    for g in genes:
        if g not in tgi: continue
        d_r, d_p = predict_target(A, rr, rp, c["eps"], tgi[g])
        out[g] = (d_r.cpu().numpy()[c["reti"]].astype(float), d_p.cpu().numpy()[c["preti"]].astype(float))
    return out

def refs_from_train(pref):
    """reference set = KOs of the TRAINING cohorts: observed responses mapped into HOLD's feature space;
    predicted responses of the same genes made by the model in HOLD's control context (label-free)."""
    c = CTX[HOLD]; ko = [(t, h) for t, h in REF_KO if h in pref]
    O = np.stack([r2train(t, obs_r(t, h))[c["reti"]] for t, h in ko])
    Pd = np.stack([np.where(c["common"] != h, pref[h][0], np.nan) for t, h in ko])
    Op = np.stack([p2train(t, CTX[t]["INFO"][h]["lfcp"])[c["preti"]] for t, h in ko]) if len(c["preti"]) else None
    Pp = np.stack([pref[h][1] for t, h in ko])
    return dict(o_r=O, p_r=Pd, o_p=Op, p_p=Pp, genes=[h for _, h in ko])

def _pef(p, o, m):
    return pe(p, o, np.asarray(m, bool) & np.isfinite(p) & np.isfinite(o))

def retrieval(genes, preds, R, kind="rr"):
    c = CTX[HOLD]; out = {}
    def _z(v): v = v - v.mean(); n = np.linalg.norm(v); return v / n if n > 0 else v
    if kind == "rr":
        S = c["S"]; gl = list(genes)
        Pm = np.stack([(preds[g][0] - _loo_mean(R["p_r"], R["genes"], g))[S] for g in gl])
        Om = np.stack([((c["INFO"][g]["lfc"] - _loo_mean(R["o_r"], R["genes"], g)) / c["sd_r"])[S] for g in gl])
    else:
        if not len(c["preti"]) or R["o_p"] is None: return out
        S = c["Sp"]; gl = [g for g in genes if c["INFO"][g]["has_p"]]
        if len(gl) < 3: return out
        Pm = np.stack([(preds[g][1] - _loo_mean(R["p_p"], R["genes"], g))[S] for g in gl])
        Om = np.stack([((c["INFO"][g]["lfcp"] - _loo_mean(R["o_p"], R["genes"], g)) / c["sd_p"])[S] for g in gl])
    f = np.isfinite(Pm).all(0) & np.isfinite(Om).all(0)
    if f.sum() < 3 or len(gl) < 3: return out
    Pm = np.stack([_z(v) for v in Pm[:, f]]); Om = np.stack([_z(v) for v in Om[:, f]]); C = Pm @ Om.T
    for i, g in enumerate(gl):
        oth = [j for j in range(len(gl)) if j != i]
        out[g] = float(np.mean(C[i, i] > C[i, oth]) + 0.5 * np.mean(C[i, i] == C[i, oth]))
    return out

def score_hold(genes, preds, R, pool=None):
    c = CTX[HOLD]; sd_r, sd_p = c["sd_r"], c["sd_p"]; pool = pool or genes
    RR = retrieval(genes, preds, R, "rr"); RP = retrieval(genes, preds, R, "rp"); rows = []
    for g in genes:
        I = c["INFO"][g]; pr, pp = preds[g]; k = I["keep"]; m = I["sig"] & k
        mu_o = _loo_mean(R["o_r"], R["genes"], g); mu_p = _loo_mean(R["p_r"], R["genes"], g)
        r = dict(cohort=HOLD, gene=g, shared=g in SH, n_de_rna=int(m.sum()), rr_retrieval=RR.get(g, np.nan), rp_retrieval=RP.get(g, np.nan))
        r["rr_auroc"] = au(I["sig"][k], np.abs(pr[k]))
        r["rr_pe"] = pe(pr, I["lfc"], m)
        r["rr_pe_cent"] = _pef(pr * sd_r - mu_p * sd_r, I["lfc"] - mu_o, m)
        r["rr_pe_meanresp"] = _pef(mu_o, I["lfc"], m)
        r["rr_auroc_meanresp"] = au(I["sig"][k], np.abs(np.nan_to_num(mu_o[k])))
        r["rr_pe_meanpred"] = _pef(mu_p * sd_r, I["lfc"], m)
        r["rr_pe_shuf"] = float(np.nanmean([pe(pr, c["INFO"][h]["lfc"], c["INFO"][h]["sig"] & c["INFO"][h]["keep"] & k) for h in pool if h != g]))
        if I["has_p"]:
            mp = I["sigp"]; r["n_de_prot"] = int(mp.sum())
            r["rp_auroc"] = au(mp, np.abs(pp)); r["rp_pe"] = pe(pp, I["lfcp"], mp); r["rp_pe_nocog"] = pe(pp, I["lfcp"], mp & ~I["cog_mask"])
            if R["o_p"] is not None:
                mo = _loo_mean(R["o_p"], R["genes"], g); mq = _loo_mean(R["p_p"], R["genes"], g)
                r["rp_pe_cent"] = _pef(pp * sd_p - mq * sd_p, I["lfcp"] - mo, mp); r["rp_pe_meanresp"] = _pef(mo, I["lfcp"], mp)
            shp = [pe(pp, c["INFO"][h]["lfcp"], c["INFO"][h]["sigp"]) for h in pool if h != g and c["INFO"][h]["has_p"]]
            r["rp_pe_shuf"] = float(np.nanmean(shp)) if shp else np.nan
        rows.append(r)
    return rows

KEYS = ["rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc", "rr_pe_shuf", "rr_pe_meanresp", "rr_auroc_meanresp", "rr_pe_meanpred",
        "rp_pe", "rp_pe_cent", "rp_retrieval", "rp_auroc", "rp_pe_nocog", "rp_pe_shuf", "rp_pe_meanresp",
        "own_rr_pe_cent", "own_rr_pe_meanresp", "own_rr_retrieval", "own_rp_pe_cent", "own_rp_pe_meanresp", "own_rp_retrieval",
        "samegene_obs_rr_pe", "samegene_obs_rr_pe_cent", "othergene_obs_rr_pe", "othergene_obs_rr_pe_cent",
        "samegene_obs_rp_pe", "othergene_obs_rp_pe"]
def summ(rows):
    out = {}
    for scope, rs in [("all", rows), ("shared", [r for r in rows if r["shared"]]), ("nonshared", [r for r in rows if not r["shared"]])]:
        s = {}
        for k in KEYS:
            v = np.array([r.get(k, np.nan) for r in rs], float); f = np.isfinite(v)
            if not f.any(): continue
            s[k] = float(v[f].mean()); s["n_" + k] = int(f.sum()); s["sem_" + k] = float(v[f].std(ddof=1) / np.sqrt(f.sum())) if f.sum() > 1 else None
        out[scope] = s
    return out

CAL = sorted(np.array(ALL[HOLD])[np.random.RandomState(0).permutation(len(ALL[HOLD]))[:max(4, round(0.2 * len(ALL[HOLD])))]].tolist())
REST = [g for g in ALL[HOLD] if g not in set(CAL)]

def evaluate():
    pT = predict_hold(sorted(set(ALL[HOLD]) | set(REF_GENES)))
    preds = {g: pT[g] for g in ALL[HOLD]}
    R_tr = refs_from_train(pT)
    rows = score_hold(ALL[HOLD], preds, R_tr)
    # oracle (label leakage, reference only): centring / mean response from the held-out cohort's own other KOs
    R_own = ref_means(CTX, {HOLD: ALL[HOLD]}, {(HOLD, g): preds[g] for g in ALL[HOLD]})[HOLD]
    own = {r["gene"]: r for r in score_hold(ALL[HOLD], preds, R_own)}
    for r in rows:
        o = own[r["gene"]]
        for k in ["rr_pe_cent", "rr_pe_meanresp", "rr_retrieval", "rp_pe_cent", "rp_pe_meanresp", "rp_retrieval"]:
            r["own_" + k] = o.get(k, np.nan)
    # calibration: 20% of held-out KOs give the reference mean; scored on the remaining 80% (vs train-ref on the same 80%)
    R_cal = ref_means(CTX, {HOLD: CAL}, {(HOLD, g): preds[g] for g in CAL})[HOLD]
    rows_cal = score_hold(REST, preds, R_cal, pool=REST); rows_trrest = score_hold(REST, preds, R_tr, pool=REST)
    return rows, dict(cal=summ(rows_cal)["all"], train_ref_same_subset=summ(rows_trrest)["all"], n_cal=len(CAL), n_rest=len(REST)), preds

# ---- model-free same-gene transfer (data level): observed response of g in training cohorts vs g in HOLD
def samegene_obs(rows):
    c = CTX[HOLD]; R0 = refs_from_train({h: (np.zeros(len(c["common"])), np.zeros(len(c["preti"]))) for h in REF_GENES})
    for r in rows:
        g = r["gene"]
        if g not in SH: continue
        I = c["INFO"][g]; m = I["sig"] & I["keep"]; mu_o = _loo_mean(R0["o_r"], R0["genes"], g)
        same = [i for i, h in enumerate(R0["genes"]) if h == g]; oth = [i for i, h in enumerate(R0["genes"]) if h != g]
        so = np.nanmean(R0["o_r"][same], 0)
        r["samegene_obs_rr_pe"] = _pef(so, I["lfc"], m); r["samegene_obs_rr_pe_cent"] = _pef(so - mu_o, I["lfc"] - mu_o, m)
        r["othergene_obs_rr_pe"] = float(np.nanmean([_pef(R0["o_r"][i], I["lfc"], m) for i in oth]))
        r["othergene_obs_rr_pe_cent"] = float(np.nanmean([_pef(R0["o_r"][i] - mu_o, I["lfc"] - mu_o, m) for i in oth]))
        if I["has_p"] and R0["o_p"] is not None:
            sp_ = np.nanmean(R0["o_p"][same], 0)
            r["samegene_obs_rp_pe"] = _pef(sp_, I["lfcp"], I["sigp"])
            r["othergene_obs_rp_pe"] = float(np.nanmean([_pef(R0["o_p"][i], I["lfcp"], I["sigp"]) for i in oth]))
    return rows

PAIR_KEYS = ["rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval", "rp_auroc"]
def paired_sub(ra, rb, k, sub=None):
    A = {r["gene"]: r.get(k, np.nan) for r in ra}
    pr = np.array([(A[r["gene"]], r.get(k, np.nan)) for r in rb if r["gene"] in A and (sub is None or r["gene"] in sub)], float).reshape(-1, 2)
    pr = pr[np.isfinite(pr).all(1)]; n = len(pr)
    if n < 3: return dict(n=n)
    d = pr[:, 1] - pr[:, 0]
    if np.allclose(d, 0): return dict(n=n, mean_diff=0.0)
    return dict(n=n, mean_diff=float(d.mean()), sem=float(d.std(ddof=1) / np.sqrt(n)), p_t=float(ttest_rel(pr[:, 1], pr[:, 0]).pvalue),
                p_wilcoxon=float(wilcoxon(d).pvalue))

OUTJ = dict(run=os.path.basename(RD), hold=HOLD, mode=MODE, train_cohorts=TRC, n_train_KO={t: len(TR_U[t]) for t in TRC}, n_test_KO=len(ALL[HOLD]),
            shared_genes=SH, S_size=len(CTX[HOLD]["S"]), Sp_size=len(CTX[HOLD]["Sp"]), mean_response_agreement=MEANCORR,
            optimiser=dict(MC, steps=STEPS, patience=PATIENCE) if MODE == "orig" else dict(MC, steps=R_STEPS, rho_max=R_RHO, eval_every=R_EVAL, patience_evals=R_PAT, inner_val_frac=0.2),
            variants={}, n_train_cells=N_TRAIN_CELLS, device=str(DEVICE))
def save_j():
    json.dump(OUTJ, open(OUT, "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else (float(o) if isinstance(o, np.floating) else str(o)))

def record(name, info, base_rows=None):
    rows, calib, _ = evaluate(); rows = samegene_obs(rows)
    st = structure(); st["rel_change_A"] = float(np.linalg.norm(assemble().detach().cpu().numpy() - A_obs) / np.linalg.norm(A_obs))
    v = dict(info=info, summary=summ(rows), calibration=calib, structure=st, per_ko=rows)
    if base_rows is not None:
        v["paired_vs_obs"] = {k: dict(all=paired_sub(base_rows, rows, k), shared=paired_sub(base_rows, rows, k, set(SH)),
                                      nonshared=paired_sub(base_rows, rows, k, set(ALL[HOLD]) - set(SH))) for k in PAIR_KEYS}
    OUTJ["variants"][name] = v; save_j()
    s = v["summary"]["all"]; ss = v["summary"]["shared"]
    log(f"[{HOLD}/{MODE}] {name}: pe {s.get('rr_pe', np.nan):+.3f} cent {s.get('rr_pe_cent', np.nan):+.3f} retr {s.get('rr_retrieval', np.nan):.3f} "
        f"auroc {s.get('rr_auroc', np.nan):.3f} shuf {s.get('rr_pe_shuf', np.nan):+.3f} meanresp(train) {s.get('rr_pe_meanresp', np.nan):+.3f} "
        f"meanresp(own) {s.get('own_rr_pe_meanresp', np.nan):+.3f} | RP pe {s.get('rp_pe', np.nan):+.3f} cent {s.get('rp_pe_cent', np.nan):+.3f} "
        f"| shared(n={len(SH)}) pe {ss.get('rr_pe', np.nan):+.3f} cent {ss.get('rr_pe_cent', np.nan):+.3f} | info {({k: info[k] for k in info if k != 'val_hist'} if info else None)}")
    return rows

model.scm.load_state_dict(SCM0)
A_obs = assemble().detach().cpu().numpy()
rows_obs = record("obs", None)
ONLY = set(x for x in os.environ.get("LOCO_ONLY", "").split(",") if x)   # optional: run only these variants (parallel split jobs)
def run(name, units=TR_U, **kw):
    if ONLY and name not in ONLY: return None
    info = FT(CTX, units, **kw); return record(name, info, rows_obs)
run("ft_standard")
for k in range(N_SHUF): run(f"ft_labelshuffle_{k}", target="shuffle", perm_seed=k)
run("ft_meantarget", target="mean")
run("ft_centered", target="centered")
# gene-specific transfer: same standard FT but WITHOUT the shared genes' KOs in the training cohorts
run("ft_standard_noshared", units={t: [g for g in TR_U[t] if g not in set(SH)] for t in TRC})
# InfoNCE: lambda chosen on an inner split of the TRAINING cohorts' KOs (retrieval within training cohorts)
if os.environ.get("LOCO_NCE", "1") == "1" and (not ONLY or "ft_infonce" in ONLY):
    rs = np.random.RandomState(7); IN_TR_L, IN_VA_L = {}, {}
    for t in TRC:
        gl = TR_U[t]; prm = rs.permutation(len(gl)); va = set(np.array(gl)[prm[:round(0.3 * len(gl))]].tolist())
        IN_TR_L[t] = [g for g in gl if g not in va]; IN_VA_L[t] = [g for g in gl if g in va]
    grid = {}
    for lam in NCE_GRID:
        finetune_orig(CTX, IN_TR_L, infonce=(lam, NCE_TAU))
        pt = predict_units(CTX, {t: IN_TR_L[t] + IN_VA_L[t] for t in TRC})
        sm = summarize(score(CTX, IN_VA_L, pt, ref_means(CTX, IN_TR_L, pt)))["overall"]
        grid[str(lam)] = dict(lam=lam, inner_retr=sm["rr_retrieval"], inner_cent=sm["rr_pe_cent"], inner_pe=sm["rr_pe"])
        log(f"  InfoNCE inner lam={lam}: retr {sm['rr_retrieval']:.3f} cent {sm['rr_pe_cent']:+.3f} pe {sm['rr_pe']:+.3f}")
    best = max(grid.values(), key=lambda v: v["inner_retr"]); OUTJ["infonce_grid"] = grid; OUTJ["infonce_selected"] = best
    run("ft_infonce", infonce=(best["lam"], NCE_TAU))
model.scm.load_state_dict(SCM0)
OUTJ["elapsed_s"] = time.time() - T0; save_j(); log("DONE ->", OUT)
