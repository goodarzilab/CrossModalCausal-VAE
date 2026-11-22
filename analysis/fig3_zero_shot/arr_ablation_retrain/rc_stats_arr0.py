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

[stats workstream copy, analysis/fig3_zero_shot/stats/rc_stats.py]
  Changes vs analysis/common/reviewer_controls.py: (1) DE ground truth can be replaced by the condition-matched pseudobulk
  definition (ST_DE=pb_bh|pb_raw|orig; ST_FRANGIEH_PROT=clr|log1p); (2) caches are read from results/_ftcache
  read-only, outputs go to paper_nov_revisions/analyses/stats; (3) the fine-tuning split is the paper's
  results/finetune_split_genes.json restricted to evaluable units; (4) predictions are dumped (npz) for the joint
  bootstrap; (5) optional eps override for the Frangieh protein-input sensitivity (ST_EPS_frangieh=path.npy);
  (6) only the variants needed for Figs 3-5 are run by default (RC_EXTRA=1 re-enables z-scale/centred fine-tunes).
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

HERE = f"{_CMC_ROOT}/analysis/common"   # [stats] import the original eval modules (unmodified)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(_CMC_ROOT, "src"))
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

P.MAX_CONTROL_CELLS = 4000
RD = os.path.abspath(sys.argv[1])
REPO = _CMC_ROOT
ST_DE = os.environ.get("ST_DE", "pb_bh"); ST_FP = os.environ.get("ST_FRANGIEH_PROT", "clr"); ST_TAG = os.environ.get("ST_TAG", "")
ST_DIR = os.path.join(_CMC_RESULTS, "paper", "analyses", "stats")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ST_DIR, "runs", f"rc_{os.path.basename(RD)}_{ST_DE}{ST_TAG}.json")
PRED_OUT = OUT.replace(".json", "_preds.npz"); _DUMP = {}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CACHE_DIR = os.environ.get("RC_CACHE_DIR", os.path.join(REPO, "results", "_ftcache"))   # [review2/arr0] overridable; content-keyed (md5 of ckpt)
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
    if os.environ.get("RC_ALLOW_BUILD") != "1" or CACHE_DIR.startswith(os.path.join(REPO, "results")):
        raise RuntimeError(f"cache miss for {tag}: {cache} (stats copy never rebuilds/writes results/)")
    log(f"[review2/arr0] cache MISS {tag}: fresh encode of control cells with {_ck}")
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
# ---- [stats] replace the DE ground truth with the condition-matched pseudobulk definition ----
PB_INFO = {}
def _pb_de(tag, al):
    import pickle as _pk
    pb = _pk.load(open(os.path.join(ST_DIR, "de", f"pb_{tag}.pkl"), "rb"))
    assert list(pb["common_genes"]) == list(al["common_genes"]), "feature order mismatch"
    out = {}
    for g, r in pb["de"].items():
        lp, pp_, qp = r["lfc_prot"], r["pvals_prot"], r["q_prot"]
        if tag == "frangieh" and ST_FP == "log1p":
            lp, pp_, qp = r["lfc_prot_log1p"], r["pvals_prot_log1p"], r["q_prot_log1p"]
        lr = np.nan_to_num(r["lfc_rna"]); lp = np.nan_to_num(lp)
        if ST_DE == "pb_raw":
            sr, sp_ = (r["pvals_rna"] < 0.05), (pp_ < 0.05)
        else:
            sr, sp_ = (r["q_rna"] < 0.05), (qp < 0.05)
        out[g] = dict(lfc_rna=lr, lfc_prot=lp, pvals_rna=r["pvals_rna"], pvals_prot=pp_,
                      sig_rna=sr & (np.abs(lr) >= LFCR), sig_prot=(sp_ & (np.abs(lp) >= LFCP)) if len(lp) else np.zeros(0, bool),
                      n_cells=r["n_cells"], n_pb=r["n_pb"])
    PB_INFO[tag] = dict(eligibility=pb["eligibility"], n_control_cells=pb["n_control_cells"], n_control_pb=pb["n_control_pb"], extra=pb["extra"])
    return out
if ST_DE != "orig":
    for _t in TAGS:
        _al, _de, _eps = RAW[_t]; RAW[_t] = (_al, _pb_de(_t, _al), _eps)
        log(f"[stats] {_t}: DE replaced by pseudobulk ({ST_DE}); units with PB DE = {len(RAW[_t][1])}")
for _t in TAGS:
    _p = os.environ.get(f"ST_EPS_{_t}")
    if _p:
        _al, _de, _eps = RAW[_t]; _e2 = np.load(_p).astype(np.float32); assert _e2.shape == _eps.shape
        RAW[_t] = (_al, _de, _e2); log(f"[stats] {_t}: eps override from {_p}")
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
_ref = json.load(open(f"{_CMC_RESULTS}/finetune_split_genes.json"))
_ev_all = set().union(*[set(CTX[t]["evaluable"]) for t in TAGS])
TRAIN_G = sorted(set(_ref["train_genes"]) & _ev_all); TEST_G = sorted(set(_ref["test_genes"]) & _ev_all)   # paper split, restricted
# sanity: the split must reproduce finetune_split_genes.json (same code, same seed)
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
ALLV = {t: sorted(g for g in RAW[t][1] if g in RAW[t][0]["train_gene_idx"] and g in RAW[t][0]["train_gene_to_idx_common"]) for t in TAGS}
@torch.no_grad()
def predict_vocab(units, A=None):
    """predictions for any vocabulary unit (not only currently evaluable ones) -> dict (t,g)->(rna_eval, prot_eval)"""
    A = (assemble() if A_OVERRIDE is None else A_OVERRIDE) if A is None else A; out = {}
    for t, genes in units.items():
        c = CTX[t]; rr, rp = control_side(A, c["eps"])
        for g in genes:
            d_r, d_p = predict_target(A, rr, rp, c["eps"], c["align"]["train_gene_idx"][g])
            out[(t, g)] = (d_r.cpu().numpy()[c["reti"]].astype(np.float32), d_p.cpu().numpy()[c["preti"]].astype(np.float32))
    return out
def dump(name, preds):
    for t in TAGS:
        ks = [g for (tt, g) in preds if tt == t]
        if not ks: continue
        _DUMP[f"{name}|{t}|genes"] = np.array(ks)
        _DUMP[f"{name}|{t}|rna"] = np.stack([np.asarray(preds[(t, g)][0], np.float32) for g in ks])
        _DUMP[f"{name}|{t}|prot"] = np.stack([np.asarray(preds[(t, g)][1], np.float32) for g in ks])
    np.savez_compressed(PRED_OUT, **_DUMP)
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

# ================================================================== RUN
RES = dict(run=os.path.basename(RD), split_matches_paper=SPLIT_OK, steps=STEPS, de_floor=[LFCR, LFCP],
           n_units={t: len(CTX[t]["evaluable"]) for t in TAGS},
           n_train_units={t: len(TRAIN[t]) for t in TAGS}, n_test_units={t: len(TEST[t]) for t in TAGS})
def save():
    json.dump(RES, open(OUT, "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else str(o))

ALL = {t: CTX[t]["evaluable"] for t in TAGS}

# ---- (A) observational model, Fig-3 set (all evaluable), all scoring variants; refs = leave-one-out over all KOs
model.scm.load_state_dict(SCM0)
P_obs_all = predict_units(CTX, ALL)
R_all = ref_means(CTX, ALL, P_obs_all)
rows_fig3 = score(CTX, ALL, P_obs_all, R_all)
with torch.no_grad():
    _A0 = assemble(); _XB = {t: control_side(_A0, CTX[t]["eps"])[0].mean(0).cpu().numpy() for t in TAGS}
for r in rows_fig3:
    gi_ = CTX[r["cohort"]]["INFO"][r["gene"]]["gidx"]; r["target_ctrl_z"] = float(_XB[r["cohort"]][gi_]); r["target_z0"] = float(Z0[gi_])
RES["shock_mode"] = SHOCK_MODE; RES["mutilate"] = MUT_DEFAULT
# decomposition of the do-op into its two terms (mutilation of A vs exogenous shock), per KO, RNA response norm
with torch.no_grad():
    _dec = []
    for t in TAGS:
        c = CTX[t]; rr, rp = control_side(_A0, c["eps"])
        for g in c["evaluable"][:60]:
            gi_ = c["INFO"][g]["gidx"]
            full = predict_target(_A0, rr, rp, c["eps"], gi_, mutilate=True)[0]
            # shock only (no mutilation)
            sh = predict_target(_A0, rr, rp, c["eps"], gi_, mutilate=False)[0]
            # mutilation only: zero shock
            s_ = (Wr[gi_].abs() / (Wr[gi_].abs().max() + 1e-8)).clamp(0, 1); Am = _A0.clone(); Am[:, :lr_] = Am[:, :lr_] * (1 - s_).unsqueeze(0)
            zr = torch.split(solve(Am, c["eps"]), [lr_, lp_], 1)[0]; mo = ((model.dec_rna(zr) + bias_rna) - rr).mean(0)
            keep = torch.as_tensor(c["INFO"][g]["keep"], device=DEVICE); ri = torch.as_tensor(c["reti"], device=DEVICE)
            f, a, b = full[ri][keep], sh[ri][keep], mo[ri][keep]
            _dec.append(dict(cohort=t, gene=g, norm_full=float(f.norm()), norm_shock_only=float(a.norm()), norm_mutil_only=float(b.norm()),
                             corr_full_mutil=float(torch.corrcoef(torch.stack([f, b]))[0, 1]), corr_full_shock=float(torch.corrcoef(torch.stack([f, a]))[0, 1]),
                             decoder_row_norm=float(Wr[gi_].norm())))
RES["doop_decomposition"] = _dec
log("do-op decomposition (median over KOs): |shock-only|/|full| = %.3f, corr(full, mutil-only) = %.3f, corr(full, shock-only) = %.3f" % (
    np.median([d["norm_shock_only"] / max(d["norm_full"], 1e-12) for d in _dec]), np.median([d["corr_full_mutil"] for d in _dec]), np.median([d["corr_full_shock"] for d in _dec])))
dump("obs", predict_vocab(ALLV))
for t in TAGS:
    c = CTX[t]; _DUMP[f"meta|{t}|sd_r"] = c["sd_r"]; _DUMP[f"meta|{t}|sd_p"] = c["sd_p"]; _DUMP[f"meta|{t}|common"] = c["common"]
    _DUMP[f"meta|{t}|pnames"] = np.array([c["align"]["train_prot_names"][i] for i in c["preti"]])
    _DUMP[f"meta|{t}|train"] = np.array(TRAIN[t]); _DUMP[f"meta|{t}|test"] = np.array(TEST[t]); _DUMP[f"meta|{t}|evaluable"] = np.array(c["evaluable"])
RES["pb_info"] = PB_INFO; RES["st_de"] = ST_DE; RES["st_frangieh_prot"] = ST_FP
RES["fig3_obs"] = dict(summary=summarize(rows_fig3), per_ko=rows_fig3, similarity=similarity(CTX, ALL, P_obs_all)); RES["structure_obs"] = structure()
log("response similarity across KOs (pred vs obs):", RES["fig3_obs"]["similarity"])
A_obs = assemble().detach().cpu().numpy()
log("fig3 obs overall:", {k: round(v, 3) for k, v in RES["fig3_obs"]["summary"]["overall"].items() if not k.startswith(("n_", "sem_"))})
save()

if os.environ.get("RC_FIG4") == "1":
    # Fig-4 block controls scored with the shared-response-aware metrics (observational model)
    g0 = torch.Generator().manual_seed(0)
    def _shuf_off(M):
        M = M.clone(); n = M.shape[0]; off = ~torch.eye(n, dtype=torch.bool, device=M.device)
        v = M[off]; M[off] = v[torch.randperm(v.numel(), generator=g0).to(M.device)]; return M
    def _shuf_all(M):
        f = M.flatten(); return f[torch.randperm(f.numel(), generator=g0).to(M.device)].reshape(M.shape)
    A0 = assemble().detach().clone(); variants = {}
    def put(rr=None, rp=None):
        A = A0.clone()
        if rr is not None: A[:lr_, :lr_] = rr
        if rp is not None: A[:lr_, lr_:] = rp
        return A
    Arr, Wrp, Wpr = A0[:lr_, :lr_], A0[:lr_, lr_:], A0[lr_:, :lr_]
    variants["Arr_transposed"] = [put(rr=Arr.T.contiguous())]
    variants["Arr_zeroed"] = [put(rr=torch.zeros_like(Arr))]
    variants["Arr_shuffled"] = [put(rr=_shuf_off(Arr)) for _ in range(3)]
    variants["Wrp_from_WprT"] = [put(rp=Wpr.T.contiguous())]
    variants["Wrp_zeroed"] = [put(rp=torch.zeros_like(Wrp))]
    variants["Wrp_shuffled"] = [put(rp=_shuf_all(Wrp)) for _ in range(3)]
    F4 = {}
    for vn, As in variants.items():
        rows_v = []
        for Av in As:
            A_OVERRIDE = Av; pv = predict_units(CTX, ALL); dump(f"fig4_{vn}_{len(rows_v)}", predict_vocab(ALLV)); A_OVERRIDE = None
            rows_v.append(score(CTX, ALL, pv, ref_means(CTX, ALL, pv)))
        # average replicate shuffles per KO
        keys = [k for k in rows_v[0][0] if k.startswith(("rr_", "rp_"))]
        avg = []
        for i, r in enumerate(rows_v[0]):
            a = dict(r)
            for k in keys:
                vals = [rv[i].get(k, np.nan) for rv in rows_v]; vals = [v for v in vals if isinstance(v, float) and v == v]
                a[k] = float(np.mean(vals)) if vals else np.nan
            avg.append(a)
        F4[vn] = dict(summary=summarize(avg), paired_learned_minus_variant={m: paired(avg, rows_fig3, m) for m in
                      ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval", "rp_pe_shuf"]})
        so = F4[vn]["summary"]["overall"]
        log(f"Fig4 {vn}: RRau {so['rr_auroc']:.3f} RRpe {so['rr_pe']:+.3f} cent {so['rr_pe_cent']:+.3f} retr {so['rr_retrieval']:.3f} | RPau {so['rp_auroc']:.3f} RPpe {so['rp_pe']:+.3f} cent {so['rp_pe_cent']:+.3f} retr {so['rp_retrieval']:.3f}")
    RES["fig4_controls"] = F4; save()
if os.environ.get("RC_FIG3_ONLY") == "1" and os.environ.get("RC_BASELINES") != "1":
    save(); log("fig3-only done"); sys.exit(0)
# ---- (A2) Fig-3 baselines rescored with knockdown-consistent signs, paired per KO
if os.environ.get("RC_BASELINES") == "1":
    log("scoring baselines on the Fig-3 set ...")
    BASE = {}
    tv = np.load(f"{_CMC_RESULTS}/baselines/totalvi_raw.npz", allow_pickle=True)
    Grr_tv = np.asarray(tv["G_rr_tv"], float); Grp_tv = np.asarray(tv["G_rp_tv"], float); np.fill_diagonal(Grr_tv, 0.0)
    tvg = {g: i for i, g in enumerate(np.asarray(tv["genes"]).astype(str))}
    ds = np.load(f"{_CMC_RESULTS}/baselines/deepsem_W_809k_clamp09.npz", allow_pickle=True)
    Wd = np.asarray(ds["W"], float); np.fill_diagonal(Wd, 0.0); dsg = {g: i for i, g in enumerate(np.asarray(ds["genes"]).astype(str))}
    Minv = np.linalg.inv(np.eye(Wd.shape[0]) - Wd.T)
    preds_b = {k: {} for k in ["corr", "totalvi_ko", "totalvi_asreported", "deepsem_ko", "deepsem_asreported"]}
    for t in TAGS:
        c = CTX[t]; al = c["align"]; common = c["common"]; g2i = c["g2i"]
        ctr, cgp = al.get("corr_train"), al.get("gene_prot_corr")
        tvcol = np.array([tvg.get(x, -1) for x in common]); dscol = np.array([dsg.get(x, -1) for x in common])
        for g in ALLV[t]:
            n_p = len(c["preti"])
            pc_r = -np.asarray(ctr[g2i[g], :], float) if ctr is not None else np.full(len(common), np.nan)
            pc_p = -np.asarray(cgp[g2i[g], :], float) if (cgp is not None and n_p) else np.zeros(n_p)
            preds_b["corr"][(t, g)] = (pc_r, pc_p)
            if g in tvg:
                vr = np.array([Grr_tv[tvg[g], j] if j >= 0 else 0.0 for j in tvcol]); vp = Grp_tv[tvg[g], c["preti"]] if n_p else np.zeros(0)
                preds_b["totalvi_ko"][(t, g)] = (-vr, -vp); preds_b["totalvi_asreported"][(t, g)] = (vr, vp)
            if g in dsg:
                vr = np.array([Minv[j, dsg[g]] if j >= 0 else 0.0 for j in dscol])
                preds_b["deepsem_ko"][(t, g)] = (-vr, np.full(n_p, np.nan)); preds_b["deepsem_asreported"][(t, g)] = (vr, np.full(n_p, np.nan))
    for name, pb in preds_b.items():
        dump(f"base_{name}", pb)
        units = {t: [g for g in CTX[t]["evaluable"] if (t, g) in pb] for t in TAGS}
        refs = ref_means(CTX, units, pb)
        rows = score(CTX, units, pb, refs)
        BASE[name] = dict(summary=summarize(rows), per_ko=rows)
        BASE[name]["paired_vs_cmc"] = {m: paired(rows, rows_fig3, m) for m in ["rr_auroc", "rr_pe", "rr_pe_z", "rp_auroc", "rp_pe", "rp_pe_z"]}
        log(f"  {name}: RR pe {BASE[name]['summary']['overall']['rr_pe']:+.3f} RP pe {BASE[name]['summary']['overall']['rp_pe']:+.3f}")
    RES["fig3_baselines"] = BASE; save()

if os.environ.get("RC_FIG3_ONLY") == "1":
    save(); log("fig3-only done"); sys.exit(0)
# ---- (B) Fig-5: observational vs fine-tuned, held-out genes, all scoring variants + controls
def run_variant(name, ctx_d=CTX, train=TRAIN, test=TEST, **kw):
    info = finetune(ctx_d, train, **kw)
    pt = predict_units(ctx_d, {t: train[t] + test[t] for t in TAGS}); dump(name + (f"_{kw['perm_seed']}" if "perm_seed" in kw else ""), pt)
    refs = ref_means(ctx_d, train, pt)              # centring / mean-response use TRAINING KOs only (no test leakage)
    rows = score(ctx_d, test, pt, refs)
    st = structure(); cur = assemble().detach().cpu().numpy()
    st["rel_change_A"] = float(np.linalg.norm(cur - A_obs) / np.linalg.norm(A_obs))
    return dict(info=info, summary=summarize(rows), per_ko=rows, structure=st)

model.scm.load_state_dict(SCM0)
P_obs_tt = predict_units(CTX, {t: TRAIN[t] + TEST[t] for t in TAGS})
R_obs_tt = ref_means(CTX, TRAIN, P_obs_tt)
rows_base = score(CTX, TEST, P_obs_tt, R_obs_tt); dump("ft_baseline", P_obs_tt)
RES["fig5_baseline"] = dict(summary=summarize(rows_base), per_ko=rows_base, structure=structure())
log("fig5 baseline:", {k: round(v, 3) for k, v in RES["fig5_baseline"]["summary"]["overall"].items() if not k.startswith(("n_", "sem_"))})

def add(name, res):
    res["paired_vs_baseline"] = {m: paired(rows_base, res["per_ko"], m, alt="greater")
                                 for m in ["rr_pe", "rr_pe_z", "rr_pe_log", "rr_pe_cent", "rr_pe_cent_z", "rr_auroc", "rr_auroc_log",
                                           "rp_pe", "rp_pe_z", "rp_pe_log", "rp_pe_cent", "rp_pe_nocog", "rp_auroc", "rr_pe_combo", "rp_pe_combo", "rr_retrieval", "rp_retrieval", "rr_pe_combofit", "rp_pe_combofit", "rr_pe_meanresp", "rp_pe_meanresp"]}
    # gene-specificity: matched minus target-shuffled, per KO (does the prediction fit ITS OWN gene better than others?)
    sp = {}
    for blk in ["rr", "rp"]:
        for scope in ["overall"] + TAGS:
            d = np.array([r.get(f"{blk}_pe", np.nan) - r.get(f"{blk}_pe_shuf", np.nan) for r in res["per_ko"]
                          if scope == "overall" or r["cohort"] == scope], float); d = d[np.isfinite(d)]
            sp[f"{blk}_{scope}"] = dict(n=int(len(d)), mean=float(d.mean()) if len(d) else None,
                                        p_gt0=float(ttest_1samp(d, 0, alternative="greater").pvalue) if len(d) > 2 else None)
    res["specificity_matched_minus_shuffled"] = sp
    RES[name] = res; save()
    s = res["summary"]["overall"]
    log(f"{name}: RR pe {s['rr_pe']:+.3f} z {s['rr_pe_z']:+.3f} log {s['rr_pe_log']:+.3f} cent {s['rr_pe_cent']:+.3f} shuf {s['rr_pe_shuf']:+.3f} "
        f"meanresp {s['rr_pe_meanresp']:+.3f} | RP pe {s['rp_pe']:+.3f} cent {s['rp_pe_cent']:+.3f} nocog {s['rp_pe_nocog']:+.3f} | "
        f"Gsym {res['structure']['corr_Grr_GrrT']:.4f}")

add("fig5_baseline_spec", dict(per_ko=rows_base, summary=RES["fig5_baseline"]["summary"], structure=RES["fig5_baseline"]["structure"]))
add("ft_standard", run_variant("ft_standard"))
if os.environ.get("RC_EXTRA") == "1": add("ft_zscale", run_variant("ft_zscale", dir_scale="z"))
if os.environ.get("RC_NEG", "1") == "1":
    for k in range(1 if QUICK else 3):
        add(f"ft_labelshuffle_{k}", run_variant("ft_labelshuffle", target="shuffle", perm_seed=k))
    add("ft_signflip", run_variant("ft_signflip", target="flip"))
    add("ft_meantarget", run_variant("ft_meantarget", target="mean"))

if os.environ.get("RC_EXTRA") == "1":
    add("ft_centered", run_variant("ft_centered", target="centered"))
    add("ft_centered_z", run_variant("ft_centered_z", target="centered", dir_scale="z"))
if os.environ.get("RC_NCE", "1") == "1":
    grid = {}
    for lam in ([1.0] if QUICK else NCE_GRID):
        finetune(CTX, IN_TR, infonce=(lam, NCE_TAU))
        pt = predict_units(CTX, {t: IN_TR[t] + IN_VA[t] for t in TAGS}); sm = summarize(score(CTX, IN_VA, pt, ref_means(CTX, IN_TR, pt)))["overall"]
        grid[str(lam)] = dict(lam=lam, inner_retr=sm["rr_retrieval"], inner_cent=sm["rr_pe_cent"], inner_pe=sm["rr_pe"], inner_rp_retr=sm["rp_retrieval"])
        log(f"  CMC InfoNCE inner lam={lam}: retr {sm['rr_retrieval']:.3f} cent {sm['rr_pe_cent']:+.3f} pe {sm['rr_pe']:+.3f}")
    best = max(grid.values(), key=lambda v: v["inner_retr"])
    res = run_variant("ft_infonce", infonce=(best["lam"], NCE_TAU)); res["inner_grid"] = grid; res["selected"] = best
    add("ft_infonce", res)

# ---- (B2) DeepSEM under the IDENTICAL harness (same split, shock magnitude, mutilation, loss, scale, optimiser,
#           steps, early stopping, spectral rule, subset draws and scorer). Only the model differs.
if os.environ.get("RC_DEEPSEM") == "1":
    dsz = np.load(f"{_CMC_RESULTS}/baselines/deepsem_W_809k_clamp09.npz", allow_pickle=True)
    W0 = torch.tensor(np.asarray(dsz["W"], float), dtype=torch.float32, device=DEVICE); W0.fill_diagonal_(0.0)
    dsg = {g: i for i, g in enumerate(np.asarray(dsz["genes"]).astype(str))}; NG = W0.shape[0]
    model.scm.load_state_dict(SCM0)
    with torch.no_grad():
        A0 = assemble(); XBAR = {t: control_side(A0, CTX[t]["eps"])[0].mean(0) for t in TAGS}   # CMC's control reconstruction (z)
    DSC = {}
    for t in TAGS:
        cols = np.array([dsg.get(x, -1) for x in CTX[t]["common"]]); assert (cols >= 0).all(), "DeepSEM vocab mismatch"
        DSC[t] = torch.as_tensor(cols, dtype=torch.long, device=DEVICE)
    def ds_shock(t, g):   # identical magnitude to CMC's do-op: drive g to zero from its control (z-space) level
        gi_ = CTX[t]["INFO"][g]["gidx"]
        return float((Z0[gi_] if SHOCK_MODE == "zero" else 0.0) - XBAR[t][gi_])
    def ds_forward(Wf, t, genes):
        gi = [dsg[g] for g in genes]; K = len(gi)
        E = torch.zeros(NG, K, device=DEVICE); Ind = torch.zeros(NG, K, device=DEVICE)
        for k, (g, j) in enumerate(zip(genes, gi)): E[j, k] = ds_shock(t, g); Ind[j, k] = 1.0
        keep = 1.0 - Ind; X = E.clone()
        for _ in range(40): X = (E + Wf.T @ X) * keep + E       # mutilated do(g): g's incoming edges cut
        return X[DSC[t]]                                        # [n_common, K]
    @torch.no_grad()
    def ds_predict_units(Wf, units):
        out = {}
        for t, genes in units.items():
            if not genes: continue
            X = ds_forward(Wf, t, genes).cpu().numpy().astype(float)
            for k, g in enumerate(genes): out[(t, g)] = (X[:, k], np.full(len(CTX[t]["preti"]), np.nan))
        return out
    def ds_mask(budget):
        a = W0.abs().clone(); a.fill_diagonal_(-1.0)
        if budget >= NG * (NG - 1): m = torch.ones_like(W0); m.fill_diagonal_(0.0); return m
        thr = torch.topk(a.flatten(), budget).values[-1]; return (a >= thr).float()
    _snd = [torch.randn(NG, 1, device=DEVICE), torch.randn(NG, 1, device=DEVICE)]
    def ds_specnorm(M):
        with torch.no_grad():
            u, v = _snd
            for _ in range(12): u = M @ v; u = u / (u.norm() + 1e-12); v = M.t() @ u; v = v / (v.norm() + 1e-12)
            _snd[0], _snd[1] = u.detach(), v.detach()
        return (_snd[0] * (M @ _snd[1])).sum()
    def ds_finetune(train_units, budget, target="real", perm_seed=0, mc=MC, infonce=None):
        mask = ds_mask(budget); dW = torch.zeros_like(W0, requires_grad=True)
        opt = torch.optim.Adam([dW], lr=mc["lr"], weight_decay=WD)
        tg = {t: [g for g in gl if CTX[t]["INFO"][g]["obs_r_raw"].numel() >= 5] for t, gl in train_units.items()}
        active = [t for t in TAGS if tg.get(t)]; n_ds = max(1, len(active)); TG = {}
        for t in active:
            c = CTX[t]; genes = tg[t]; src = list(genes)
            if target == "shuffle": src = list(np.array(genes)[np.random.RandomState(1000 + perm_seed).permutation(len(genes))])
            O = np.stack([np.where(c["INFO"][h]["keep"], c["INFO"][h]["lfc"], np.nan) for h in genes]) if target == "centered" else None
            for j, (g, h) in enumerate(zip(genes, src)):
                I = c["INFO"][g]; J = c["INFO"][h]; pos = J["de_pos"]
                if target == "centered":
                    raw = (I["lfc"] - np.nanmean(O[[i for i in range(len(genes)) if i != j]], 0))[I["de_pos"]]; pos = I["de_pos"]
                else:
                    raw = J["lfc"][pos]
                zz = raw / np.clip(np.asarray(c["align"]["train_rna_std"])[c["reti"][pos]], STD_FLOOR, None)
                TG[(t, g)] = (torch.as_tensor(pos, dtype=torch.long, device=DEVICE), torch.as_tensor(raw, dtype=torch.float32, device=DEVICE),
                              torch.as_tensor(zz, dtype=torch.float32, device=DEVICE))
        best, best_dW, since, best_step = float("inf"), None, 0, -1
        NCE_O = {t: nce_targets(CTX, t, tg[t]) for t in active} if infonce is not None else {}
        for step in range(STEPS):
            opt.zero_grad(); total = 0.0; Wf = W0 + mask * dW
            for t in active:
                X = ds_forward(Wf, t, tg[t]); L = []
                for k, g in enumerate(tg[t]):
                    pos, raw, zz = TG[(t, g)]; pr = X[pos, k]; l = torch.zeros((), device=DEVICE)
                    if pr.numel() >= 5 and torch.std(pr) > 1e-12: l = l + mc["w_dir"] * (-pearson_t(pr, raw))
                    l = l + mc["w_mag"] * ((pr - zz) ** 2).mean(); L.append(l)
                ds = torch.stack(L).mean()
                if infonce is not None and len(tg[t]) >= 3:
                    ds = ds + infonce[0] * info_nce(X[torch.as_tensor(CTX[t]["S"], device=DEVICE)].T, NCE_O[t], infonce[1])
                (ds / n_ds).backward(retain_graph=True); total += float(ds.detach()) / n_ds
            rl = mc["w_rho"] * torch.relu(ds_specnorm(W0 + mask * dW) - mc["rho_pen"]) ** 2; rl.backward(); total += float(rl.detach())
            torch.nn.utils.clip_grad_norm_([dW], 1.0); opt.step()
            with torch.no_grad(): dW.data = torch.nan_to_num(dW.data)
            rho = float(torch.linalg.eigvals((W0 + mask * dW).detach()).abs().max()) if step % 10 == 9 else 0.0
            if total < best - 1e-4: best, since, best_step, best_dW = total, 0, step, dW.detach().clone()
            else: since += 1
            if rho > mc["rho_guard"] or since >= PATIENCE: break
        Wf = (W0 + mask * (best_dW if best_dW is not None else dW.detach())).detach()
        return Wf, dict(best_step=best_step, best_loss=best, budget=int(mask.sum()), rho=float(torch.linalg.eigvals(Wf).abs().max()))
    def ds_struct(Wf):
        W = Wf.cpu().numpy(); off = ~np.eye(W.shape[0], dtype=bool); out = dict(corr_W_WT=float(np.corrcoef(W[off], W.T[off])[0, 1]))
        if CTRI is not None:
            gmap = np.array([dsg.get(g, -1) for g in _genes_train]); a, b = gmap[CTRI[0]], gmap[CTRI[1]]; k = (a >= 0) & (b >= 0)
            f, r = np.abs(W[a[k], b[k]]), np.abs(W[b[k], a[k]])
            out.update(collectri_fwd_over_rev_median=float(np.median(f) / (np.median(r) + 1e-12)), collectri_frac_fwd_gt_rev=float(np.mean(f > r)))
        return out
    TT = {t: TRAIN[t] + TEST[t] for t in TAGS}
    def ds_eval(Wf):
        pt = ds_predict_units(Wf, TT); rows = score(CTX, TEST, pt, ref_means(CTX, TRAIN, pt)); return rows, pt
    rows_ds0, pt0 = ds_eval(W0)
    DSR = dict(n_scm_params_cmc=int(sum(p.numel() for p in model.scm.parameters())),
               baseline=dict(summary=summarize(rows_ds0), per_ko=rows_ds0, structure=ds_struct(W0)))
    pa = ds_predict_units(W0, ALL); rows_ds_fig3 = score(CTX, ALL, pa, ref_means(CTX, ALL, pa))
    DSR["fig3_obs_knockdown"] = dict(summary=summarize(rows_ds_fig3), per_ko=rows_ds_fig3, similarity=similarity(CTX, ALL, pa),
                                    paired_vs_cmc={m: paired(rows_ds_fig3, rows_fig3, m) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_pe_combo"]})
    sb = DSR["baseline"]["summary"]["overall"]; log(f"DeepSEM baseline (knockdown, held-out): RR pe {sb['rr_pe']:+.3f} cent {sb['rr_pe_cent']:+.3f} shuf {sb['rr_pe_shuf']:+.3f} combo {sb['rr_pe_combo']:+.3f}")
    BUD = {"16k": 16384, "cmc_matched": DSR["n_scm_params_cmc"], "full": NG * (NG - 1)}
    for bn, b in BUD.items():
        for tgt in (["real", "centered", "shuffle"] if bn == "cmc_matched" else ["real"]):
            Wf, info = ds_finetune(TRAIN, b, target=tgt); rows, _ = ds_eval(Wf)
            key = f"ft_{bn}_{tgt}"
            DSR[key] = dict(info=info, summary=summarize(rows), per_ko=rows, structure=ds_struct(Wf),
                            paired_vs_baseline={m: paired(rows_ds0, rows, m, alt="greater") for m in ["rr_pe", "rr_pe_cent", "rr_pe_combo", "rr_auroc", "rr_retrieval"]},
                            paired_vs_cmc_ft={m: paired(rows, RES["ft_standard" if tgt != "centered" else "ft_centered"]["per_ko"], m) for m in ["rr_pe", "rr_pe_cent", "rr_pe_combo", "rr_auroc"]})
            s0 = DSR[key]["summary"]["overall"]
            log(f"DeepSEM {key} (budget {info['budget']}, step {info['best_step']}): RR pe {s0['rr_pe']:+.3f} cent {s0['rr_pe_cent']:+.3f} shuf {s0['rr_pe_shuf']:+.3f} combo {s0['rr_pe_combo']:+.3f} auroc {s0['rr_auroc']:.3f}")
            RES["deepsem_matched"] = DSR; save()
    if os.environ.get("RC_NCE", "1") == "1":
        grid = {}
        for lam in ([1.0] if QUICK else NCE_GRID):
            Wf, _ = ds_finetune(IN_TR, BUD["cmc_matched"], infonce=(lam, NCE_TAU))
            pt = ds_predict_units(Wf, {t: IN_TR[t] + IN_VA[t] for t in TAGS}); sm = summarize(score(CTX, IN_VA, pt, ref_means(CTX, IN_TR, pt)))["overall"]
            grid[str(lam)] = dict(lam=lam, inner_retr=sm["rr_retrieval"], inner_cent=sm["rr_pe_cent"], inner_pe=sm["rr_pe"])
            log(f"  DeepSEM InfoNCE inner lam={lam}: retr {sm['rr_retrieval']:.3f} cent {sm['rr_pe_cent']:+.3f}")
        best = max(grid.values(), key=lambda v: v["inner_retr"])
        Wf, info = ds_finetune(TRAIN, BUD["cmc_matched"], infonce=(best["lam"], NCE_TAU)); rows, _ = ds_eval(Wf)
        DSR["ft_cmc_matched_infonce"] = dict(info=info, selected=best, inner_grid=grid, summary=summarize(rows), per_ko=rows, structure=ds_struct(Wf),
                                            paired_vs_baseline={m: paired(rows_ds0, rows, m, alt="greater") for m in ["rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc"]},
                                            paired_vs_cmc_infonce={m: paired(rows, RES["ft_infonce"]["per_ko"], m) for m in ["rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc"]} if "ft_infonce" in RES else {})
        s0 = DSR["ft_cmc_matched_infonce"]["summary"]["overall"]
        log(f"DeepSEM InfoNCE (lam {best['lam']}): RR pe {s0['rr_pe']:+.3f} cent {s0['rr_pe_cent']:+.3f} retr {s0['rr_retrieval']:.3f}")
        RES["deepsem_matched"] = DSR; save()
    if os.environ.get("RC_LC", "1") == "1":
        dlc = {}
        for size in ([5, 25] if QUICK else [5, 10, 25, 50, 100, 200, "all"]):
            runs = []
            for sidx in range(1 if QUICK else {5: 10, 10: 10, 25: 10, 50: 5, 100: 5, 200: 5, "all": 1}[size]):
                if size == "all": gsel = set(TRAIN_G)
                else: gsel = set(np.array(TRAIN_G)[np.random.RandomState(SEED + 1000 * sidx + size).permutation(len(TRAIN_G))[:size]].tolist())
                Wf, _ = ds_finetune({t: [g for g in TRAIN[t] if g in gsel] for t in TAGS}, BUD["cmc_matched"])
                runs.append(summarize(ds_eval(Wf)[0])["overall"])
            dlc[str(size)] = {m: dict(mean=float(np.nanmean([r[m] for r in runs])), sd=float(np.nanstd([r[m] for r in runs], ddof=1)) if len(runs) > 1 else 0.0)
                              for m in ["rr_pe", "rr_pe_cent", "rr_pe_shuf", "rr_pe_combo", "rr_auroc", "rr_retrieval"]}
            log(f"DeepSEM LC size={size}: RR pe {dlc[str(size)]['rr_pe']['mean']:+.3f} cent {dlc[str(size)]['rr_pe_cent']['mean']:+.3f} combo {dlc[str(size)]['rr_pe_combo']['mean']:+.3f}")
            DSR["learning_curve"] = dlc; RES["deepsem_matched"] = DSR; save()
    model.scm.load_state_dict(SCM0)

# ---- (C) learning curve with all scoring variants + more subsets
if os.environ.get("RC_LC", "1") == "1" and os.environ.get("RC_CMC_LC", "1") == "1":
    SIZES = [5, 25] if QUICK else [5, 10, 25, 50, 100, 200, "all"]
    NSUB = {5: 10, 10: 10, 25: 10, 50: 5, 100: 5, 200: 5, "all": 1}
    curve = {}
    for size in SIZES:
        runs = []
        for s in range(1 if QUICK else NSUB[size]):
            if size == "all": gsel = set(TRAIN_G)
            else:
                rng = np.random.RandomState(SEED + 1000 * s + size)
                gsel = set(np.array(TRAIN_G)[rng.permutation(len(TRAIN_G))[:size]].tolist())
            tu = {t: [g for g in TRAIN[t] if g in gsel] for t in TAGS}
            info = finetune(CTX, tu)
            pt = predict_units(CTX, {t: TRAIN[t] + TEST[t] for t in TAGS})
            rows = score(CTX, TEST, pt, ref_means(CTX, TRAIN, pt))
            sm = summarize(rows)["overall"]; sm["genes"] = sorted(gsel) if size != "all" else "all"
            sm["structure"] = structure(); runs.append(sm)
        agg = {}
        for mtr in METRICS:
            v = np.array([r[mtr] for r in runs], float)
            agg[mtr] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0, n_subsets=len(v))
        for sk in ["corr_Grr_GrrT", "collectri_fwd_over_rev_median"]:
            v = np.array([r["structure"].get(sk, np.nan) for r in runs], float); agg[sk] = dict(mean=float(np.nanmean(v)))
        curve[str(size)] = dict(agg=agg, runs=runs)
        log(f"LC size={size}: RR pe {agg['rr_pe']['mean']:+.3f}±{agg['rr_pe']['sd']:.3f} cent {agg['rr_pe_cent']['mean']:+.3f} "
            f"shuf {agg['rr_pe_shuf']['mean']:+.3f} | RP pe {agg['rp_pe']['mean']:+.3f} cent {agg['rp_pe_cent']['mean']:+.3f}")
        RES["learning_curve"] = curve; save()

# ---- (D) nested selection of the fine-tuning weights (inner split of TRAIN genes only)
if os.environ.get("RC_NESTED", "1") == "1":
    rng = np.random.RandomState(7); tg = np.array(TRAIN_G); prm = rng.permutation(len(tg))
    inner_val = set(tg[prm[:round(0.3 * len(tg))]].tolist())
    IN_TR = {t: [g for g in TRAIN[t] if g not in inner_val] for t in TAGS}; IN_VA = {t: [g for g in TRAIN[t] if g in inner_val] for t in TAGS}
    grid = {}
    for wd_ in ([1.0] if QUICK else [0.5, 1.0, 2.0]):
        for wm_ in ([1.0] if QUICK else [0.5, 1.0, 2.0]):
            mc = dict(MC); mc["w_dir"], mc["w_mag"] = wd_, wm_
            finetune(CTX, IN_TR, mc=mc)
            pt = predict_units(CTX, {t: IN_TR[t] + IN_VA[t] for t in TAGS})
            sm = summarize(score(CTX, IN_VA, pt, ref_means(CTX, IN_TR, pt)))["overall"]
            grid[f"{wd_}_{wm_}"] = dict(w_dir=wd_, w_mag=wm_, inner_rr_pe=sm["rr_pe"], inner_rp_pe=sm["rp_pe"],
                                        inner_score=np.nanmean([sm["rr_pe"], sm["rp_pe"]]))
    best = max(grid.values(), key=lambda v: v["inner_score"])
    mc = dict(MC); mc["w_dir"], mc["w_mag"] = best["w_dir"], best["w_mag"]
    res = run_variant("ft_nested", mc=mc); res["inner_grid"] = grid; res["selected"] = best
    add("ft_nested", res)

# ---- (E) BH-FDR DE ground truth (re-derive labels AND targets), principal results
if os.environ.get("RC_BH", "1") == "1":
    CB = {t: build_ctx(t, "bh") for t in TAGS}
    ALLB = {t: CB[t]["evaluable"] for t in TAGS}
    trB = {t: [g for g in CB[t]["evaluable"] if g in set(TRAIN_G)] for t in TAGS}
    teB = {t: [g for g in CB[t]["evaluable"] if g in set(TEST_G)] for t in TAGS}
    model.scm.load_state_dict(SCM0)
    pa = predict_units(CB, ALLB); r3 = score(CB, ALLB, pa, ref_means(CB, ALLB, pa))
    pb = predict_units(CB, {t: trB[t] + teB[t] for t in TAGS}); rb = score(CB, teB, pb, ref_means(CB, trB, pb))
    resB = run_variant("ft_bh", ctx_d=CB, train=trB, test=teB)
    resB["paired_vs_baseline"] = {m: paired(rb, resB["per_ko"], m, alt="greater") for m in ["rr_pe", "rr_pe_cent", "rp_pe", "rp_pe_cent"]}
    RES["bh"] = dict(n_units={t: len(ALLB[t]) for t in TAGS}, n_train={t: len(trB[t]) for t in TAGS}, n_test={t: len(teB[t]) for t in TAGS},
                     fig3_obs=summarize(r3), fig5_baseline=summarize(rb), ft=resB["summary"], paired=resB["paired_vs_baseline"],
                     per_ko_fig3=r3, per_ko_base=rb, per_ko_ft=resB["per_ko"])
    if os.environ.get("RC_BASELINES") == "1":
        rows = score(CB, {t: [g for g in ALLB[t] if (t, g) in preds_b["corr"]] for t in TAGS}, preds_b["corr"],
                     ref_means(CB, {t: [g for g in ALLB[t] if (t, g) in preds_b["corr"]] for t in TAGS}, preds_b["corr"]))
        RES["bh"]["fig3_corr"] = summarize(rows)
        RES["bh"]["fig3_corr_paired"] = {m: paired(rows, r3, m) for m in ["rr_auroc", "rr_pe", "rp_auroc", "rp_pe"]}
    log("BH:", RES["bh"]["n_units"], "fig5 base RR", round(RES["bh"]["fig5_baseline"]["overall"]["rr_pe"], 3), "-> ft", round(resB["summary"]["overall"]["rr_pe"], 3))
    save()

model.scm.load_state_dict(SCM0)
RES["elapsed_s"] = time.time() - T0; save()
log("DONE ->", OUT)
