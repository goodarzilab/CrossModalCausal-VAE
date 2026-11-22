"""Combined-pool interventional fine-tune of the latent SCM adjacency A (encoder + decoder FROZEN).

The realistic "use ALL my perturbation data" setting. Unlike leave-one-dataset-out (finetune_multidataset.py),
here we POOL all evaluable (gene,dataset) KO units across Frangieh+iccite+arce and HOLD OUT BY GENE,
STRATIFIED BY DATASET (each dataset contributes ~30% of its units to a held-out test, ~70% to train),
so every dataset's manifold is represented in BOTH train and test. This tests whether a shared A can be
oriented by pooled interventions and generalize to NEW KO targets within the same datasets.

Design:
  1. Pool all evaluable units (Frangieh 181 / iccite 214 / arce 23 = 418; ~401 unique genes).
  2. By-gene 70/30 split stratified per dataset (all instances of a held-out gene -> test; no target leakage).
  3. Fine-tune ONLY A with the COMBINED loss (dir Pearson + z-space magnitude MSE), dataset-BALANCED
     (per-dataset mean, then averaged across datasets), w_dir=1 w_mag=1 w_rp=0.5 w_rho=0.1, lr 5e-4, rho_guard 0.92.
  4. Evaluate held-out KOs, RR and RP, pearDE + AUROC, OVERALL (pooled) and PER-DATASET, vs baseline (frozen A).
  5. LEARNING CURVE: fine-tune on {5,10,25,50,100,200,all} train genes (mean+-SEM over random subsets),
     fixed held-out test -> does the observational model need MANY or FEW perturbations to orient A?
  6. PARETO: (RR pearDE, RR AUROC) and (RP pearDE, RP AUROC) scatter over #train-genes + a w_dir x w_mag
     sweep at #train=all, tracing the direction<->ranking trade-off.

Machinery (soft-do predict_target/control_side, combined loss, spectral-norm penalty, metrics) is adapted
from eval/finetune_interventional.py; multi-dataset assembly/eval from eval/finetune_multidataset.py.
Neither file is modified (both are top-level scripts, so their helpers are copied here). /tmp data caches
use the IDENTICAL key format, so any existing v2/multi caches are reused.

Usage:
  CMC_DE_LFC_RNA=0.10 CMC_DE_LFC_PROT=0.05 \
  python analysis/fig5_interventions/citeseq_finetune/finetune_combined.py results/linfb_2c_fb10_b20_v4
Env knobs: CMC_FT_STEPS(150) CMC_FT_PATIENCE(30) CMC_FT_WD(1e-4) CMC_FT_TRAIN_CELLS(cuda:3000)
  CMC_FC_SIZES="5,10,25,50,100,200,all"  CMC_FC_SEEDS(3)  combined weights CMC_FT_WDIR/WMAG/WRP/WRHO/LR_COMB/RHO_COMB
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
import os, sys, json, gc, hashlib
os.environ.setdefault("CMC_DE_LFC_RNA", "0.10")   # locked DE filter
os.environ.setdefault("CMC_DE_LFC_PROT", "0.05")

import pickle
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(_CMC_ROOT, "analysis", "common"))
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE

# ------------------------------------------------------------------- config
P.MAX_CONTROL_CELLS = 4000
RD = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
    os.path.join(_CMC_RESULTS, "linfb_2c_fb10_b20_v4")
REPO_RESULTS = _CMC_RESULTS
OUT_JSON = os.environ.get("CMC_FT_OUT") or os.path.join(REPO_RESULTS, "finetune_combined.json")
FIG_DIR = os.path.join(REPO_RESULTS, "figures", "slides")
LFCR, LFCP = os.environ["CMC_DE_LFC_RNA"], os.environ["CMC_DE_LFC_PROT"]
DEVICE = P.DEVICE
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

DATA = f"{_CMC_DATA}"
DATASETS = {
    "frangieh": (f"{DATA}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad",
                 f"{DATA}/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad"),
    "iccite":   (f"{DATA}/chen_perturb_icciteseq/export/iccite_rna.h5ad",
                 f"{DATA}/chen_perturb_icciteseq/export/iccite_prot.h5ad"),
    "arce":     (f"{DATA}/arce_perturb_citeseq/arce_rna.h5ad",
                 f"{DATA}/arce_perturb_citeseq/arce_prot.h5ad"),
}
ALL_TAGS = ["frangieh", "iccite", "arce"]

STEPS    = int(os.environ.get("CMC_FT_STEPS", "150"))
PATIENCE = int(os.environ.get("CMC_FT_PATIENCE", "30"))
WD       = float(os.environ.get("CMC_FT_WD", "1e-4"))
N_TRAIN_CELLS = (3000 if DEVICE.type == "cuda" else 1500)
N_TRAIN_CELLS = int(os.environ.get("CMC_FT_TRAIN_CELLS", str(N_TRAIN_CELLS)))
N_SEEDS = int(os.environ.get("CMC_FC_SEEDS", "3"))
SIZES_ENV = os.environ.get("CMC_FC_SIZES", "5,10,25,50,100,200,all")

MC = dict(lr=float(os.environ.get("CMC_FT_LR_COMB", "5e-4")), rho_guard=float(os.environ.get("CMC_FT_RHO_COMB", "0.92")),
          w_dir=float(os.environ.get("CMC_FT_WDIR", "1.0")), w_mag=float(os.environ.get("CMC_FT_WMAG", "1.0")),
          w_rp=float(os.environ.get("CMC_FT_WRP", "0.5")),  w_rho=float(os.environ.get("CMC_FT_WRHO", "0.1")),
          rho_pen=float(os.environ.get("CMC_FT_RHO_PEN", "0.9")))
STD_FLOOR = 0.05

print(f"=== COMBINED-POOL interventional fine-tune | run={os.path.basename(RD)} device={DEVICE} ===")
print(f"    DE filter LFC_RNA={LFCR} LFC_PROT={LFCP} | steps={STEPS} patience={PATIENCE} wd={WD} train_cells={N_TRAIN_CELLS}")
print(f"    combined cfg = {MC} | sizes={SIZES_ENV} seeds={N_SEEDS}")

cfg = json.load(open(f"{RD}/config.json"))
_use_cache = os.environ.get("CMC_FT_CACHE", "1") == "1"

# ------------------------------------------------------------------- build model (n_g/n_p from ckpt shapes)
_sd0 = torch.load(P.resolve_ckpt(RD), map_location="cpu")
N_G, N_P = _sd0["dec_rna.weight"].shape[0], _sd0["dec_prot.weight"].shape[0]
del _sd0
model = P.load_ctx_model(CrossModalCausalLDVAE, RD, N_G, N_P, cfg)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
for p in model.scm.parameters():
    p.requires_grad_(True)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_scm = sum(p.numel() for p in model.scm.parameters())
print(f"    trainable params: {n_train:,} (scm={n_scm:,}) / total {sum(p.numel() for p in model.parameters()):,}")
assert n_train == n_scm, "only scm params must be trainable"

_frozen_snap = {k: dict(model.named_parameters())[k].detach().clone() for k in
                ["dec_rna.weight", "dec_prot.weight", "bias_rna", "bias_prot",
                 "enc_rna.fc_mu.weight", "enc_prot.fc_mu.weight"]}
_scm_init_state = {k: v.detach().clone() for k, v in model.scm.state_dict().items()}

lr_, lp_ = model.latent_r, model.latent_p
Wr = model._dec_r_w().detach().to(DEVICE)
bias_rna = model.bias_rna.detach().to(DEVICE)
bias_prot = model.bias_prot.detach().to(DEVICE)
egate = model._eps_gate()[0].to(DEVICE) if getattr(model, "eps_bottleneck", False) else None
solve = lambda Amat, e: model.scm._scm_solve(Amat, e)

def assemble():
    A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
    return torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0).to(DEVICE)

def control_side(A_full, e):
    z_c = solve(A_full, e)
    zr_c, zp_c = torch.split(z_c, [lr_, lp_], 1)
    return model.dec_rna(zr_c) + bias_rna, model.dec_prot(zp_c) + bias_prot

MUT = os.environ.get("CMC_FT_MUTILATE", "1") == "1"   # 1 = mutilated soft-do (cut incoming edges);
#   0 = non-mutilated shock-and-propagate, IDENTICAL to perturb_validation (Fig 3). Applied to BOTH
#   the training do-loss and every eval so the do-operation is exactly the same everywhere.
def predict_target(A_full, rec_r_c, rec_p_c, e, gidx, mutilate=True):
    """Torch replica of mutilation_directed_eval.predict(): decoder-loading soft-do, propagate, decode."""
    B = e.shape[0]
    dx = torch.zeros_like(rec_r_c); dx[:, gidx] = -rec_r_c[:, gidx]
    deps = torch.cat([dx @ Wr, torch.zeros(B, lp_, device=DEVICE)], 1)
    if egate is not None:
        deps = deps * egate.unsqueeze(0)
    Amat = A_full
    if mutilate:
        s = (Wr[gidx].abs() / (Wr[gidx].abs().max() + 1e-8)).clamp(0, 1)
        Amat = A_full.clone(); Amat[:, :lr_] = Amat[:, :lr_] * (1 - s).unsqueeze(0)
    z_p = solve(Amat, e + deps)
    zr_p, zp_p = torch.split(z_p, [lr_, lp_], 1)
    d_r = ((model.dec_rna(zr_p) + bias_rna) - rec_r_c).mean(0)
    d_p = ((model.dec_prot(zp_p) + bias_prot) - rec_p_c).mean(0)
    return d_r, d_p

_sn_u = torch.randn(lr_ + lp_, 1, device=DEVICE); _sn_v = torch.randn(lr_ + lp_, 1, device=DEVICE)
def spectral_norm(A):
    global _sn_u, _sn_v
    with torch.no_grad():
        u, v = _sn_u, _sn_v
        for _ in range(12):
            u = A @ v; u = u / (u.norm() + 1e-12)
            v = A.t() @ u; v = v / (v.norm() + 1e-12)
        _sn_u, _sn_v = u.detach(), v.detach()
    return (_sn_u * (A @ _sn_v)).sum()

def spectral_radius(A):
    with torch.no_grad():
        return float(torch.linalg.eigvals(A.detach().cpu()).abs().max())

def au(lab, sc):
    lab = lab.astype(int)
    if lab.sum() < 3 or lab.sum() >= len(lab):
        return (np.nan, np.nan)
    return roc_auc_score(lab, sc), average_precision_score(lab, sc) / lab.mean()

def pe(pred, obs, m):
    if m.sum() < 5 or np.std(pred[m]) == 0 or np.std(obs[m]) == 0:
        return np.nan
    return float(np.corrcoef(pred[m], obs[m])[0, 1])

def pearson_t(pred, obs):
    pc = pred - pred.mean(); oc = obs - obs.mean()
    return (pc * oc).sum() / (pc.norm() * oc.norm() + 1e-8)

# ------------------------------------------------------------------- dataset loading (cached to /tmp)
def load_dataset(tag):
    rna_path, prot_path = DATASETS[tag]
    # cache key MUST be content-unique per model: the old key = basename(ckpt) = "final_model.pt" collided
    # across every model (all share that filename) -> a stale encode from a DIFFERENT model was silently
    # reused, giving wrong eps. Key on model-dir + a content hash of the checkpoint file.
    _ckpt_p = P.resolve_ckpt(RD)
    _ckpt_h = hashlib.md5(open(_ckpt_p, "rb").read()).hexdigest()[:12]
    ck = f"{os.path.basename(RD)}_{_ckpt_h}"
    cache = f"/tmp/ft_interv_{tag}_{ck}_{LFCR}_{LFCP}.pkl"
    if _use_cache and os.path.exists(cache):
        print(f"    [data cache HIT] {tag}: {cache}")
        b = pickle.load(open(cache, "rb"))
        return b["align"], b["de"], b["eps"]
    print(f"    [loading] {tag} ...")
    P.PERTURB_RNA_PATH = rna_path; P.PERTURB_PROT_PATH = prot_path
    pr, pp, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
    de = P.compute_observed_de(pr, pp, align)
    _er, _ep, *_ = P.encode_control_cells(model, pr, pp, align)
    eps_np = torch.cat([_er, _ep], 1).numpy()[:4000]
    del pr, pp; gc.collect()
    if _use_cache:
        try:
            pickle.dump({"align": align, "de": de, "eps": eps_np}, open(cache, "wb"))
            print(f"    [data cache SAVE] {tag}: {cache}")
        except Exception as e:
            print(f"    [cache save failed] {str(e)[:80]}")
    return align, de, eps_np

def build_ctx(tag):
    align, de, eps_np = load_dataset(tag)
    common = np.array(align["common_genes"])
    reti = align["rna_eval_train_indices"]; preti = align["prot_eval_train_indices"]
    tgi = align["train_gene_idx"]; g2i = align["train_gene_to_idx_common"]
    tr_std = np.asarray(align["train_rna_std"]); tp_std = np.asarray(align["train_prot_std"])
    INFO, evaluable = {}, []
    for g, d in de.items():
        if g not in tgi or g not in g2i:
            continue
        keep = common != g
        lab = d["sig_rna"][keep].astype(int)
        if lab.sum() < 3:
            continue
        lfc = np.asarray(d["lfc_rna"])[keep]; dm = lab.astype(bool)
        labp = d.get("sig_prot", np.array([])); has_p = len(labp) and labp.sum() >= 3
        de_pos = np.where(keep & d["sig_rna"].astype(bool))[0]
        tr_idx = reti[de_pos]
        obs_raw = np.asarray(d["lfc_rna"])[de_pos]
        obs_z = obs_raw / np.clip(tr_std[tr_idx], STD_FLOOR, None)
        info = dict(gidx=tgi[g], keep=keep, lab=lab, lfc=lfc, dm=dm, has_p=bool(has_p),
                    labp=labp.astype(int) if has_p else None,
                    lfcp=np.asarray(d["lfc_prot"]) if has_p else None,
                    dmp=labp.astype(bool) if has_p else None,
                    sel_r=torch.as_tensor(tr_idx, dtype=torch.long, device=DEVICE),
                    obs_r_raw=torch.as_tensor(obs_raw, dtype=torch.float32, device=DEVICE),
                    obs_r_z=torch.as_tensor(obs_z, dtype=torch.float32, device=DEVICE),
                    n_de=int(dm.sum()), has_p_loss=False)
        if has_p and len(preti) > 0:
            dp_pos = np.where(labp.astype(bool))[0]
            tp_idx = preti[dp_pos]
            op_raw = np.asarray(d["lfc_prot"])[dp_pos]
            op_z = op_raw / np.clip(tp_std[tp_idx], STD_FLOOR, None)
            info.update(sel_p=torch.as_tensor(tp_idx, dtype=torch.long, device=DEVICE),
                        obs_p_raw=torch.as_tensor(op_raw, dtype=torch.float32, device=DEVICE),
                        obs_p_z=torch.as_tensor(op_z, dtype=torch.float32, device=DEVICE),
                        has_p_loss=len(dp_pos) >= 3)
        INFO[g] = info
        evaluable.append(g)
    evaluable = sorted(evaluable)
    return dict(tag=tag, eps=torch.as_tensor(eps_np, dtype=torch.float32, device=DEVICE)[:4000].detach(),
                reti=reti, preti=preti, tgi=tgi, common=common, INFO=INFO, evaluable=evaluable,
                n_prot=len(preti))

def _summ(loc):
    nm = lambda x: float(np.nanmean(x)) if len(x) and not np.all(np.isnan(x)) else float("nan")
    _lst = lambda x: [None if (v != v) else float(v) for v in x]   # per-KO array (nan -> None), order-aligned
    return {"rr_pearDE": nm(loc["rr_p"]), "rr_auroc": nm(loc["rr_a"]),
            "rp_pearDE": nm(loc["rp_p"]), "rp_auroc": nm(loc["rp_a"]),
            "n_rr": int(np.sum(~np.isnan(loc["rr_p"]))), "n_rp": int(np.sum(~np.isnan(loc["rp_p"]))),
            "rr_pearDE_list": _lst(loc["rr_p"]), "rp_pearDE_list": _lst(loc["rp_p"])}

def eval_pooled(units):
    """units: dict tag -> list of genes. Returns overall (pooled per-KO) + per-dataset metrics."""
    agg = {m: [] for m in ["rr_p", "rr_a", "rp_p", "rp_a"]}
    per = {}
    with torch.no_grad():
        A_full = assemble()
        for t, genes in units.items():
            if not genes:
                continue
            ctx = CTX[t]; reti, preti, INFO = ctx["reti"], ctx["preti"], ctx["INFO"]
            rec_r_c, rec_p_c = control_side(A_full, ctx["eps"])
            loc = {m: [] for m in agg}
            for g in genes:
                info = INFO[g]
                d_r, d_p = predict_target(A_full, rec_r_c, rec_p_c, ctx["eps"], info["gidx"], mutilate=MUT)
                dr = d_r.cpu().numpy(); dp = d_p.cpu().numpy()
                a, _ = au(info["lab"], np.abs(dr[reti])[info["keep"]]); loc["rr_a"].append(a)
                loc["rr_p"].append(pe(dr[reti][info["keep"]], info["lfc"], info["dm"]))
                if info["has_p"] and len(preti) > 0:
                    a2, _ = au(info["labp"], np.abs(dp[preti])); loc["rp_a"].append(a2)
                    loc["rp_p"].append(pe(dp[preti], info["lfcp"], info["dmp"]))
            for m in agg:
                agg[m] += loc[m]
            per[t] = _summ(loc)
    return {"overall": _summ(agg), "per_dataset": per}

# ------------------------------------------------------------------- build contexts + stratified split
CTX = {t: build_ctx(t) for t in ALL_TAGS}
n_units = {t: len(CTX[t]["evaluable"]) for t in ALL_TAGS}
all_units = [(t, g) for t in ALL_TAGS for g in CTX[t]["evaluable"]]
uniq_genes = sorted({g for _, g in all_units})
print(f"    units: " + " ".join(f"{t}={n_units[t]}" for t in ALL_TAGS) +
      f" | total={len(all_units)} unique_genes={len(uniq_genes)} shared={len(all_units) - len(uniq_genes)}")

# by-gene 70/30 split, stratified per dataset (all instances of a held-out gene -> test)
test_genes, train_genes = set(), set()
for t in ALL_TAGS:
    genes_t = CTX[t]["evaluable"]
    already_test = [g for g in genes_t if g in test_genes]
    undecided = [g for g in genes_t if g not in test_genes and g not in train_genes]
    n_test_target = round(0.30 * len(genes_t))
    n_more = max(0, n_test_target - len(already_test))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(undecided))
    pick = set(undecided[i] for i in perm[:n_more])
    for g in undecided:
        (test_genes if g in pick else train_genes).add(g)
train_genes = sorted(train_genes); test_genes = sorted(test_genes)
TRAIN_UNITS = {t: [g for g in CTX[t]["evaluable"] if g in set(train_genes)] for t in ALL_TAGS}
TEST_UNITS  = {t: [g for g in CTX[t]["evaluable"] if g in set(test_genes)] for t in ALL_TAGS}
n_train_units = sum(len(v) for v in TRAIN_UNITS.values())
n_test_units = sum(len(v) for v in TEST_UNITS.values())
print("    split (by gene, per-dataset stratified 70/30):")
for t in ALL_TAGS:
    print(f"      {t:9s} train_units={len(TRAIN_UNITS[t]):3d}  test_units={len(TEST_UNITS[t]):3d} "
          f"({len(TEST_UNITS[t]) / max(1, n_units[t]):.0%} held out)")
print(f"      TOTAL train_units={n_train_units} test_units={n_test_units} | train_genes={len(train_genes)} test_genes={len(test_genes)}")

# dump the exact by-gene split so baselines (e.g. DeepSEM finetune) can evaluate on the IDENTICAL
# held-out gene set -> fair pareto. FT_DUMP_SPLIT_ONLY=1 exits here (fast: no fine-tuning).
_split_dump = {"seed": SEED, "de_lfc_rna": LFCR, "de_lfc_prot": LFCP,
               "train_genes": train_genes, "test_genes": test_genes,
               "train_units": {t: TRAIN_UNITS[t] for t in ALL_TAGS},
               "test_units": {t: TEST_UNITS[t] for t in ALL_TAGS}}
json.dump(_split_dump, open(os.path.join(REPO_RESULTS, "finetune_split_genes.json"), "w"), indent=2)
print(f"      [split dump] wrote results/finetune_split_genes.json")
if os.environ.get("FT_DUMP_SPLIT_ONLY") == "1":
    print("      FT_DUMP_SPLIT_ONLY=1 -> exiting after split dump"); sys.exit(0)

# ------------------------------------------------------------------- sanity vs reference mutil jsons
def ref_json(tag):
    p = f"{RD}/mutil_directed_{tag}.json"
    if os.path.exists(p):
        r = json.load(open(p)); return {k: r.get(f"mutil_{k}") for k in ["rr_pearDE", "rr_auroc", "rp_pearDE", "rp_auroc"]}
    return {}

model.scm.load_state_dict(_scm_init_state)
print("\n[SANITY] baseline (frozen init A) over ALL evaluable per dataset vs mutil_directed_<tag>.json:")
sanity = {}; sanity_ok = True
base_full = eval_pooled({t: CTX[t]["evaluable"] for t in ALL_TAGS})
for t in ALL_TAGS:
    b = base_full["per_dataset"][t]; r = ref_json(t); sanity[t] = {"mine": b, "ref": r}
    ok = all(abs(b[k] - r[k]) < 1e-3 for k in ["rr_pearDE", "rr_auroc", "rp_pearDE", "rp_auroc"] if r.get(k) is not None)
    sanity_ok = sanity_ok and ok
    print(f"  {t:9s} RRpearDE {b['rr_pearDE']:+.4f}/{r.get('rr_pearDE')} RRauroc {b['rr_auroc']:.4f}/{r.get('rr_auroc')} "
          f"RPpearDE {b['rp_pearDE']:+.4f}/{r.get('rp_pearDE')} RPauroc {b['rp_auroc']:.4f}/{r.get('rp_auroc')} [{'OK' if ok else 'MISMATCH'}]")
print(f"  --> baseline reproduces mutil reference: {sanity_ok}")

# baseline on the held-out TEST set (overall + per dataset)
BASE_TEST = eval_pooled(TEST_UNITS)
print(f"[baseline held-out] overall RR pearDE={BASE_TEST['overall']['rr_pearDE']:+.4f} AUROC={BASE_TEST['overall']['rr_auroc']:.4f} "
      f"| RP pearDE={BASE_TEST['overall']['rp_pearDE']:+.4f} AUROC={BASE_TEST['overall']['rp_auroc']:.4f}")

# ------------------------------------------------------------------- fine-tune (dataset-balanced combined loss)
def finetune(train_units, mc):
    """train_units: dict tag -> list of genes. Balanced per-dataset gradient accumulation."""
    model.scm.load_state_dict(_scm_init_state)
    opt = torch.optim.Adam([p for p in model.scm.parameters() if p.requires_grad], lr=mc["lr"], weight_decay=WD)
    tgts = {t: [g for g in gl if CTX[t]["INFO"][g]["obs_r_raw"].numel() >= 5] for t, gl in train_units.items()}
    active = [t for t in ALL_TAGS if tgts.get(t)]
    n_ds = max(1, len(active))
    best, best_state, best_step, since = float("inf"), None, -1, 0
    rho0 = spectral_radius(assemble())
    for step in range(STEPS):
        opt.zero_grad(); total = 0.0; per_ds = {}
        for t in active:
            e_tr = CTX[t]["eps"][:N_TRAIN_CELLS]
            A_full = assemble()
            rec_r_c, rec_p_c = control_side(A_full, e_tr)
            L_rr, L_rp = [], []
            for g in tgts[t]:
                info = CTX[t]["INFO"][g]
                d_r, d_p = predict_target(A_full, rec_r_c, rec_p_c, e_tr, info["gidx"], mutilate=MUT)
                pr = d_r[info["sel_r"]]
                l = torch.zeros((), device=DEVICE)
                if mc["w_dir"] > 0 and pr.numel() >= 5 and torch.std(pr) > 1e-12:
                    l = l + mc["w_dir"] * (-pearson_t(pr, info["obs_r_raw"]))
                if mc["w_mag"] > 0:
                    l = l + mc["w_mag"] * ((pr - info["obs_r_z"]) ** 2).mean()
                L_rr.append(l)
                if mc["w_rp"] > 0 and info["has_p_loss"]:
                    pp_ = d_p[info["sel_p"]]
                    lp = torch.zeros((), device=DEVICE)
                    if mc["w_dir"] > 0 and pp_.numel() >= 3 and torch.std(pp_) > 1e-12:
                        lp = lp + mc["w_dir"] * (-pearson_t(pp_, info["obs_p_raw"]))
                    if mc["w_mag"] > 0:
                        lp = lp + mc["w_mag"] * ((pp_ - info["obs_p_z"]) ** 2).mean()
                    L_rp.append(lp)
            if not L_rr:
                continue
            ds_loss = torch.stack(L_rr).mean()
            if L_rp:
                ds_loss = ds_loss + mc["w_rp"] * torch.stack(L_rp).mean()
            (ds_loss / n_ds).backward()
            per_ds[t] = float(ds_loss.detach().cpu()); total += per_ds[t] / n_ds
        if not per_ds:
            break
        if mc["w_rho"] > 0:
            A_full = assemble()
            rho_loss = mc["w_rho"] * torch.relu(spectral_norm(A_full) - mc["rho_pen"]) ** 2
            rho_loss.backward(); total += float(rho_loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_([p for p in model.scm.parameters() if p.requires_grad], 1.0)
        opt.step()
        rho = spectral_radius(assemble())
        if total < best - 1e-4:
            best, best_step, since = total, step, 0
            best_state = {k: v.detach().clone() for k, v in model.scm.state_dict().items()}
        else:
            since += 1
        if rho > mc["rho_guard"]:
            break
        if since >= PATIENCE:
            break
    if best_state is not None:
        model.scm.load_state_dict(best_state)
    return dict(n_train_targets={t: len(tgts.get(t, [])) for t in ALL_TAGS}, best_train_loss=best,
                best_step=best_step, rho_init=rho0, rho_final=spectral_radius(assemble()))

def subsample_train(size, seed):
    if size == "all" or size >= len(train_genes):
        gsel = set(train_genes)
    else:
        rng = np.random.RandomState(SEED + 1000 * seed + size)
        gsel = set(np.array(train_genes)[rng.permutation(len(train_genes))[:size]].tolist())
    return {t: [g for g in TRAIN_UNITS[t] if g in gsel] for t in ALL_TAGS}

# ------------------------------------------------------------------- MAIN: full-train combined tune
print("\n########## MAIN: fine-tune on ALL train genes (combined loss)")
ft_all = finetune(TRAIN_UNITS, MC)
TUNED_TEST = eval_pooled(TEST_UNITS)
TUNED_TRAIN = eval_pooled(TRAIN_UNITS)   # overfitting watch
TUNED_FULL = eval_pooled({t: CTX[t]["evaluable"] for t in ALL_TAGS})   # tuned A on ALL perturbations (train+held-out)
_scm_tuned_path = OUT_JSON.replace(".json", "_scm.pt")   # save tuned SCM so the Fig-3 harness can route it
torch.save(model.scm.state_dict(), _scm_tuned_path); print(f"    [saved tuned SCM] {_scm_tuned_path}")
print(f"    [tuned held-out] overall RR pearDE={TUNED_TEST['overall']['rr_pearDE']:+.4f} AUROC={TUNED_TEST['overall']['rr_auroc']:.4f} "
      f"| RP pearDE={TUNED_TEST['overall']['rp_pearDE']:+.4f} AUROC={TUNED_TEST['overall']['rp_auroc']:.4f}  (best_step={ft_all['best_step']}, rho={ft_all['rho_final']:.3f})")
print(f"    [tuned TRAIN]    overall RR pearDE={TUNED_TRAIN['overall']['rr_pearDE']:+.4f} AUROC={TUNED_TRAIN['overall']['rr_auroc']:.4f} "
      f"| RP pearDE={TUNED_TRAIN['overall']['rp_pearDE']:+.4f} AUROC={TUNED_TRAIN['overall']['rp_auroc']:.4f}")

# ------------------------------------------------------------------- LEARNING CURVE
SIZES = []
for s in SIZES_ENV.split(","):
    s = s.strip()
    SIZES.append("all" if s == "all" else min(int(s), len(train_genes)))
# dedupe preserving order
seen = set(); SIZES = [s for s in SIZES if not (s in seen or seen.add(s))]
print(f"\n########## LEARNING CURVE sizes={SIZES} (n_seeds={N_SEEDS}; 'all'->1 seed)")
_FAST = os.environ.get("CMC_FT_FAST") == "1"   # skip learning-curve + pareto (only need baseline/tuned per-KO)
_NOPARETO = _FAST or os.environ.get("CMC_FT_NOPARETO") == "1"   # skip ONLY the pareto w-grid (keep curve)
curve = {}
for size in ([] if _FAST else SIZES):
    ns = 1 if size == "all" else N_SEEDS
    runs = []
    for seed in range(ns):
        tu = subsample_train(size, seed)
        n_used = sum(len(v) for v in tu.values())
        if n_used == 0:
            continue
        finetune(tu, MC)
        ev = eval_pooled(TEST_UNITS)
        runs.append(ev["overall"])
    if not runs:
        continue
    agg = {}
    for m in ["rr_pearDE", "rr_auroc", "rp_pearDE", "rp_auroc"]:
        vals = np.array([r[m] for r in runs], dtype=float)
        agg[m] = {"mean": float(np.nanmean(vals)),
                  "sem": float(np.nanstd(vals) / np.sqrt(max(1, len(vals)))) if len(vals) > 1 else 0.0}
    curve[str(size)] = {"n_seeds": len(runs), "metrics": agg}
    print(f"    size={str(size):>4}  RRpearDE={agg['rr_pearDE']['mean']:+.4f}+-{agg['rr_pearDE']['sem']:.4f}  "
          f"RRauroc={agg['rr_auroc']['mean']:.4f}  RPpearDE={agg['rp_pearDE']['mean']:+.4f}  RPauroc={agg['rp_auroc']['mean']:.4f}")

# ------------------------------------------------------------------- PARETO: w_dir x w_mag sweep @ all-train
print("\n########## PARETO w_dir x w_mag sweep @ #train=all")
WGRID = [0.5, 1.0, 2.0]
sweep = {}
for wd_ in ([] if _NOPARETO else WGRID):
    for wm_ in WGRID:
        mc = dict(MC); mc["w_dir"] = wd_; mc["w_mag"] = wm_
        finetune(TRAIN_UNITS, mc)
        ev = eval_pooled(TEST_UNITS)["overall"]
        sweep[f"wdir{wd_}_wmag{wm_}"] = {"w_dir": wd_, "w_mag": wm_, **{k: ev[k] for k in ["rr_pearDE", "rr_auroc", "rp_pearDE", "rp_auroc"]}}
        print(f"    w_dir={wd_} w_mag={wm_}: RRpearDE={ev['rr_pearDE']:+.4f} RRauroc={ev['rr_auroc']:.4f} "
              f"RPpearDE={ev['rp_pearDE']:+.4f} RPauroc={ev['rp_auroc']:.4f}")

# ------------------------------------------------------------------- verify frozen
model.scm.load_state_dict(_scm_init_state)
frozen_ok = all(torch.allclose(dict(model.named_parameters())[k].detach(), v) for k, v in _frozen_snap.items())
print(f"\n[CHECK] frozen (encoder/decoder/bias) weights unchanged: {frozen_ok}")

# ------------------------------------------------------------------- save
results = {"run": os.path.basename(RD), "design": "combined-pool, hold-out-by-gene stratified-by-dataset 70/30",
           "device": str(DEVICE), "de_lfc_rna": float(LFCR), "de_lfc_prot": float(LFCP),
           "n_units": n_units, "n_unique_genes": len(uniq_genes),
           "split": {"train_units": {t: len(TRAIN_UNITS[t]) for t in ALL_TAGS},
                     "test_units": {t: len(TEST_UNITS[t]) for t in ALL_TAGS},
                     "n_train_genes": len(train_genes), "n_test_genes": len(test_genes)},
           "hyperparams": {"steps": STEPS, "patience": PATIENCE, "wd": WD, "train_cells": N_TRAIN_CELLS,
                           "std_floor": STD_FLOOR, "mode_cfg": MC, "loss_pooling": "balanced (per-dataset mean, averaged across datasets)"},
           "sanity_vs_reference": sanity, "baseline_reproduces_mutil": bool(sanity_ok),
           "baseline_heldout": BASE_TEST, "tuned_heldout": TUNED_TEST, "tuned_train": TUNED_TRAIN,
           "baseline_allpert": base_full, "tuned_allpert": TUNED_FULL,
           "finetune_info_all": ft_all, "learning_curve": curve, "pareto_wsweep": sweep,
           "frozen_weights_unchanged": bool(frozen_ok)}
os.makedirs(REPO_RESULTS, exist_ok=True)
json.dump(results, open(OUT_JSON, "w"), indent=2)

# ------------------------------------------------------------------- comparison table
METS = ["rr_pearDE", "rr_auroc", "rp_pearDE", "rp_auroc"]
def row(lbl, d): return f"  {lbl:<22} " + " ".join(f"{d[m]:>+10.4f}" for m in METS)
print("\n" + "=" * 92)
print(f"HELD-OUT KO transfer (combined-pool)  |  baseline (frozen A) vs combined-pool-tuned A")
print("=" * 92)
print(f"  {'':<22} " + " ".join(f"{m:>10}" for m in METS))
print("[OVERALL pooled held-out]")
print(row("baseline", BASE_TEST["overall"])); print(row("combined-pool", TUNED_TEST["overall"]))
for t in ALL_TAGS:
    print(f"[{t} held-out KOs]")
    print(row("baseline", BASE_TEST["per_dataset"][t])); print(row("combined-pool", TUNED_TEST["per_dataset"][t]))

# ------------------------------------------------------------------- figures (deck style)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
C_MAIN, C_ALT, C_BASE = "#1f4e79", "#b5651d", "#9aa0a6"
os.makedirs(FIG_DIR, exist_ok=True)

def size_x(s): return len(train_genes) if s == "all" else int(s)
xs = sorted([size_x(s) for s in curve], key=lambda v: v)
skeys = sorted(curve, key=lambda s: size_x(s))

# --- learning curves (2x2) ---
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
panels = [("rr_pearDE", "RNA→RNA direction (pearDE)", 0.0),
          ("rr_auroc", "RNA→RNA AUROC", 0.5),
          ("rp_pearDE", "RNA→Prot direction (pearDE)", 0.0),
          ("rp_auroc", "RNA→Prot AUROC", 0.5)]
for ax, (k, ttl, hl) in zip(axs.ravel(), panels):
    xv = [size_x(s) for s in skeys]
    mv = [curve[s]["metrics"][k]["mean"] for s in skeys]
    sv = [curve[s]["metrics"][k]["sem"] for s in skeys]
    ax.errorbar(xv, mv, yerr=sv, marker="o", ms=5, lw=2, color=C_MAIN, capsize=3, label="combined-pool-tuned A")
    ax.axhline(BASE_TEST["overall"][k], ls="--", c=C_BASE, lw=1.4, label="baseline (frozen A)")
    ax.axhline(hl, ls=":", c="k", lw=.7)
    ax.set_xscale("log"); ax.set_xticks(xv); ax.set_xticklabels([str(int(v)) for v in xv], fontsize=10)
    ax.set_xlabel("# train genes (held-out test fixed)"); ax.set_ylabel(k.replace("_", " ")); ax.set_title(ttl)
axs[0, 0].legend(fontsize=9, loc="best")
fig.suptitle("Combined-pool interventional fine-tune: held-out KO metrics vs # training perturbations",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIG_DIR, "fig_finetune_combined_curve.png"), dpi=130); plt.close(fig)

# --- Pareto (RR + RP) ---
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, (px, py, ttl) in zip(axs, [("rr_pearDE", "rr_auroc", "RNA→RNA: direction vs ranking"),
                                    ("rp_pearDE", "rp_auroc", "RNA→Prot: direction vs ranking")]):
    # learning-curve trajectory (by #train genes)
    xv = [curve[s]["metrics"][px]["mean"] for s in skeys]
    yv = [curve[s]["metrics"][py]["mean"] for s in skeys]
    ax.plot(xv, yv, "-o", color=C_MAIN, ms=6, lw=1.6, label="# train genes (5→all)")
    for s, xx, yy in zip(skeys, xv, yv):
        ax.annotate(s, (xx, yy), fontsize=8, color=C_MAIN, xytext=(3, 3), textcoords="offset points")
    # w_dir x w_mag sweep @ all-train
    sx = [v[px] for v in sweep.values()]; sy = [v[py] for v in sweep.values()]
    ax.scatter(sx, sy, color=C_ALT, s=42, marker="s", label="w_dir×w_mag sweep (all-train)", zorder=3)
    # baseline
    ax.scatter([BASE_TEST["overall"][px]], [BASE_TEST["overall"][py]], color=C_BASE, s=90, marker="*",
               label="baseline (frozen A)", zorder=4, edgecolor="k", linewidth=.5)
    ax.axvline(0.0, ls=":", c="k", lw=.7); ax.axhline(0.5, ls=":", c="k", lw=.7)
    ax.set_xlabel(px.replace("_", " ")); ax.set_ylabel(py.replace("_", " ")); ax.set_title(ttl)
axs[0].legend(fontsize=9, loc="best")
fig.suptitle("Pareto: direction (pearDE) ↔ ranking (AUROC) trade-off on held-out KOs", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG_DIR, "fig_finetune_combined_pareto.png"), dpi=130); plt.close(fig)

json.dump(results, open(OUT_JSON, "w"), indent=2)
print(f"\nSaved: {OUT_JSON}")
print(f"Saved: {os.path.join(FIG_DIR, 'fig_finetune_combined_curve.png')}")
print(f"Saved: {os.path.join(FIG_DIR, 'fig_finetune_combined_pareto.png')}")
print("DONE")
