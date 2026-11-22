"""Cross-validated GENERALIZATION analysis of interventional fine-tuning of the latent SCM adjacency A.

Replaces the coarse 40/60 cross-fit in eval/finetune_interventional.py with a robust CV estimate of how
well fine-tuning ONLY model.scm (encoder + decoder FROZEN) on a mutilation do-loss generalizes to UNSEEN
Frangieh KO targets. Reuses that script's machinery verbatim (differentiable mutilation predict_target /
control_side, /tmp-cached data assembly, the COMBINED loss w_dir=1/w_mag=1/w_rp=0.5/w_rho=0.1 + spectral
penalty, evaluate(), finetune()). FRANGIEH only, combined mode only.

Three analyses (all held-out = targets never in that fit's train set):
  1) K-FOLD CV (K=10) over Frangieh evaluable targets -> held-out RR pearDE/AUROC, RP pearDE/AUROC as
     mean +/- SEM across folds. Robust version of the v2 40/60 cross-fit.
  2) True LEAVE-ONE-TARGET-OUT on a random 20-target subset (fit on the other ~180, eval the 1) to
     confirm 10-fold ~= LOTO.
  3) LEARNING CURVE: fixed 30% held-out test set; fit A on {10,25,50,75,100,all} of the remaining
     training targets (averaged over a few random draws per size) -> held-out RR pearDE & RP AUROC vs
     #train targets. Shows how many perturbations are needed / whether it has saturated.

Read-out is IDENTICAL to eval/mutilation_directed_eval.py. Seed 42 everywhere.

Usage:
  CMC_DE_LFC_RNA=0.10 CMC_DE_LFC_PROT=0.05 \
  python analysis/fig5_interventions/citeseq_finetune/finetune_loto.py results/linfb_2c_fb10_b20_v4
Env knobs (defaults in parens): CMC_LOTO_K(10) CMC_LOTO_N(20) CMC_LC_SIZES(10,25,50,75,100,all)
  CMC_LC_SEEDS(3) CMC_LC_TEST_FRAC(0.30) CMC_FT_STEPS(200) CMC_FT_PATIENCE(40) CMC_FT_TRAIN_CELLS
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
import os, sys, json, time, gc
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
OUT_JSON = os.path.join(REPO_RESULTS, "finetune_loto.json")
FIG_DIR = os.path.join(REPO_RESULTS, "figures", "slides")
FIG_PATH = os.path.join(FIG_DIR, "fig_finetune_loto.png")
V2_JSON = os.path.join(REPO_RESULTS, "finetune_interventional_v2.json")
LFCR, LFCP = os.environ["CMC_DE_LFC_RNA"], os.environ["CMC_DE_LFC_PROT"]
DEVICE = P.DEVICE
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

TRAIN_TAG = "frangieh"
DATA = f"{_CMC_DATA}"
DATASETS = {
    "frangieh": (f"{DATA}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad",
                 f"{DATA}/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad"),
}

# analysis knobs
K_FOLDS   = int(os.environ.get("CMC_LOTO_K", "10"))
LOTO_N    = int(os.environ.get("CMC_LOTO_N", "20"))
LC_SIZES  = os.environ.get("CMC_LC_SIZES", "10,25,50,75,100,all").split(",")
LC_SEEDS  = int(os.environ.get("CMC_LC_SEEDS", "3"))
LC_TEST_FRAC = float(os.environ.get("CMC_LC_TEST_FRAC", "0.30"))

# fine-tuning config (identical to v2 "combined" mode)
STEPS    = int(os.environ.get("CMC_FT_STEPS", "200"))
PATIENCE = int(os.environ.get("CMC_FT_PATIENCE", "40"))
WD       = float(os.environ.get("CMC_FT_WD", "1e-4"))
N_TRAIN_CELLS = 4000 if DEVICE.type == "cuda" else int(os.environ.get("CMC_FT_TRAIN_CELLS", "1500"))
MC = dict(lr=float(os.environ.get("CMC_FT_LR_COMB", "5e-4")), rho_guard=float(os.environ.get("CMC_FT_RHO_COMB", "0.92")),
          w_dir=float(os.environ.get("CMC_FT_WDIR", "1.0")), w_mag=float(os.environ.get("CMC_FT_WMAG", "1.0")),
          w_rp=float(os.environ.get("CMC_FT_WRP", "0.5")), w_rho=float(os.environ.get("CMC_FT_WRHO", "0.1")),
          rho_pen=float(os.environ.get("CMC_FT_RHO_PEN", "0.9")))
STD_FLOOR = 0.05

print(f"=== finetune LOTO/CV generalization | run={os.path.basename(RD)} train={TRAIN_TAG} device={DEVICE} ===", flush=True)
print(f"    DE filter LFC_RNA={LFCR} LFC_PROT={LFCP} | K={K_FOLDS} LOTO_N={LOTO_N} LC_SIZES={LC_SIZES} LC_SEEDS={LC_SEEDS}", flush=True)
print(f"    combined-mode cfg={MC} steps={STEPS} patience={PATIENCE} wd={WD} train_cells={N_TRAIN_CELLS}", flush=True)

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
print(f"    trainable params: {n_train:,} (scm={n_scm:,}) / total {sum(p.numel() for p in model.parameters()):,}", flush=True)
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

# differentiable spectral norm (sigma_max >= spectral radius) via warm-started power iteration.
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

# ------------------------------------------------------------------- metrics (identical to mutil eval)
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
    ck = os.path.basename(P.resolve_ckpt(RD))
    cache = f"/tmp/ft_interv_{tag}_{ck}_{LFCR}_{LFCP}.pkl"
    if _use_cache and os.path.exists(cache):
        print(f"    [data cache HIT] {tag}: {cache}", flush=True)
        b = pickle.load(open(cache, "rb"))
        return b["align"], b["de"], b["eps"]
    print(f"    [loading] {tag} ...", flush=True)
    P.PERTURB_RNA_PATH = rna_path; P.PERTURB_PROT_PATH = prot_path
    pr, pp, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
    de = P.compute_observed_de(pr, pp, align)
    _er, _ep, *_ = P.encode_control_cells(model, pr, pp, align)
    eps_np = torch.cat([_er, _ep], 1).numpy()[:4000]
    del pr, pp; gc.collect()
    if _use_cache:
        try:
            pickle.dump({"align": align, "de": de, "eps": eps_np}, open(cache, "wb"))
            print(f"    [data cache SAVE] {tag}: {cache}", flush=True)
        except Exception as e:
            print(f"    [cache save failed] {str(e)[:80]}", flush=True)
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

def evaluate(ctx, target_list):
    reti, preti, INFO = ctx["reti"], ctx["preti"], ctx["INFO"]
    rr_p, rr_a, rp_p, rp_a = [], [], [], []
    with torch.no_grad():
        A_full = assemble()
        rec_r_c, rec_p_c = control_side(A_full, ctx["eps"])
        for g in target_list:
            info = INFO[g]
            d_r, d_p = predict_target(A_full, rec_r_c, rec_p_c, ctx["eps"], info["gidx"], mutilate=True)
            dr = d_r.cpu().numpy(); dp = d_p.cpu().numpy()
            a, _ = au(info["lab"], np.abs(dr[reti])[info["keep"]]); rr_a.append(a)
            rr_p.append(pe(dr[reti][info["keep"]], info["lfc"], info["dm"]))
            if info["has_p"] and len(preti) > 0:
                a2, _ = au(info["labp"], np.abs(dp[preti])); rp_a.append(a2)
                rp_p.append(pe(dp[preti], info["lfcp"], info["dmp"]))
    nm = lambda x: float(np.nanmean(x)) if len(x) and not np.all(np.isnan(x)) else float("nan")
    return {"rr_pearDE": nm(rr_p), "rr_auroc": nm(rr_a), "rp_pearDE": nm(rp_p), "rp_auroc": nm(rp_a),
            "n_rr": int(np.sum(~np.isnan(rr_p))), "n_rp": int(np.sum(~np.isnan(rp_p)))}

def evaluate_per_target(ctx, target_list):
    """Return per-target metric dicts (for LOTO single-target aggregation), keyed by gene."""
    reti, preti, INFO = ctx["reti"], ctx["preti"], ctx["INFO"]
    out = {}
    with torch.no_grad():
        A_full = assemble()
        rec_r_c, rec_p_c = control_side(A_full, ctx["eps"])
        for g in target_list:
            info = INFO[g]
            d_r, d_p = predict_target(A_full, rec_r_c, rec_p_c, ctx["eps"], info["gidx"], mutilate=True)
            dr = d_r.cpu().numpy(); dp = d_p.cpu().numpy()
            a, _ = au(info["lab"], np.abs(dr[reti])[info["keep"]])
            rr_p = pe(dr[reti][info["keep"]], info["lfc"], info["dm"])
            rp_a = rp_p = np.nan
            if info["has_p"] and len(preti) > 0:
                a2, _ = au(info["labp"], np.abs(dp[preti])); rp_a = a2
                rp_p = pe(dp[preti], info["lfcp"], info["dmp"])
            out[g] = {"rr_pearDE": rr_p, "rr_auroc": a, "rp_pearDE": rp_p, "rp_auroc": rp_a}
    return out

# ------------------------------------------------------------------- fine-tuning (identical to v2 combined)
def finetune(train_list, mc, tag=""):
    model.scm.load_state_dict(_scm_init_state)
    opt = torch.optim.Adam([p for p in model.scm.parameters() if p.requires_grad], lr=mc["lr"], weight_decay=WD)
    tgts = [g for g in train_list if fr["INFO"][g]["obs_r_raw"].numel() >= 5]
    e_tr = fr["eps"][:N_TRAIN_CELLS]
    best, best_state, best_step, since = float("inf"), None, -1, 0
    rho0 = spectral_radius(assemble())
    for step in range(STEPS):
        opt.zero_grad()
        A_full = assemble()
        rec_r_c, rec_p_c = control_side(A_full, e_tr)
        L_rr, L_rp = [], []
        for g in tgts:
            info = fr["INFO"][g]
            d_r, d_p = predict_target(A_full, rec_r_c, rec_p_c, e_tr, info["gidx"], mutilate=True)
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
            print("    [warn] no usable train targets"); break
        loss = torch.stack(L_rr).mean()
        if L_rp:
            loss = loss + mc["w_rp"] * torch.stack(L_rp).mean()
        if mc["w_rho"] > 0:
            loss = loss + mc["w_rho"] * torch.relu(spectral_norm(A_full) - mc["rho_pen"]) ** 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.scm.parameters() if p.requires_grad], 1.0)
        opt.step()
        lv = float(loss.detach().cpu()); rho = spectral_radius(assemble())
        if lv < best - 1e-4:
            best, best_step, since = lv, step, 0
            best_state = {k: v.detach().clone() for k, v in model.scm.state_dict().items()}
        else:
            since += 1
        if rho > mc["rho_guard"]:
            print(f"      [{tag}] rho guard {rho:.3f} > {mc['rho_guard']}; stop @ {step}", flush=True); break
        if since >= PATIENCE:
            break
    if best_state is not None:
        model.scm.load_state_dict(best_state)
    return dict(n_train_targets=len(tgts), best_train_loss=best, best_step=best_step,
                rho_init=rho0, rho_final=spectral_radius(assemble()))

# ------------------------------------------------------------------- aggregation helpers
def agg(vals):
    a = np.asarray(vals, dtype=float); a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"mean": float("nan"), "sem": float("nan"), "n": 0}
    m = float(a.mean())
    s = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return {"mean": m, "sem": s, "n": int(len(a))}

METS = ["rr_pearDE", "rr_auroc", "rp_pearDE", "rp_auroc"]

# ------------------------------------------------------------------- build frangieh context
CTX = {TRAIN_TAG: build_ctx(TRAIN_TAG)}
fr = CTX[TRAIN_TAG]
n_prot_eval = sum(fr["INFO"][g]["has_p"] for g in fr["evaluable"])
print(f"    evaluable frangieh targets = {len(fr['evaluable'])} (with sig protein = {n_prot_eval})", flush=True)

# ------------------------------------------------------------------- sanity vs reference mutil json
model.scm.load_state_dict(_scm_init_state)
def ref_json(tag):
    p = f"{RD}/mutil_directed_{tag}.json"
    if os.path.exists(p):
        r = json.load(open(p)); return {k: r.get(f"mutil_{k}") for k in METS}
    return {}
base_all = evaluate(fr, fr["evaluable"]); ref = ref_json(TRAIN_TAG)
print("\n[SANITY] baseline over ALL evaluable vs reference mutil_directed_frangieh.json:", flush=True)
print(f"  mine RRpearDE={base_all['rr_pearDE']:+.4f} RRauroc={base_all['rr_auroc']:.4f} "
      f"RPpearDE={base_all['rp_pearDE']:+.4f} RPauroc={base_all['rp_auroc']:.4f}", flush=True)
print(f"  ref  RRpearDE={ref.get('rr_pearDE')} RRauroc={ref.get('rr_auroc')} "
      f"RPpearDE={ref.get('rp_pearDE')} RPauroc={ref.get('rp_auroc')}", flush=True)
sanity_ok = ref and abs(base_all["rr_pearDE"] - ref["rr_pearDE"]) < 1e-3 and abs(base_all["rr_auroc"] - ref["rr_auroc"]) < 1e-3

results = {"run": os.path.basename(RD), "train_tag": TRAIN_TAG, "device": str(DEVICE), "seed": SEED,
           "de_lfc_rna": float(LFCR), "de_lfc_prot": float(LFCP), "n_evaluable": len(fr["evaluable"]),
           "n_evaluable_prot": int(n_prot_eval),
           "hyperparams": {"K_folds": K_FOLDS, "loto_n": LOTO_N, "lc_sizes": LC_SIZES, "lc_seeds": LC_SEEDS,
                           "lc_test_frac": LC_TEST_FRAC, "steps": STEPS, "patience": PATIENCE, "wd": WD,
                           "train_cells": N_TRAIN_CELLS, "std_floor": STD_FLOOR, "mode_cfg": MC},
           "sanity": {"baseline_all": base_all, "reference": ref, "matches_reference": bool(sanity_ok)}}

evaluable = list(fr["evaluable"])
NE = len(evaluable)

# =================================================================== 1) K-FOLD CV
print("\n" + "=" * 90 + f"\n[1] {K_FOLDS}-FOLD CV over {NE} Frangieh evaluable targets\n" + "=" * 90, flush=True)
rng = np.random.RandomState(SEED)
perm = rng.permutation(NE)
folds_idx = [perm[i::K_FOLDS] for i in range(K_FOLDS)]   # interleaved -> balanced sizes
cv_folds = []
t0 = time.time()
for fi, test_ii in enumerate(folds_idx):
    test_list = sorted(evaluable[i] for i in test_ii)
    train_list = sorted(evaluable[i] for i in perm if i not in set(test_ii))
    model.scm.load_state_dict(_scm_init_state)
    base_ev = evaluate(fr, test_list)
    ft = finetune(train_list, MC, tag=f"cv{fi}")
    tune_ev = evaluate(fr, test_list)
    cv_folds.append({"fold": fi, "n_train": len(train_list), "n_test": len(test_list),
                     "baseline": base_ev, "tuned": tune_ev, "finetune_info": ft})
    print(f"  fold {fi:2d} n_test={len(test_list):3d} | base RR={base_ev['rr_pearDE']:+.3f}/{base_ev['rr_auroc']:.3f} "
          f"RP={base_ev['rp_pearDE']:+.3f}/{base_ev['rp_auroc']:.3f}  ->  tuned RR={tune_ev['rr_pearDE']:+.3f}/"
          f"{tune_ev['rr_auroc']:.3f} RP={tune_ev['rp_pearDE']:+.3f}/{tune_ev['rp_auroc']:.3f} "
          f"[rho {ft['rho_init']:.2f}->{ft['rho_final']:.2f} best@{ft['best_step']}]", flush=True)
cv_summary = {}
for m in METS:
    cv_summary[m] = {"baseline": agg([f["baseline"][m] for f in cv_folds]),
                     "tuned": agg([f["tuned"][m] for f in cv_folds])}
results["kfold_cv"] = {"folds": cv_folds, "summary": cv_summary}
print(f"  [K-fold done in {time.time()-t0:.0f}s]", flush=True)
print("  --- CV mean +/- SEM across folds (held-out) ---", flush=True)
for m in METS:
    b, t = cv_summary[m]["baseline"], cv_summary[m]["tuned"]
    print(f"    {m:11s}  baseline {b['mean']:+.4f}+/-{b['sem']:.4f}   tuned {t['mean']:+.4f}+/-{t['sem']:.4f}", flush=True)

# =================================================================== 2) TRUE LOTO on 20-subset
print("\n" + "=" * 90 + f"\n[2] TRUE LEAVE-ONE-TARGET-OUT on random {LOTO_N}-target subset\n" + "=" * 90, flush=True)
rng2 = np.random.RandomState(SEED + 1)
loto_targets = sorted(evaluable[i] for i in rng2.choice(NE, size=min(LOTO_N, NE), replace=False))
loto_rows = []
t0 = time.time()
model.scm.load_state_dict(_scm_init_state)
base_per = evaluate_per_target(fr, loto_targets)
for j, g in enumerate(loto_targets):
    train_list = sorted(x for x in evaluable if x != g)
    ft = finetune(train_list, MC, tag=f"loto{j}")
    tune_g = evaluate_per_target(fr, [g])[g]
    loto_rows.append({"target": g, "n_train": len(train_list), "baseline": base_per[g], "tuned": tune_g,
                      "finetune_info": ft})
    print(f"  [{j+1:2d}/{len(loto_targets)}] {g:12s} base RRpearDE={base_per[g]['rr_pearDE']:+.3f} "
          f"-> tuned {tune_g['rr_pearDE']:+.3f} | base RRauroc={base_per[g]['rr_auroc']:.3f} "
          f"-> {tune_g['rr_auroc']:.3f}", flush=True)
loto_summary = {}
for m in METS:
    loto_summary[m] = {"baseline": agg([r["baseline"][m] for r in loto_rows]),
                       "tuned": agg([r["tuned"][m] for r in loto_rows])}
results["true_loto"] = {"targets": loto_targets, "rows": loto_rows, "summary": loto_summary}
print(f"  [LOTO done in {time.time()-t0:.0f}s]", flush=True)
print("  --- LOTO mean +/- SEM over subset (held-out single target) ---", flush=True)
for m in METS:
    b, t = loto_summary[m]["baseline"], loto_summary[m]["tuned"]
    print(f"    {m:11s}  baseline {b['mean']:+.4f}+/-{b['sem']:.4f} (n={b['n']})   tuned {t['mean']:+.4f}+/-{t['sem']:.4f} (n={t['n']})", flush=True)

# =================================================================== 3) LEARNING CURVE
print("\n" + "=" * 90 + f"\n[3] LEARNING CURVE (fixed {int(LC_TEST_FRAC*100)}% held-out test)\n" + "=" * 90, flush=True)
rng3 = np.random.RandomState(SEED + 2)
perm3 = rng3.permutation(NE)
n_test = int(round(LC_TEST_FRAC * NE))
test_ii = perm3[:n_test]
pool_ii = perm3[n_test:]
lc_test = sorted(evaluable[i] for i in test_ii)
lc_pool = [evaluable[i] for i in pool_ii]
print(f"    fixed test={len(lc_test)}  train-pool={len(lc_pool)}", flush=True)
sizes = []
for s in LC_SIZES:
    sizes.append(len(lc_pool) if s.strip().lower() == "all" else min(int(s), len(lc_pool)))
sizes = sorted(set(sizes))

model.scm.load_state_dict(_scm_init_state)
lc_base = evaluate(fr, lc_test)
print(f"    baseline(frozen A) on test: RRpearDE={lc_base['rr_pearDE']:+.4f} RPauroc={lc_base['rp_auroc']:.4f}", flush=True)
lc_points = []
t0 = time.time()
for sz in sizes:
    is_all = sz >= len(lc_pool)
    nseed = 1 if is_all else LC_SEEDS
    draws = []
    for s in range(nseed):
        rseed = np.random.RandomState(10_000 + 137 * s + sz)
        sub = sorted(lc_pool if is_all else [lc_pool[i] for i in rseed.choice(len(lc_pool), size=sz, replace=False)])
        ft = finetune(sub, MC, tag=f"lc{sz}_s{s}")
        ev = evaluate(fr, lc_test)
        draws.append({"seed": s, "n_train_used": ft["n_train_targets"], "eval": ev, "finetune_info": ft})
    pt = {"size": sz, "is_all": bool(is_all), "n_draws": nseed, "draws": draws}
    for m in METS:
        pt[m] = agg([dr["eval"][m] for dr in draws])
    lc_points.append(pt)
    print(f"  size={sz:3d} (draws={nseed}) held-out RRpearDE={pt['rr_pearDE']['mean']:+.4f}+/-{pt['rr_pearDE']['sem']:.4f} "
          f"| RPauroc={pt['rp_auroc']['mean']:.4f}+/-{pt['rp_auroc']['sem']:.4f} "
          f"| RRauroc={pt['rr_auroc']['mean']:.4f}", flush=True)
results["learning_curve"] = {"test_targets": lc_test, "pool_size": len(lc_pool), "sizes": sizes,
                             "baseline": lc_base, "points": lc_points}
print(f"  [learning curve done in {time.time()-t0:.0f}s]", flush=True)

# =================================================================== v2 reference (40/60)
if os.path.exists(V2_JSON):
    v2 = json.load(open(V2_JSON))
    def v2_combined_mean(setkey, m):
        f = v2["modes"]["combined"]["folds"]
        return float(np.nanmean([f["fold1_train40_test60"]["eval"][setkey][m],
                                 f["fold2_train60_test40"]["eval"][setkey][m]]))
    results["v2_reference_40_60"] = {
        "combined_frangieh_heldout": {m: v2_combined_mean("frangieh_test", m) for m in METS},
        "baseline_frangieh": {m: float(np.nanmean([v2["baseline"]["frangieh_fold1_test"][m],
                                                   v2["baseline"]["frangieh_fold2_test"][m]])) for m in METS}}

# =================================================================== verify only scm changed
model.scm.load_state_dict(_scm_init_state)
frozen_ok = all(torch.allclose(dict(model.named_parameters())[k].detach(), v) for k, v in _frozen_snap.items())
results["frozen_weights_unchanged"] = bool(frozen_ok)
print(f"\n[CHECK] frozen (encoder/decoder/bias) weights unchanged: {frozen_ok}", flush=True)

os.makedirs(REPO_RESULTS, exist_ok=True)
json.dump(results, open(OUT_JSON, "w"), indent=2)
print(f"Saved: {OUT_JSON}", flush=True)

# =================================================================== figure (deck style)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False})
    C_BASE, C_TUNE = "#9aa0a6", "#1f4e79"
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))

    # Panel A: CV mean+/-SEM bars, baseline vs tuned, for RR pearDE & RP AUROC
    axA = axs[0]
    groups = [("rr_pearDE", "RNA→RNA\ndirection (pearDE)"), ("rp_auroc", "RNA→Prot\nAUROC")]
    x = np.arange(len(groups)); w = 0.34
    bmean = [cv_summary[m]["baseline"]["mean"] for m, _ in groups]
    bsem = [cv_summary[m]["baseline"]["sem"] for m, _ in groups]
    tmean = [cv_summary[m]["tuned"]["mean"] for m, _ in groups]
    tsem = [cv_summary[m]["tuned"]["sem"] for m, _ in groups]
    axA.bar(x - w/2, bmean, w, yerr=bsem, capsize=4, label="baseline (frozen A)", color=C_BASE,
            error_kw=dict(lw=1.2))
    axA.bar(x + w/2, tmean, w, yerr=tsem, capsize=4, label=f"tuned A ({K_FOLDS}-fold CV)", color=C_TUNE,
            error_kw=dict(lw=1.2))
    for xi, (bm, tm) in enumerate(zip(tmean, bmean)):
        axA.text(xi - w/2, bmean[xi] + bsem[xi] + 0.012, f"{bmean[xi]:.2f}", ha="center", fontsize=9, color="#555")
        axA.text(xi + w/2, tmean[xi] + tsem[xi] + 0.012, f"{tmean[xi]:.2f}", ha="center", fontsize=9,
                 color=C_TUNE, fontweight="bold")
    axA.axhline(0.0, ls="-", c="k", lw=.6)
    axA.axhline(0.5, ls="--", c="#bbb", lw=.8)
    axA.set_xticks(x); axA.set_xticklabels([g[1] for g in groups], fontsize=11)
    axA.set_ylabel("metric (held-out targets)")
    axA.set_title(f"{K_FOLDS}-fold CV generalization  (mean ± SEM)")
    axA.legend(fontsize=10, loc="upper left")
    axA.set_ylim(min(0, min(bmean + tmean) - 0.05), max(tmean + bmean) + 0.18)

    # Panel B: learning curve of held-out RR pearDE vs #train targets (+ RP AUROC on twin axis)
    axB = axs[1]
    xs = [p["size"] for p in lc_points]
    rr = [p["rr_pearDE"]["mean"] for p in lc_points]
    rrse = [p["rr_pearDE"]["sem"] for p in lc_points]
    axB.errorbar(xs, rr, yerr=rrse, marker="o", ms=6, lw=2, color=C_TUNE, capsize=3, label="held-out RR pearDE")
    axB.axhline(lc_base["rr_pearDE"], ls="--", c="#9aa0a6", lw=1.2, label="baseline RR pearDE (frozen A)")
    axB.set_xlabel("# training targets (fixed 30% held-out test)")
    axB.set_ylabel("held-out RR pearDE", color=C_TUNE)
    axB.tick_params(axis="y", labelcolor=C_TUNE)
    axB.set_title("Learning curve: direction recovery vs #perturbations")
    axB2 = axB.twinx(); axB2.spines["top"].set_visible(False)
    rpa = [p["rp_auroc"]["mean"] for p in lc_points]
    rpse = [p["rp_auroc"]["sem"] for p in lc_points]
    axB2.errorbar(xs, rpa, yerr=rpse, marker="s", ms=5, lw=1.6, ls=":", color="#e8963a", capsize=3,
                  label="held-out RP AUROC")
    axB2.set_ylabel("held-out RP AUROC", color="#e8963a")
    axB2.tick_params(axis="y", labelcolor="#e8963a")
    h1, l1 = axB.get_legend_handles_labels(); h2, l2 = axB2.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower right")

    fig.suptitle("Interventional fine-tune of A generalizes to UNSEEN Frangieh KO targets (encoder/decoder frozen)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=130); plt.close(fig)
    print(f"Saved figure: {FIG_PATH}", flush=True)
except Exception as e:
    print(f"[figure failed] {e}", flush=True)

print("DONE", flush=True)
