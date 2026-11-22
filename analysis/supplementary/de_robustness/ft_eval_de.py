"""Replogle 2022 test of interventional fine-tuning of the RNA-only CMC-VAE latent SCM.

Metrics (RNA block) copied from analysis/common/reviewer_controls.py: rr_pe, rr_pe_cent, rr_pe_shuf, rr_pe_meanresp, rr_retrieval,
rr_auroc, rr_auroc_meanresp (+ rr_pe_z, rr_pe_meanpred, rr_pe_combo). DE ground truth = P.compute_observed_de
(MW p<0.05 & |dmean log-norm|>=0.10, >=30 cells), evaluable if >=3 DE genes excluding the KO gene.

Operators (z-space, encoder/decoder frozen):
  paper : reviewer_controls.predict_target (shock dx_g = -rec_c[:,g], projected eps shift dx@W, soft mutilation of the
          columns of A by |W_g|/max|W_g|), i.e. the manuscript do-op.
  shift : correctly-scaled shift intervention: minimum-norm latent eps shift whose propagated, decoded effect on the KO
          gene itself equals the knockdown delta_g = Z0_g - mean rec_c[g] (drive g to zero expression); no mutilation.
Both are linear in eps, so the cell-averaged response is computed exactly from the mean control eps (checked against a
per-cell copy of predict_target at start-up).

Fine-tuning (only the SCM; lr 5e-4, Adam wd 1e-4, clip 1, w_dir=w_mag=1, w_rho=0.1 spectral-norm hinge at 0.9 as in
reviewer_controls) with PROJECTED spectral radius (after each step rescale W so rho(A)<=0.95, no stopping) for up to
300 steps; the number of steps is selected on an inner 70/30 split of the training genes (criterion: real-target
loss on the inner-validation genes, the same criterion for every variant), then refit on all training genes.
Variants: real, shuffle (targets permuted among training KOs), mean (cohort-mean response target), centered,
infonce (real + lambda*InfoNCE, lambda from {0.3,1,3} by inner retrieval).

Usage: python ft_eval.py LINE SEED [OPS=shift,paper]     env RP_PARTS=obs,ft,lc,xctx  RP_QUICK=1

nov-2026 revision copy (paper_nov_revisions/code/perturbseq). Changes vs eval/replogle/ft_eval.py:
  * PS_PRE=obs|ctrl (+ PS_TAG) selects the pretrained model: obs = NT + training-perturbation cells (seeds 42/1/2 read
    from results/replogle/models, k562 s1 from the revision models dir), ctrl = non-targeting cells only. The z-scoring
    stats are the ones the model was trained with (config["stats"]).
  * extra metrics on ONE canonical scale (log units divided by the control-cell sd of each feature):
    rr_pe_cent_canon, rr_retrieval_canon, rr_pcor (partial corr(p, o | m), m = leave-one-out mean measured response of
    training KOs), rr_systema_cacc (Systema centroid accuracy, see systema_cacc()).
  * paired tests are two-sided (t and Wilcoxon) + bootstrap-over-KO 95% CI.
  * new parts: rrr (reduced-rank ridge restricted to span(D), rank sweep), nblc (ridge / RRR learning curves on the same
    gene subsets as the SCM learning curve), ds (DeepSEM with top-73,728 / top-36,672 masks and a rank-12 update across
    all genes; InfoNCE weight chosen on the inner split like CMC; InfoNCE shuffled arm), xctx2 (cross-context regime).
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
# ==== review2/de_robust COPY (reviewer M10). Changes vs analysis/fig5_interventions/perturbseq/ft_eval.py:
#  * PS_DE_TRAIN=orig|bh|pb|pbgem selects the DE definition that builds the FINE-TUNING targets / evaluable KOs / S
#    (default orig = identical to the paper); evaluation under the alternative DE definitions is done offline from
#    the saved predictions (rescore_perturbseq.py).
#  * Only the Fig.5 arms are fitted (obs, real, infonce, 3 label-shuffled, 3 InfoNCE-shuffled; DeepSEM top73728 all
#    arms); every fitted SCM / DeepSEM graph is used to predict ALL KOs of {line}_de.npz and saved to preds/*.npz.
#  * Fitting code, seeds, inner split, step selection and the order of DeepSEM fits are unchanged.
import os, sys, json, time
import numpy as np, pandas as pd, torch
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import ttest_rel, wilcoxon, ttest_1samp
from sklearn.metrics import roc_auc_score
HERE = f"{_CMC_ROOT}/analysis/fig5_interventions/perturbseq"; sys.path.insert(0, HERE)
from rna_model import RnaCMC

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

LINE, SEED = sys.argv[1], int(sys.argv[2]); OPS = (sys.argv[3] if len(sys.argv) > 3 else "shift,paper").split(",")
OTHER = {"jurkat": "k562", "k562": "jurkat"}[LINE]
PARTS = os.environ.get("RP_PARTS", "obs,ft,lc,xctx").split(","); QUICK = os.environ.get("RP_QUICK") == "1"
REPO = f"{_CMC_ROOT}"; D = f"{_CMC_RESULTS}/replogle/data"
AN = f"{_CMC_RESULTS}/paper/analyses/perturbseq"
PRE = os.environ.get("PS_PRE", "obs"); TAG = os.environ.get("PS_TAG", PRE)
MD = f"{AN}/models/{LINE}_{TAG}_s{SEED}"
if PRE == "obs" and not os.path.exists(f"{MD}/final_model.pt"): MD = f"{_CMC_RESULTS}/replogle/models/{LINE}_s{SEED}"
DER = f"{_CMC_RESULTS}/paper/analyses/review2/de_robust"; DE_TRAIN = os.environ.get("PS_DE_TRAIN", "orig")
OUT = os.environ["RP_OUT"]; PRED_OUT = os.environ["RP_PRED_OUT"]
LFCR, PTHR, STD_FLOOR, JIT = 0.10, 0.05, 0.05, 1e-6
STEPS = 30 if QUICK else 300
MC = dict(lr=5e-4, w_dir=1.0, w_mag=1.0, w_rho=0.1, rho_pen=0.9, rho_proj=0.95); WD = 1e-4
NCE_GRID, NCE_TAU = [0.3, 1.0, 3.0], 0.1
DEV = torch.device("cpu"); torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
torch.manual_seed(SEED); np.random.seed(SEED)

FEATS = json.load(open(f"{D}/features.json"))["features"]; G = len(FEATS); FI = {g: i for i, g in enumerate(FEATS)}
SPL = json.load(open(f"{D}/split.json")); TRS, TES = set(SPL["train_genes"]), set(SPL["test_genes"])
cfg = json.load(open(f"{MD}/config.json"))
STATS_F = cfg.get("stats", f"{D}/{LINE}_stats.npz")
ST = np.load(STATS_F); MEAN, STD = ST["mean"].astype(float), ST["std"].astype(float)
SD_R = np.clip(STD, 1e-8, None)
SD_NT = np.clip(np.load(f"{AN}/data/{LINE}_ntstats.npz")["std"].astype(float), 1e-8, None)   # canonical scale

# ------------------------------------------------------------------ model
model = RnaCMC(G, cfg["latent"], cfg["hidden"]); model.load_state_dict(torch.load(f"{MD}/final_model.pt", map_location="cpu")); model.eval()
for p in model.parameters(): p.requires_grad_(False)
for p in model.scm.parameters(): p.requires_grad_(True)
SCM0 = {k: v.detach().clone() for k, v in model.scm.state_dict().items()}
L = model.latent_r; Wr = model.dec_rna.weight.detach(); bias = model.bias_rna.detach(); I_L = torch.eye(L)
Z0 = torch.as_tensor(-MEAN / SD_R, dtype=torch.float32)
obsA = pd.read_csv(f"{D}/{LINE}_obs.csv"); ntA = np.where(obsA.is_nt.values)[0]
ctrl = np.random.RandomState(42).choice(ntA, min(4000, len(ntA)), replace=False)
Xc = ((np.load(f"{D}/{LINE}_X.npy", mmap_mode="r")[np.sort(ctrl)] - MEAN) / (STD + 1e-6)).astype(np.float32)
with torch.no_grad(): EPS = model.enc_rna(torch.from_numpy(Xc))[0]
EBAR = EPS.mean(0)
CORR = np.load(f"{D}/{LINE}_corr.npy").astype(float)
log(f"model {MD} [{PRE}/{TAG}] stats {STATS_F}: G={G} L={L}; control cells encoded {len(EPS)}; ops={OPS} parts={PARTS} steps={STEPS}")

def get_A(): return model.scm.get_A()
def Ninv(A): return torch.linalg.inv(I_L - A + JIT * I_L)        # z = eps @ Ninv(A)  (== solve(I-A^T+jI, eps^T)^T)

def predict_batch(A, gidx, op):
    """[K,G] mean response (z-space) for KOs of features gidx (LongTensor [K])."""
    N = Ninv(A); U = N @ Wr.T                                      # [L,G]: decoded effect of a unit eps shift
    rbar = EBAR @ U + bias                                         # mean control reconstruction
    if op == "shift":
        u = U[:, gidx].T                                           # [K,L]
        delta = Z0[gidx] - rbar[gidx]
        deps = delta[:, None] * u / (u.pow(2).sum(1, keepdim=True) + 1e-12)
        return deps @ U
    out = []
    for s0 in range(0, len(gidx), 256):
        gi = gidx[s0:s0 + 256]; Wg = Wr[gi]                         # [k,L]
        s = (Wg.abs() / (Wg.abs().max(1, keepdim=True).values + 1e-8)).clamp(0, 1)
        deps = (-rbar[gi])[:, None] * Wg
        # z (I - A diag(1-s)) = v  solved by the convergent fixed point z <- v + (z A)(1-s)  (rho<1; batched LAPACK solve
        # is broken on these CPUs: MKL SLASWP error). Exact at convergence (checked vs per-cell solve in check_ops).
        v = EBAR.unsqueeze(0) + deps; keep_ = 1 - s; zp = v
        for _it in range(1000):
            zn = v + (zp @ A) * keep_
            err = (zn - zp).abs().max(1).values; zp = zn
            if not torch.isfinite(zp).all(): break
            if float(err.max()) <= 1e-7 * (float(zn.abs().max()) + 1e-12): break
        bad = ~(torch.isfinite(zp).all(1)) | (err > 1e-5 * (zp.abs().max(1).values + 1e-12))
        if bad.any():   # rows whose mutilated system did not converge: exact single (non-batched) solves
            rows_ = []
            for k in range(len(gi)):
                if bool(bad[k]):
                    Mk = (I_L - A * keep_[k].unsqueeze(0) + JIT * I_L).T
                    rows_.append(torch.linalg.solve(Mk, v[k].unsqueeze(-1)).squeeze(-1))
                else: rows_.append(zp[k])
            zp = torch.stack(rows_)
        out.append((zp - EBAR @ N) @ Wr.T)
    return torch.cat(out, 0)

@torch.no_grad()
def check_ops():
    """per-cell reviewer_controls.predict_target (paper op) vs the mean-eps batched version."""
    A = get_A(); res = {}
    for gi in [0, 7, 123]:
        e = EPS; zc = model.scm._scm_solve(A, e); rc = model.dec_rna(zc) + bias
        dx = torch.zeros_like(rc); dx[:, gi] = -rc[:, gi]; deps = dx @ Wr
        s = (Wr[gi].abs() / (Wr[gi].abs().max() + 1e-8)).clamp(0, 1); Am = A.clone() * (1 - s).unsqueeze(0)
        zp = model.scm._scm_solve(Am, e + deps); ref = ((model.dec_rna(zp) + bias) - rc).mean(0)
        b = predict_batch(A, torch.tensor([gi]), "paper")[0]
        res[str(gi)] = float((ref - b).abs().max() / (ref.abs().max() + 1e-12))
        sb = predict_batch(A, torch.tensor([gi]), "shift")[0]; res[f"shift_own_{gi}"] = [float(sb[gi]), float(Z0[gi] - (EBAR @ Ninv(A) @ Wr.T + bias)[gi])]
    return res

# ------------------------------------------------------------------ DE contexts
def build_ctx(line):
    """DE of `line` (vs its NT cells) on the model's z-scale (model-line std)."""
    de = np.load(f"{D}/{line}_de.npz"); INFO = {}; ncell = {}
    if DE_TRAIN == "orig":
        SIGS = [(pv < PTHR) & (np.abs(lfc) >= LFCR) for lfc, pv in zip(de["lfc"].astype(float), de["pvals"].astype(float))]; LFCS = de["lfc"].astype(float)
    else:
        dv = np.load(f"{DER}/de/{line}_de_variants.npz"); assert list(dv["genes"].astype(str)) == list(de["genes"].astype(str))
        LFCS = np.nan_to_num(dv[{"bh": "lfc", "pb": "lfc_pb", "pbgem": "lfc_pbgem", "pbmask": "lfc"}[DE_TRAIN]].astype(float))
        SIGS = dv[{"pbmask": "sig_pb"}.get(DE_TRAIN, "sig_" + DE_TRAIN)]
    PVS = np.where(np.stack(SIGS), 0.0, 1.0)
    for g, lfc, pv, nc, sig in zip(de["genes"].astype(str), LFCS, PVS, de["n_cells"], SIGS):
        sig = np.asarray(sig, bool); keep = np.array(FEATS) != g
        if sig[keep].sum() < 3: continue
        de_pos = np.where(keep & sig)[0]
        INFO[g] = dict(g=g, gidx=FI[g], keep=keep, sig=sig, lfc=lfc, de_pos=de_pos, n_cells=int(nc),
                       own_lfc=float(lfc[FI[g]]), own_p=float(pv[FI[g]]))
    return dict(line=line, INFO=INFO, evaluable=sorted(INFO), n_kos_tested=len(de["genes"]), sd_r=SD_R)

def set_S(c, train_genes):
    u = np.zeros(G, bool)
    for g in train_genes: u |= c["INFO"][g]["sig"]
    u &= ~np.isin(np.array(FEATS), c["evaluable"]); c["S"] = np.where(u)[0]

# ------------------------------------------------------------------ metrics (RNA block of reviewer_controls.score)
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

def ref_means(c, genes, preds):
    O = np.stack([np.where(c["INFO"][g]["keep"], c["INFO"][g]["lfc"], np.nan) for g in genes])
    Pd = np.stack([np.where(c["INFO"][g]["keep"], preds[g], np.nan) for g in genes])
    R = dict(o_r=O, p_r=Pd, genes=list(genes), gi={g: i for i, g in enumerate(genes)})
    for k in ["o_r", "p_r"]:
        R[k + "_sum"] = np.nansum(R[k], 0); R[k + "_cnt"] = np.isfinite(R[k]).sum(0)
    return R

def loo(R, k, g):
    """== reviewer_controls._loo_mean (nanmean over reference KOs other than g)."""
    s, n = R[k + "_sum"].copy(), R[k + "_cnt"].astype(float).copy()
    if g in R["gi"]:
        v = R[k][R["gi"][g]]; f = np.isfinite(v); s[f] -= v[f]; n[f] -= 1
    with np.errstate(invalid="ignore", divide="ignore"): return np.where(n > 0, s / n, np.nan)

def shuf_pe(c, genes, preds, pool):
    """vectorised == mean_h pe(pred_g, lfc_h, sig_h & keep_h & keep_g) over h in pool, h != g."""
    P = np.stack([preds[g] for g in genes]).astype(float); gidx = np.array([c["INFO"][g]["gidx"] for g in genes])
    M = np.stack([(c["INFO"][h]["sig"] & c["INFO"][h]["keep"]) for h in pool]).astype(float)
    O = np.stack([c["INFO"][h]["lfc"] for h in pool]).astype(float); OM = O * M
    n = np.broadcast_to(M.sum(1)[None], (len(genes), len(pool))).copy()
    Sp, Spp, Spo = P @ M.T, (P ** 2) @ M.T, P @ OM.T
    So = np.broadcast_to(OM.sum(1)[None], n.shape).copy(); Soo = np.broadcast_to((O ** 2 * M).sum(1)[None], n.shape).copy()
    ex = M[:, gidx].T > 0                                              # pool h has g's own gene in its mask -> drop it
    pg = P[np.arange(len(genes)), gidx][:, None]; og = O[:, gidx].T
    n -= ex; Sp -= ex * pg; Spp -= ex * pg ** 2; So -= ex * og; Soo -= ex * og ** 2; Spo -= ex * pg * og
    with np.errstate(invalid="ignore", divide="ignore"):
        vp = Spp - Sp ** 2 / n; vo = Soo - So ** 2 / n
        r = (Spo - Sp * So / n) / np.sqrt(vp * vo)
    r[(n < 5) | ~(vp > 1e-18) | ~(vo > 1e-18)] = np.nan
    same = np.array([[g == h for h in pool] for g in genes]); r[same] = np.nan
    return np.nanmean(r, 1)

def pcor(p, o, z, m):
    """partial correlation corr(p, o | z) over features m (residualise p and o on [1, z])."""
    m = np.asarray(m, bool) & np.isfinite(p) & np.isfinite(o) & np.isfinite(z)
    if m.sum() < 8: return np.nan
    Z = np.column_stack([np.ones(m.sum()), z[m]])
    rp = p[m] - Z @ np.linalg.lstsq(Z, p[m], rcond=None)[0]; ro = o[m] - Z @ np.linalg.lstsq(Z, o[m], rcond=None)[0]
    if np.std(rp) < 1e-12 or np.std(ro) < 1e-12: return np.nan
    return float(np.corrcoef(rp, ro)[0, 1])

def retrieval(Pm, Om, genes):
    C = np.nan_to_num(Pm @ Om.T); out = {}
    for i, g in enumerate(genes):
        oth = np.arange(len(genes)) != i
        out[g] = float(np.mean(C[i, i] > C[i, oth]) + 0.5 * np.mean(C[i, i] == C[i, oth]))
    return out

TGT_FEAT = None
def systema_cacc(c, genes, preds, sd, hvg_only=False):
    """Systema centroid accuracy (Vinas Torne et al., Nat Biotechnol 2025; mlbio-epfl/systema
    evaluation/centroid_accuracy.py::calculate_centroid_accuracies): Euclidean distance between the predicted mean
    post-perturbation profile (control mean + predicted shift, log-normalised units, all features) and the ground-truth
    centroid of EVERY perturbation of the dataset (here: all evaluable KOs of the line, train + test); score_g = fraction
    of the other centroids that are farther than g's own centroid (chance 0.5). The control mean cancels in the distance,
    so it is computed on shifts. hvg_only=True: same, restricted to the 1,315 non-target HVG features (prevents trivial
    identification through the knocked-down target transcript itself)."""
    global TGT_FEAT
    if TGT_FEAT is None:
        tg = set(SPL["train_genes"]) | set(SPL["test_genes"]); TGT_FEAT = np.array([f in tg for f in FEATS])
    cols = ~TGT_FEAT if hvg_only else np.ones(G, bool)
    cand = c["evaluable"]; O = np.stack([c["INFO"][h]["lfc"][cols] for h in cand]); ci = {h: i for i, h in enumerate(cand)}
    P = np.stack([(preds[g] * sd)[cols] for g in genes])
    Dm = np.sqrt(np.clip((P ** 2).sum(1)[:, None] + (O ** 2).sum(1)[None] - 2 * P @ O.T, 0, None)); out = {}
    for i, g in enumerate(genes):
        if g not in ci or not np.isfinite(Dm[i]).all(): continue
        j = ci[g]; oth = np.arange(len(cand)) != j
        out[g] = float(np.mean(Dm[i, oth] > Dm[i, j]))
    return out

def score(c, genes, preds, R, pool=None):
    S = c["S"]; sd = c["sd_r"]; RET = {}; RETC = {}; SYS = {}; SYSH = {}
    def _z(v): v = v - v.mean(); nn_ = np.linalg.norm(v); return v / nn_ if nn_ > 0 else v
    if len(genes) >= 3:
        Pm = np.stack([_z((preds[g] - loo(R, "p_r", g))[S]) for g in genes])
        Om = np.stack([_z(((c["INFO"][g]["lfc"] - loo(R, "o_r", g)) / sd)[S]) for g in genes])
        RET = retrieval(Pm, Om, genes)
        # canonical scale: log units / control-cell sd for BOTH prediction and measurement
        Pc = np.stack([_z(((preds[g] - loo(R, "p_r", g)) * sd / SD_NT)[S]) for g in genes])
        Oc = np.stack([_z(((c["INFO"][g]["lfc"] - loo(R, "o_r", g)) / SD_NT)[S]) for g in genes])
        RETC = retrieval(Pc, Oc, genes)
        SYS = systema_cacc(c, genes, preds, sd); SYSH = systema_cacc(c, genes, preds, sd, hvg_only=True)
    sh = shuf_pe(c, genes, preds, pool or genes); rows = []
    for j, g in enumerate(genes):
        I = c["INFO"][g]; pr = preds[g]; k = I["keep"]; m = I["sig"] & k
        mu_o, mu_p = loo(R, "o_r", g), loo(R, "p_r", g)
        r = dict(gene=g, n_de_rna=int(m.sum()), rr_retrieval=RET.get(g, np.nan))
        r["rr_auroc"] = au(I["sig"][k], np.abs(pr[k]))
        r["rr_pe"] = pe(pr, I["lfc"], m)
        r["rr_pe_z"] = pe(pr, I["lfc"] / sd, m)
        r["rr_pe_cent"] = pe(pr * sd - mu_p * sd, I["lfc"] - mu_o, m)
        r["rr_pe_meanresp"] = pe(mu_o, I["lfc"], m)
        r["rr_auroc_meanresp"] = au(I["sig"][k], np.abs(np.nan_to_num(mu_o[k])))
        r["rr_pe_meanpred"] = pe(mu_p * sd, I["lfc"], m)
        r["rr_pe_combo"] = pe(mu_o + (pr - mu_p) * sd, I["lfc"], m)
        r["rr_pe_shuf"] = float(sh[j])
        r["rr_retrieval_canon"] = RETC.get(g, np.nan)
        r["rr_pe_cent_canon"] = pe((pr * sd - mu_p * sd) / SD_NT, (I["lfc"] - mu_o) / SD_NT, m)
        r["rr_pcor"] = pcor(pr * sd / SD_NT, I["lfc"] / SD_NT, mu_o / SD_NT, m)
        r["rr_systema_cacc"] = SYS.get(g, np.nan); r["rr_systema_cacc_hvg"] = SYSH.get(g, np.nan)
        rows.append(r)
    return rows

METRICS = ["rr_retrieval", "rr_pe_cent", "rr_pe", "rr_pe_shuf", "rr_auroc", "rr_pe_meanresp", "rr_auroc_meanresp",
           "rr_pe_z", "rr_pe_meanpred", "rr_pe_combo", "rr_retrieval_canon", "rr_pe_cent_canon", "rr_pcor", "rr_systema_cacc", "rr_systema_cacc_hvg"]
PMX = ["rr_retrieval", "rr_pe_cent", "rr_pe", "rr_auroc", "rr_retrieval_canon", "rr_pe_cent_canon", "rr_pcor", "rr_systema_cacc", "rr_systema_cacc_hvg"]
def summarize(rows):
    s = {}
    for mt in METRICS:
        v = np.array([r.get(mt, np.nan) for r in rows], float); f = np.isfinite(v)
        s[mt] = float(v[f].mean()) if f.any() else None; s["n_" + mt] = int(f.sum())
        s["sem_" + mt] = float(v[f].std(ddof=1) / np.sqrt(f.sum())) if f.sum() > 1 else None
    s["spec_pe_minus_shuf"] = float(np.nanmean([r["rr_pe"] - r["rr_pe_shuf"] for r in rows]))
    return s

def paired(rows_a, rows_b, mt):
    """b vs a on the same KOs; TWO-SIDED paired t / Wilcoxon p-values + bootstrap-over-KO 95% CI of the mean difference
    (p_t1 / p_wilcoxon1 = one-sided b > a, kept for comparison with the submitted version)."""
    A = {r["gene"]: r.get(mt, np.nan) for r in rows_a}
    pr = np.array([(A[r["gene"]], r.get(mt, np.nan)) for r in rows_b if r["gene"] in A], float).reshape(-1, 2)
    pr = pr[np.isfinite(pr).all(1)]; n = len(pr)
    if n < 3 or np.allclose(pr[:, 0], pr[:, 1]): return dict(n=n)
    d = pr[:, 1] - pr[:, 0]
    bs = d[np.random.RandomState(0).randint(0, n, (2000, n))].mean(1)
    return dict(n=n, mean_a=float(pr[:, 0].mean()), mean_b=float(pr[:, 1].mean()), mean_diff=float(d.mean()),
                ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                p_t=float(ttest_rel(pr[:, 1], pr[:, 0]).pvalue),
                p_wilcoxon=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                p_t1=float(ttest_rel(pr[:, 1], pr[:, 0], alternative="greater").pvalue),
                p_wilcoxon1=float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0)

def avg_rows(list_of_rows):
    out = []
    for i, r in enumerate(list_of_rows[0]):
        a = dict(r)
        for k in r:
            if k.startswith("rr_"): a[k] = float(np.nanmean([rr[i][k] for rr in list_of_rows]))
        out.append(a)
    return out

# ------------------------------------------------------------------ fine-tuning
def make_T(c, genes, target="real", perm_seed=0):
    genes = [g for g in genes if len(c["INFO"][g]["de_pos"]) >= 5]
    src = list(genes)
    if target in ("shuffle", "centered_shuffle"):
        src = list(np.array(genes)[np.random.RandomState(1000 + perm_seed).permutation(len(genes))])
    if target in ("mean", "centered", "centered_shuffle"):
        Ob = np.stack([np.where(c["INFO"][h]["keep"], c["INFO"][h]["lfc"], np.nan) for h in genes])
        Rm = dict(o_r=Ob, gi={g: i for i, g in enumerate(genes)}, o_r_sum=np.nansum(Ob, 0), o_r_cnt=np.isfinite(Ob).sum(0))
        meanvec = np.nanmean(Ob, 0)
    K = len(genes); M = np.zeros((K, G), np.float32); RAW = np.zeros((K, G), np.float32); RZ = np.zeros((K, G), np.float32)
    for k, (g, h) in enumerate(zip(genes, src)):
        I, J = c["INFO"][g], c["INFO"][h]
        if target in ("real", "shuffle"): pos = J["de_pos"]; raw = J["lfc"][pos]
        elif target == "mean": pos = I["de_pos"]; raw = meanvec[pos]
        elif target == "centered": pos = I["de_pos"]; raw = (I["lfc"] - loo(Rm, "o_r", g))[pos]
        elif target == "centered_shuffle": pos = J["de_pos"]; raw = (J["lfc"] - loo(Rm, "o_r", h))[pos]
        else: raise ValueError(target)
        M[k, pos] = 1; RAW[k, pos] = raw; RZ[k, pos] = raw / np.clip(STD[pos], STD_FLOOR, None)
    S = c["S"]
    NCE_O = np.stack([(c["INFO"][h]["lfc"] / c["sd_r"])[S] for h in src])
    return dict(genes=genes, gidx=torch.tensor([c["INFO"][g]["gidx"] for g in genes]), M=torch.from_numpy(M),
                RAW=torch.from_numpy(RAW), RZ=torch.from_numpy(RZ), NCE_O=torch.as_tensor(NCE_O, dtype=torch.float32),
                S=torch.as_tensor(S))

def info_nce(P, O, tau):   # == reviewer_controls.info_nce
    P = P - P.mean(0, keepdim=True); O = O - O.mean(0, keepdim=True)
    P = (P - P.mean(1, keepdim=True)); P = P / (P.norm(dim=1, keepdim=True) + 1e-8)
    O = (O - O.mean(1, keepdim=True)); O = O / (O.norm(dim=1, keepdim=True) + 1e-8)
    return torch.nn.functional.cross_entropy((P @ O.T) / tau, torch.arange(len(P)))

OFF = None   # learnable per-cell-line offset added to every prediction during fine-tuning (absorbs the shared response)
def ft_objective(A, T, op, lam=None):
    P = predict_batch(A, T["gidx"], op); M = T["M"]; n = M.sum(1)
    if OFF is not None: P = P + OFF
    pc = (P - (P * M).sum(1, keepdim=True) / n[:, None]) * M
    tc = (T["RAW"] - (T["RAW"] * M).sum(1, keepdim=True) / n[:, None]) * M
    r = (pc * tc).sum(1) / (pc.norm(dim=1) * tc.norm(dim=1) + 1e-8)
    mse = (((P - T["RZ"]) ** 2) * M).sum(1) / n
    l = (MC["w_dir"] * (-r) + MC["w_mag"] * mse).mean()
    if lam is not None and len(T["genes"]) >= 3: l = l + lam * info_nce(P[:, T["S"]], T["NCE_O"], NCE_TAU)
    return l

def rho_of(A):
    A = A.detach()
    if not torch.isfinite(A).all(): return float("inf")
    return float(torch.linalg.eigvals(A).abs().max())

def run_ft(T, op, n_steps, lam=None, val=None, val_lam=None, offset=False):
    """train SCM on T for n_steps; if val given, return per-step validation criterion (index = #updates).
    offset=True: a learnable [G] offset (lr 1e-2, no wd) is added to all predictions in the objective; it is kept in
    the global OFF afterwards and must be cleared (SCM-only evaluation)."""
    global OFF
    model.scm.load_state_dict(SCM0); OFF = None
    groups = [dict(params=list(model.scm.parameters()), lr=MC["lr"], weight_decay=WD)]
    if offset: OFF = torch.zeros(G, requires_grad=True); groups.append(dict(params=[OFF], lr=1e-2, weight_decay=0.0))
    opt = torch.optim.Adam(groups)
    crit = []; nproj = 0
    def vc():
        with torch.no_grad(): return float(ft_objective(get_A(), val, op, val_lam))
    if val is not None: crit.append(vc())
    for step in range(n_steps):
        opt.zero_grad(); A = get_A(); loss = ft_objective(A, T, op, lam)
        if not torch.isfinite(loss): continue
        if MC["w_rho"] > 0: loss = loss + MC["w_rho"] * torch.relu(torch.linalg.svdvals(A)[0] - MC["rho_pen"]) ** 2
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.scm.parameters()), 1.0)
        if OFF is not None: torch.nn.utils.clip_grad_norm_([OFF], 1.0)
        opt.step()
        with torch.no_grad():
            r = rho_of(get_A())
            if r > MC["rho_proj"]: model.scm.weights_rr.mul_(MC["rho_proj"] / r); nproj += 1
        if val is not None: crit.append(vc())
    return crit, nproj

def inner_split(genes):
    g = np.array(sorted(genes)); prm = np.random.RandomState(7).permutation(len(g)); nv = max(1, round(0.3 * len(g)))
    return sorted(g[prm[nv:]].tolist()), sorted(g[prm[:nv]].tolist())

def fit(c_tr, genes, op, target="real", perm_seed=0, lam=None, offset=False):
    """inner split -> #steps -> refit on all genes. Leaves model.scm in the fitted state (offset discarded:
    evaluation is SCM-only)."""
    global OFF
    itr, iva = inner_split(genes)
    Ttr = make_T(c_tr, itr, target, perm_seed); Tva = make_T(c_tr, iva, "real")
    if len(Ttr["genes"]) >= 1 and len(Tva["genes"]) >= 1:
        crit, _ = run_ft(Ttr, op, STEPS, lam, val=Tva, val_lam=lam if lam is not None and len(Tva["genes"]) >= 3 else None, offset=offset)
        best = int(np.nanargmin(crit))
    else: crit, best = [], STEPS
    T = make_T(c_tr, genes, target, perm_seed); _, nproj = run_ft(T, op, best, lam, offset=offset)
    off_norm = float(OFF.detach().norm()) if OFF is not None else None; OFF = None
    with torch.no_grad(): A = get_A(); a = A.numpy(); off = ~np.eye(L, dtype=bool); a0 = (torch.sigmoid(SCM0["logits_rr"]) * SCM0["mask_rr"] * SCM0["weights_rr"]).numpy()
    return dict(n_steps=best, inner_crit_first=crit[0] if crit else None, inner_crit_best=float(np.nanmin(crit)) if crit else None,
                n_train=len(T["genes"]), n_inner_val=len(Tva["genes"]), n_proj=nproj, offset_norm=off_norm, rho=rho_of(A),
                corr_A_AT=float(np.corrcoef(a[off], a.T[off])[0, 1]), rel_change_A=float(np.linalg.norm(a - a0) / np.linalg.norm(a0)))

@torch.no_grad()
def preds_for(c, genes, op):
    A = get_A(); P = predict_batch(A, torch.tensor([c["INFO"][g]["gidx"] for g in genes]), op).numpy().astype(float)
    return {g: P[i] for i, g in enumerate(genes)}

def evaluate(c, train_genes, test_genes, op, preds=None):
    pt = preds or preds_for(c, train_genes + test_genes, op)
    return score(c, test_genes, pt, ref_means(c, train_genes, pt))


# ================================================================== RUN (review2/de_robust)
RES = dict(line=LINE, seed=SEED, ops=OPS, steps=STEPS, de_train=DE_TRAIN, de=dict(p=PTHR, lfc_floor=LFCR, min_cells=30, min_de=3), mc=MC, model=MD)
def save(): json.dump(RES, open(OUT, "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else str(o))
CA = build_ctx(LINE); TRAIN = [g for g in CA["evaluable"] if g in TRS]; TEST = [g for g in CA["evaluable"] if g in TES]
set_S(CA, TRAIN)
ALLG = sorted(set(np.load(f"{D}/{LINE}_de.npz")["genes"].astype(str)) & set(FEATS))
nde = np.array([len(CA["INFO"][g]["de_pos"]) for g in CA["evaluable"]])
RES["data"] = dict(n_evaluable=len(CA["evaluable"]), n_train=len(TRAIN), n_test=len(TEST), de_per_ko_median=float(np.median(nde)), n_S=int(len(CA["S"])), n_pred_genes=len(ALLG))
log("data:", RES["data"]); save()
PRED = {}
GALL = torch.tensor([FI[g] for g in ALLG])
@torch.no_grad()
def preds_all(op):
    P_ = predict_batch(get_A(), GALL, op).numpy().astype(float); return {g: P_[i] for i, g in enumerate(ALLG)}
@torch.no_grad()
def ds_preds_all(Wf):
    P_ = ds_predict(Wf, GALL).cpu().numpy().astype(float); return {g: P_[i] for i, g in enumerate(ALLG)}
def store(name, P_):
    PRED[name] = np.stack([P_[g] for g in ALLG]).astype(np.float32)
def save_preds():
    np.savez(PRED_OUT, genes=np.array(ALLG), **PRED)
KEYM = ["rr_retrieval", "rr_pe_cent", "rr_pe", "rr_pe_shuf", "rr_auroc"]
def fmt(s): return " ".join(f"{m.replace('rr_', '')} {s[m]:+.3f}" for m in KEYM if s.get(m) is not None)
op = "shift"
model.scm.load_state_dict(SCM0)
pobs = preds_all(op); store("cmc_obs", pobs)
rows = evaluate(CA, TRAIN, TEST, op, preds=pobs); RES.setdefault("summ", {})["cmc_obs"] = summarize(rows); log(f"OBS: {fmt(summarize(rows))}")
store("corr", {g: -CORR[FI[g]] for g in ALLG})
if "ft" in PARTS:
    def go(name, **kw):
        info = fit(CA, TRAIN, op, **kw); P_ = preds_all(op); store(name, P_)
        s = summarize(evaluate(CA, TRAIN, TEST, op, preds=P_)); RES["summ"][name] = s; RES.setdefault("info", {})[name] = info
        log(f"FT {name} [{info['n_steps']} steps]: {fmt(s)}"); save(); save_preds()
    go("real")
    grid = {}; itr, iva = inner_split(TRAIN)
    for lam in NCE_GRID:
        fit(CA, itr, op, target="real", lam=lam); pt = preds_for(CA, itr + iva, op)
        sm = summarize(score(CA, iva, pt, ref_means(CA, itr, pt))); grid[str(lam)] = dict(lam=lam, inner_retr=sm["rr_retrieval"], inner_cent=sm["rr_pe_cent"])
    lam1 = max(grid.values(), key=lambda v: v["inner_retr"])["lam"]; RES["infonce_grid"] = grid; log("InfoNCE grid", grid, "->", lam1)
    go("infonce", lam=lam1)
    for k in range(3): go(f"shuffle_{k}", target="shuffle", perm_seed=k)
    for k in range(3): go(f"infonce_shuffle_{k}", target="shuffle", perm_seed=k, lam=lam1)
save_preds()

if "ds" in PARTS:
    DS_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dz = np.load(os.environ["PS_DS_W"], allow_pickle=True); assert list(np.asarray(dz["genes"]).astype(str)) == FEATS
    W0 = torch.tensor(np.asarray(dz["W"], float), dtype=torch.float32, device=DS_DEV); W0.fill_diagonal_(0.0)
    XBAR = torch.as_tensor(Xc.mean(0), dtype=torch.float32, device=DS_DEV); Z0d = Z0.to(DS_DEV); IG = torch.eye(G, device=DS_DEV)
    OFFD = 1.0 - IG
    N_PARAM_CMC = int(sum(p.numel() for p in model.scm.parameters())); N_POS_CMC = int(L * L - L)
    def ds_predict(Wf, gidx):
        Mi = torch.linalg.inv(IG - Wf.T); gidx = gidx.to(DS_DEV); dlt = Z0d[gidx] - XBAR[gidx]
        return (Mi[:, gidx] / Mi[gidx, gidx].unsqueeze(0) * dlt.unsqueeze(0)).T
    def to_dev(T): return {k: (v.to(DS_DEV) if torch.is_tensor(v) else v) for k, v in T.items()}
    def ds_obj(Wf, T, lam=None):
        P = ds_predict(Wf, T["gidx"]); M = T["M"]; n = M.sum(1)
        pc = (P - (P * M).sum(1, keepdim=True) / n[:, None]) * M; tc = (T["RAW"] - (T["RAW"] * M).sum(1, keepdim=True) / n[:, None]) * M
        r = (pc * tc).sum(1) / (pc.norm(dim=1) * tc.norm(dim=1) + 1e-8); mse = (((P - T["RZ"]) ** 2) * M).sum(1) / n
        l = (MC["w_dir"] * (-r) + MC["w_mag"] * mse).mean()
        if lam is not None and len(T["genes"]) >= 3:
            Pq = P[:, T["S"]]; O = T["NCE_O"]
            Pq = Pq - Pq.mean(0, keepdim=True); O = O - O.mean(0, keepdim=True); Pq = Pq - Pq.mean(1, keepdim=True); Pq = Pq / (Pq.norm(dim=1, keepdim=True) + 1e-8)
            O = O - O.mean(1, keepdim=True); O = O / (O.norm(dim=1, keepdim=True) + 1e-8)
            l = l + lam * torch.nn.functional.cross_entropy((Pq @ O.T) / NCE_TAU, torch.arange(len(Pq), device=DS_DEV))
        return l
    _sv = [torch.randn(G, 1, device=DS_DEV), torch.randn(G, 1, device=DS_DEV)]
    def specnorm(Mx):
        with torch.no_grad():
            u, v = _sv
            for _ in range(12): u = Mx @ v; u = u / (u.norm() + 1e-12); v = Mx.t() @ u; v = v / (v.norm() + 1e-12)
            _sv[0], _sv[1] = u, v
        return (_sv[0] * (Mx @ _sv[1])).sum()
    def ds_variant(mode):
        if mode.startswith("top"):
            k = int(mode[3:]); a_ = W0.abs().clone(); a_.fill_diagonal_(-1.0); thr = torch.topk(a_.flatten(), k).values[-1]; MASK = (a_ >= thr).float()
            init = lambda: [torch.zeros_like(W0, requires_grad=True)]
            W_of = lambda P: W0 + MASK * P[0]; shrink = lambda P: P[0].mul_(0.7)
            return dict(init=init, W_of=W_of, shrink=shrink, n_param=k, n_pos=int(MASK.sum()))
        r = int(mode[2:])
        def init():
            g_ = torch.Generator().manual_seed(0)
            return [torch.zeros(G, r, device=DS_DEV, requires_grad=True), (torch.randn(G, r, generator=g_) * 0.01).to(DS_DEV).requires_grad_()]
        W_of = lambda P: W0 + (P[0] @ P[1].T) * OFFD; shrink = lambda P: P[0].mul_(0.7)
        return dict(init=init, W_of=W_of, shrink=shrink, n_param=2 * G * r, n_pos=G * G - G)
    def ds_run(V, T, n_steps, lam=None, val=None):
        P = V["init"](); opt = torch.optim.Adam(P, lr=MC["lr"], weight_decay=WD); crit = []; nshr = 0
        def vc():
            with torch.no_grad(): return float(ds_obj(V["W_of"](P), val, lam if len(val["genes"]) >= 3 else None))
        if val is not None: crit.append(vc())
        for step in range(n_steps):
            opt.zero_grad(); Wf = V["W_of"](P); loss = ds_obj(Wf, T, lam) + MC["w_rho"] * torch.relu(specnorm(Wf) - MC["rho_pen"]) ** 2
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(P, 1.0); opt.step()
            if step % 10 == 9:
                with torch.no_grad():
                    for _ in range(8):
                        if float(torch.linalg.eigvals(V["W_of"](P)).abs().max()) <= MC["rho_proj"]: break
                        V["shrink"](P); nshr += 1
            if val is not None: crit.append(vc())
        return V["W_of"](P).detach(), crit, nshr
    def ds_fit(V, genes, target="real", perm_seed=0, lam=None):
        itr, iva = inner_split(genes)
        Ttr = to_dev(make_T(CA, itr, target, perm_seed)); Tva = to_dev(make_T(CA, iva, "real"))
        _, crit, _ = ds_run(V, Ttr, STEPS, lam, val=Tva); best = int(np.nanargmin(crit))
        Wf, _, nshr = ds_run(V, to_dev(make_T(CA, genes, target, perm_seed)), best, lam)
        return Wf, dict(n_steps=best, n_shrink=nshr, rho=float(torch.linalg.eigvals(Wf).abs().max()))
    @torch.no_grad()
    def ds_preds(Wf, genes):
        P = ds_predict(Wf, torch.tensor([CA["INFO"][g]["gidx"] for g in genes])).cpu().numpy().astype(float); return {g: P[i] for i, g in enumerate(genes)}
    DSR = dict(ds_w=os.environ["PS_DS_W"], device=str(DS_DEV), rho_W0=float(torch.linalg.eigvals(W0).abs().max()),
               cmc_n_param=N_PARAM_CMC, cmc_n_positions=N_POS_CMC)
    CR = {}
    rows0 = evaluate(CA, TRAIN, TEST, None, preds=ds_preds(W0, TRAIN + TEST)); DSR["obs"] = dict(summary=summarize(rows0), per_ko=rows0)
    store("ds_obs", ds_preds_all(W0))
    log(f"DS obs: {fmt(DSR['obs']['summary'])}")
    for mode in os.environ.get("PS_DS_VARIANTS", "top73728,top36672,lr12").split(","):
        V = ds_variant(mode); X = dict(n_param=V["n_param"], n_positions=V["n_pos"]); DROWS = {"obs": rows0}
        def ds_add(name, Wf, info=None):
            rows = evaluate(CA, TRAIN, TEST, None, preds=ds_preds(Wf, TRAIN + TEST)); DROWS[name] = rows
            store(f"ds_{mode}_{name}", ds_preds_all(Wf)); save_preds()
            X[name] = dict(info=info, summary=summarize(rows), per_ko=rows)
            log(f"DS[{mode}] {name}{'' if info is None else ' [%d steps]' % info['n_steps']}: {fmt(X[name]['summary'])}")
            DSR[mode] = X; RES["deepsem"] = DSR; save(); return rows
        lams = {}
        for tgt in ["real", "centered"]:      # InfoNCE weight chosen on the inner split (same criterion as CMC), not fixed
            itr, iva = inner_split(TRAIN); grid = {}
            for lam in NCE_GRID:
                Wf, _ = ds_fit(V, itr, target=tgt, lam=lam); pt = ds_preds(Wf, itr + iva)
                grid[lam] = summarize(score(CA, iva, pt, ref_means(CA, itr, pt)))["rr_retrieval"]
            lams[tgt] = max(grid, key=grid.get); X[f"lam_grid_{tgt}"] = {str(k): v for k, v in grid.items()}
            log(f"DS[{mode}] InfoNCE[{tgt}] inner grid {grid} -> {lams[tgt]}")
        for name, kw in [("real", {}), ("centered", dict(target="centered")), ("infonce", dict(lam=lams["real"])),
                         ("centered_infonce", dict(target="centered", lam=lams["centered"]))]:
            Wf, info = ds_fit(V, TRAIN, **kw); ds_add(name, Wf, info)
        for base, kw in [("shuffle", dict(target="shuffle")), ("centered_shuffle", dict(target="centered_shuffle")),
                         ("infonce_shuffle", dict(target="shuffle", lam=lams["real"])), ("centered_infonce_shuffle", dict(target="centered_shuffle", lam=lams["centered"]))]:
            rr_ = []
            for k in range(2):
                Wf, info = ds_fit(V, TRAIN, perm_seed=k, **kw); rr_.append(ds_add(f"{base}_{k}", Wf, info))
            DROWS[base] = avg_rows(rr_); X[f"{base}_avg"] = dict(summary=summarize(DROWS[base]), per_ko=DROWS[base])
        X["paired"] = {f"{x}_vs_{y}": {m: paired(DROWS[y], DROWS[x], m) for m in PMX} for x, y in
                       [("real", "obs"), ("real", "shuffle"), ("centered", "centered_shuffle"), ("infonce", "infonce_shuffle"),
                        ("centered_infonce", "centered_infonce_shuffle")]}
        X["paired_cmc_minus_deepsem"] = {a: {m: paired(DROWS[a], CR[a], m) for m in PMX} for a in CR if a in DROWS}
        for k, v in {**X["paired"], **{"cmc-ds " + a: b for a, b in X["paired_cmc_minus_deepsem"].items()}}.items():
            log(f"  DS[{mode}] paired {k}: " + " | ".join(f"{m.replace('rr_','')} d={v[m].get('mean_diff', 0):+.3f} p={v[m].get('p_wilcoxon', 1):.2g}" for m in ["rr_retrieval", "rr_pe_cent", "rr_pe_cent_canon"]))
        DSR[mode] = X; RES["deepsem"] = DSR; save()


save_preds(); model.scm.load_state_dict(SCM0); RES["elapsed_s"] = time.time() - T0; save(); log("DONE ->", OUT, PRED_OUT)
