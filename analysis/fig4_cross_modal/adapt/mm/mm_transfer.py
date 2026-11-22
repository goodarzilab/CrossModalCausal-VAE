# [adapt] COPY of code/xmodal/mm/mm_transfer.py (= code/multimodal/mm_transfer.py). Scoring, folds, arms, FT protocol unchanged.
# Additions (information-matched CONTROL-CELL baselines, same 90% control cells the adapted models saw):
#   map 'ctrl'   : ridge RNA->protein fitted on the screen's control cells (RidgeCV per protein), in training-z units
#   feat 'ctrlcx': gene features = correlation row of g across the screen's control cells (ridge on RNA/protein labels)
#   sets ridgeR_ctrlcx(+ctrl), ridgeR_cx+ctrl, ftR+ctrl, obs+ctrl, measRNA+ctrl, zs_ctrlcorr(+ctrl)
#   part 'ridgelc': protein-label ridge learning curve (n KOs with protein labels) on cx / ctrlcx features.
"""Experiment 1 (cross-modal transfer) + few-shot protein learning curves.

5-fold knockout-gene cross-validation (a gene is held out in every screen at once). In each fold the latent graph
(model.scm) is fine-tuned on the training knockouts with one of several supervision regimes and the held-out knockouts
are predicted with a signed soft operator (clamp_zero | cutclamp_zero). Out-of-fold predictions for every evaluable KO
are pooled and scored with the op_controls scorer (pearDE / centred / AUROC / retrieval, RNA and protein).

Arms (fine-tuning protocol = Perturb-seq 'project' protocol: inner 70/30 split of training genes selects the number of
steps on the SUPERVISED modality's validation loss, then refit on all training genes):
  obs        no fine-tuning
  ftR        RNA loss only (w_rp = 0)                     <- key: no protein label anywhere
  ftR_shuf   same, KO labels permuted within screen
  ftP        protein loss only (w_rr = 0)                 <- upper reference (protein) / reverse direction (RNA)
  ftP_shuf   same, permuted
  ftRP       paper loss (RNA + 0.5 protein)
Label-free protein baselines (protein = map(RNA response)):  maps = training-corpus RNA-protein correlation ('corr'),
corpus multivariate ridge ('cridge'), totalVI denoised correlation ('tv'), cognate transcript ('cog'); applied to
(a) the ftR-predicted RNA response, (b) a ridge-on-RNA-labels predicted RNA response, (c) the MEASURED RNA response.
Protein-label references: ridge on gene features (decoder loadings 'fp', corpus co-expression row 'cx') trained on the
same knockouts that carry the protein loss in ftP.
Few-shot: protein labels on n KOs per screen (n grid), arms ftP(n), ftRP(n) (RNA labels on all training KOs), ridge(n).

env: MM_RD MM_OUT MM_OP MM_PARTS=main,fewshot MM_NFOLD=5 MM_STEPS=300
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_core import *          # noqa
from mm_core import _cog_map, _project

OPN = os.environ.get("MM_OP", "clamp_zero")
NF = int(os.environ.get("MM_NFOLD", "5"))
MPARTS = os.environ.get("MM_PARTS", "main,ridgelc").split(",")
MSTEPS = int(os.environ.get("MM_STEPS", "300"))
if QUICK: MSTEPS = 20
FS_N = [int(x) for x in os.environ.get("MM_FS_N", "3,6,12,25,50").split(",")]
FS_DRAWS = int(os.environ.get("MM_FS_DRAWS", "2"))
ALLU = {t: list(CTX[t]["evaluable"]) for t in TAGS}
set_S(CTX, ALLU)            # evaluation feature sets defined from all evaluable KOs (labels define S only)

MC_R = dict(MC, w_rr=1.0, w_rp=0.0)
MC_P = dict(MC, w_rr=0.0, w_rp=1.0)
MC_RP = dict(MC, w_rr=1.0, w_rp=0.5)

# ------------------------------------------------------------------ folds over genes
genes_all = sorted(set().union(*[set(v) for v in ALLU.values()]))
perm = np.random.RandomState(0).permutation(len(genes_all))
FOLD = {genes_all[i]: k % NF for k, i in enumerate(perm)}
def fold_units(k):
    tr = {t: [g for g in ALLU[t] if FOLD[g] != k] for t in TAGS}; te = {t: [g for g in ALLU[t] if FOLD[g] == k] for t in TAGS}
    trg = sorted(set().union(*[set(v) for v in tr.values()]))
    rng = np.random.RandomState(7 + k); pr = rng.permutation(len(trg)); iv = set(np.array(trg)[pr[:round(0.3 * len(trg))]].tolist())
    itr = {t: [g for g in tr[t] if g not in iv] for t in TAGS}; iva = {t: [g for g in tr[t] if g in iv] for t in TAGS}
    return tr, te, itr, iva

# ------------------------------------------------------------------ fine-tuning with optional protein-label mask
def build_targets_mm(ctx_d, units, target, perm_seed, p_keep=None):
    TG = build_targets(ctx_d, units, target, perm_seed)
    if p_keep is not None:
        for (t, g), e in TG["TGT"].items():
            if g not in p_keep.get(t, set()): e["has_p"] = False
    return TG

def ft_project(ctx_d, units, mc, target, perm_seed, n_steps, val_units=None, p_keep=None, p_keep_val=None):
    model.scm.load_state_dict(SCM0)
    opt = torch.optim.Adam([p for p in model.scm.parameters() if p.requires_grad], lr=mc["lr"], weight_decay=WD)
    TG = build_targets_mm(ctx_d, units, target, perm_seed, p_keep)
    TV = build_targets_mm(ctx_d, val_units, "real", 0, p_keep_val) if val_units is not None else None
    crit = []
    if TV is not None:
        with torch.no_grad(): crit.append(objective(ctx_d, TV, OPN, mc))
    for step in range(n_steps):
        opt.zero_grad(); objective(ctx_d, TG, OPN, mc, backward=True)
        if mc["w_rho"] > 0:
            rl = mc["w_rho"] * torch.relu(spectral_norm(assemble()) - mc["rho_pen"]) ** 2; rl.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.scm.parameters() if p.requires_grad], 1.0); opt.step(); _project()
        if TV is not None:
            with torch.no_grad(): crit.append(objective(ctx_d, TV, OPN, mc))
    return crit

def fit_arm(tr, itr, iva, mc, target="real", perm_seed=0, p_keep=None):
    """inner split selects #steps on the supervised validation loss; refit on all training KOs."""
    pk_tr = None if p_keep is None else {t: set(g for g in p_keep.get(t, ()) if g in set(itr[t])) for t in TAGS}
    pk_va = None if p_keep is None else {t: set(g for g in p_keep.get(t, ()) if g in set(iva[t])) for t in TAGS}
    crit = ft_project(CTX, itr, mc, target, perm_seed, MSTEPS, val_units=iva, p_keep=pk_tr, p_keep_val=pk_va)
    crit = np.array(crit, float)
    best = int(np.nanargmin(crit)) if np.isfinite(crit).any() and np.ptp(crit[np.isfinite(crit)]) > 0 else 0
    if best > 0: ft_project(CTX, tr, mc, target, perm_seed, best, p_keep=p_keep)
    else: model.scm.load_state_dict(SCM0)
    return dict(steps=best, crit0=float(crit[0]), critbest=float(np.nanmin(crit)))

def predict(units):
    pr, _ = predict_units_op(CTX, units, OPN); return pr

# ------------------------------------------------------------------ label-free maps RNA(z) -> protein(z), per screen
JS = np.load(f"{_CMC_RESULTS}/paper/analyses/baselines/train_stats/joint_stats.npz", allow_pickle=True)
CJ = JS["C"].astype(np.float64); NGV = len(JS["genes"])
TV = np.load(f"{REPO}/results/baselines/totalvi_raw.npz", allow_pickle=True)
GTV = np.asarray(TV["G_rp_tv"], float)
MAPS = {}
for t in TAGS:
    c = CTX[t]; reti, preti = c["reti"], c["preti"]
    if not len(preti): continue
    Crp = CJ[np.ix_(reti, NGV + preti)]; Crr = CJ[np.ix_(reti, reti)]
    lam = 1.0
    B_ridge = np.linalg.solve(Crr + lam * len(reti) / 100.0 * np.eye(len(reti)), Crp)   # corpus multivariate ridge (std units)
    pn = [c["align"]["train_prot_names"][i] for i in preti]; com = list(c["common"])
    Bcog = np.zeros((len(reti), len(preti)))
    for j, p_ in enumerate(pn):
        g = _cog_map.get(p_)
        if g in com: Bcog[com.index(g), j] = 1.0
    MAPS[t] = dict(corr=Crp / len(reti), cridge=B_ridge, tv=GTV[np.ix_(reti, preti)] / len(reti), cog=Bcog,
                   has_cog=Bcog.sum(0) > 0)
    log(f"maps {t}: {len(preti)} proteins, {int(MAPS[t]['has_cog'].sum())} with measured cognate transcript")

def rna_z_measured(t, g):
    I = CTX[t]["INFO"][g]; v = np.nan_to_num(I["lfc"] / CTX[t]["sd_r"]); return v

def map_preds(rna_of, mapname, rna_part_of=None):
    """protein prediction = rna_z @ MAP for every KO of every screen; rna_of(t,g)->z RNA vector over reti."""
    out = {}
    for t in TAGS:
        for g in ALLU[t]:
            r = rna_of(t, g); p = r @ MAPS[t][mapname] if t in MAPS else np.zeros(0)
            out[(t, g)] = (rna_part_of(t, g) if rna_part_of else r, p)
    return out

# ------------------------------------------------------------------ ridge baselines on gene features
FEATS = {"fp": Wr.cpu().numpy().astype(float), "cx": CJ[:NGV, :]}
# ---- [adapt] control-cell information (seed/model independent; cached): 90% split identical to code/adapt/adapt_lib.split_idx
ADOUT = f"{_CMC_SCRATCH}/nov_adapt/diag"
def _ctrl_info(t):
    f = f"{ADOUT}/ctrlinfo_{t}.npz"
    if os.path.exists(f): z = np.load(f); return z["B"], z["C"]
    from sklearn.linear_model import RidgeCV
    pb = pickle.load(open(f"{_CMC_RESULTS}/paper/analyses/stats/de/pb_{t}.pkl", "rb"))
    Xc = pb["ct"]["cells_r"].astype(np.float64); Yc = pb["ct"]["cells_p"].astype(np.float64)
    prm = np.random.RandomState(12345).permutation(len(Xc)); tr = np.sort(prm[int(round(0.1 * len(Xc))):])
    Xc, Yc = Xc[tr], Yc[tr]
    B = RidgeCV(alphas=np.logspace(0, 8, 17), alpha_per_target=True).fit(Xc - Xc.mean(0), Yc - Yc.mean(0)).coef_.T if Yc.shape[1] else np.zeros((Xc.shape[1], 0))
    sd = Xc.std(0); Z = (Xc - Xc.mean(0)) / np.where(sd > 0, sd, 1); C = (Z.T @ Z) / len(Z)
    np.savez(f, B=B, C=C); return B, C
CTRLI = {t: _ctrl_info(t) for t in TAGS}
for t in TAGS:
    c = CTX[t]
    F_ = np.zeros((NGV, len(c["reti"]))); F_[c["reti"]] = CTRLI[t][1]; FEATS.setdefault("ctrlcx_by_tag", {})[t] = F_
    if t in MAPS:   # raw log-RNA -> raw CLR protein coefficients, converted to training-z in and out
        MAPS[t]["ctrl"] = CTRLI[t][0] * c["sd_r"][:, None] / c["sd_p"][None, :]
def ridge_model(Xf, Y, alpha):
    mu_x = Xf.mean(0); mu_y = Y.mean(0); Xc_ = Xf - mu_x
    if Xc_.shape[1] <= Xc_.shape[0]: B = np.linalg.solve(Xc_.T @ Xc_ + alpha * np.eye(Xc_.shape[1]), Xc_.T @ (Y - mu_y))
    else: B = Xc_.T @ np.linalg.solve(Xc_ @ Xc_.T + alpha * np.eye(len(Xc_)), Y - mu_y)
    return lambda Xq: (Xq - mu_x) @ B + mu_y
RIDGE_GRID = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
def _targets(t, genes, mod):
    c = CTX[t]
    if mod == "r": return np.stack([np.nan_to_num(c["INFO"][g]["lfc"] / c["sd_r"]) for g in genes])
    return np.stack([np.nan_to_num(c["INFO"][g]["lfcp"] / c["sd_p"]) for g in genes])
def ridge_fit_predict(t, tr_genes, pred_genes, mod, feat):
    """alpha by K-fold (K=min(5,n)) CV MSE on the training KOs; returns {g: pred}."""
    F = FEATS["ctrlcx_by_tag"][t] if feat == "ctrlcx" else FEATS[feat]
    X = lambda gs: np.stack([F[CTX[t]["INFO"][g]["gidx"]] for g in gs]); tr_genes = list(tr_genes)
    Y = _targets(t, tr_genes, mod); n = len(tr_genes)
    if n == 0: return {}
    if n < 3: alpha = 100.0
    else:
        K = min(5, n); fid = np.arange(n) % K; np.random.RandomState(0).shuffle(fid); err = {}
        for a in RIDGE_GRID:
            e = 0.0
            for k in range(K):
                trm = fid != k
                if trm.sum() < 1: continue
                f = ridge_model(X([g for g, m in zip(tr_genes, trm) if m]), Y[trm], a); e += float(((f(X([g for g, m in zip(tr_genes, ~trm) if m])) - Y[~trm]) ** 2).sum())
            err[a] = e
        alpha = min(err, key=err.get)
    f = ridge_model(X(tr_genes), Y, alpha); P_ = f(X(pred_genes))
    return {g: P_[i] for i, g in enumerate(pred_genes)}

# ------------------------------------------------------------------ scoring helpers
def score_pooled(preds, units=None):
    units = units or {t: [g for g in ALLU[t] if (t, g) in preds] for t in TAGS}
    return score(CTX, units, preds, ref_means(CTX, units, preds))
def slim(rows):
    K = ["cohort", "gene", "n_de_rna", "n_de_prot", "rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rr_pe_meanresp", "rr_auroc_meanresp",
         "rp_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval", "rp_pe_meanresp", "rp_auroc_meanresp", "rp_pe_nocog"]
    return [{k: r.get(k) for k in K if k in r} for r in rows]
SUMK = ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval", "rp_pe_meanresp", "rp_auroc_meanresp", "rr_pe_meanresp"]
def brief2(rows):
    s = summarize(rows); return {sc: {k: round(s[sc][k], 4) for k in SUMK if s[sc].get(k) == s[sc].get(k)} for sc in ["overall"] + TAGS}

RES = dict(run=os.path.basename(RD), de_floor=[LFCR, LFCP])
def save():
    json.dump(RES, open(OUT, "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else (o.item() if hasattr(o, "item") else str(o)))
RES.update(op=OPN, nfold=NF, msteps=MSTEPS, fold_of_gene=FOLD, n_units={t: len(ALLU[t]) for t in TAGS},
           n_units_prot={t: int(sum(CTX[t]["INFO"][g]["has_p"] for g in ALLU[t])) for t in TAGS})

# ======================================================================================== main
if "main" in MPARTS:
    ARMS = [("ftR", MC_R, "real"), ("ftR_shuf", MC_R, "shuffle"), ("ftP", MC_P, "real"), ("ftP_shuf", MC_P, "shuffle"), ("ftRP", MC_RP, "real")]
    OOF = {a: {} for a in ["obs"] + [a for a, _, _ in ARMS]}
    RIDGE = {k: {} for k in ["ridgeR_cx", "ridgeR_fp", "ridgeP_cx", "ridgeP_fp", "ridgeR_ctrlcx", "ridgeP_ctrlcx"]}
    INFO_FT = {}
    model.scm.load_state_dict(SCM0); P0 = predict(ALLU)
    for (t, g), v in P0.items(): OOF["obs"][(t, g)] = v
    for k in range(NF):
        tr, te, itr, iva = fold_units(k)
        for arm, mc, tgt in ARMS:
            info = fit_arm(tr, itr, iva, mc, target=tgt); INFO_FT[f"{arm}|{k}"] = info
            pr = predict(te)
            for key, v in pr.items(): OOF[arm][key] = v
            log(f"fold {k} {arm}: {info}")
        model.scm.load_state_dict(SCM0)
        # ridge baselines (per screen)
        for t in TAGS:
            if not te[t]: continue
            lab_p = [g for g in tr[t] if CTX[t]["INFO"][g]["has_p_loss"] and CTX[t]["INFO"][g]["obs_r_raw"].numel() >= 5]
            lab_r = [g for g in tr[t] if CTX[t]["INFO"][g]["obs_r_raw"].numel() >= 5]
            for feat in ["cx", "fp", "ctrlcx"]:
                pr_r = ridge_fit_predict(t, lab_r, te[t], "r", feat)
                pr_p = ridge_fit_predict(t, lab_p, te[t], "p", feat) if len(CTX[t]["preti"]) and lab_p else {g: np.zeros(len(CTX[t]["preti"])) for g in te[t]}
                for g in te[t]:
                    RIDGE[f"ridgeR_{feat}"][(t, g)] = pr_r[g]; RIDGE[f"ridgeP_{feat}"][(t, g)] = pr_p[g]
        RES["ft_info"] = INFO_FT; save()
    # ---- assemble prediction sets
    PS = {a: OOF[a] for a in OOF}
    ftR_r = lambda t, g: OOF["ftR"][(t, g)][0]
    ridgeR_r = lambda t, g: RIDGE["ridgeR_cx"][(t, g)]
    for mp in ["corr", "cridge", "tv", "cog"]:
        PS[f"ftR+{mp}"] = map_preds(ftR_r, mp)
        PS[f"ridgeR+{mp}"] = map_preds(ridgeR_r, mp)
        PS[f"measRNA+{mp}"] = map_preds(rna_z_measured, mp, rna_part_of=ftR_r)
    for feat in ["cx", "fp", "ctrlcx"]:
        PS[f"ridgeP_{feat}"] = {(t, g): (RIDGE[f"ridgeR_{feat}"][(t, g)], RIDGE[f"ridgeP_{feat}"][(t, g)]) for t in TAGS for g in ALLU[t]}
    # [adapt] control-cell (information-matched) sets
    obs_r = lambda t, g: OOF["obs"][(t, g)][0]
    ridgeC_r = lambda t, g: RIDGE["ridgeR_ctrlcx"][(t, g)]
    zs_r = lambda t, g: -FEATS["ctrlcx_by_tag"][t][CTX[t]["INFO"][g]["gidx"]]
    if all(t in MAPS for t in TAGS):
        PS["ftR+ctrl"] = map_preds(ftR_r, "ctrl"); PS["obs+ctrl"] = map_preds(obs_r, "ctrl")
        PS["ridgeR_cx+ctrl"] = map_preds(ridgeR_r, "ctrl"); PS["ridgeR_ctrlcx+ctrl"] = map_preds(ridgeC_r, "ctrl")
        PS["measRNA+ctrl"] = map_preds(rna_z_measured, "ctrl", rna_part_of=ftR_r); PS["zs_ctrlcorr+ctrl"] = map_preds(zs_r, "ctrl")
        PS["ridgeR_ctrlcx+corr"] = map_preds(ridgeC_r, "corr")
    ROWS = {}
    for name, pr in PS.items():
        rows = score_pooled(pr); ROWS[name] = rows
        log(f"[{name}] {brief2(rows)['overall']}")
    RES["main"] = {name: dict(summary=summarize(rows), per_ko=slim(rows)) for name, rows in ROWS.items()}
    # protein subset with a measured cognate transcript / without (per-KO pearDE restricted to those proteins)
    SUB = {}
    for name in ["obs", "ftR", "ftR_shuf", "ftP", "ftRP", "measRNA+cog", "measRNA+corr", "ftR+cog", "ftR+corr", "ridgeP_cx"]:
        pr = PS[name]; vals = {"cog": [], "nocog": []}
        for t in TAGS:
            if t not in MAPS: continue
            hc = MAPS[t]["has_cog"]
            for g in ALLU[t]:
                I = CTX[t]["INFO"][g]
                if not I["has_p"]: continue
                for lab, m in [("cog", hc), ("nocog", ~hc)]:
                    vals[lab].append(dict(cohort=t, gene=g, pe=pe(pr[(t, g)][1], I["lfcp"], I["sigp"] & m), n=int((I["sigp"] & m).sum())))
        SUB[name] = vals
    RES["cog_subsets"] = SUB
    save()

# ======================================================================================== [adapt] protein-label ridge learning curve
if "ridgelc" in MPARTS:
    LC = {}
    for k in range(NF):
        tr, te, itr, iva = fold_units(k)
        labp = {t: [g for g in tr[t] if CTX[t]["INFO"][g]["has_p_loss"] and CTX[t]["INFO"][g]["obs_r_raw"].numel() >= 5] for t in TAGS}
        for n in [3, 6, 12, 25, 50, 10 ** 6]:
            for d in range(3 if n < 10 ** 6 else 1):
                for t in TAGS:
                    if not te[t] or not len(CTX[t]["preti"]): continue
                    L = labp[t]; rng = np.random.RandomState(1000 * k + 10 * d + n % 997)
                    keep = sorted(np.array(L)[rng.permutation(len(L))[:min(n, len(L))]].tolist()) if L else []
                    for feat in ["cx", "ctrlcx"]:
                        pp = ridge_fit_predict(t, keep, te[t], "p", feat) if keep else {g: np.zeros(len(CTX[t]["preti"])) for g in te[t]}
                        for g in te[t]: LC.setdefault(f"ridgeP_{feat}|{n}|{d}", {})[(t, g)] = (np.zeros(len(CTX[t]["reti"])), pp[g])
    OUTLC = {}
    for name, pr in LC.items():
        rows = score_pooled(pr); OUTLC[name] = dict(summary=summarize(rows), per_ko=slim(rows))
    RES["ridgelc"] = OUTLC; save(); log("ridgelc done")

# ======================================================================================== few-shot protein labels
if "fewshot" in MPARTS:
    FS = {}
    for k in range(NF):
        tr, te, itr, iva = fold_units(k)
        labp = {t: [g for g in tr[t] if CTX[t]["INFO"][g]["has_p_loss"] and CTX[t]["INFO"][g]["obs_r_raw"].numel() >= 5] for t in TAGS}
        for n in FS_N + [10 ** 6]:
            for d in range(FS_DRAWS if n < 10 ** 6 else 1):
                keep = {}
                for t in TAGS:
                    L = labp[t]; rng = np.random.RandomState(1000 * k + 10 * d + n % 997)
                    keep[t] = set(np.array(L)[rng.permutation(len(L))[:min(n, len(L))]].tolist()) if L else set()
                nlab = {t: len(keep[t]) for t in TAGS}
                for arm, mc in [("ftP", MC_P), ("ftRP", MC_RP)]:
                    info = fit_arm(tr, itr, iva, mc, p_keep=keep)
                    pr = predict(te)
                    for key, v in pr.items(): FS.setdefault(f"{arm}|{n}|{d}", {})[key] = v
                    log(f"fewshot fold {k} n={n} d={d} {arm}: nlab={nlab} {info}")
                model.scm.load_state_dict(SCM0)
                for t in TAGS:
                    if not te[t] or not len(CTX[t]["preti"]): continue
                    for feat in ["cx", "fp"]:
                        pp = ridge_fit_predict(t, sorted(keep[t]), te[t], "p", feat) if keep[t] else {g: np.zeros(len(CTX[t]["preti"])) for g in te[t]}
                        for g in te[t]: FS.setdefault(f"ridgeP_{feat}|{n}|{d}", {})[(t, g)] = (np.zeros(len(CTX[t]["reti"])), pp[g])
                FS.setdefault("_nlab", {})[f"{k}|{n}|{d}"] = nlab
        log(f"fewshot fold {k} done")
    OUTFS = {"nlab": FS.pop("_nlab")}
    for name, pr in FS.items():
        rows = score_pooled(pr); OUTFS[name] = dict(summary=summarize(rows), per_ko=slim(rows))
        log(f"[FS {name}] {brief2(rows)['overall']}")
    RES["fewshot"] = OUTFS; save()

model.scm.load_state_dict(SCM0)
RES["elapsed_s"] = time.time() - T0; save(); log("DONE ->", OUT)
