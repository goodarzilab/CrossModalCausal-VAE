"""Fig. 3 zero-shot, information-matched comparators (review M1 / crux C1).

For every evaluable knockout unit (Frangieh/Chen/Arce; RNA and protein read-outs) we score, with the SAME DE labels,
target-gene exclusion and scorer as analysis/common/reviewer_controls.py:
  KO-agnostic rankers (no perturbation data, no knockout identity):
     screen control-cell mean / variance / s.d., training-corpus mean / s.d., decoder-row norm ||D[f,:]||,
     CMC-generic = leave-one-out mean |prediction| of CMC-VAE over the other knockouts (KO-agnostic model ranking)
  KO-specific zero-shot predictors (knockdown sign = negated row):
     CMC-VAE (do-op), training-corpus correlation (as in the paper), Ledoit-Wolf partial correlation (training corpus),
     within cell-type x cohort correlation (training corpus), screen-control-cell correlation (screen's own controls)
Additional analyses: own-vs-other-knockout AUROC (KO specificity of the ranking), expression-stratified AUROC,
cross-validated logistic combinations (does CMC add to an expression/variance/row-norm ranking?).
Usage: python zeroshot_fig3.py results/sweep_sparse/sp_dec_gl1000
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
import os, sys
exec(open(f"{_CMC_ROOT}/analysis/fig3_zero_shot/zeroshot_baselines/_rc_header.py").read())
import anndata as ad
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

BASE = f"{_CMC_RESULTS}/paper/analyses/baselines"
JS = np.load(f"{BASE}/train_stats/joint_stats.npz", allow_pickle=True)
NGt = len(JS["genes"]); assert list(JS["genes"]) == list(RAW["frangieh"][0]["train_gene_names"])
Cj, Pj, Cct = JS["C"], JS["P_lw"], JS["C_ct"]
Wp = model.dec_prot.weight.detach().cpu().numpy(); Wr_np = Wr.cpu().numpy()

# ---------------------------------------------------------------- CMC predictions (observational model, Fig-3 set)
model.scm.load_state_dict(SCM0)
ALL = {t: CTX[t]["evaluable"] for t in TAGS}
PRED = {"CMC-VAE": predict_units(CTX, ALL)}
log("CMC predictions done")

# ---------------------------------------------------------------- screen control cells
def screen_controls(t):
    al = CTX[t]["align"]
    r = ad.read_h5ad(DATASETS[t][0], backed="r"); p = ad.read_h5ad(DATASETS[t][1], backed="r")
    common_obs = sorted(set(r.obs_names) & set(p.obs_names))
    tg = P.parse_sgRNA_to_gene(r.obs.loc[common_obs, "sgRNA"]); ctrl = np.asarray(tg == "control")
    assert ctrl.sum() == int(np.asarray(al["control_mask"]).sum()), (ctrl.sum(), int(np.asarray(al["control_mask"]).sum()))
    names = np.array(common_obs)[ctrl]
    ri = {n: i for i, n in enumerate(r.obs_names)}; pi = {n: i for i, n in enumerate(p.obs_names)}
    ir = np.sort([ri[n] for n in names]); ip = np.array([pi[n] for n in np.array(r.obs_names)[ir]])
    Xr = r.X[ir]; Xr = Xr.toarray() if hasattr(Xr, "toarray") else np.asarray(Xr); Xr = Xr[:, al["rna_eval_perturb_indices"]].astype(np.float64)
    order = np.argsort(ip); Xp = p.X[np.sort(ip)]; Xp = Xp.toarray() if hasattr(Xp, "toarray") else np.asarray(Xp)
    Xp = np.empty_like(Xp) if False else Xp
    Xp_full = np.zeros_like(Xp); Xp_full[order] = Xp            # restore row order to match Xr
    pe_idx = al["prot_eval_perturb_indices"]
    Xp = Xp_full[:, pe_idx].astype(np.float64) if len(pe_idx) else np.zeros((len(ir), 0))
    def zc(M):
        M = M - M.mean(0); s = M.std(0); s[s < 1e-9] = np.inf; return M / s
    Z = np.concatenate([zc(Xr), zc(Xp)], 1); Cs = np.nan_to_num(Z.T @ Z / len(Z))
    return dict(n=len(ir), mean_r=Xr.mean(0), var_r=Xr.var(0), mean_p=Xp.mean(0), var_p=Xp.var(0), C=Cs, nr=Xr.shape[1])
SCR = {t: screen_controls(t) for t in TAGS}
log("screen controls:", {t: SCR[t]["n"] for t in TAGS})

# ---------------------------------------------------------------- KO-specific zero-shot predictors (knockdown sign)
for name in ["corr", "partial_corr", "corr_within_ct", "screen_ctrl_corr"]: PRED[name] = {}
chk = []
for t in TAGS:
    c = CTX[t]; reti, preti = c["reti"], c["preti"]; S = SCR[t]; g2i = c["g2i"]
    for g in c["evaluable"]:
        gi = c["INFO"][g]["gidx"]; gc_ = g2i[g]
        PRED["corr"][(t, g)] = (-Cj[gi, reti].astype(float), -Cj[gi, NGt + preti].astype(float))
        PRED["partial_corr"][(t, g)] = (-Pj[gi, reti].astype(float), -Pj[gi, NGt + preti].astype(float))
        PRED["corr_within_ct"][(t, g)] = (-Cct[gi, reti].astype(float), -Cct[gi, NGt + preti].astype(float))
        PRED["screen_ctrl_corr"][(t, g)] = (-S["C"][gc_, :S["nr"]], -S["C"][gc_, S["nr"]:])
        if c["align"].get("corr_train") is not None:
            chk.append(float(np.max(np.abs(np.asarray(c["align"]["corr_train"][gc_], float) - Cj[gi, reti]))))
log("max |corr_train(paper) - joint corr| =", max(chk) if chk else None)

# ---------------------------------------------------------------- KO-agnostic rankers
def agnostic(t):
    c = CTX[t]; al = c["align"]; reti, preti = c["reti"], c["preti"]; S = SCR[t]
    tr_m = np.asarray(al["train_rna_mean"], float)[reti]; tr_s = np.asarray(al["train_rna_std"], float)[reti]
    tp_m = np.asarray(al["train_prot_mean"], float)[preti] if len(preti) else np.zeros(0)
    tp_s = np.asarray(al["train_prot_std"], float)[preti] if len(preti) else np.zeros(0)
    return {"ctrl_mean": (S["mean_r"], S["mean_p"]), "ctrl_var": (S["var_r"], S["var_p"]),
            "ctrl_cv2": (S["var_r"] / np.maximum(S["mean_r"], 1e-3) ** 2, S["var_p"] / np.maximum(np.abs(S["mean_p"]), 1e-3) ** 2),
            "train_mean": (tr_m, tp_m), "train_sd": (tr_s, tp_s),
            "decoder_row_norm": (np.linalg.norm(Wr_np[reti], axis=1), np.linalg.norm(Wp[preti], axis=1) if len(preti) else np.zeros(0))}
AG = {t: agnostic(t) for t in TAGS}
AG_NAMES = list(AG["frangieh"].keys()) + ["cmc_generic"]
SPEC_NAMES = ["CMC-VAE", "corr", "partial_corr", "corr_within_ct", "screen_ctrl_corr"]

def loo_abs(name, t, g, mod):
    genes = [h for h in CTX[t]["evaluable"] if h != g]
    return np.mean([np.abs(PRED[name][(t, h)][mod]) for h in genes], 0)

# ---------------------------------------------------------------- per-KO AUROCs for every ranker
rows = []
for t in TAGS:
    c = CTX[t]
    for g in c["evaluable"]:
        I = c["INFO"][g]; k = I["keep"]; r = dict(cohort=t, gene=g, n_de_rna=int((I["sig"] & k).sum()))
        for nm, (vr, vp) in AG[t].items():
            r[f"rr_auroc__{nm}"] = au(I["sig"][k], np.asarray(vr)[k])
            if I["has_p"]: r[f"rp_auroc__{nm}"] = au(I["sigp"], np.asarray(vp))
        r["rr_auroc__cmc_generic"] = au(I["sig"][k], loo_abs("CMC-VAE", t, g, 0)[k])
        if I["has_p"]: r["rp_auroc__cmc_generic"] = au(I["sigp"], loo_abs("CMC-VAE", t, g, 1))
        for nm in SPEC_NAMES:
            pr, pp = PRED[nm][(t, g)]
            r[f"rr_auroc__{nm}"] = au(I["sig"][k], np.abs(pr[k]))
            if I["has_p"]: r[f"rp_auroc__{nm}"] = au(I["sigp"], np.abs(pp))
        rows.append(r)
log("per-KO AUROCs done")

# ---------------------------------------------------------------- shared-response metrics (retrieval, centred pearDE) for KO-specific predictors
SR = {}
for nm in SPEC_NAMES:
    rs = score(CTX, ALL, PRED[nm], ref_means(CTX, ALL, PRED[nm]))
    SR[nm] = {(r["cohort"], r["gene"]): r for r in rs}
for r in rows:
    for nm in SPEC_NAMES:
        s = SR[nm][(r["cohort"], r["gene"])]
        for m in ["rr_retrieval", "rp_retrieval", "rr_pe", "rp_pe", "rr_pe_cent", "rp_pe_cent", "rr_auroc", "rp_auroc"]:
            r[f"{m}__{nm}" if not m.endswith("auroc") else f"{m}_chk__{nm}"] = s.get(m, np.nan)
log("shared-response metrics done")

# ---------------------------------------------------------------- own-vs-other KO AUROC (does the ranking carry KO-specific info?)
def own_other(nm, t, mod):
    c = CTX[t]; genes = [g for g in c["evaluable"] if (mod == 0 or c["INFO"][g]["has_p"])]
    if len(genes) < 3: return {}
    Pm = np.stack([np.abs(PRED[nm][(t, g)][mod]) for g in genes])
    out = {}
    for i, g in enumerate(genes):
        I = c["INFO"][g]
        lab, msk = (I["sig"], I["keep"]) if mod == 0 else (I["sigp"], np.ones(len(I["sigp"]), bool))
        y = lab[msk].astype(int)
        if y.sum() < 3 or y.sum() >= len(y): continue
        Rk = rankdata(Pm[:, msk], axis=1); npos = y.sum(); nneg = len(y) - npos
        A = (Rk[:, y == 1].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)
        out[g] = (float(A[i]), float(np.mean(np.delete(A, i))))
    return out
OO = {}
for nm in SPEC_NAMES:
    for t in TAGS:
        for mod, blk in [(0, "rr"), (1, "rp")]:
            for g, (a_own, a_oth) in own_other(nm, t, mod).items():
                OO[(nm, t, g, blk)] = (a_own, a_oth)
for r in rows:
    for nm in SPEC_NAMES:
        for blk in ["rr", "rp"]:
            v = OO.get((nm, r["cohort"], r["gene"], blk))
            if v: r[f"{blk}_auroc_own__{nm}"], r[f"{blk}_auroc_other__{nm}"] = v
log("own-vs-other done")

# ---------------------------------------------------------------- expression-stratified AUROC (within control-mean quintiles)
def strat_auroc(lab, sc_, strat, nb=5):
    q = np.quantile(strat, np.linspace(0, 1, nb + 1)); b = np.clip(np.searchsorted(q, strat, side="right") - 1, 0, nb - 1)
    vals, w = [], []
    for k_ in range(nb):
        m = b == k_; y = lab[m]
        if y.sum() >= 2 and (len(y) - y.sum()) >= 2:
            vals.append(roc_auc_score(y, sc_[m])); w.append(y.sum())
    return float(np.average(vals, weights=w)) if vals else np.nan
for r in rows:
    t, g = r["cohort"], r["gene"]; c = CTX[t]; I = c["INFO"][g]; k = I["keep"]; sm = AG[t]["ctrl_mean"][0][k]
    for nm in SPEC_NAMES + ["ctrl_var", "decoder_row_norm", "cmc_generic", "train_sd"]:
        if nm in SPEC_NAMES: s_ = np.abs(PRED[nm][(t, g)][0])[k]
        elif nm == "cmc_generic": s_ = loo_abs("CMC-VAE", t, g, 0)[k]
        else: s_ = np.asarray(AG[t][nm][0])[k]
        r[f"rr_auroc_strat__{nm}"] = strat_auroc(I["sig"][k].astype(int), s_, sm)
log("stratified done")

# ---------------------------------------------------------------- cross-validated logistic combinations (5-fold grouped by KO, within cohort, RNA)
def pct(v): return (rankdata(v) - 0.5) / len(v)
COMBOS = {"expr": ["ctrl_mean", "ctrl_var", "train_sd"], "expr+rownorm": ["ctrl_mean", "ctrl_var", "train_sd", "decoder_row_norm"]}
for extra in ["CMC-VAE", "cmc_generic", "CMC-VAE_shuffled", "corr", "partial_corr", "screen_ctrl_corr"]:
    COMBOS[f"expr+rownorm+{extra}"] = COMBOS["expr+rownorm"] + [extra]
COMBOS["expr+rownorm+corr+partial_corr+screen_ctrl_corr"] = COMBOS["expr+rownorm"] + ["corr", "partial_corr", "screen_ctrl_corr"]
COMBOS["expr+rownorm+corr+partial_corr+screen_ctrl_corr+CMC-VAE"] = COMBOS["expr+rownorm+corr+partial_corr+screen_ctrl_corr"] + ["CMC-VAE"]
LOGIT = {}
for t in TAGS:
    c = CTX[t]; genes = c["evaluable"]; rs = np.random.RandomState(0)
    der = rs.permutation(len(genes))
    while np.any(der == np.arange(len(genes))): der = rs.permutation(len(genes))
    feats = {}
    for i, g in enumerate(genes):
        k = c["INFO"][g]["keep"]; F = {}
        for nm in ["ctrl_mean", "ctrl_var", "train_sd", "decoder_row_norm"]: F[nm] = pct(np.asarray(AG[t][nm][0])[k])
        for nm in ["CMC-VAE", "corr", "partial_corr", "screen_ctrl_corr"]: F[nm] = pct(np.abs(PRED[nm][(t, g)][0])[k])
        F["cmc_generic"] = pct(loo_abs("CMC-VAE", t, g, 0)[k])
        F["CMC-VAE_shuffled"] = pct(np.abs(PRED["CMC-VAE"][(t, genes[der[i]])][0])[k])   # another KO's prediction
        feats[g] = (F, c["INFO"][g]["sig"][k].astype(int))
    folds = np.arange(len(genes)) % 5; np.random.RandomState(1).shuffle(folds)
    for cname, cols in COMBOS.items():
        for f in range(5):
            tr = [g for g, ff in zip(genes, folds) if ff != f]; te = [g for g, ff in zip(genes, folds) if ff == f]
            Xtr = np.concatenate([np.stack([feats[g][0][cc] for cc in cols], 1) for g in tr]); ytr = np.concatenate([feats[g][1] for g in tr])
            lr_m = LogisticRegression(C=1.0, max_iter=500).fit(Xtr, ytr)
            for g in te:
                Xte = np.stack([feats[g][0][cc] for cc in cols], 1)
                LOGIT[(t, g, cname)] = au(feats[g][1], lr_m.decision_function(Xte))
for r in rows:
    for cname in COMBOS: r[f"rr_auroc_logit__{cname}"] = LOGIT.get((r["cohort"], r["gene"], cname), np.nan)
log("logistic combos done")

import pandas as pd
df = pd.DataFrame(rows); df.insert(0, "model_run", os.path.basename(RD))
df.to_csv(OUT.replace(".json", "_per_ko.csv"), index=False)
json.dump(dict(run=os.path.basename(RD), n_units={t: len(ALL[t]) for t in TAGS}, n_ctrl={t: SCR[t]["n"] for t in TAGS},
               corr_check_maxabs=max(chk) if chk else None, lw_delta=float(JS["lw_delta"]),
               means=df.drop(columns=["model_run", "gene"]).groupby("cohort").mean(numeric_only=True).to_dict()),
          open(OUT, "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else str(o))
log("DONE ->", OUT)
