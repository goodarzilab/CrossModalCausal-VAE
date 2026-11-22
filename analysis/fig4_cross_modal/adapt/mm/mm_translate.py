# [adapt] COPY of code/xmodal/mm/mm_translate.py (= code/multimodal/mm_translate.py). Scoring unchanged. Changes: XM_RUNS may map
# seed -> {screen: model_dir} (each screen encoded/decoded by its own control-cell-adapted model); added 'ctrlridge90' = the
# control-cell ridge fitted on the same 90% of control cells the adapted models were trained on (information-matched).
"""Experiment 2 (+3): predicting the measured PROTEIN response of a knockout from its MEASURED RNA response.

For every pseudobulk-DE unit (data_prep/pseudobulk_de/pb_de.py; condition-matched guide x stratum pseudobulks, limma-trend) with >=3
DE proteins, the protein LFC is predicted from RNA only, by
  CMC-enc   : encode each KO / control pseudobulk mean RNA profile (protein input at the training mean = absent),
              solve the latent SCM, decode protein; then the SAME limma design as the ground truth on the imputed
              protein pseudobulks  (the model's full cross-modal map, RNA -> eps -> SCM -> protein decoder)
  CMC-Wrp   : linear map through the learned cross-modal block only:  dz_r = ridge-pinv(D_r) dx_r (observed genes),
              dz_p = dz_r W_rp (I - A_pp)^-1,  dp = dz_p D_p^T   (+ controls: W_rp shuffled, W_rp <- W_pr^T)
  corr      : training-corpus RNA-protein Pearson map applied to dx (z)
  cridge    : training-corpus multivariate ridge RNA -> protein (from the joint correlation matrix)
  totalVI   : totalVI denoised RNA-protein correlation map
  cognate   : protein change = change of its own transcript (proteins without a measured cognate get 0)
  ctrlridge : ridge RNA -> protein fitted on the screen's own CONTROL cells only (alpha by 5-fold CV)
  koridge   : (upper reference, uses protein labels) ridge dx -> dp fitted on the OTHER knockouts of the screen
              (leave-one-KO-out), i.e. a perturbation-trained translation map.
Scale: everything in training-z units (LFC / training SD). Metrics per KO: pearDE over DE proteins, AUROC(|pred|, DE),
centred pearDE (leave-one-KO-out mean removed from prediction and observation), retrieval (centred, among KOs of the
screen); per protein: Pearson across KOs (all units). Exp 3: stratification by cognate availability and by
transcript-protein discordance.

usage: python -I mm_translate.py OUT_JSON      (all three seeds inside)
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
import os, sys, json, pickle, time
import numpy as np, torch
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeCV
REPO = f"{_CMC_ROOT}"
sys.path.insert(0, os.path.join(_CMC_ROOT, "analysis", "common")); sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"))
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE
from pblimma import limma_ko
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
OUT = sys.argv[1]
DEVICE = P.DEVICE
TAGS = ["frangieh", "iccite", "arce"]
SEEDS = json.loads(os.environ["XM_RUNS"])
HASH = {"s42": "3a5d257c9323", "s1": "7ff5ef2f5880", "s2": "69407bfd5193"}
LFCP = 0.05; STD_FLOOR = 0.05
torch.manual_seed(0)

COG = json.load(open(f"{_CMC_RESULTS}/baselines/_cmcvae_cognate.json"))
cog_map = {p: g for p, g in zip(COG["protein_names"], COG["cognate_gene"]) if g}
JS = np.load(f"{_CMC_RESULTS}/paper/analyses/baselines/train_stats/joint_stats.npz", allow_pickle=True)
CJ = JS["C"].astype(np.float64); NGV = len(JS["genes"])
GTV = np.asarray(np.load(f"{_CMC_RESULTS}/baselines/totalvi_raw.npz", allow_pickle=True)["G_rp_tv"], float)

def pe(a, b, m=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if m is not None: a, b = a[m], b[m]
    f = np.isfinite(a) & np.isfinite(b); a, b = a[f], b[f]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])
def au(lab, sc):
    lab = np.asarray(lab).astype(int); sc = np.asarray(sc, float)
    if lab.sum() < 1 or lab.sum() >= len(lab) or not np.isfinite(sc).all(): return np.nan
    return float(roc_auc_score(lab, sc))

# ------------------------------------------------------------------ data
D = {}
for t in TAGS:
    pb = pickle.load(open(f"{_CMC_RESULTS}/paper/analyses/stats/de/pb_{t}.pkl", "rb"))
    al = pickle.load(open(f"{_CMC_RESULTS}/_ftcache/rc_{t}_sp_dec_gl1000_{HASH['s42']}_0.1_0.05.pkl", "rb"))["align"]
    assert list(pb["common_genes"]) == list(al["common_genes"])
    reti = np.asarray(al["rna_eval_train_indices"]); preti = np.asarray(al["prot_eval_train_indices"])
    sd_r = np.clip(np.asarray(al["train_rna_std"], float)[reti], STD_FLOOR, None); sd_p = np.clip(np.asarray(al["train_prot_std"], float)[preti], STD_FLOOR, None)
    units = {}
    for g, r in pb["de"].items():
        lp = np.nan_to_num(r["lfc_prot"]); sig = (r["q_prot"] < 0.05) & (np.abs(lp) >= LFCP)
        lr = np.nan_to_num(r["lfc_rna"]); sigr = (r["q_rna"] < 0.05) & (np.abs(lr) >= 0.10)
        units[g] = dict(lfc_r=lr, q_r=r["q_rna"], sig_r=sigr, lfc_p=lp, sig_p=sig, q_p=r["q_prot"], has_p=bool(sig.sum() >= 3))
    pn = [al["train_prot_names"][i] for i in preti]; com = list(pb["common_genes"])
    cog_idx = np.array([com.index(cog_map[p]) if cog_map.get(p) in com else -1 for p in pn])
    D[t] = dict(pb=pb, al=al, reti=reti, preti=preti, sd_r=sd_r, sd_p=sd_p, units=units, pn=pn, cog_idx=cog_idx,
                tr_mean_r=np.asarray(al["train_rna_mean"], float), tr_std_r=np.clip(np.asarray(al["train_rna_std"], float), 1e-8, None),
                tr_mean_p=np.asarray(al["train_prot_mean"], float), tr_std_p=np.clip(np.asarray(al["train_prot_std"], float), 1e-8, None))
    log(t, "units", len(units), "with >=3 DE proteins", sum(u["has_p"] for u in units.values()), "proteins", len(pn), "with cognate", int((cog_idx >= 0).sum()))

def dx_z(t, g): return D[t]["units"][g]["lfc_r"] / D[t]["sd_r"]
def obs_pz(t, g): return D[t]["units"][g]["lfc_p"] / D[t]["sd_p"]

# ------------------------------------------------------------------ predictions
PRED = {}          # PRED[method][t][g] = predicted protein LFC (z) over preti
def add(name, t, g, v): PRED.setdefault(name, {}).setdefault(t, {})[g] = np.asarray(v, float)

# label-free corpus maps and cognate
for t in TAGS:
    d = D[t]; reti, preti = d["reti"], d["preti"]
    Crp = CJ[np.ix_(reti, NGV + preti)]; Crr = CJ[np.ix_(reti, reti)]
    Bridge = np.linalg.solve(Crr + len(reti) / 100.0 * np.eye(len(reti)), Crp)
    Gtv = GTV[np.ix_(reti, preti)]
    for g in d["units"]:
        x = dx_z(t, g)
        add("corr", t, g, x @ Crp / len(reti)); add("cridge", t, g, x @ Bridge); add("totalVI", t, g, x @ Gtv / len(reti))
        cg = np.array([x[i] if i >= 0 else 0.0 for i in d["cog_idx"]]); add("cognate", t, g, cg)

# control-cell ridge (screen's own control cells; raw log-norm RNA -> CLR protein) and KO-trained ridge (LOKO)
CTRL_FIT = {}
for t in TAGS:
    d = D[t]; Xc = d["pb"]["ct"]["cells_r"].astype(np.float64); Yc = d["pb"]["ct"]["cells_p"].astype(np.float64)
    rng = np.random.RandomState(0); idx = rng.permutation(len(Xc)); ntr = int(0.8 * len(idx))
    mx, my = Xc[idx[:ntr]].mean(0), Yc[idx[:ntr]].mean(0)
    rc = RidgeCV(alphas=np.logspace(0, 8, 17), alpha_per_target=True).fit(Xc[idx[:ntr]] - mx, Yc[idx[:ntr]] - my)
    ho = rc.predict(Xc[idx[ntr:]] - mx) + my
    CTRL_FIT[t] = dict(alpha=np.atleast_1d(rc.alpha_).tolist(), heldout_cell_perprot_r=[pe(ho[:, j], Yc[idx[ntr:], j]) for j in range(Yc.shape[1])])
    rc = RidgeCV(alphas=np.logspace(0, 8, 17), alpha_per_target=True).fit(Xc - Xc.mean(0), Yc - Yc.mean(0)); B = rc.coef_.T      # [genes, prots] raw units
    for g, u in d["units"].items():
        add("ctrlridge", t, g, (u["lfc_r"] @ B) / d["sd_p"])
    log(f"ctrlridge {t}: alpha median {np.median(CTRL_FIT[t]['alpha']):.3g} held-out control-cell per-protein r median {np.nanmedian(CTRL_FIT[t]['heldout_cell_perprot_r']):.3f}")
    # perturbation-trained translation (uses protein labels of the other KOs): ridge on dx (z) -> dp (z), leave-one-KO-out
    G = list(d["units"]); X = np.stack([dx_z(t, g) for g in G]); Y = np.stack([obs_pz(t, g) for g in G])
    for i, g in enumerate(G):
        m = np.arange(len(G)) != i
        r_ = RidgeCV(alphas=np.logspace(-1, 7, 17)).fit(X[m], Y[m]); add("koridge", t, g, r_.predict(X[i:i + 1])[0])
    log(f"koridge {t} done")

# CMC model maps (per seed)
def load_model(run):
    rd = run; cfg = json.load(open(f"{rd}/config.json"))
    sd0 = torch.load(P.resolve_ckpt(rd), map_location="cpu"); ng, np_ = sd0["dec_rna.weight"].shape[0], sd0["dec_prot.weight"].shape[0]; del sd0
    m = P.load_ctx_model(CrossModalCausalLDVAE, rd, ng, np_, cfg); m.eval(); return m

# [adapt] information-matched control-cell ridge (same 90% split as the adapted models; cached by analysis/fig4_cross_modal/adapt/mm/mm_transfer.py)
for t in TAGS:
    zc = np.load(f"{_CMC_SCRATCH}/nov_adapt/diag/ctrlinfo_{t}.npz"); B90 = zc["B"]
    for g, u in D[t]["units"].items(): add("ctrlridge90", t, g, (u["lfc_r"] @ B90) / D[t]["sd_p"])
CHECK = {}
def _runs_of(run): return run if isinstance(run, dict) else {t: run for t in TAGS}
for sk, run in SEEDS.items():
  for t_model, rd_model in _runs_of(run).items():
    model = load_model(rd_model); lr_, lp_ = model.latent_r, model.latent_p; DL = lr_ + lp_
    TAGS_M = [t_model]
    with torch.no_grad():
        A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
        A = torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0).to(DEVICE)
        Wr = model._dec_r_w().detach().to(DEVICE); Wp = model.dec_prot.weight.detach().to(DEVICE)
        ngv, npv = Wr.shape[0], Wp.shape[0]
    def impute(Xr_log, t):
        """pseudobulk / cell log-norm RNA over the screen's eval genes -> imputed protein (training-z) over preti."""
        d = D[t]; Xf = np.zeros((len(Xr_log), ngv), np.float32)
        Xf[:, d["reti"]] = (Xr_log - d["tr_mean_r"][d["reti"]]) / d["tr_std_r"][d["reti"]]
        out = []
        with torch.no_grad():
            for s in range(0, len(Xf), 1024):
                xr = torch.as_tensor(Xf[s:s + 1024], device=DEVICE); xp = torch.zeros(len(xr), npv, device=DEVICE)
                mu_r, _ = model.enc_rna(xr); mu_p, _ = model.enc_prot(xp)
                z = model.scm._scm_solve(A, torch.cat([mu_r, mu_p], 1)); zp = z[:, lr_:]
                out.append((model.dec_prot(zp) + model.bias_prot).cpu().numpy()[:, d["preti"]])
        return np.concatenate(out, 0)
    for t in TAGS_M:
        d = D[t]; pb = d["pb"]; ct = pb["ct"]
        Ict = impute(ct["Y_r"], t)
        # sanity: imputing the pseudobulk mean vs mean of per-cell imputations (control pseudobulks)
        Icell = impute(ct["cells_r"], t)
        Icm = np.stack([Icell[ct["cell_pb"] == k].mean(0) for k in range(len(ct["Y_r"]))])
        CHECK[f"{sk}:{t}"] = dict(pb_vs_cellmean_r=pe(Ict.ravel(), Icm.ravel()),
                                  ctrlcell_perprot_r=[pe(Icell[:, j], (ct["cells_p"][:, j] - d["tr_mean_p"][d["preti"][j]]) / d["tr_std_p"][d["preti"][j]]) for j in range(len(d["preti"]))])
        for g in d["units"]:
            kp = pb["ko_pb"][g]; Ik = impute(kp["Y_r"], t)
            b, _, _ = limma_ko(Ik, kp["strata"], Ict, ct["strata"])
            add(f"CMC-enc:{sk}", t, g, b)
        log(f"{sk} {t}: CMC-enc done; pb-vs-cellmean r={CHECK[f'{sk}:{t}']['pb_vs_cellmean_r']:.3f} ctrl-cell per-protein imputation r median {np.nanmedian(CHECK[f'{sk}:{t}']['ctrlcell_perprot_r']):.3f}")
    # linear W_rp route
    with torch.no_grad():
        g0 = torch.Generator().manual_seed(0)
        Wrp_sh = W_rp.flatten()[torch.randperm(W_rp.numel(), generator=g0)].reshape(W_rp.shape).to(DEVICE)
        variants = {"CMC-Wrp": W_rp.to(DEVICE), "CMC-Wrp-shuffled": Wrp_sh, "CMC-Wrp<-WprT": W_pr.T.contiguous().to(DEVICE)}
        Npp = torch.linalg.inv(torch.eye(lp_, device=DEVICE) - A_pp.to(DEVICE))
    for t in TAGS_M:
        d = D[t]; Wo = Wr[torch.as_tensor(d["reti"], device=DEVICE)]                 # observed-gene decoder rows
        lam = 0.01 * float(torch.trace(Wo.T @ Wo)) / lr_
        Pinv = torch.linalg.solve(Wo.T @ Wo + lam * torch.eye(lr_, device=DEVICE), Wo.T)   # [l_r, genes]
        for g in d["units"]:
            x = torch.as_tensor(dx_z(t, g), dtype=torch.float32, device=DEVICE); dzr = Pinv @ x
            for vn, M in variants.items():
                dzp = (dzr @ M) @ Npp; dp = (dzp @ Wp.T).cpu().numpy()[d["preti"]]; add(f"{vn}:{sk}", t, g, dp)
    log(f"{sk}: Wrp maps done")
    del model; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

# seed-mean CMC predictions
for base in ["CMC-enc", "CMC-Wrp", "CMC-Wrp-shuffled", "CMC-Wrp<-WprT"]:
    for t in TAGS:
        for g in D[t]["units"]:
            add(base, t, g, np.mean([PRED[f"{base}:{s}"][t][g] for s in SEEDS], 0))

# ------------------------------------------------------------------ scoring
def score_method(name):
    rows = []
    for t in TAGS:
        d = D[t]; G = [g for g in d["units"] if d["units"][g]["has_p"]]
        if len(d["preti"]) < 3 or len(G) < 3: continue
        Pm = np.stack([PRED[name][t][g] for g in G]); Om = np.stack([obs_pz(t, g) for g in G])
        sP, sO = Pm.sum(0), Om.sum(0); n = len(G)
        Pc = Pm - (sP - Pm) / (n - 1); Oc = Om - (sO - Om) / (n - 1)
        def zr(M):
            M = M - M.mean(1, keepdims=True); nr = np.linalg.norm(M, axis=1, keepdims=True); return M / np.where(nr > 0, nr, 1)
        C = zr(Pc) @ zr(Oc).T
        for i, g in enumerate(G):
            u = d["units"][g]; m = u["sig_p"]; oth = np.arange(n) != i
            rows.append(dict(cohort=t, gene=g, n_de=int(m.sum()),
                             pe=pe(Pm[i], Om[i], m), pe_all=pe(Pm[i], Om[i]), auroc=au(m, np.abs(Pm[i])),
                             pe_cent=pe(Pc[i], Oc[i], m), retrieval=float(np.mean(C[i, i] > C[i, oth]) + 0.5 * np.mean(C[i, i] == C[i, oth])),
                             sign_acc=float(np.mean(np.sign(Pm[i][m]) == np.sign(Om[i][m])))))
    return rows
def perprot(name):
    out = []
    for t in TAGS:
        d = D[t]; G = list(d["units"])
        if len(d["preti"]) < 1: continue
        Pm = np.stack([PRED[name][t][g] for g in G]); Om = np.stack([obs_pz(t, g) for g in G])
        Sig = np.stack([d["units"][g]["sig_p"] for g in G])
        for j, p_ in enumerate(d["pn"]):
            out.append(dict(cohort=t, protein=p_, has_cog=bool(d["cog_idx"][j] >= 0), n_de_ko=int(Sig[:, j].sum()),
                            r=pe(Pm[:, j], Om[:, j]), r_deKO=pe(Pm[:, j], Om[:, j], Sig[:, j]) if Sig[:, j].sum() >= 5 else np.nan))
    return out
def summ(rows, keys):
    o = {}
    for sc in ["overall"] + TAGS:
        rs = rows if sc == "overall" else [r for r in rows if r["cohort"] == sc]
        o[sc] = {}
        for k in keys:
            v = np.array([r[k] for r in rs], float); v = v[np.isfinite(v)]
            o[sc][k] = float(v.mean()) if len(v) else None; o[sc]["n_" + k] = int(len(v))
    return o

METHODS = list(PRED)
RES = dict(check=CHECK, ctrl_fit=CTRL_FIT, methods=METHODS, n_units={t: len(D[t]["units"]) for t in TAGS},
           n_units_p={t: int(sum(u["has_p"] for u in D[t]["units"].values())) for t in TAGS}, proteins={t: D[t]["pn"] for t in TAGS},
           has_cog={t: (D[t]["cog_idx"] >= 0).tolist() for t in TAGS})
RES["per_ko"] = {m: score_method(m) for m in METHODS}
RES["per_protein"] = {m: perprot(m) for m in METHODS}
RES["summary"] = {m: summ(RES["per_ko"][m], ["pe", "pe_all", "auroc", "pe_cent", "retrieval", "sign_acc"]) for m in METHODS}
RES["summary_perprot"] = {m: summ(RES["per_protein"][m], ["r", "r_deKO"]) for m in METHODS}
for m in METHODS:
    s = RES["summary"][m]["overall"]; sp_ = RES["summary_perprot"][m]["overall"]
    log(f"{m:24s} pe {s['pe']:+.3f} cent {s['pe_cent']:+.3f} auroc {s['auroc']:.3f} retr {s['retrieval']:.3f} sign {s['sign_acc']:.3f} | per-protein r {sp_['r']:+.3f} (n={sp_['n_r']})")

# ------------------------------------------------------------------ exp 3: discordance between protein and its transcript
DISC = {}
for m in METHODS:
    recs = []
    for t in TAGS:
        d = D[t]
        for g, u in d["units"].items():
            for j, ci in enumerate(d["cog_idx"]):
                if ci < 0 or not u["sig_p"][j] or d["pb"]["common_genes"][ci] == g: continue   # skip the KO target's own protein
                rz = u["lfc_r"][ci] / d["sd_r"][ci]; pz = u["lfc_p"][j] / d["sd_p"][j]
                cat = "concordant" if (u["sig_r"][ci] and np.sign(rz) == np.sign(pz)) else ("opposite" if (u["sig_r"][ci] and np.sign(rz) != np.sign(pz)) else "transcript_unchanged")
                recs.append(dict(cohort=t, gene=g, protein=d["pn"][j], cat=cat, obs=float(pz), pred=float(PRED[m][t][g][j])))
    out = {}
    for cat in ["concordant", "opposite", "transcript_unchanged", "all"]:
        rs = [r for r in recs if cat == "all" or r["cat"] == cat]
        if not rs: continue
        o_ = np.array([r["obs"] for r in rs]); p_ = np.array([r["pred"] for r in rs])
        out[cat] = dict(n=len(rs), sign_acc=float(np.mean(np.sign(p_) == np.sign(o_))), r=pe(p_, o_),
                        n_ko=len(set((r["cohort"], r["gene"]) for r in rs)))
    DISC[m] = dict(summary=out, records=recs if m in ("CMC-enc", "cognate", "ctrlridge", "corr", "koridge", "CMC-Wrp") else None)
RES["discordance"] = {m: DISC[m] for m in METHODS}
for m in METHODS:
    log("disc", m, {k: (v["n"], round(v["sign_acc"], 3)) for k, v in DISC[m]["summary"].items()})
# store raw predictions of the main methods for figures
RES["pred"] = {m: {t: {g: v.tolist() for g, v in PRED[m][t].items()} for t in TAGS} for m in ["CMC-enc", "CMC-Wrp", "cognate", "ctrlridge", "corr", "totalVI", "koridge", "cridge"]}
RES["obs"] = {t: {g: obs_pz(t, g).tolist() for g in D[t]["units"]} for t in TAGS}
RES["sig"] = {t: {g: D[t]["units"][g]["sig_p"].tolist() for g in D[t]["units"]} for t in TAGS}
json.dump(RES, open(OUT, "w"), default=lambda o: None if (isinstance(o, float) and o != o) else (o.item() if hasattr(o, "item") else str(o)))
log("DONE", OUT)
