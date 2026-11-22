#!/usr/bin/env python
"""Reviewer M10, Papalexi 2021 ECCITE-seq (untouched screen): rescore the frozen zero-shot predictions (saved in
analyses/untouched/rc_sp_dec_gl1000{,_s1,_s2}_preds.npz; no model is re-run) under alternative DE ground truths.

DE definitions (RNA; 2,031 model-vocabulary evaluation genes; KO vs non-targeting cells; LFC floor 0.10):
  orig   : published = cell-level two-sided Mann-Whitney raw P<0.05 & |LFC|>=0.10 (P, LFC read from the run cache)
  bh     : same P, BH across the 2,031 genes within each KO, q<0.05 & |LFC|>=0.10
  pb     : pseudobulk = guide x replicate (rep1-3) mean log-normalised expression (>=10 cells, as data_prep/pseudobulk_de/pb_de.py),
           controls = non-targeting guide x replicate; pblimma.limma_ko (KO + replicate design, binned limma-trend
           empirical Bayes, two-sided), BH within KO, q<0.05 & |LFC_pb|>=0.10; observed response = LFC_pb
  pbmask : sensitivity: pb significance mask, original cell-level LFC as the observed response
  orig_pbunits : sensitivity: original DE but only the units evaluable under pb
Evaluable unit: >=3 DE genes excluding the knocked-out gene, among the 13 units that have saved predictions.
Metrics re-implemented exactly as rc_untouched.score (validated against the published numbers in the 'orig' block):
  rr_auroc   = AUROC(DE label, |pred|) over non-own genes (>=3 positives)
  rr_pe_cent = Pearson(pred*sd - LOO mean pred*sd, LFC - LOO mean LFC) over DE genes (>=5)
  rr_retrieval = rank of own response among evaluable units, centred profiles on S (genes DE in >=1 unit, minus targets)
usage: python -I papalexi_de_rescore.py
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
import os, sys, json, pickle
import numpy as np, pandas as pd, anndata as ad
from scipy.stats import ttest_rel, wilcoxon, ttest_1samp
from sklearn.metrics import roc_auc_score
sys.path.insert(0, f"{_CMC_ROOT}/analysis/fig3_zero_shot/stats")
from pblimma import limma_ko, bh

UNT = f"{_CMC_RESULTS}/paper/analyses/untouched"
DER = f"{_CMC_RESULTS}/paper/analyses/review2/de_robust"
RUNS = ["sp_dec_gl1000", "sp_dec_gl1000_s1", "sp_dec_gl1000_s2"]
LFCR, MINC = 0.10, 10
rng = np.random.default_rng(0)

C = pickle.load(open(f"{UNT}/cache/rc_papalexi_sp_dec_gl1000_3a5d257c9323_0.1_0.05.pkl", "rb"))
al, de0 = C["align"], C["de"]
common = np.array(al["common_genes"]); sd = np.clip(np.asarray(al["train_rna_std"], float)[np.asarray(al["rna_eval_train_indices"])], 1e-8, None)
J0 = json.load(open(f"{UNT}/rc_sp_dec_gl1000.json")); UNITS13 = J0["papalexi_units"]["evaluable"]
ALLU = sorted(de0)                                         # 18 in-vocabulary targets with >=30 cells

# ---------------------------------------------------------------- DE variants
DE = {v: {} for v in ["orig", "bh", "pb", "pbmask"]}
for g in ALLU:
    l, p = np.asarray(de0[g]["lfc_rna"], float), np.asarray(de0[g]["pvals_rna"], float)
    DE["orig"][g] = (l, (p < 0.05) & (np.abs(l) >= LFCR))
    assert (DE["orig"][g][1] == np.asarray(de0[g]["sig_rna"], bool)).all()
    DE["bh"][g] = (l, (bh(p) < 0.05) & (np.abs(l) >= LFCR))
a = ad.read_h5ad(f"{UNT}/data/papalexi_proc/papalexi_rna.h5ad")
X = a[:, list(common)].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X); X = X.astype(np.float64)
o = a.obs; tg = o["gene_target"].astype(str).values; gid = o["guide_ID"].astype(str).values; rep = o["replicate"].astype(str).values
def pbs(mask):
    Y, S, N = [], [], []
    for (gd, r), idx in pd.Series(np.where(mask)[0]).groupby([gid[mask], rep[mask]]):
        if len(idx) >= MINC: Y.append(X[idx.values].mean(0)); S.append(r); N.append(len(idx))
    return (np.stack(Y) if Y else np.zeros((0, X.shape[1]))), np.array(S), N
Yc, Sc, Nc = pbs(tg == "NT")
PBINFO = dict(n_control_pb=len(Yc), control_pb_cells_median=float(np.median(Nc)), units={})
for g in ALLU:
    Yk, Sk, Nk = pbs(tg == g)
    PBINFO["units"][g] = dict(n_pb=len(Yk), n_rep=len(set(Sk.tolist())), cells_in_pb=int(sum(Nk)))
    if len(Yk) < 2:
        DE["pb"][g] = (np.full(len(common), np.nan), np.zeros(len(common), bool)); DE["pbmask"][g] = (DE["orig"][g][0], np.zeros(len(common), bool)); continue
    b, p, info = limma_ko(Yk, Sk, Yc, Sc); q = bh(p); PBINFO["units"][g].update(df=info["df"], d0=info.get("d0_median"))
    s_ = (q < 0.05) & (np.abs(np.nan_to_num(b)) >= LFCR)
    DE["pb"][g] = (np.nan_to_num(b), s_); DE["pbmask"][g] = (DE["orig"][g][0], s_)
    PBINFO["units"][g]["r_lfc_cell_pb"] = float(np.corrcoef(b, DE["orig"][g][0])[0, 1])

def n_de(v, g): return int((DE[v][g][1] & (common != g)).sum())
# unit-matched sensitivity: original DE, restricted to the units evaluable under pb (separates unit drop-out from DE choice)
DE["orig_pbunits"] = {g: (DE["orig"][g][0], DE["orig"][g][1] if n_de("pb", g) >= 3 else np.zeros(len(common), bool)) for g in ALLU}
DECOUNT = {v: {g: n_de(v, g) for g in ALLU} for v in DE}

# ---------------------------------------------------------------- predictions
P = {}
for r in RUNS:
    z = np.load(f"{UNT}/rc_{r}_preds.npz")
    P[f"CMC-VAE {r}"] = {g: z[f"cmc_obs|papalexi|{g}|r"] for g in UNITS13}
    if r == RUNS[0]:
        for b_ in ["corr", "totalvi", "deepsem", "zs_ctrl_expr", "zs_ctrl_var"]:
            P[b_] = {g: z[f"{b_}|papalexi|{g}|r"] for g in UNITS13 if f"{b_}|papalexi|{g}|r" in z}

def au(lab, sc):
    lab = np.asarray(lab).astype(int); sc = np.asarray(sc, float)
    if lab.sum() < 3 or lab.sum() >= len(lab) or not np.isfinite(sc).all(): return np.nan
    return float(roc_auc_score(lab, sc))

def score(v, preds):
    units = [g for g in UNITS13 if n_de(v, g) >= 3 and g in preds]
    K = len(units); keep = np.stack([common != g for g in units])
    lfc = np.stack([DE[v][g][0] for g in units]); sig = np.stack([DE[v][g][1] for g in units])
    Pr = np.stack([preds[g] for g in units]).astype(float)
    O = np.where(keep, lfc, np.nan); Pn = np.where(keep, Pr, np.nan)
    loo = lambda M, i: np.nanmean(np.delete(M, i, 0), 0)
    S = sig.any(0) & ~np.isin(common, units)
    out = {}; Pm, Om = [], []
    zz = lambda x: (x - x.mean()) / (np.linalg.norm(x - x.mean()) + 1e-300)
    for i, g in enumerate(units):
        mo, mp = loo(O, i), loo(Pn, i); m = sig[i] & keep[i]; pc = np.nan
        if m.sum() >= 5:
            x, y = ((Pr[i] - mp) * sd)[m], (lfc[i] - mo)[m]
            if np.std(x) > 0 and np.std(y) > 0: pc = float(np.corrcoef(x, y)[0, 1])
        out[g] = dict(rr_pe_cent=pc, rr_auroc=au(sig[i][keep[i]], np.abs(Pr[i][keep[i]])),
                      rr_auroc_meanresp=au(sig[i][keep[i]], np.abs(np.nan_to_num(mo[keep[i]]))), n_de=int(m.sum()))
        Pm.append(zz((Pr[i] - mp)[S])); Om.append(zz(((lfc[i] - mo) / sd)[S]))
    if K >= 3:
        Cm = np.nan_to_num(np.stack(Pm) @ np.stack(Om).T)
        for i, g in enumerate(units):
            oth = np.delete(Cm[i], i); out[g]["rr_retrieval"] = float(np.mean(Cm[i, i] > oth) + 0.5 * np.mean(Cm[i, i] == oth))
    return out, dict(units=units, n_S=int(S.sum()))

MET = ["rr_auroc", "rr_pe_cent", "rr_retrieval"]
def seed_mean(lst):
    return {g: {m: float(np.nanmean([d[g][m] for d in lst])) if np.isfinite([d[g][m] for d in lst]).any() else np.nan for m in MET} for g in lst[0]}
def summ(pu, m):
    v = np.array([pu[g][m] for g in pu], float); v = v[np.isfinite(v)]
    if not len(v): return dict(n=0)
    bs = rng.choice(v, (10000, len(v))).mean(1)
    return dict(n=int(len(v)), mean=float(v.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
def one(pu, m, mu0):
    v = np.array([pu[g][m] for g in pu], float); v = v[np.isfinite(v)]
    if len(v) < 3: return dict(n=int(len(v)))
    return dict(n=int(len(v)), p_t=float(ttest_1samp(v, mu0).pvalue), p_wilcoxon=float(wilcoxon(v - mu0).pvalue) if np.any(v != mu0) else 1.0)
def paired(pa, pb, m):
    g = [x for x in pa if x in pb and np.isfinite(pa[x][m]) and np.isfinite(pb[x][m])]
    if len(g) < 3: return dict(n=len(g))
    d = np.array([pb[x][m] - pa[x][m] for x in g])
    if np.allclose(d, 0): return dict(n=len(g), mean_diff=0.0, p_t=1.0)
    return dict(n=len(g), mean_diff=float(d.mean()), p_t=float(ttest_rel([pb[x][m] for x in g], [pa[x][m] for x in g]).pvalue),
                p_wilcoxon=float(wilcoxon(d).pvalue))

RES = dict(de_counts=DECOUNT, pb_info=PBINFO, variants={})
for v in DE:
    sc_ = {k: score(v, pr) for k, pr in P.items()}
    pu = {k: s[0] for k, s in sc_.items()}
    pu["CMC-VAE (seed mean)"] = seed_mean([pu[f"CMC-VAE {r}"] for r in RUNS])
    mr = {g: dict(rr_auroc=pu[f"CMC-VAE {RUNS[0]}"][g]["rr_auroc_meanresp"], rr_pe_cent=np.nan, rr_retrieval=np.nan) for g in pu[f"CMC-VAE {RUNS[0]}"]}
    pu["mean_response_ref"] = mr
    tab = {k: {m: summ(x, m) for m in MET} for k, x in pu.items()}
    cmc = pu["CMC-VAE (seed mean)"]
    comp = {f"CMC-VAE minus {k}": {m: paired(x, cmc, m) for m in MET} for k, x in pu.items() if not k.startswith("CMC-VAE")}
    for m in MET:                                  # BH within metric across comparisons (as stats_untouched.add_bh)
        ks = [k for k in comp if "p_t" in comp[k][m]]
        for k, q in zip(ks, bh(np.array([comp[k][m]["p_t"] for k in ks]))): comp[k][m]["q_bh"] = float(q)
    units = sc_[f"CMC-VAE {RUNS[0]}"][1]["units"]
    RES["variants"][v] = dict(units=units, n_units=len(units), n_S=sc_[f"CMC-VAE {RUNS[0]}"][1]["n_S"],
                              de_per_unit_median=float(np.median([DECOUNT[v][g] for g in units])) if units else None,
                              de_per_unit={g: DECOUNT[v][g] for g in UNITS13}, table=tab, paired=comp,
                              vs_chance={k: dict(pe_cent_vs0=one(x, "rr_pe_cent", 0.0), retrieval_vs05=one(x, "rr_retrieval", 0.5), auroc_vs05=one(x, "rr_auroc", 0.5))
                                         for k, x in pu.items() if k in ("CMC-VAE (seed mean)", "corr", "totalvi", "deepsem")},
                              per_unit={k: x for k, x in pu.items() if k in ("CMC-VAE (seed mean)", "corr", "totalvi", "deepsem")})
    print(f"== {v}: units {len(units)} median DE {RES['variants'][v]['de_per_unit_median']}")
    for k in ["CMC-VAE (seed mean)"] + [f"CMC-VAE {r}" for r in RUNS] + ["corr", "totalvi", "deepsem", "zs_ctrl_expr", "zs_ctrl_var", "mean_response_ref"]:
        print(f"  {k:28s} " + "  ".join(f"{m.replace('rr_','')} {tab[k][m].get('mean', float('nan')):.3f}" for m in MET))
# published numbers (stats_papalexi.json) for the harness check
S0 = json.load(open(f"{UNT}/stats_papalexi.json"))["zeroshot_table"]
RES["published_check"] = {k: {m: [S0[k][m].get("mean"), RES["variants"]["orig"]["table"][k][m].get("mean")] for m in MET}
                          for k in ["CMC-VAE (seed mean)", "corr", "totalvi", "deepsem", "zs_ctrl_expr", "zs_ctrl_var"]}
print(json.dumps(RES["published_check"], indent=0))
json.dump(RES, open(f"{DER}/runs/papalexi_rescore.json", "w"), indent=1, default=lambda x: None if (isinstance(x, float) and x != x) else (x.item() if isinstance(x, np.generic) else str(x)))
print("WROTE", f"{DER}/runs/papalexi_rescore.json")
