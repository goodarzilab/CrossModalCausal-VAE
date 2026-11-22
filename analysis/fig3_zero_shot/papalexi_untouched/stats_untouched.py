#!/usr/bin/env python
"""Statistics for the untouched-screen evaluation (pre-registered; see analyses/untouched/PREREGISTRATION.md).
Reads rc_<run>.json (+ _preds.npz) for the three seeds, writes stats.json and source-data CSVs.
Usage: python -I stats_untouched.py [tag=papalexi] [B_joint=200]"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import sys, os, json, glob, pickle, warnings
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon, ttest_1samp, rankdata, norm
warnings.filterwarnings("ignore")
UNT = f"{_CMC_RESULTS}/paper/analyses/untouched"
TAG = sys.argv[1] if len(sys.argv) > 1 else "papalexi"
BJ = int(sys.argv[2]) if len(sys.argv) > 2 else 200
RESD = os.path.join(UNT, sys.argv[3]) if len(sys.argv) > 3 else UNT   # results dir ('clean' for the clean-control deviation)
RUNS = ["sp_dec_gl1000", "sp_dec_gl1000_s1", "sp_dec_gl1000_s2"]
J = {r: json.load(open(f"{RESD}/rc_{r}.json")) for r in RUNS if os.path.exists(f"{RESD}/rc_{r}.json")}
RUNS = list(J)
MET = ["rr_auroc", "rr_pe", "rr_pe_log", "rr_pe_cent", "rr_retrieval", "rr_pe_meanresp", "rr_auroc_meanresp", "rr_pe_shuf",
       "rp_auroc", "rp_pe", "rp_retrieval"]
rng = np.random.default_rng(0)

def num(v):
    try: v = float(v); return v if np.isfinite(v) else np.nan
    except Exception: return np.nan

def per_unit(rows, metrics=MET):
    return {r["gene"]: {m: num(r.get(m)) for m in metrics} | {"dev_target": r.get("dev_target") in (True, "True")} for r in rows}

def seed_mean(list_of_pu):
    genes = sorted(set().union(*[set(p) for p in list_of_pu]))
    out = {}
    for g in genes:
        out[g] = {}
        for m in MET:
            v = [p[g][m] for p in list_of_pu if g in p and np.isfinite(p[g][m])]
            out[g][m] = float(np.mean(v)) if v else np.nan
        out[g]["dev_target"] = any(p[g]["dev_target"] for p in list_of_pu if g in p)
    return out

def summ(pu, m, genes=None):
    v = np.array([pu[g][m] for g in (genes or pu) if g in pu], float); v = v[np.isfinite(v)]
    if not len(v): return dict(n=0)
    bs = rng.choice(v, (10000, len(v))).mean(1) if len(v) > 1 else np.array([v[0]])
    return dict(n=int(len(v)), mean=float(v.mean()), sem=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
                ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])

def one_sample(pu, m, mu0):
    v = np.array([pu[g][m] for g in pu], float); v = v[np.isfinite(v)]
    if len(v) < 3: return dict(n=int(len(v)))
    return dict(n=int(len(v)), mean=float(v.mean()), p_t=float(ttest_1samp(v, mu0).pvalue),
                p_wilcoxon=float(wilcoxon(v - mu0).pvalue) if np.any(v != mu0) else 1.0)

def paired(pa, pb, m):
    """b minus a over shared units; two-sided."""
    g = [x for x in pa if x in pb and np.isfinite(pa[x][m]) and np.isfinite(pb[x][m])]
    if len(g) < 3: return dict(n=len(g))
    a = np.array([pa[x][m] for x in g]); b = np.array([pb[x][m] for x in g]); d = b - a
    if np.allclose(d, 0): return dict(n=len(g), mean_diff=0.0, p_t=1.0, p_wilcoxon=1.0)
    bs = rng.choice(d, (10000, len(d))).mean(1)
    return dict(n=len(g), mean_a=float(a.mean()), mean_b=float(b.mean()), mean_diff=float(d.mean()),
                ci95_boot=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                p_t=float(ttest_rel(b, a).pvalue), p_wilcoxon=float(wilcoxon(d).pvalue))

def bh(ps):
    ps = np.asarray(ps, float); ok = np.isfinite(ps); q = np.full(len(ps), np.nan)
    if ok.sum():
        p = ps[ok]; n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1); r = np.minimum.accumulate(r[::-1])[::-1]
        qq = np.empty(n); qq[o] = np.clip(r, 0, 1); q[ok] = qq
    return q

def add_bh(panel):
    """panel: {comparison: {metric: test}} -> BH within each metric across comparisons."""
    mets = sorted({m for c in panel.values() for m in c})
    for m in mets:
        keys = [k for k in panel if m in panel[k] and "p_t" in panel[k][m]]
        q = bh([panel[k][m]["p_t"] for k in keys])
        for k, qq in zip(keys, q): panel[k][m]["q_bh"] = float(qq)
    return panel

OUT = dict(tag=TAG, runs=RUNS)
if not RUNS: sys.exit("no results")
J0 = J[RUNS[0]]
OUT["units"] = J0.get(f"{TAG}_units")
OUT["dev_fidelity"] = {r: J[r].get("dev_fig3_fidelity") for r in RUNS}

# =============================== (A) zero-shot primary
ZS = {r: J[r][f"{TAG}_zeroshot"] for r in RUNS}
cmc_seed = {r: per_unit(ZS[r]["cmc"]["per_ko"]) for r in RUNS}
cmc = seed_mean(list(cmc_seed.values()))
methods = {"CMC-VAE (seed mean)": cmc}
for r in RUNS: methods[f"CMC-VAE {r}"] = cmc_seed[r]
for b in ["corr", "totalvi", "deepsem", "zs_ctrl_expr", "zs_ctrl_var", "zs_decoder_rownorm", "dev_copy"]:
    if "per_ko" not in ZS[RUNS[0]].get(b, {}): continue
    lst = [per_unit(ZS[r][b]["per_ko"]) for r in RUNS]
    methods[b] = seed_mean(lst) if b == "zs_decoder_rownorm" else lst[0]
# model-free references as methods (taken from the CMC rows, identical for any method)
methods["mean_response_ref"] = {g: {m: np.nan for m in MET} | {"rr_pe": cmc_seed[RUNS[0]][g]["rr_pe_meanresp"], "rr_auroc": cmc_seed[RUNS[0]][g]["rr_auroc_meanresp"],
                                   "dev_target": cmc_seed[RUNS[0]][g]["dev_target"]} for g in cmc_seed[RUNS[0]]}
novel = [g for g in cmc if not cmc[g]["dev_target"]]; shared = [g for g in cmc if cmc[g]["dev_target"]]
OUT["zeroshot_table"] = {k: {m: summ(v, m) for m in MET} for k, v in methods.items()}
OUT["zeroshot_table_novel"] = {k: {m: summ(v, m, novel) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]} for k, v in methods.items()}
OUT["zeroshot_table_devtarget"] = {k: {m: summ(v, m, shared) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]} for k, v in methods.items()}
OUT["zeroshot_vs_chance"] = {k: dict(rr_pe_cent_vs0=one_sample(v, "rr_pe_cent", 0.0), rr_retrieval_vs05=one_sample(v, "rr_retrieval", 0.5),
                                     rr_auroc_vs05=one_sample(v, "rr_auroc", 0.5), rr_pe_vs0=one_sample(v, "rr_pe", 0.0)) for k, v in methods.items()}
comp = {}
for k, v in methods.items():
    if k.startswith("CMC-VAE"): continue
    comp[f"CMC-VAE minus {k}"] = {m: paired(v, cmc, m) for m in ["rr_auroc", "rr_pe", "rr_pe_log", "rr_pe_cent", "rr_retrieval", "rp_auroc"]}
OUT["zeroshot_paired"] = add_bh(comp)
OUT["zeroshot_similarity"] = {r: ZS[r]["cmc"].get("similarity") for r in RUNS}
OUT["doop_shock_fraction_median"] = {r: float(np.median([d["shock_over_full"] for d in ZS[r]["cmc"]["doop"]])) for r in RUNS}
# per-unit source data
rows = []
for k, v in methods.items():
    for g, d in v.items(): rows.append(dict(method=k, gene=g, **{m: d.get(m) for m in MET}, dev_target=d["dev_target"]))
pd.DataFrame(rows).to_csv(f"{UNT}/source_data_{TAG}_zeroshot_per_unit.csv", index=False)

# =============================== (A2) joint unit + cell bootstrap for CMC P1/P2 (seed-mean)
def joint_bootstrap():
    import anndata as ad
    sys.path.insert(0, f"{_CMC_ROOT}/analysis/common")
    pk = glob.glob(f"{UNT}/cache/rc_{TAG}_sp_dec_gl1000_*_0.1_0.05.pkl"); pk = [p for p in pk if "_s1_" not in p and "_s2_" not in p][0]
    al = pickle.load(open(pk, "rb"))["align"]
    common = np.array(al["common_genes"]); sd = np.clip(np.asarray(al["train_rna_std"], float)[np.asarray(al["rna_eval_train_indices"])], 1e-8, None)
    PD = TAG
    a = ad.read_h5ad(f"{UNT}/data/papalexi_proc/{PD}_rna.h5ad")
    X = a[:, list(common)].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    tg = a.obs["gene_target"].astype(str).values
    units = sorted(cmc)
    preds = []
    for r in RUNS:
        z = np.load(f"{RESD}/rc_{r}_preds.npz"); preds.append(np.stack([z[f"cmc_obs|{TAG}|{g}|r"] for g in units]))
    ctl = np.where(tg == "NT")[0]; idx = {g: np.where(tg == g)[0] for g in units}
    keep = np.stack([common != g for g in units]); tset = np.isin(common, units)
    def mw(Xp, Xc):
        n1, n2 = len(Xp), len(Xc); R = rankdata(np.vstack([Xp, Xc]), axis=0); U = R[:n1].sum(0) - n1 * (n1 + 1) / 2
        return 2 * norm.sf(np.abs((U - n1 * n2 / 2) / (np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) + 1e-10)))
    def metrics(lfc, sig, Pr):
        K = len(units); O = np.where(keep, lfc, np.nan); Pn = np.where(keep, Pr, np.nan)
        loo = lambda M, i: np.nanmean(np.delete(M, i, 0), 0)
        S = (sig.any(0)) & ~tset
        pc = np.full(K, np.nan); Pm, Om = [], []
        for i in range(K):
            mo, mp = loo(O, i), loo(Pn, i); m = sig[i] & keep[i]
            if m.sum() >= 5:
                x, y = ((Pr[i] - mp) * sd)[m], (lfc[i] - mo)[m]
                if np.std(x) > 0 and np.std(y) > 0: pc[i] = np.corrcoef(x, y)[0, 1]
            zz = lambda v: (v - v.mean()) / (np.linalg.norm(v - v.mean()) + 1e-300)
            Pm.append(zz((Pr[i] - mp)[S])); Om.append(zz(((lfc[i] - mo) / sd)[S]))
        C = np.nan_to_num(np.stack(Pm) @ np.stack(Om).T); ret = np.array([np.mean(C[i, i] > np.delete(C[i], i)) + 0.5 * np.mean(C[i, i] == np.delete(C[i], i)) for i in range(K)])
        return pc, ret
    def de_all(cidx, uidx):
        Xc = X[cidx]; mc = Xc.mean(0); L, Sg = [], []
        for g in units:
            Xp = X[uidx[g]]; l = Xp.mean(0) - mc; p = mw(Xp, Xc); L.append(l); Sg.append((p < 0.05) & (np.abs(l) >= 0.10))
        return np.stack(L), np.stack(Sg)
    lfc0, sig0 = de_all(ctl, idx)
    base = [metrics(lfc0, sig0, P_) for P_ in preds]
    pc0 = np.nanmean([b[0] for b in base], 0); rt0 = np.mean([b[1] for b in base], 0)
    check = dict(pe_cent_reimpl=float(np.nanmean(pc0)), pe_cent_json=OUT["zeroshot_table"]["CMC-VAE (seed mean)"]["rr_pe_cent"].get("mean"),
                 retr_reimpl=float(np.mean(rt0)), retr_json=OUT["zeroshot_table"]["CMC-VAE (seed mean)"]["rr_retrieval"].get("mean"))
    bs_pc, bs_rt = [], []
    for b in range(BJ):
        cidx = rng.choice(ctl, len(ctl)); uidx = {g: rng.choice(idx[g], len(idx[g])) for g in units}
        lfc, sig = de_all(cidx, uidx); res = [metrics(lfc, sig, P_) for P_ in preds]
        pc = np.nanmean([r_[0] for r_ in res], 0); rt = np.mean([r_[1] for r_ in res], 0)
        u = rng.choice(len(units), len(units)); bs_pc.append(np.nanmean(pc[u])); bs_rt.append(np.mean(rt[u]))
    return dict(B=BJ, check=check, rr_pe_cent_ci95=[float(np.nanpercentile(bs_pc, 2.5)), float(np.nanpercentile(bs_pc, 97.5))],
                rr_retrieval_ci95=[float(np.percentile(bs_rt, 2.5)), float(np.percentile(bs_rt, 97.5))],
                frac_pe_cent_le0=float(np.mean(np.array(bs_pc) <= 0)), frac_retr_le05=float(np.mean(np.array(bs_rt) <= 0.5)))
if BJ > 0:
    try: OUT["joint_bootstrap_cmc"] = joint_bootstrap()
    except Exception as e: OUT["joint_bootstrap_cmc"] = f"failed: {e!r}"

# =============================== (B) S2 transfer
if all("transfer" in J[r] for r in RUNS):
    T = {}
    arms = list(J[RUNS[0]]["transfer"])
    pu = {a: seed_mean([per_unit(J[r]["transfer"][a][TAG]["per_ko"]) for r in RUNS]) for a in arms}
    shuf = [a for a in arms if "labelshuffle" in a]
    pu["ft_labelshuffle_mean"] = seed_mean([pu[a] for a in shuf]) if shuf else {}
    pu["obs"] = cmc
    rt = J[RUNS[0]].get("ridge_transfer", {}).get(TAG, {})
    for nm in rt: pu[nm] = per_unit(rt[nm]["per_ko"])
    T["table"] = {a: {m: summ(v, m) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]} for a, v in pu.items()}
    T["dev_heldout"] = {r: {a: dict(this=J[r]["transfer"][a]["dev_heldout"], paper=J[r]["transfer"][a].get("dev_heldout_paper")) for a in arms} for r in RUNS}
    pan = {}
    for b in ["obs", "ft_labelshuffle_mean", "ft_meantarget"] + list(rt):
        if b in pu: pan[f"ft_standard minus {b}"] = {m: paired(pu[b], pu["ft_standard"], m) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]}
    for b in rt: pan[f"obs minus {b}"] = {m: paired(pu[b], cmc, m) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]}
    T["paired"] = add_bh(pan)
    OUT["transfer"] = T
    rows = [dict(arm=a, gene=g, **{m: d.get(m) for m in MET}) for a, v in pu.items() for g, d in v.items()]
    pd.DataFrame(rows).to_csv(f"{UNT}/source_data_{TAG}_transfer_per_unit.csv", index=False)

# =============================== (C) S1 within-screen CV fine-tuning
if all(TAG in J[r].get("cv", {}) for r in RUNS):
    C = {}
    arms = list(J[RUNS[0]]["cv"][TAG]["arms"])
    pu = {}
    for a in arms:
        lst = [per_unit(J[r]["cv"][TAG]["arms"][a]["per_ko"]) for r in RUNS]
        pu[a] = lst[0] if a.startswith("ridge") and False else seed_mean(lst)
    shuf = [a for a in arms if a.startswith("ft_shuffle")]
    pu["ft_shuffle_mean"] = seed_mean([pu[a] for a in shuf])
    C["folds"] = J[RUNS[0]]["cv"][TAG]["folds"]
    C["table"] = {a: {m: summ(v, m) for m in ["rr_auroc", "rr_pe", "rr_pe_log", "rr_pe_cent", "rr_retrieval", "rr_pe_meanresp"]} for a, v in pu.items()}
    C["table_per_seed"] = {r: {a: {m: summ(per_unit(J[r]["cv"][TAG]["arms"][a]["per_ko"]), m) for m in ["rr_pe", "rr_pe_cent", "rr_retrieval"]} for a in arms} for r in RUNS}
    pan = {f"ft_real minus {b}": {m: paired(pu[b], pu["ft_real"], m) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]}
           for b in ["obs", "ft_shuffle_mean", "ft_mean", "ridge_decoder", "ridge_coexpr"]}
    pan.update({f"obs minus {b}": {m: paired(pu[b], pu["obs"], m) for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]} for b in ["ridge_decoder", "ridge_coexpr"]})
    C["paired"] = add_bh(pan)
    OUT["cv"] = C
    rows = [dict(arm=a, gene=g, **{m: d.get(m) for m in MET}) for a, v in pu.items() for g, d in v.items()]
    pd.DataFrame(rows).to_csv(f"{UNT}/source_data_{TAG}_cv_per_unit.csv", index=False)

json.dump(OUT, open(f"{UNT}/stats_{TAG}.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else (o if not isinstance(o, np.generic) else o.item()))
print("WROTE", f"{UNT}/stats_{TAG}.json")
