#!/usr/bin/env python
"""Reviewer M10, Replogle Perturb-seq: rescore saved predictions (preds/*.npz written by ft_eval_de.py) under the
alternative DE ground truths of de/{line}_de_variants.npz.  The scorer is a verbatim copy of the RNA metrics of
analysis/fig5_interventions/perturbseq/ft_eval.py (score / ref_means / loo / retrieval / pe / au); only the DE context changes:
for evaluation DE `ev` the evaluable KOs (>=3 DE genes excluding the target), the training / held-out KO lists
(evaluable ∩ global split), the retrieval feature set S (genes DE in >=1 training KO minus all evaluable targets),
the leave-one-out reference means (training KOs) and the DE masks of centred pearDE are all recomputed.
Observed response: cell-level LFC for orig / bh / pbmask, pseudobulk LFC for pb / pbgem.

usage: python -I rescore_perturbseq.py   -> runs/perturbseq_rescore.json (+ per-KO rows in runs/perturbseq_rescore_perko.json.gz)
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
import os, sys, json, glob, gzip
import numpy as np
from scipy.stats import wilcoxon, ttest_1samp
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

REPO = f"{_CMC_ROOT}"; D = f"{_CMC_RESULTS}/replogle/data"
AN = f"{_CMC_RESULTS}/paper/analyses/perturbseq"; DER = f"{_CMC_RESULTS}/paper/analyses/review2/de_robust"
LFCR = 0.10
FEATS = json.load(open(f"{D}/features.json"))["features"]; G = len(FEATS); FI = {g: i for i, g in enumerate(FEATS)}; FA = np.array(FEATS)
SPL = json.load(open(f"{D}/split.json")); TRS, TES = set(SPL["train_genes"]), set(SPL["test_genes"])
EVS = ["orig", "bh", "pb", "pbmask", "pbgem"]

def model_dir(line, pre, seed):
    md = f"{AN}/models/{line}_{pre}_s{seed}"
    if pre == "obs" and not os.path.exists(f"{md}/final_model.pt"): md = f"{_CMC_RESULTS}/replogle/models/{line}_s{seed}"
    return md

def ctx(line, ev, sd_r):
    dv = np.load(f"{DER}/de/{line}_de_variants.npz"); genes = dv["genes"].astype(str)
    if ev == "orig":
        de = np.load(f"{D}/{line}_de.npz"); LF = de["lfc"].astype(float); SG = (de["pvals"].astype(float) < 0.05) & (np.abs(LF) >= LFCR)
    else:
        LF = np.nan_to_num(dv[{"bh": "lfc", "pb": "lfc_pb", "pbgem": "lfc_pbgem", "pbmask": "lfc"}[ev]].astype(float))
        SG = dv[{"pbmask": "sig_pb"}.get(ev, "sig_" + ev)].astype(bool)
    INFO = {}
    for k, g in enumerate(genes):
        if g not in FI: continue
        keep = FA != g; sig = SG[k]
        if sig[keep].sum() < 3: continue
        INFO[g] = dict(g=g, keep=keep, sig=sig, lfc=LF[k], n_de=int((sig & keep).sum()))
    ev_ = sorted(INFO); tr = [g for g in ev_ if g in TRS]; te = [g for g in ev_ if g in TES]
    u = np.zeros(G, bool)
    for g in tr: u |= INFO[g]["sig"]
    u &= ~np.isin(FA, ev_)
    return dict(INFO=INFO, evaluable=ev_, train=tr, test=te, S=np.where(u)[0], sd_r=sd_r)

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
    s, n = R[k + "_sum"].copy(), R[k + "_cnt"].astype(float).copy()
    if g in R["gi"]:
        v = R[k][R["gi"][g]]; f = np.isfinite(v); s[f] -= v[f]; n[f] -= 1
    with np.errstate(invalid="ignore", divide="ignore"): return np.where(n > 0, s / n, np.nan)

def retrieval(Pm, Om, genes):
    C = np.nan_to_num(Pm @ Om.T); out = {}
    for i, g in enumerate(genes):
        oth = np.arange(len(genes)) != i
        out[g] = float(np.mean(C[i, i] > C[i, oth]) + 0.5 * np.mean(C[i, i] == C[i, oth]))
    return out

def score(c, preds, SD_NT):
    genes = c["test"]; R = ref_means(c, c["train"], preds); S = c["S"]; sd = c["sd_r"]
    def _z(v): v = v - v.mean(); nn_ = np.linalg.norm(v); return v / nn_ if nn_ > 0 else v
    Pm = np.stack([_z((preds[g] - loo(R, "p_r", g))[S]) for g in genes])
    Om = np.stack([_z(((c["INFO"][g]["lfc"] - loo(R, "o_r", g)) / sd)[S]) for g in genes])
    RET = retrieval(Pm, Om, genes)
    Pc = np.stack([_z(((preds[g] - loo(R, "p_r", g)) * sd / SD_NT)[S]) for g in genes])
    Oc = np.stack([_z(((c["INFO"][g]["lfc"] - loo(R, "o_r", g)) / SD_NT)[S]) for g in genes])
    RETC = retrieval(Pc, Oc, genes); rows = []
    for g in genes:
        I = c["INFO"][g]; pr = preds[g]; k = I["keep"]; m = I["sig"] & k
        mu_o, mu_p = loo(R, "o_r", g), loo(R, "p_r", g)
        rows.append(dict(gene=g, n_de_rna=int(m.sum()), rr_retrieval=RET[g], rr_retrieval_canon=RETC[g],
                         rr_auroc=au(I["sig"][k], np.abs(pr[k])), rr_pe=pe(pr, I["lfc"], m),
                         rr_pe_cent=pe(pr * sd - mu_p * sd, I["lfc"] - mu_o, m),
                         rr_pe_cent_canon=pe((pr * sd - mu_p * sd) / SD_NT, (I["lfc"] - mu_o) / SD_NT, m)))
    return rows

MET = ["rr_retrieval", "rr_pe_cent", "rr_pe", "rr_auroc", "rr_retrieval_canon", "rr_pe_cent_canon"]
def avg_rows(L):
    out = []
    for i, r in enumerate(L[0]):
        a = dict(r)
        for k in r:
            if k.startswith("rr_"): a[k] = float(np.nanmean([rr[i][k] for rr in L]))
        out.append(a)
    return out
def mean_of(rows, m):
    v = np.array([r[m] for r in rows], float); return float(np.nanmean(v))

def kovec(rows, m): return {r["gene"]: r.get(m) for r in rows if r.get(m) is not None and np.isfinite(r.get(m))}
def seed_contrast(A, B, m, rng=None, nb=4000):
    """b - a; A/B {seed: rows}; == aggregate_rev.seed_contrast (seed-averaged per-KO Wilcoxon, KO + hierarchical bootstrap)."""
    rng = np.random.RandomState(0) if rng is None else rng
    seeds = [s for s in A if s in B]; genes = None
    for s in seeds:
        a, b = kovec(A[s], m), kovec(B[s], m); g = sorted(set(a) & set(b)); genes = g if genes is None else sorted(set(genes) & set(g))
    Dm = np.array([[kovec(B[s], m)[g] - kovec(A[s], m)[g] for g in genes] for s in seeds]); n = Dm.shape[1]; dbar = Dm.mean(0)
    bs = dbar[rng.randint(0, n, (nb, n))].mean(1)
    hb = np.array([Dm[rng.randint(0, len(seeds), len(seeds))][:, rng.randint(0, n, n)].mean() for _ in range(1000)])
    sdm = Dm.mean(1)
    return dict(n_ko=n, n_seeds=len(seeds), mean_diff=float(dbar.mean()), ci95_ko=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                ci95_hier=[float(np.percentile(hb, 2.5)), float(np.percentile(hb, 97.5))],
                p_wilcoxon_seedavg=float(wilcoxon(dbar).pvalue) if np.any(dbar != 0) else 1.0, seed_diffs=[float(x) for x in sdm],
                max_p_per_seed=float(max(wilcoxon(d).pvalue if np.any(d != 0) else 1.0 for d in Dm)),
                sign_consistent=bool(np.all(sdm > 0) or np.all(sdm < 0)))
def add_bh(panel):
    ks = [k for k in panel if "p_wilcoxon_seedavg" in panel[k]]
    if ks:
        for k, q in zip(ks, multipletests([panel[k]["p_wilcoxon_seedavg"] for k in ks], method="fdr_bh")[1]): panel[k]["q_bh"] = float(q)
    return panel
def did(rows_a, rows_sa, m):
    """per-KO (arm - shuffled) differences as rows (same KO order)."""
    sa = kovec(rows_sa, m); return [dict(gene=r["gene"], **{m: r[m] - sa[r["gene"]]}) for r in rows_a if np.isfinite(r[m]) and r["gene"] in sa]

# ================================================================== score everything
OUT = dict(ev_variants=EVS, de_summary={}, runs={}, arms={}, contrasts={}, repro={})
PERKO = {}
for line in ["jurkat", "k562"]:
    dv = np.load(f"{DER}/de/{line}_de_variants.npz"); OUT["de_summary"][line] = json.loads(str(dv["summary"]))
    SD_NT = np.clip(np.load(f"{AN}/data/{line}_ntstats.npz")["std"].astype(float), 1e-8, None)
    for f in sorted(glob.glob(f"{DER}/preds/{line}_*.npz")):
        tag = os.path.basename(f)[:-4]                        # line_pre_sSEED_detrain[_ds]
        parts = tag.split("_"); pre, seed, detr = parts[1], int(parts[2][1:]), parts[3]; isds = tag.endswith("_ds")
        cfg = json.load(open(f"{model_dir(line, pre, seed)}/config.json"))
        st = np.load(cfg.get("stats", f"{D}/{line}_stats.npz")); SD_R = np.clip(st["std"].astype(float), 1e-8, None)
        z = np.load(f); gl = list(z["genes"].astype(str)); arms = [k for k in z.files if k != "genes"]
        for ev in EVS:
            c = ctx(line, ev, SD_R); R = {}
            for a in arms:
                P = z[a].astype(float); preds = {g: P[i] for i, g in enumerate(gl)}
                R[a] = score(c, preds, SD_NT)
            PERKO[(tag, ev)] = R
            OUT["runs"].setdefault(tag, {})[ev] = dict(n_train=len(c["train"]), n_test=len(c["test"]), n_S=int(len(c["S"])),
                                                       de_median_eval=float(np.median([c["INFO"][g]["n_de"] for g in c["evaluable"]])),
                                                       summary={a: {m: mean_of(R[a], m) for m in MET} for a in R})
        print(tag, "scored", flush=True)

def rows_arm(tag, ev, arm):
    R = PERKO.get((tag, ev))
    if R is None: return None
    pfx = "" if not tag.endswith("_ds") else "ds_top73728_"
    if arm == "obs": return R["ds_obs" if tag.endswith("_ds") else "cmc_obs"]
    if arm.endswith("_avg"):
        base = arm[:-4]; ks = sorted(k for k in R if k.startswith(pfx + base + "_") and k[len(pfx + base) + 1:].isdigit())
        return avg_rows([R[k] for k in ks]) if ks else None
    return R.get(pfx + arm)

ARMS = ["obs", "real", "shuffle_avg", "infonce", "infonce_shuffle_avg"]
for line in ["jurkat", "k562"]:
    for pre in ["obs", "ctrl"]:
        for detr in ["orig", "bh", "pb"]:
            seeds = [s for s in [42, 1, 2] if (f"{line}_{pre}_s{s}_{detr}", "orig") in PERKO]
            if not seeds: continue
            key = f"{line}_{pre}_train{detr}"
            for ev in EVS:
                T = {s: {a: rows_arm(f"{line}_{pre}_s{s}_{detr}", ev, a) for a in ARMS} for s in seeds}
                OUT["arms"].setdefault(key, {})[ev] = {a: {m: dict(mean=float(np.mean([mean_of(T[s][a], m) for s in seeds])),
                                                              sd=float(np.std([mean_of(T[s][a], m) for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0,
                                                              per_seed=[mean_of(T[s][a], m) for s in seeds]) for m in MET} for a in ARMS}
                pan = {}
                for b, a in [("real", "shuffle_avg"), ("real", "obs"), ("infonce", "infonce_shuffle_avg"), ("infonce", "obs")]:
                    for m in ["rr_pe_cent", "rr_retrieval", "rr_pe_cent_canon"]:
                        if len(seeds) >= 1: pan[f"{b}_vs_{a}|{m}"] = seed_contrast({s: T[s][a] for s in seeds}, {s: T[s][b] for s in seeds}, m)
                # DeepSEM (top73728) difference-in-differences, same cells, same seeds
                dseeds = [s for s in seeds if (f"{line}_{pre}_s{s}_{detr}_ds", ev) in PERKO]
                ds_arms = {}
                if dseeds:
                    DT = {s: {a: rows_arm(f"{line}_{pre}_s{s}_{detr}_ds", ev, a) for a in ARMS} for s in dseeds}
                    ds_arms = {a: {m: dict(mean=float(np.mean([mean_of(DT[s][a], m) for s in dseeds])), per_seed=[mean_of(DT[s][a], m) for s in dseeds])
                                   for m in MET} for a in ARMS}
                    for arm, sh in [("real", "shuffle_avg"), ("infonce", "infonce_shuffle_avg")]:
                        for m in ["rr_retrieval", "rr_pe_cent"]:
                            pan[f"DiD_CMCgain_minus_DSgain:{arm}|{m}"] = seed_contrast({s: did(DT[s][arm], DT[s][sh], m) for s in dseeds},
                                                                                       {s: did(T[s][arm], T[s][sh], m) for s in dseeds}, m)
                            pan[f"DS_{arm}_vs_{sh}|{m}"] = seed_contrast({s: DT[s][sh] for s in dseeds}, {s: DT[s][arm] for s in dseeds}, m)
                OUT["contrasts"].setdefault(key, {})[ev] = add_bh(pan)
                if ds_arms: OUT.setdefault("ds_arms", {}).setdefault(key, {})[ev] = ds_arms
        # control-only vs observational pretraining: gain (FT - shuffled) under each, and their difference
    for ev in EVS:
        ko = {}; T = {}
        for pre in ["obs", "ctrl"]:
            seeds = [s for s in [42, 1, 2] if (f"{line}_{pre}_s{s}_orig", ev) in PERKO]
            T[pre] = {s: {a: rows_arm(f"{line}_{pre}_s{s}_orig", ev, a) for a in ["real", "shuffle_avg", "infonce", "infonce_shuffle_avg"]} for s in seeds}
        if T.get("obs") and T.get("ctrl"):
            pan = {}
            for arm, sh in [("real", "shuffle_avg"), ("infonce", "infonce_shuffle_avg")]:
                for m in ["rr_pe_cent", "rr_retrieval"]:
                    pan[f"ctrl_minus_obs_gain:{arm}|{m}"] = seed_contrast({s: did(T["obs"][s][arm], T["obs"][s][sh], m) for s in T["obs"]},
                                                                          {s: did(T["ctrl"][s][arm], T["ctrl"][s][sh], m) for s in T["ctrl"]}, m)
            OUT["contrasts"].setdefault(f"{line}_ctrl_vs_obs", {})[ev] = add_bh(pan)

# ---- harness reproduction: rescored 'orig' vs published summary.json and vs the original per-KO rows
PSUM = json.load(open(f"{AN}/summary.json"))
for line in ["jurkat", "k562"]:
    for pre in ["obs", "ctrl"]:
        k = f"{line}_{pre}_trainorig"
        if k not in OUT["arms"]: continue
        pub = PSUM["ft"].get(f"{line}_{pre}", {}).get("arms", {})
        OUT["repro"][k] = {a: {m: dict(published=pub.get(a, {}).get(m, {}).get("mean"), rescored=OUT["arms"][k]["orig"][a][m]["mean"])
                               for m in ["rr_retrieval", "rr_pe_cent"]} for a in ARMS}
        # per-KO exactness vs the original run json (seed 42 real arm)
        try:
            r0 = json.load(open(f"{AN}/runs/{line}_{pre}_s42.json"))["ft"]["shift"]["real"]["per_ko"]
            a = kovec(r0, "rr_pe_cent"); b = kovec(rows_arm(f"{line}_{pre}_s42_orig", "orig", "real"), "rr_pe_cent")
            d = np.array([a[g] - b[g] for g in a if g in b]); OUT["repro"][k]["perko_maxabs_diff_real_s42_pe_cent"] = float(np.abs(d).max())
            a = kovec(r0, "rr_retrieval"); b = kovec(rows_arm(f"{line}_{pre}_s42_orig", "orig", "real"), "rr_retrieval")
            d = np.array([a[g] - b[g] for g in a if g in b]); OUT["repro"][k]["perko_maxabs_diff_real_s42_retrieval"] = float(np.abs(d).max())
        except Exception as e: OUT["repro"][k]["perko_check"] = repr(e)
    try:
        dsp = PSUM["ds"][f"{line}_obs"]["variants"]["top73728"]
        OUT["repro"][f"{line}_obs_ds"] = {a: dict(published=dsp["arms"][a]["rr_retrieval"]["mean"], rescored=OUT["ds_arms"][f"{line}_obs_trainorig"]["orig"][a]["rr_retrieval"]["mean"]) for a in ARMS}
        for arm in ["real", "infonce"]:
            cpub = dsp["contrasts"][f"DiD_CMCgain_minus_DSgain:{arm}|rr_retrieval"]
            cres = OUT["contrasts"][f"{line}_obs_trainorig"]["orig"][f"DiD_CMCgain_minus_DSgain:{arm}|rr_retrieval"]
            OUT["repro"][f"{line}_obs_ds"][f"DiD_{arm}_retrieval"] = dict(published=[cpub["mean_diff"], cpub["p_wilcoxon_seedavg"], cpub.get("q_bh")],
                                                                          rescored=[cres["mean_diff"], cres["p_wilcoxon_seedavg"], cres.get("q_bh")])
    except Exception as e: OUT["repro"][f"{line}_obs_ds"] = repr(e)

json.dump(OUT, open(f"{DER}/runs/perturbseq_rescore.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else str(o))
with gzip.open(f"{DER}/runs/perturbseq_rescore_perko.json.gz", "wt") as fh:
    json.dump({f"{t}|{e}": R for (t, e), R in PERKO.items()}, fh, default=lambda o: None if (isinstance(o, float) and o != o) else str(o))
print("WROTE", f"{DER}/runs/perturbseq_rescore.json")
