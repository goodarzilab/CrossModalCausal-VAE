#!/usr/bin/env python
"""Joint bootstrap over knockouts and control cells for Figs 3-5 under the pseudobulk DE definition.

Replicate b (b=0: no resampling = point estimate):
  1. per screen, resample control pseudobulks within stratum (cluster bootstrap) and, inside each, resample its cells
     with replacement -> new control pseudobulk means;
  2. re-run the pseudobulk limma test for every knockout -> new measured LFC, BH q and DE sets;
  3. re-score every saved prediction set (Fig 3 CMC 3 seeds + baselines; Fig 4 block variants; Fig 5 fine-tuning
     variants x 3 seeds) with score_lib (mirrors the harness);
  4. resample knockouts with replacement within screen and record means / paired differences.
Predictions are held fixed (models are not refitted inside the bootstrap).
usage: python -I boot.py TASK_ID N_REPS [OUTDIR]
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
import os, sys, json, pickle, time
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pblimma import limma_ko, bh
import score_lib as SL

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
TASK, NREP = int(sys.argv[1]), int(sys.argv[2])
ST = f"{_CMC_RESULTS}/paper/analyses/stats"
OUTD = sys.argv[3] if len(sys.argv) > 3 else f"{ST}/boot"; os.makedirs(OUTD, exist_ok=True)
DEMODE = os.environ.get("ST_DE", "pb_bh")
TAGS = ["frangieh", "iccite", "arce"]
SEEDS = {"s42": f"{ST}/runs/rc_sp_dec_gl1000_{DEMODE}_preds.npz", "s1": f"{ST}/runs/rc_sp_dec_gl1000_s1_{DEMODE}_preds.npz",
         "s2": f"{ST}/runs/rc_sp_dec_gl1000_s2_{DEMODE}_preds.npz"}
PB = {t: pickle.load(open(f"{ST}/de/pb_{t}.pkl", "rb")) for t in TAGS}
NPZ = {s: dict(np.load(f, allow_pickle=True)) for s, f in SEEDS.items() if os.path.exists(f)}
log("seeds loaded:", list(NPZ))
COG = json.load(open(f"{_CMC_RESULTS}/baselines/_cmcvae_cognate.json"))
cog_map = {p: g for p, g in zip(COG["protein_names"], COG["cognate_gene"]) if g}

def getset(npz, name, t):
    k = f"{name}|{t}|genes"
    if k not in npz: return None
    return {g: (r.astype(float), p.astype(float)) for g, r, p in zip(npz[k], npz[f"{name}|{t}|rna"], npz[f"{name}|{t}|prot"])}

M0 = NPZ["s42"]
META = {t: dict(sd_r=M0[f"meta|{t}|sd_r"], sd_p=M0[f"meta|{t}|sd_p"], common=M0[f"meta|{t}|common"], pnames=list(M0[f"meta|{t}|pnames"]),
                train=list(M0[f"meta|{t}|train"]), test=list(M0[f"meta|{t}|test"])) for t in TAGS}
for t in TAGS:
    assert list(META[t]["common"]) == list(PB[t]["common_genes"])
# ---- prediction sets ----
F3 = {}; F4 = {}; F5 = {}
for s, npz in NPZ.items():
    for t in TAGS:
        F3.setdefault(f"cmc_{s}", {})[t] = getset(npz, "obs", t)
        for v in ["ft_baseline", "ft_standard", "ft_labelshuffle_0", "ft_labelshuffle_1", "ft_labelshuffle_2", "ft_signflip", "ft_meantarget"]:
            x = getset(npz, v, t)
            if x is not None: F5.setdefault(f"{v}|{s}", {})[t] = x
for t in TAGS:
    for b in ["corr", "totalvi_ko", "deepsem_ko"]:
        x = getset(M0, f"base_{b}", t)
        if x is not None: F3.setdefault(b, {})[t] = x
    for vn in ["Arr_transposed", "Arr_zeroed", "Arr_shuffled", "Wrp_from_WprT", "Wrp_zeroed", "Wrp_shuffled"]:
        reps = [getset(M0, f"fig4_{vn}_{k}", t) for k in range(3)]
        reps = [r for r in reps if r is not None]
        if reps: F4.setdefault(vn, {})[t] = reps
log("sets: F3", list(F3), "F4", list(F4), "F5", list(F5))

# ---- DE realisation ----
def realise(rng):
    C = {}
    for t in TAGS:
        pb = PB[t]; ct = pb["ct"]; strata = np.asarray(ct["strata"]); n_ct = len(strata)
        if rng is None:
            Yr, Yp, sc = ct["Y_r"], ct["Y_p"], strata
        else:
            if os.environ.get("BOOT_PB", "0") == "1":     # optional extra level: resample control pseudobulks (cluster)
                sel = []
                for st in np.unique(strata):
                    ids = np.where(strata == st)[0]; sel.extend(rng.choice(ids, len(ids), replace=True))
                sel = np.array(sel)
            else:                                          # default: keep control pseudobulk structure, resample cells within each
                sel = np.arange(n_ct); rows, cols, vals = [], [], []
            cell_ids = [np.where(ct["cell_pb"] == k)[0] for k in range(n_ct)]
            for r_, k in enumerate(sel):
                idx = cell_ids[k]; w = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)))
                nz = w > 0; rows.append(np.full(nz.sum(), r_)); cols.append(idx[nz]); vals.append(w[nz] / len(idx))
            W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(sel), len(ct["cell_pb"])))
            Yr = np.asarray(W @ ct["cells_r"]); Yp = np.asarray(W @ ct["cells_p"]) if ct["cells_p"].shape[1] else np.zeros((len(sel), 0)); sc = strata[sel]
        common = np.asarray(pb["common_genes"]); genes = [g for g in pb["de"] if g in set(common)]
        F = len(common); Pn = len(pb["prot_names"])
        L = np.zeros((len(genes), F)); Sg = np.zeros((len(genes), F), bool); Lp = np.zeros((len(genes), Pn)); Sp_ = np.zeros((len(genes), Pn), bool)
        for i, g in enumerate(genes):
            kp = pb["ko_pb"][g]
            if rng is None:
                d = pb["de"][g]; b, q, bp_, qp = d["lfc_rna"], d["q_rna"], d["lfc_prot"], d["q_prot"]
            else:
                b, p, _ = limma_ko(kp["Y_r"], np.asarray(kp["strata"]), Yr, sc); q = bh(p)
                if Pn: bp_, pp_, _ = limma_ko(kp["Y_p"], np.asarray(kp["strata"]), Yp, sc); qp = bh(pp_)
                else: bp_, qp = np.zeros(0), np.zeros(0)
            b = np.nan_to_num(b); L[i] = b
            Sg[i] = ((q < 0.05) if DEMODE != "pb_raw" else (pb["de"][g]["pvals_rna"] < 0.05)) & (np.abs(b) >= SL.LFCR)
            if Pn:
                bp_ = np.nan_to_num(bp_); Lp[i] = bp_; Sp_[i] = (qp < 0.05) & (np.abs(bp_) >= SL.LFCP)
        pn = META[t]["pnames"]
        cogm = {g: np.array([cog_map.get(x) == g for x in pn], bool) for g in genes}
        C[t] = SL.Cohort(genes, common, L, Sg, Lp, Sp_, META[t]["sd_r"], META[t]["sd_p"], cogm, META[t]["train"])
    return C

METRICS = ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rr_pe_meanresp", "rr_auroc_meanresp",
           "rp_auroc", "rp_auroc_ge5", "rp_pe", "rp_pe_cent", "rp_retrieval", "rp_pe_meanresp", "rp_auroc_meanresp"]

def score_all(C):
    """-> per-KO table: rows[(set, t, g)] = {metric: v}"""
    T = {}
    for nm, d in F3.items():
        for t in TAGS:
            if not d.get(t): continue
            ev = [g for g in C[t].evaluable if g in d[t]]
            for g, r in SL.score_set(C[t], d[t], ev, ev).items(): T[(f"f3:{nm}", t, g)] = r
    for vn, d in F4.items():
        for t in TAGS:
            if not d.get(t): continue
            ev = [g for g in C[t].evaluable if g in d[t][0]]
            reps = [SL.score_set(C[t], rp, ev, ev) for rp in d[t]]
            for g in reps[0]:
                T[(f"f4:{vn}", t, g)] = {m: float(np.nanmean([rr[g].get(m, np.nan) for rr in reps])) if any(np.isfinite(rr[g].get(m, np.nan)) for rr in reps) else np.nan for m in METRICS}
    for nm, d in F5.items():
        for t in TAGS:
            if not d.get(t): continue
            ev = set(C[t].evaluable)
            te = [g for g in META[t]["test"] if g in ev and g in d[t]]; tr = [g for g in META[t]["train"] if g in ev and g in d[t]]
            for g, r in SL.score_set(C[t], d[t], te, tr).items(): T[(f"f5:{nm}", t, g)] = r
    # seed-averaged Fig-5 sets (labelshuffle averaged over its 3 permutations first)
    seeds = sorted(NPZ)
    for v in ["ft_baseline", "ft_standard", "ft_labelshuffle", "ft_signflip", "ft_meantarget"]:
        for t in TAGS:
            for g in META[t]["test"]:
                vals = {}
                for m in METRICS:
                    per_seed = []
                    for s in seeds:
                        if v == "ft_labelshuffle":
                            xs = [T.get((f"f5:ft_labelshuffle_{k}|{s}", t, g), {}).get(m, np.nan) for k in range(3)]
                            x = np.nanmean(xs) if np.isfinite(xs).any() else np.nan
                        else:
                            x = T.get((f"f5:{v}|{s}", t, g), {}).get(m, np.nan)
                        per_seed.append(x)
                    vals[m] = float(np.nanmean(per_seed)) if np.isfinite(per_seed).any() else np.nan
                    vals[m + "|nseed"] = int(np.isfinite(per_seed).sum())
                if any(np.isfinite(vals[m]) for m in METRICS): T[(f"f5:{v}|mean", t, g)] = vals
        for s in seeds:
            if v == "ft_labelshuffle":
                for t in TAGS:
                    for g in META[t]["test"]:
                        xs = [T.get((f"f5:ft_labelshuffle_{k}|{s}", t, g)) for k in range(3)]
                        xs = [x for x in xs if x]
                        if xs: T[(f"f5:ft_labelshuffle|{s}", t, g)] = {m: float(np.nanmean([x.get(m, np.nan) for x in xs])) if any(np.isfinite(x.get(m, np.nan)) for x in xs) else np.nan for m in METRICS}
    # Fig-3 cross-seed mean of CMC
    for t in TAGS:
        for g in C[t].evaluable:
            xs = [T.get((f"f3:cmc_{s}", t, g)) for s in seeds]; xs = [x for x in xs if x]
            if xs: T[("f3:cmc_mean", t, g)] = {m: float(np.nanmean([x.get(m, np.nan) for x in xs])) if any(np.isfinite(x.get(m, np.nan)) for x in xs) else np.nan for m in METRICS}
    return T

def pooled(C, seed=None):
    out = {}
    for nm in ["cmc_s42", "corr", "totalvi_ko"]:
        for ti, t in enumerate(TAGS):
            d = F3.get(nm, {}).get(t)
            if not d: continue
            gl = [g for g in C[t].evaluable if C[t].has_p[g] and g in d]
            if seed is not None and gl:
                idx = np.random.default_rng(seed * 7 + ti).choice(len(gl), len(gl), replace=True); gl = [gl[i] for i in idx]
            out[f"{nm}|{t}"] = SL.pooled_protein(C[t], d, gl)
    return out

def summarise(T, rng):
    """KO-resampled means and paired differences for every set/metric/scope present."""
    sets = sorted(set(k[0] for k in T))
    units = {t: sorted(set(g for (_, tt, g) in T if tt == t)) for t in TAGS}
    samp = {t: (rng.choice(len(units[t]), len(units[t]), replace=True) if rng is not None else np.arange(len(units[t]))) for t in TAGS}
    S = {}
    def vec(st, t, m):
        return np.array([T.get((st, t, units[t][i]), {}).get(m, np.nan) for i in samp[t]], float)
    cache = {}
    for st in sets:
        for m in METRICS:
            allv = []
            for t in TAGS:
                v = vec(st, t, m); cache[(st, t, m)] = v; allv.append(v)
                f = np.isfinite(v)
                if f.any(): S[f"mean|{st}|{m}|{t}"] = float(v[f].mean())
            v = np.concatenate(allv); f = np.isfinite(v)
            if f.any(): S[f"mean|{st}|{m}|overall"] = float(v[f].mean())
    return S, cache, samp, units

COMPS = []
for b in ["corr", "totalvi_ko", "deepsem_ko"]:
    for m in METRICS[:4] + METRICS[6:11]:
        COMPS.append(("f3:cmc_s42", m, f"f3:{b}", m)); COMPS.append(("f3:cmc_mean", m, f"f3:{b}", m))
for st in ["f3:cmc_s42", "f3:cmc_mean"]:
    COMPS += [(st, "rr_auroc", st, "rr_auroc_meanresp"), (st, "rr_pe", st, "rr_pe_meanresp"),
              (st, "rp_auroc", st, "rp_auroc_meanresp"), (st, "rp_pe", st, "rp_pe_meanresp")]
for vn in ["Arr_transposed", "Arr_zeroed", "Arr_shuffled", "Wrp_from_WprT", "Wrp_zeroed", "Wrp_shuffled"]:
    for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval"]:
        COMPS.append(("f3:cmc_s42", m, f"f4:{vn}", m))
for s in ["mean", "s42", "s1", "s2"]:
    for m in ["rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc", "rp_pe", "rp_pe_cent", "rp_auroc", "rp_retrieval"]:
        COMPS += [(f"f5:ft_standard|{s}", m, f"f5:ft_baseline|{s}", m), (f"f5:ft_standard|{s}", m, f"f5:ft_labelshuffle|{s}", m),
                  (f"f5:ft_standard|{s}", m, f"f5:ft_meantarget|{s}", m), (f"f5:ft_labelshuffle|{s}", m, f"f5:ft_baseline|{s}", m),
                  (f"f5:ft_meantarget|{s}", m, f"f5:ft_baseline|{s}", m)]
    COMPS.append((f"f5:ft_baseline|{s}", "rr_pe_meanresp", f"f5:ft_standard|{s}", "rr_pe"))

def diffs(cache, S):
    for a, ma, b, mb in COMPS:
        allv = []
        for t in TAGS:
            if (a, t, ma) not in cache or (b, t, mb) not in cache: continue
            d = cache[(a, t, ma)] - cache[(b, t, mb)]; allv.append(d); f = np.isfinite(d)
            if f.any(): S[f"diff|{a}|{ma}|{b}|{mb}|{t}"] = float(d[f].mean())
        if allv:
            d = np.concatenate(allv); f = np.isfinite(d)
            if f.any(): S[f"diff|{a}|{ma}|{b}|{mb}|overall"] = float(d[f].mean())
    return S

out_rows = []
for rep in range(NREP):
    b = TASK * NREP + rep
    rng = None if b == 0 else np.random.default_rng(100003 + b)
    C = realise(rng)
    T = score_all(C)
    S, cache, samp, units = summarise(T, rng)
    S = diffs(cache, S)
    # pooled protein analysis on the same KO resample
    for k, v in pooled(C, None if b == 0 else b).items():
        for kk, vv in v.items(): S[f"pooled|{k}|{kk}"] = vv
    S["_b"] = b
    S["_n_eval"] = {t: len(C[t].evaluable) for t in TAGS}; S["_n_has_p"] = {t: int(sum(C[t].has_p[g] for g in C[t].evaluable)) for t in TAGS}
    out_rows.append(S)
    if b == 0:
        pickle.dump(dict(T=T, evaluable={t: C[t].evaluable for t in TAGS}, has_p={t: {g: C[t].has_p[g] for g in C[t].evaluable} for t in TAGS},
                         n_de={t: {g: int((C[t].sig[C[t].gi[g]] & C[t].keep[C[t].gi[g]]).sum()) for g in C[t].genes} for t in TAGS},
                         n_dep={t: {g: int(C[t].sigp[C[t].gi[g]].sum()) for g in C[t].genes} for t in TAGS}), open(f"{OUTD}/point_{DEMODE}.pkl", "wb"))
    log(f"rep {b} done; n_eval {S['_n_eval']}")
json.dump(out_rows, open(f"{OUTD}/boot_{DEMODE}_{TASK:03d}.json", "w"), default=lambda o: None if (isinstance(o, float) and o != o) else o)
log("WROTE", f"{OUTD}/boot_{DEMODE}_{TASK:03d}.json")
