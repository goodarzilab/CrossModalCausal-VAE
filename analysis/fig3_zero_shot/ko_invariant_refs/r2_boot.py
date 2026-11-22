#!/usr/bin/env python
"""Round-2 unified joint bootstrap (KO x control-cell, DE recomputed per replicate, model seeds resampled in the
hierarchical variant). Protocol/RNG identical to code/reviewfix/rf_boot.py (= code/bootclamp, analysis/fig3_zero_shot/stats/boot.py):
replicate b uses default_rng(100003+b) for control-cell + KO resampling and default_rng(700001+b) for seeds; b=0 = point.
Sections
  F3  : Fig 3 zero-shot sets (paired clamp, RNA-only clamp, control-adapted clamp, corr/totalVI/DeepSEM/edge cut,
        knockout-invariant control mean / control s.d. rankings recomputed from the resampled controls), metrics incl.
        expression-stratified and residualised AUROC (task 1/3/5).
  LG  : LFC-floor grid (RNA 0.05/0.10/0.15/0.25 at protein 0.05; protein 0.025/0.05/0.10 at RNA 0.10) for clamp,
        RNA-only clamp and correlation (task 6).
  F5  : Fig 5b,c fine-tuning arms on the paper split (paired, RNA-only, per-screen adapted / unadapted) (task 1/5/8).
  ST4 : Supp Table 4 pooled table: paper split + 6 gene splits x 3 models, unique held-out units (task 8).
Output: OUTD/boot_<task>.npz with matrix V[nrep, nkey] and keys.
usage: python -I r2_boot.py TASK NREP OUTD"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, pickle, time, glob
import numpy as np, scipy.sparse as sp
from scipy.stats import rankdata
B = f"{_CMC_RESULTS}/paper"
sys.path.insert(0, f"{_CMC_ROOT}/analysis/fig3_zero_shot/stats"); sys.path.insert(0, f"{_CMC_ROOT}/data_prep/pseudobulk_de")
from pblimma import limma_ko, bh
import score_lib as SL
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
TASK, NREP, OUTD = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; os.makedirs(OUTD, exist_ok=True)
ONLY = os.environ.get("R2_SECTIONS", "F3,LG,F5,ST4").split(",")
ST = f"{B}/analyses/stats"; TAGS = ["frangieh", "iccite", "arce"]; SEEDS = ["s42", "s1", "s2"]
RF = f"{_CMC_SCRATCH}/nov_reviewfix/preds"; R2 = f"{_CMC_SCRATCH}/nov_round2/preds"
PB = {t: pickle.load(open(f"{ST}/de/pb_{t}.pkl", "rb")) for t in TAGS}
M0 = dict(np.load(f"{ST}/runs/rc_sp_dec_gl1000_pb_bh_preds.npz", allow_pickle=True))
COG = json.load(open(f"{_CMC_RESULTS}/baselines/_cmcvae_cognate.json"))
cog_map = {p: g for p, g in zip(COG["protein_names"], COG["cognate_gene"]) if g}
META = {t: dict(sd_r=M0[f"meta|{t}|sd_r"], sd_p=M0[f"meta|{t}|sd_p"], common=M0[f"meta|{t}|common"], pnames=list(M0[f"meta|{t}|pnames"]),
                train=list(M0[f"meta|{t}|train"])) for t in TAGS}

def getset(npz, name, t, noprot=False):
    """noprot=True for RNA-only models: protein predictions set to NaN so no protein metric is scored."""
    k = f"{name}|{t}|genes"
    if k not in npz: return None
    g_, r_, p_ = npz[k], npz[f"{name}|{t}|rna"], npz[f"{name}|{t}|prot"]
    return {g: (r.astype(float), np.full(p.shape, np.nan) if noprot else p.astype(float)) for g, r, p in zip(g_, r_, p_)}
def ld(p): return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None

# ---------------- Fig 3 sets
F3 = {}; FAM3 = {}   # family -> list of per-seed set names
for s in SEEDS:
    Z = ld(f"{RF}/fig34_{s}.npz"); F3[f"clamp_{s}"] = {t: getset(Z, "clamp", t) for t in TAGS}
    Z = ld(f"{R2}/rnaonly/fig34_{s}.npz")
    if Z is not None: F3[f"rnaonly_{s}"] = {t: getset(Z, "clamp", t, noprot=True) for t in TAGS}
    ad = {t: ld(f"{R2}/adapt_{t}/fig34_{s}.npz") for t in TAGS}
    if all(v is not None for v in ad.values()): F3[f"adapt_{s}"] = {t: getset(ad[t], "clamp", t) for t in TAGS}
for fam in ["clamp", "rnaonly", "adapt"]:
    if all(f"{fam}_{s}" in F3 for s in SEEDS): FAM3[fam] = [f"{fam}_{s}" for s in SEEDS]
for b in ["corr", "totalvi_ko", "deepsem_ko"]:
    F3[b] = {t: getset(M0, f"base_{b}", t) for t in TAGS}
F3["edgecut"] = {t: getset(M0, "obs", t) for t in TAGS}
SINGLE3 = ["corr", "totalvi_ko", "deepsem_ko", "edgecut", "ctrl_mean", "ctrl_sd"]
# second model class (task 1): denoised-RNA correlation read-out of scVI (no pairing) vs totalVI (paired), same input/split;
# read-out identical to base_totalvi_ko (pred = -G_rr[g, common], verified r = -1.000 against the stored totalVI preds)
for nm, fn in [("scvi_ko", "scvi_s42"), ("totalvi_fresh", "totalvi_s42")]:
    p = f"{_CMC_SCRATCH}/nov_round2/scvi/{fn}.npz"
    if not os.path.exists(p): continue
    z = np.load(p, allow_pickle=True); Gm = z["G_rr"].astype(float); gi = {g: i for i, g in enumerate(z["genes"].astype(str))}
    F3[nm] = {}
    for t in TAGS:
        com = META[t]["common"]; col = np.array([gi.get(c, -1) for c in com]); npn = len(META[t]["pnames"])
        F3[nm][t] = {g: (np.where(col >= 0, -Gm[gi[g], np.clip(col, 0, None)], 0.0), np.full(npn, np.nan))
                     for g in PB[t]["de"] if g in gi}
    SINGLE3.append(nm)
log("F3 families", FAM3)

# ---------------- Fig 5 runs: family -> seed -> list of (run_id, {t: {arm: preds}}, train{t}, test{t})
def load_arms(paths_by_tag, noprot=False):
    arms = {}; train = {}; test = {}
    for t, p in paths_by_tag.items():
        Z = ld(p)
        if Z is None: return None
        for a in ["baseline", "real", "shuffle_0", "shuffle_1", "shuffle_2", "mean"]:
            k = f"{a}|{t}|genes"
            if k not in Z: continue
            arms.setdefault(a, {})[t] = getset(Z, a, t, noprot)
            it = Z[f"{a}|{t}|is_test"]; gl = list(Z[k])
            test[t] = [g for g, x in zip(gl, it) if x]; train[t] = [g for g, x in zip(gl, it) if not x]
    return arms, train, test
F5 = {}
for s in SEEDS:
    x = load_arms({t: f"{RF}/fig5arms_{s}.npz" for t in TAGS}); F5.setdefault("paired", {})[s] = [("paper",) + x] if x else []
    x = load_arms({t: f"{R2}/rnaonly/fig5arms_{s}.npz" for t in TAGS}, noprot=True)
    if x: F5.setdefault("rnaonly", {})[s] = [("paper",) + x]
    x = load_arms({t: f"{R2}/adapt_{t}/fig5arms_{s}.npz" for t in TAGS})
    if x: F5.setdefault("adapt", {})[s] = [("paper",) + x]
    x = load_arms({t: f"{R2}/unadpt_{t}/fig5arms_{s}.npz" for t in TAGS})
    if x: F5.setdefault("unadpt", {})[s] = [("paper",) + x]
    L = [("paper",) + load_arms({t: f"{RF}/fig5arms_{s}.npz" for t in TAGS})]
    for k in range(1, 7):
        x = load_arms({t: f"{R2}/st4/{s}_split{k}/fig5arms_{s}.npz" for t in TAGS})
        if x: L.append((f"split{k}",) + x)
    F5.setdefault("st4", {})[s] = L
F5 = {f: d for f, d in F5.items() if all(d.get(s) for s in SEEDS)}
log("F5 families", {f: {s: len(v) for s, v in d.items()} for f, d in F5.items()})

# ---------------- DE realisation (verbatim rf_boot.realise, but returns raw arrays so floors can vary)
def realise_raw(rng):
    R = {}
    for t in TAGS:
        pb = PB[t]; ct = pb["ct"]; strata = np.asarray(ct["strata"]); n_ct = len(strata)
        if rng is None: Yr, Yp, sc = ct["Y_r"], ct["Y_p"], strata
        else:
            rows, cols, vals = [], [], []
            cell_ids = [np.where(ct["cell_pb"] == k)[0] for k in range(n_ct)]
            for r_, k in enumerate(range(n_ct)):
                idx = cell_ids[k]; w = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)))
                nz = w > 0; rows.append(np.full(nz.sum(), r_)); cols.append(idx[nz]); vals.append(w[nz] / len(idx))
            W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_ct, len(ct["cell_pb"])))
            Yr = np.asarray(W @ ct["cells_r"]); Yp = np.asarray(W @ ct["cells_p"]) if ct["cells_p"].shape[1] else np.zeros((n_ct, 0)); sc = strata
        common = np.asarray(pb["common_genes"]); genes = [g for g in pb["de"] if g in set(common)]
        F = len(common); Pn = len(pb["prot_names"])
        L = np.zeros((len(genes), F)); Q = np.ones((len(genes), F)); Lp = np.zeros((len(genes), Pn)); Qp = np.ones((len(genes), Pn))
        for i, g in enumerate(genes):
            kp = pb["ko_pb"][g]
            if rng is None:
                d = pb["de"][g]; b, q, bp_, qp = d["lfc_rna"], d["q_rna"], d["lfc_prot"], d["q_prot"]
            else:
                b, p, _ = limma_ko(kp["Y_r"], np.asarray(kp["strata"]), Yr, sc); q = bh(p)
                if Pn: bp_, pp_, _ = limma_ko(kp["Y_p"], np.asarray(kp["strata"]), Yp, sc); qp = bh(pp_)
                else: bp_, qp = np.zeros(0), np.zeros(0)
            L[i] = np.nan_to_num(b); Q[i] = np.nan_to_num(q, nan=1.0)
            if Pn: Lp[i] = np.nan_to_num(bp_); Qp[i] = np.nan_to_num(qp, nan=1.0)
        R[t] = dict(genes=genes, common=common, L=L, Q=Q, Lp=Lp, Qp=Qp, Yr=Yr, Yp=Yp)
    return R

def cohort(R, t, fr=0.10, fp=0.05):
    r = R[t]; pn = META[t]["pnames"]; cogm = {g: np.array([cog_map.get(x) == g for x in pn], bool) for g in r["genes"]}
    Sg = (r["Q"] < 0.05) & (np.abs(r["L"]) >= fr); Sp_ = (r["Qp"] < 0.05) & (np.abs(r["Lp"]) >= fp)
    return SL.Cohort(r["genes"], r["common"], r["L"], Sg, r["Lp"], Sp_, META[t]["sd_r"], META[t]["sd_p"], cogm, META[t]["train"])

# ---------------- expression-stratified / residualised AUROC (task 3)
def _qbins(x, n):
    e = np.quantile(x, np.linspace(0, 1, n + 1)[1:-1]); return np.searchsorted(e, x, side="right")
def strat_auroc(lab, sc, bins):
    """pair-weighted within-stratum AUROC (concordant DE/non-DE pairs counted only inside a stratum); vectorised."""
    if lab.sum() < 3: return np.nan
    o = np.lexsort((sc, bins)); bo = bins[o]; lo = lab[o]
    nb = int(bins.max()) + 1; cnt = np.bincount(bins, minlength=nb); start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    wr = np.arange(len(sc)) - start[bo] + 1.0                 # within-stratum rank (1-based; ties broken by order)
    n1 = np.bincount(bo, weights=lo.astype(float), minlength=nb); n0 = cnt - n1
    U = np.bincount(bo, weights=wr * lo, minlength=nb) - n1 * (n1 + 1) / 2
    ok = (n1 > 0) & (n0 > 0); den = (n1 * n0)[ok].sum()
    return float(U[ok].sum() / den) if den > 0 else np.nan
def strat_ctx(R, t):
    cm = R[t]["Yr"].mean(0); cs = R[t]["Yr"].std(0, ddof=1)
    rm = rankdata(cm) / len(cm); rs = rankdata(cs) / len(cs)
    X = np.column_stack([np.ones_like(rm), rm, rs, rm ** 2, rs ** 2, rm * rs])
    return dict(cm=cm, cs=cs, b_ms=_qbins(cm, 5) * 5 + _qbins(cs, 5), b_ms10=_qbins(cm, 10) * 10 + _qbins(cs, 10), b_s10=_qbins(cs, 10), b_m10=_qbins(cm, 10), X=X, Xp=np.linalg.pinv(X))
def extra_metrics(C, preds, ev, SX):
    """expression-stratified AUROC (control-mean x control-s.d. quintile 5x5 strata; s.d. deciles; mean deciles) and
    residualised AUROC (|pred| residualised on a quadratic in control-mean and control-s.d. ranks, fit over all features)."""
    out = {}
    ev = [g for g in ev if np.isfinite(preds[g][0]).all()]
    if not ev: return out
    Sc = np.abs(np.stack([preds[g][0] for g in ev])); Res = Sc - (Sc @ SX["Xp"].T) @ SX["X"].T
    for j, g in enumerate(ev):
        i = C.gi[g]; k = C.keep[i]; lab = C.sig[i][k]; sc = Sc[j][k]
        if lab.sum() < 3: continue
        out[g] = dict(rr_au_strat_ms=strat_auroc(lab, sc, SX["b_ms"][k]), rr_au_strat_ms10=strat_auroc(lab, sc, SX["b_ms10"][k]), rr_au_strat_sd10=strat_auroc(lab, sc, SX["b_s10"][k]),
                      rr_au_strat_m10=strat_auroc(lab, sc, SX["b_m10"][k]), rr_au_resid=SL.auroc(lab, Res[j][k]))
    return out

M3 = ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc_meanresp", "rp_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval",
      "rr_au_strat_ms", "rr_au_strat_ms10", "rr_au_strat_sd10", "rr_au_strat_m10", "rr_au_resid"]
MLG = ["rr_auroc", "rr_pe_cent", "rr_retrieval", "rr_pe", "rp_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval"]
M5 = ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval"]
GRID = [(0.05, 0.05), (0.10, 0.05), (0.15, 0.05), (0.25, 0.05), (0.10, 0.025), (0.10, 0.10)]

def agg_family(vals_by_seed, sdraw):
    P = np.stack(vals_by_seed)
    with np.errstate(all="ignore"): return np.nanmean(P, 0), np.nanmean(P[sdraw], 0)

def summarise_units(S, prefix, units_t, getv, sets_fam, sets_single, metrics, comps, samp, sdraw, scopes=True):
    """units_t: {t: [unit ids]}; getv(setname, t, unit, metric) -> value; sets_fam: {fam: [per-seed names]}.
    Writes mean|prefix|set|metric|scope and diff|prefix|a|b|metric|scope (scope = screen or 'all' pooled)."""
    for m in metrics:
        V = {}
        for t in TAGS:
            U = units_t.get(t, []);
            if not U: continue
            idx = samp[t]
            def vec(st): return np.array([getv(st, t, U[i], m) for i in idx], float)
            for st in sets_single: V[(st, t)] = vec(st)
            for fam, names in sets_fam.items():
                per = [vec(n) for n in names]
                V[(fam + "3", t)], V[(fam + "3h", t)] = agg_family(per, sdraw)
                for n, v in zip(names, per): V[(n, t)] = v
        allsets = sorted(set(k[0] for k in V))
        for st in allsets:
            vs = [V[(st, t)] for t in TAGS if (st, t) in V]
            for t in TAGS:
                if (st, t) in V:
                    v = V[(st, t)]; f = np.isfinite(v)
                    if f.any(): S[f"mean|{prefix}|{st}|{m}|{t}"] = float(v[f].mean())
            if scopes and vs:
                v = np.concatenate(vs); f = np.isfinite(v)
                if f.any(): S[f"mean|{prefix}|{st}|{m}|all"] = float(v[f].mean())
        for a, b in comps:
            ds = []
            for t in TAGS:
                if (a, t) not in V or (b, t) not in V: continue
                d = V[(a, t)] - V[(b, t)]; f = np.isfinite(d); ds.append(d)
                if f.any(): S[f"diff|{prefix}|{a}|{b}|{m}|{t}"] = float(d[f].mean()); S[f"n|{prefix}|{a}|{b}|{m}|{t}"] = int(f.sum())
            if scopes and ds:
                d = np.concatenate(ds); f = np.isfinite(d)
                if f.any(): S[f"diff|{prefix}|{a}|{b}|{m}|all"] = float(d[f].mean()); S[f"n|{prefix}|{a}|{b}|{m}|all"] = int(f.sum())

def hcomps(pairs):   # add hierarchical versions for family-vs-family / family-vs-single
    out = list(pairs)
    for a, b in pairs:
        if a.endswith("3"): out.append((a + "h", b + "h" if b.endswith("3") else b))
    return out

def one_rep(b):
    rng = None if b == 0 else np.random.default_rng(100003 + b); rs = None if b == 0 else np.random.default_rng(700001 + b)
    R = realise_raw(rng); S = {"_b": b}
    sdraw = rs.choice(3, 3, replace=True) if rs is not None else np.arange(3)
    C0 = {t: cohort(R, t) for t in TAGS}; SX = {t: strat_ctx(R, t) for t in TAGS}
    # KO resampling: one draw per screen over the union universe of each section (drawn in fixed order from rng)
    def draw(units_t):
        return {t: (rng.choice(len(units_t[t]), len(units_t[t]), replace=True) if (rng is not None and len(units_t[t])) else np.arange(len(units_t[t]))) for t in TAGS}
    if "F3" in ONLY:
        T = {}
        for t in TAGS:
            C = C0[t]; cm, cs = SX[t]["cm"], SX[t]["cs"]
            pm = R[t]["Yp"].mean(0) if R[t]["Yp"].shape[1] else np.zeros(0); ps = R[t]["Yp"].std(0, ddof=1) if R[t]["Yp"].shape[1] else np.zeros(0)
            loc = dict(F3); loc["ctrl_mean"] = {tt: None for tt in TAGS}; loc["ctrl_sd"] = {tt: None for tt in TAGS}
            loc["ctrl_mean"][t] = {g: (np.abs(cm), np.abs(pm)) for g in C.genes}; loc["ctrl_sd"][t] = {g: (cs, ps) for g in C.genes}
            for nm, d in loc.items():
                if not d.get(t): continue
                ev = [g for g in C.evaluable if g in d[t]]
                sc = SL.score_set(C, d[t], ev, ev); ex = extra_metrics(C, d[t], ev, SX[t])
                for g in ev:
                    r = dict(sc.get(g, {})); r.update(ex.get(g, {})); T[(nm, t, g)] = r
        units = {t: sorted(set(g for (_, tt, g) in T if tt == t)) for t in TAGS}
        fams = {f: v for f, v in FAM3.items()}
        comps = [(f"{f}3", x) for f in fams for x in SINGLE3] + [(a + "3", b + "3") for a in fams for b in fams if a != b and (a, b) in
                 [("clamp", "rnaonly"), ("adapt", "clamp"), ("adapt", "rnaonly"), ("rnaonly", "clamp")]] + [("corr", "ctrl_mean"), ("corr", "ctrl_sd")] + [(a, b) for a in ["totalvi_ko", "totalvi_fresh"] for b in ["scvi_ko"] if a in F3 and b in F3]
        summarise_units(S, "F3", units, lambda st, t, g, m: T.get((st, t, g), {}).get(m, np.nan), fams, SINGLE3, M3, hcomps(comps), draw(units), sdraw)
    if "LG" in ONLY:
        T = {}
        lgsets = [n for f in ["clamp", "rnaonly"] if f in FAM3 for n in FAM3[f]] + ["corr"]
        for fr, fp in GRID:
            tag = f"r{fr}_p{fp}"
            for t in TAGS:
                C = C0[t] if (fr, fp) == (0.10, 0.05) else cohort(R, t, fr, fp)
                for nm in lgsets:
                    d = F3[nm][t]; ev = [g for g in C.evaluable if g in d]
                    for g, r in SL.score_set(C, d, ev, ev).items(): T[(tag, nm, t, g)] = r
            units = {t: sorted(set(g for (tg, _, tt, g) in T if tt == t and tg == tag)) for t in TAGS}
            fams = {f: FAM3[f] for f in ["clamp", "rnaonly"] if f in FAM3}
            comps = hcomps([(f + "3", "corr") for f in fams] + ([("clamp3", "rnaonly3")] if "rnaonly" in fams else []))
            summarise_units(S, f"LG_{tag}", units, lambda st, t, g, m: T.get((tag, st, t, g), {}).get(m, np.nan), fams, ["corr"], MLG, comps, draw(units), sdraw)
    if "F5" in ONLY or "ST4" in ONLY:
        for fam, bys in F5.items():
            if (fam == "st4") != ("ST4" in ONLY and fam == "st4") and "F5" not in ONLY: continue
            if fam == "st4" and "ST4" not in ONLY: continue
            # per seed: per-unit metric averaged over the runs in which that unit was held out
            U = {}   # (seed, arm, t, g) -> metric dict list
            for s in SEEDS:
                for run in bys[s]:
                    _, arms, train, test = run
                    for a, d in arms.items():
                        for t in TAGS:
                            if not d.get(t): continue
                            C = C0[t]; ev = set(C.evaluable)
                            tst = [g for g in test[t] if g in ev]; trn = [g for g in train[t] if g in ev]
                            for g, r in SL.score_set(C, d[t], tst, trn).items(): U.setdefault((s, a, t, g), []).append(r)
            def gv(st, t, g, m):
                s, a = st.split("@"); L = U.get((s, a, t, g))
                if not L: return np.nan
                v = np.array([x.get(m, np.nan) for x in L], float); v = v[np.isfinite(v)]
                return float(v.mean()) if len(v) else np.nan
            units = {t: sorted(set(g for (s, a, tt, g) in U if tt == t)) for t in TAGS}
            arms_ = ["baseline", "real", "shuffle_0", "shuffle_1", "shuffle_2", "mean"]
            # 'shuf' = mean of the 3 shuffles computed per seed via a pseudo-set
            def gv2(st, t, g, m):
                s, a = st.split("@")
                if a == "shuf": return np.nanmean([gv(f"{s}@shuffle_{k}", t, g, m) for k in range(3)]) if any(np.isfinite(gv(f"{s}@shuffle_{k}", t, g, m)) for k in range(3)) else np.nan
                return gv(st, t, g, m)
            fams = {f"{fam}:{a}": [f"{s}@{a}" for s in SEEDS] for a in ["baseline", "real", "shuf", "mean"]}
            comps = hcomps([(f"{fam}:real3", f"{fam}:baseline3"), (f"{fam}:real3", f"{fam}:shuf3"), (f"{fam}:real3", f"{fam}:mean3"),
                            (f"{fam}:shuf3", f"{fam}:baseline3"), (f"{fam}:mean3", f"{fam}:baseline3")])
            summarise_units(S, f"F5_{fam}", units, gv2, fams, [], M5 if fam != "st4" else ["rr_pe_cent", "rr_retrieval", "rr_pe"], comps, draw(units), sdraw)
        # cross-family contrasts on the paper split (paired vs rnaonly; adapt vs unadpt): common unit draw
        for fa, fb in [("paired", "rnaonly"), ("adapt", "unadpt"), ("adapt", "paired")]:
            if "F5" not in ONLY or fa not in F5 or fb not in F5: continue
            UU = {}
            for fam in (fa, fb):
                for s in SEEDS:
                    _, arms, train, test = F5[fam][s][0]
                    for a, d in arms.items():
                        for t in TAGS:
                            if not d.get(t): continue
                            C = C0[t]; ev = set(C.evaluable)
                            tst = [g for g in test[t] if g in ev]; trn = [g for g in train[t] if g in ev]
                            for g, r in SL.score_set(C, d[t], tst, trn).items(): UU[(fam, s, a, t, g)] = r
            def gx(st, t, g, m):
                fam, s, a = st.split("@")
                if a == "shuf":
                    v = [UU.get((fam, s, f"shuffle_{k}", t, g), {}).get(m, np.nan) for k in range(3)]
                    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                if a == "rs":   # real - shuffled (per unit)
                    return gx(f"{fam}@{s}@real", t, g, m) - gx(f"{fam}@{s}@shuf", t, g, m)
                if a == "rb":   # real - baseline
                    return gx(f"{fam}@{s}@real", t, g, m) - gx(f"{fam}@{s}@baseline", t, g, m)
                return UU.get((fam, s, a, t, g), {}).get(m, np.nan)
            units = {t: sorted(set(g for (f, s, a, tt, g) in UU if tt == t)) for t in TAGS}
            fams = {f"{f}:{a}": [f"{f}@{s}@{a}" for s in SEEDS] for f in (fa, fb) for a in ["baseline", "real", "shuf", "rs", "rb"]}
            comps = hcomps([(f"{fa}:{a}3", f"{fb}:{a}3") for a in ["baseline", "real", "shuf", "rs", "rb"]])
            summarise_units(S, f"X5_{fa}_{fb}", units, gx, fams, [], M5, comps, draw(units), sdraw)
    return S

keys = None; rows = []
reps = list(range(TASK * NREP, (TASK + 1) * NREP))
if 0 not in reps: reps = [0] + reps    # point estimate defines the key set in every task
for b in reps:
    S = one_rep(b)
    if keys is None: keys = sorted(k for k in S if k != "_b")
    rows.append([S.get(k, np.nan) for k in keys]); log("rep", b, "nkeys", len(S))
V = np.array(rows, dtype=np.float64); bs = np.array(reps)
np.savez_compressed(f"{OUTD}/boot_{TASK:03d}.npz", V=V, b=bs, keys=np.array(keys))
log("WROTE")
