"""Task 9: gene-block (node) bootstrap and within-gene label-permutation tests for the Fig 2a/b AUROC differences.

Statistic (unchanged from Fig 2a/b): AUROC of |map| for reference pairs (positives) vs all other ordered off-diagonal pairs
(negatives) in the same universe.
  2a  universe = genes in both the model vocabulary and the reference resource; STRING / co-expression (undirected): score
      M[i,j]+M[j,i]; DoRothEA / CollecTRI (directed): score M[i,j] (regulator i -> target j).
  2b  universe = all model genes; positives = ordered pairs of distinct members of an immune program; score M[i,j].
Resampling unit = GENE (node), never the edge:
  node    each replicate draws gene multiplicities w ~ Multinomial(N_genes, 1/N_genes) over the full vocabulary (one draw
          shared by every map, reference and program -> joint, paired); pair (i,j) gets weight w_i*w_j and BOTH positives
          and negatives are re-weighted, so the AUROC is recomputed on the resampled gene set.
  source  (2a only) regulator-block variant: weight w_i on the source/regulator gene only (targets fixed).
AUROCs are computed from 65,536 rank-quantile bins of each map's score within the universe (ties within a bin count 1/2;
bin error < 1e-4, checked against the exact point estimates). Replicate b uses default_rng(910000+b); b=0 = point estimate.
perm    within-gene label permutation for a contrast (A vs B): both maps are rank-quantile transformed within the universe
        (common marginal); for each gene i independently with prob. 1/2 the score rows of A and B (all pairs with source i)
        are swapped; d_perm = AUROC(A') - AUROC(B'). Two-sided P = (1 + #{|d_perm| >= |d_obs|}) / (B + 1).
usage: python -I task9_gbboot.py MODE(boot|perm) TASK NREP OUTDIR MAPS.npz [MAPS2.npz ...]
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
import sys, os, json, time, numpy as np
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
MODE, TASK, NREP, OUTD = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; MAPF = sys.argv[5:]
os.makedirs(OUTD, exist_ok=True)
R = f"{_CMC_ROOT}"; NBIN = 65536
JS = np.load(f"{_CMC_RESULTS}/paper/analyses/baselines/train_stats/joint_stats.npz", allow_pickle=True)
genes = list(map(str, JS["genes"])); NG = len(genes); gi = {g: i for i, g in enumerate(genes)}
MAPS = {}
for k, key in [("corr", "C"), ("partial_corr", "P_lw"), ("corr_within_ct", "C_ct"), ("partial_within_ct", "P_ct")]:
    M = np.abs(JS[key][:NG, :NG].astype(np.float64)); np.fill_diagonal(M, 0); MAPS[k] = M
for f in MAPF:
    z = np.load(f, allow_pickle=True); assert list(map(str, z["genes"])) == genes
    for k in z.files:
        if k != "genes": MAPS[k] = z[k].astype(np.float64)
NAMES = list(MAPS); log("maps", NAMES)
ref = np.load(f"{_CMC_RESULTS}/ref_nets_v3_2c.npz", allow_pickle=True)
rg = list(map(str, ref["genes"])); ridx = {g: i for i, g in enumerate(rg)}
common = [g for g in rg if g in gi]; ci = np.array([gi[g] for g in common]); ri = np.array([ridx[g] for g in common]); nc = len(common)
REFS = {"coexpr": False, "STRING": False, "DoRothEA": True, "CollecTRI": True}
offc = ~np.eye(nc, dtype=bool); offG = ~np.eye(NG, dtype=bool)
POS = {}
for rn, directed in REFS.items():
    mk = ref[rn][np.ix_(ri, ri)].astype(bool); lab = (mk if directed else (mk | mk.T)) & offc
    POS[rn] = np.flatnonzero(lab)
pw = json.load(open(f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000/pathway_recovery.json"))["pathways"]
PROG = {}
for nm, d in pw.items():
    mem = np.array(sorted({gi[g] for g in d["members"] if g in gi})); inp = np.zeros(NG, bool); inp[mem] = True
    PROG[nm] = np.flatnonzero(np.outer(inp, inp) & offG)
def qbins(score, offmask):
    s = score.ravel(); o = np.argsort(s[offmask.ravel()], kind="stable"); E = len(o)
    b = np.zeros(s.size, np.uint16); idx = np.flatnonzero(offmask.ravel())
    r = np.empty(E, np.int64); r[o] = np.arange(E)
    # tie-aware: equal scores share the bin of their first rank
    ss = s[idx][o]; first = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]; grp = np.repeat(first, np.diff(np.r_[first, E])); r[o] = grp
    b[idx] = (r * NBIN // E).astype(np.uint16); return b
BINS = {}
for m, M in MAPS.items():
    Mc = M[np.ix_(ci, ci)]
    BINS[(m, "sym")] = qbins(Mc + Mc.T, offc); BINS[(m, "dir")] = qbins(Mc, offc); BINS[(m, "full")] = qbins(M, offG)
log("binned; nc", nc, "NG", NG, {k: len(v) for k, v in POS.items()})
UNIV = {"coexpr": "sym", "STRING": "sym", "DoRothEA": "dir", "CollecTRI": "dir"}
def auc_h(hp, hall):
    hn = hall - hp; cum = np.cumsum(hn) - hn; den = hp.sum() * hn.sum()
    return float((hp * (cum + 0.5 * hn)).sum() / den) if den > 0 else np.nan
def aucs_weighted(Wc, Wf, variant):
    """returns dict (map, item) -> auroc for weights Wc (nc*nc flat) and Wf (NG*NG flat; None to skip programs)."""
    out = {}
    for m in NAMES:
        for u in ("sym", "dir"):
            b = BINS[(m, u)]; hall = np.bincount(b, weights=Wc, minlength=NBIN)
            for rn in REFS:
                if UNIV[rn] != u: continue
                p = POS[rn]; out[(m, rn)] = auc_h(np.bincount(b[p], weights=Wc[p], minlength=NBIN), hall)
        if Wf is not None:
            b = BINS[(m, "full")]; hall = np.bincount(b, weights=Wf, minlength=NBIN)
            for nm, p in PROG.items(): out[(m, nm)] = auc_h(np.bincount(b[p], weights=Wf[p], minlength=NBIN), hall)
    return out
mc = offc.ravel().astype(float); mf = offG.ravel().astype(float)
ITEMS = list(REFS) + list(PROG)
res = []
if MODE == "boot":
    for rep in range(NREP):
        b = TASK * NREP + rep
        if b == 0: w = np.ones(NG)
        else: w = np.random.default_rng(910000 + b).multinomial(NG, np.full(NG, 1.0 / NG)).astype(float)
        wc = w[ci]
        A = aucs_weighted(np.outer(wc, wc).ravel() * mc, np.outer(w, w).ravel() * mf, "node")
        S = aucs_weighted(np.outer(wc, np.ones(nc)).ravel() * mc, None, "source")
        res.append(dict(b=b, node={f"{m}|{it}": v for (m, it), v in A.items()}, source={f"{m}|{it}": v for (m, it), v in S.items()}))
        if rep % 10 == 0: log("rep", b)
else:   # perm
    CMCs = sorted(k for k in NAMES if k.startswith("CMC_")); RNAs = sorted(k for k in NAMES if k.startswith("RNA_"))
    BLs = ["corr", "partial_corr", "corr_within_ct", "partial_within_ct"]
    PAIRS = [(c, bl) for c in CMCs for bl in BLs] + [(c, "RNA_" + c.split("_")[1]) for c in CMCs if "RNA_" + c.split("_")[1] in NAMES] + \
            [(r, bl) for r in RNAs for bl in BLs]
    ones_c = mc; ones_f = mf
    def contrast(a, bm, swap):
        """d = AUROC(a') - AUROC(b') for row-swap mask over genes (None = observed)."""
        o = {}
        for u, n, msk, items, wts in [("sym", nc, offc, [r for r in REFS if UNIV[r] == "sym"], ones_c), ("dir", nc, offc, [r for r in REFS if UNIV[r] == "dir"], ones_c),
                                      ("full", NG, offG, list(PROG), ones_f)]:
            Ba, Bb = BINS[(a, u)].reshape(n, n), BINS[(bm, u)].reshape(n, n)
            if swap is not None:
                s = swap[ci] if n == nc else swap
                if u == "sym":   # symmetric score: swap whole (i, j) pair blocks by source gene i (ordered pairs)
                    pass
                Aa = np.where(s[:, None], Bb, Ba).ravel(); Ab = np.where(s[:, None], Ba, Bb).ravel()
            else: Aa, Ab = Ba.ravel(), Bb.ravel()
            ha, hb = np.bincount(Aa, weights=wts, minlength=NBIN), np.bincount(Ab, weights=wts, minlength=NBIN)
            for it in items:
                p = (POS[it] if u != "full" else PROG[it])
                o[it] = auc_h(np.bincount(Aa[p], minlength=NBIN).astype(float), ha) - auc_h(np.bincount(Ab[p], minlength=NBIN).astype(float), hb)
        return o
    for rep in range(NREP):
        b = TASK * NREP + rep
        sw = None if b == 0 else (np.random.default_rng(920000 + b).random(NG) < 0.5)
        res.append(dict(b=b, d={f"{a}|{bm}|{it}": v for a, bm in PAIRS for it, v in contrast(a, bm, sw).items()}))
        if rep % 10 == 0: log("perm", b)
json.dump(res, open(f"{OUTD}/{MODE}_{TASK:03d}.json", "w"))
log("WROTE")
