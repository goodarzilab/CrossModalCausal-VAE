"""Training-corpus (809,130-cell) statistics for information-matched baselines.

Writes paper_nov_revisions/analyses/baselines/train_stats/:
  joint_stats.npz : genes, prots, mean/std (log-norm RNA, CLR protein), C (Pearson, joint RNA+protein, all cells),
                    P_lw (Ledoit-Wolf partial correlation; shrinkage delta estimated on a 200k-cell subsample and
                    applied to the full-data correlation), C_ct (within cell-type x cohort correlation: per-stratum
                    means removed), P_ct (LW partial correlation of the within-stratum covariance),
                    P_prot_glasso (graphical-lasso CV partial correlation, protein panel only, 150k cells)
  subsample.json  : regulon/STRING AUROC of |corr| and |partial corr| from n-cell subsamples (3 reps) and
                    top-1% edge Jaccard between replicate subsamples, cell-disjoint halves and donor-disjoint halves.
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, time
import numpy as np, anndata as ad
from sklearn.covariance import LedoitWolf, GraphicalLassoCV
from sklearn.metrics import roc_auc_score

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
R = f"{_CMC_ROOT}"
D = f"{_CMC_DATA}/combined_v4"
OUT = f"{_CMC_RESULTS}/paper/analyses/baselines/train_stats"; os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(0)

ar = ad.read_h5ad(f"{D}/rna_combined.h5ad", backed="r"); ap = ad.read_h5ad(f"{D}/citeseq_combined.h5ad", backed="r")
assert (ar.obs_names == ap.obs_names).all()
genes = list(ar.var_names.astype(str)); prots = list(ap.var_names.astype(str)); NG, NP = len(genes), len(prots)
n = ar.n_obs
strata = (ar.obs["dataset"].astype(str) + "|" + ar.obs["celltype"].astype(str)).values
donor = ar.obs["donor"].astype(str).values
log(f"{n} cells, {NG} genes, {NP} proteins, {len(np.unique(strata))} strata, {len(np.unique(donor))} donors")
X = np.empty((n, NG + NP), np.float32)
CH = 50000
for i in range(0, n, CH):
    xc = ar.X[i:i + CH]; xc = xc.toarray() if hasattr(xc, "toarray") else np.asarray(xc)
    X[i:i + CH, :NG] = xc
    pc = ap.X[i:i + CH]; pc = pc.toarray() if hasattr(pc, "toarray") else np.asarray(pc)
    X[i:i + CH, NG:] = pc
log("loaded X", X.shape)
mu = X.mean(0, dtype=np.float64); sd = X.std(0, dtype=np.float64); sd[sd < 1e-9] = 1e-9

def corr_of(idx=None, Z=None):
    """Pearson correlation of the rows idx of X (or of Z) via float64 Gram accumulation."""
    M = X if Z is None else Z
    ii = np.arange(M.shape[0]) if idx is None else np.sort(idx)
    s1 = np.zeros(M.shape[1]); S = np.zeros((M.shape[1], M.shape[1]))
    for k in range(0, len(ii), CH):
        xc = M[ii[k:k + CH]].astype(np.float64); s1 += xc.sum(0); S += xc.T @ xc
    m = len(ii); cov = (S - np.outer(s1, s1) / m) / (m - 1)
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None)); return np.nan_to_num(cov / np.outer(d, d)), cov

def lw_delta(idx):
    Z = (X[np.sort(idx)] - mu) / sd
    lw = LedoitWolf(assume_centered=True, block_size=2000).fit(Z.astype(np.float64)); return float(lw.shrinkage_)

def partial_from_corr(C, delta):
    S = (1 - delta) * C + delta * np.eye(C.shape[0])            # LW target = mu*I with mu = mean(diag)=1 for a correlation
    P = np.linalg.inv(S); d = np.sqrt(np.diag(P)); pc = -P / np.outer(d, d); np.fill_diagonal(pc, 0.0); return pc

C, _ = corr_of(); log("full corr done")
sub = rng.choice(n, 200000, replace=False)
delta = lw_delta(sub); log("LW shrinkage on 200k subsample:", delta)
P_lw = partial_from_corr(C, delta); log("partial corr done")

# within cell-type x cohort: remove stratum means, then correlation
Xc = X.copy()
for s in np.unique(strata):
    m = strata == s; Xc[m] -= Xc[m].mean(0, keepdims=True)
C_ct, _ = corr_of(Z=Xc); log("within-stratum corr done")
P_ct = partial_from_corr(C_ct, delta); del Xc

# protein-only graphical lasso (as in analysis/fig2_networks/ppi_surface_complexes.py)
try:
    sp = np.sort(rng.choice(n, 150000, replace=False)); Zp = (X[sp, NG:] - mu[NG:]) / sd[NG:]
    gl = GraphicalLassoCV(alphas=6, cv=3, max_iter=200).fit(Zp.astype(np.float64))
    Pp = gl.precision_; d = np.sqrt(np.diag(Pp)); P_prot_glasso = -Pp / np.outer(d, d); np.fill_diagonal(P_prot_glasso, 0.0)
    gl_alpha = float(gl.alpha_)
except Exception as e:
    log("glasso failed", e); P_prot_glasso = np.full((NP, NP), np.nan); gl_alpha = float("nan")
log("protein glasso alpha", gl_alpha)
np.savez(f"{OUT}/joint_stats.npz", genes=np.array(genes), prots=np.array(prots), mean=mu, std=sd, C=C.astype(np.float32),
         P_lw=P_lw.astype(np.float32), C_ct=C_ct.astype(np.float32), P_ct=P_ct.astype(np.float32),
         P_prot_glasso=P_prot_glasso.astype(np.float32), lw_delta=delta, glasso_alpha=gl_alpha, n_cells=n)
log("saved joint_stats.npz")

# ------------------------------------------------------------------ subsampling / reproducibility of correlation
ref = np.load(f"{_CMC_RESULTS}/ref_nets_v3_2c.npz", allow_pickle=True); meta = json.load(open(f"{_CMC_RESULTS}/ref_nets_v3_2c.json"))
rg = list(map(str, ref["genes"])); ridx = {g: i for i, g in enumerate(rg)}; gi = {g: i for i, g in enumerate(genes)}
common = [g for g in rg if g in gi]; ci = np.array([gi[g] for g in common]); ri = np.array([ridx[g] for g in common])
off = ~np.eye(len(common), dtype=bool)
LAB = {}
for rn in ["STRING", "DoRothEA", "CollecTRI"]:
    mk = ref[rn][np.ix_(ri, ri)].astype(bool)
    LAB[rn] = (mk if meta[rn]["directed"] else (mk | mk.T))[off]
def rec(M):
    Ms = np.abs(M[np.ix_(ci, ci)]); out = {}
    for rn, lab in LAB.items():
        s = Ms[off] if meta[rn]["directed"] else (Ms + Ms.T)[off]
        out[rn] = float(roc_auc_score(lab, s))
    return out
offG = ~np.eye(NG, dtype=bool)
def topset(M, frac=0.01):
    a = np.abs(M[:NG, :NG])[offG]; k = int(round(frac * a.size)); return set(np.argpartition(a, -k)[-k:].tolist())
def jac(a, b): return len(a & b) / len(a | b)

res = dict(full=dict(corr=rec(C), partial_lw=rec(P_lw), corr_within_ct=rec(C_ct), partial_within_ct=rec(P_ct)), lw_delta=delta, subsample={})
log("full-data recovery", res["full"])
for nn in [500, 2000, 10000, 50000]:
    reps = []; tops = []
    for r in range(3):
        idx = np.random.RandomState(100 + r + nn).choice(n, nn, replace=False)
        Cs, _ = corr_of(idx); Ps = partial_from_corr(Cs, lw_delta(idx))
        reps.append(dict(corr=rec(Cs), partial=rec(Ps))); tops.append(topset(Cs))
    J = [jac(tops[a], tops[b]) for a, b in [(0, 1), (0, 2), (1, 2)]]
    res["subsample"][str(nn)] = dict(reps=reps, top1pct_jaccard_corr=J); log(nn, reps[0], "jac", np.round(J, 3))
# cell-disjoint halves and donor-disjoint halves
perm = rng.permutation(n); h1, h2 = perm[:n // 2], perm[n // 2:]
Ca, _ = corr_of(h1); Cb, _ = corr_of(h2)
res["halves_cells"] = dict(top1pct_jaccard=jac(topset(Ca), topset(Cb)), rec_a=rec(Ca), rec_b=rec(Cb),
                           spearman_offdiag=float(np.corrcoef(np.argsort(np.argsort(Ca[:NG, :NG][offG])), np.argsort(np.argsort(Cb[:NG, :NG][offG])))[0, 1]))
ud = np.unique(donor); pd_ = rng.permutation(ud); da = set(pd_[:len(ud) // 2])
ma = np.array([d in da for d in donor])
Ca, _ = corr_of(np.where(ma)[0]); Cb, _ = corr_of(np.where(~ma)[0])
res["halves_donors"] = dict(top1pct_jaccard=jac(topset(Ca), topset(Cb)), n_a=int(ma.sum()), n_b=int((~ma).sum()), rec_a=rec(Ca), rec_b=rec(Cb))
# also donor-disjoint within the Hao cohort only (8 donors; 4 vs 4) -- the small-cohort regime
hao = np.array([d.startswith("hao") for d in donor]); hd = sorted(set(donor[hao])); hA = set(hd[::2])
ia = np.where(hao & np.isin(donor, list(hA)))[0]; ib = np.where(hao & ~np.isin(donor, list(hA)))[0]
Ca, _ = corr_of(ia); Cb, _ = corr_of(ib)
res["halves_donors_hao"] = dict(top1pct_jaccard=jac(topset(Ca), topset(Cb)), n_a=len(ia), n_b=len(ib))
log("halves", {k: res[k]["top1pct_jaccard"] for k in ["halves_cells", "halves_donors", "halves_donors_hao"]})
json.dump(res, open(f"{OUT}/subsample.json", "w"), indent=1)
log("DONE")
