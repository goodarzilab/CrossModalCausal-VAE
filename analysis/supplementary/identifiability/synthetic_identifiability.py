"""SYNTHETIC IDENTIFIABILITY PROBE  (G4 Part 1 — gate for a real-data non-Gaussian sweep)

Rigorously demonstrate the identifiability boundary for OUR model class:
    latent SCM   z = (I - A^T)^{-1} eps      (linear; src/model.py BlockStructuredScmLayer)
    decoder      x = D z + obs_noise         (random linear decoder, G >> P genes)
    read-out     G_rr = D A D^T              (the second-order / covariance-dominated projection)

Core claim under test (Shimizu 2006 LiNGAM; PRIORS_FOR_DIRECTION.md sec. D):
  * a linear SCM with GAUSSIAN eps is direction-UNIDENTIFIABLE  -> our observed asymmetry ~= 1.0 /
    pearDE ~= 0 is THEORY, not a bug;
  * with NON-GAUSSIAN eps the orientation becomes recoverable by a higher-order (ICA / LiNGAM) method,
    but NOT by a second-order read-out (which is the class our G_rr = D A D^T mutilation read-out is in).

We sweep the noise family (a continuous non-Gaussianity / excess-kurtosis axis), the sample size n, and
several random DAGs, and for each cell compare FIVE orientation read-outs on the latent z and on the
decode->recover latent z_hat = D^+ x:

  1. secondorder_partialcorr   precision / partial-correlation (pure 2nd-order)     -> CAN NEVER orient
  2. pairwise_gaussian         our model class, GAUSSIAN eps prior  (||eps||^2)      -> 2nd-order == chance
  3. pairwise_laplace          our model class, LAPLACE  eps prior  (||eps||_1)      -> higher-order
  4. pairwise_entropy          LiNGAM pairwise likelihood ratio (Hyvarinen entropy)  -> higher-order
  5. directlingam              full DirectLiNGAM (Shimizu 2011), skeleton+order      -> higher-order

Read-outs 2/3/4 are GIVEN the true skeleton (orientation-only, the fairest analog of our asymmetry test
which is computed on known CollecTRI directed edges). DirectLiNGAM discovers skeleton+order from scratch.
2 vs 3/4 isolates the epsilon-PRIOR lever; 1 vs 3/4 on the SAME non-Gaussian data isolates the
READ-OUT/LOSS lever -> this answers the gate's part (c): what must Part 2 change.

lingam is not installed in base env -> DirectLiNGAM + the Hyvarinen entropy measure are implemented here.

Outputs:
  results/synthetic_identifiability.json
  results/figures/slides/fig_synthetic_identifiability.png
Usage:
  python analysis/supplementary/identifiability/synthetic_identifiability.py            # full sweep
  python analysis/supplementary/identifiability/synthetic_identifiability.py --quick     # fast smoke
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
import os, sys, json, time, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, pearsonr, kurtosis

HERE = os.path.dirname(os.path.abspath(__file__))
RES = _CMC_RESULTS
OUTFIG = os.path.join(RES, "figures", "slides")
os.makedirs(OUTFIG, exist_ok=True)

# ============================================================================
# 1. SIMULATION  (matches src/model.py conventions)
#    B[i,j] = weight of edge j->i  (LiNGAM convention x = B x + e).  In model.py the latent adjacency
#    is A_full with A_full[j,i] = weight j->i, i.e. B = A_full^T; both give z = (I - A^T)^{-1} eps.
# ============================================================================
def random_dag(P, parents_per_node, wlo, whi, rng):
    """Strictly-lower-triangular B in a random topological order. B[i,j] = weight parent j -> child i."""
    order = rng.permutation(P)                      # topological order (order[0] is a root)
    pos = np.empty(P, dtype=int); pos[order] = np.arange(P)   # pos[v] = rank of v in the order
    B = np.zeros((P, P))
    for c in range(P):                              # for each node, pick parents strictly earlier in order
        earlier = order[:pos[c]]
        if len(earlier) == 0:
            continue
        k = min(parents_per_node, len(earlier))
        pa = rng.choice(earlier, size=k, replace=False)
        w = rng.uniform(wlo, whi, size=k) * rng.choice([-1.0, 1.0], size=k)
        B[c, pa] = w
    return B

def gen_noise(family, param, n, P, rng):
    """Return eps [n,P] standardized to ~unit variance and its mean measured excess kurtosis."""
    if family == "gaussian":
        e = rng.standard_normal((n, P))
    elif family == "uniform":
        e = rng.uniform(-1.0, 1.0, (n, P)) * np.sqrt(3.0)          # excess kurtosis -1.2
    elif family == "laplace":
        e = rng.laplace(0.0, 1.0 / np.sqrt(2.0), (n, P))            # excess kurtosis 3.0
    elif family == "exp":
        e = rng.standard_exponential((n, P)) - 1.0                  # skewed, excess kurtosis 6.0
    elif family == "t":
        e = rng.standard_t(param, (n, P)) / np.sqrt(param / (param - 2.0))  # excess kurtosis 6/(df-4)
    else:
        raise ValueError(family)
    exk = float(np.mean(kurtosis(e, axis=0, fisher=True)))
    return e, exk

def simulate_latent(B, eps):
    """z = (I - A^T)^{-1} eps.  Row convention: Z = E @ inv(I - B).T  (see model.py _scm_solve)."""
    P = B.shape[0]
    M = np.linalg.inv(np.eye(P) - B)          # (I-B)^{-1}
    return eps @ M.T

def decode_recover(Z, G, obs_noise_frac, rng):
    """x = D z + obs_noise (random linear decoder, G genes) ; recover z_hat = D^+ x.
    Tests whether the decoder mixing D (an extra linear mixing, like ICA's) destroys latent identifiability."""
    P = Z.shape[1]
    D = rng.standard_normal((G, P))                 # decoder [G, P] (genes x latents)
    X = Z @ D.T                                      # [n, G]
    sd = obs_noise_frac * X.std()
    X = X + rng.standard_normal(X.shape) * sd
    Zhat = X @ np.linalg.pinv(D).T                  # (pinv(D) @ X.T).T ; = Z + projected obs noise
    return Zhat

def edge_list(B):
    """True directed edges as (child c, parent p): B[c,p] != 0 means p -> c."""
    cc, pp = np.where(np.abs(B) > 1e-9)
    return list(zip(cc.tolist(), pp.tolist()))

# ============================================================================
# 2. HYVARINEN ENTROPY + DirectLiNGAM (Shimizu 2011) — kernel-free 'pwling' measure
# ============================================================================
_K1, _K2, _GAMMA, _HG = 79.047, 7.4129, 0.37457, 0.5 * (1.0 + np.log(2 * np.pi))
def _entropy(u):
    """Hyvarinen (1998) differential-entropy approximation for zero-mean unit-variance u."""
    return _HG - _K1 * (np.mean(np.log(np.cosh(u))) - _GAMMA) ** 2 \
               - _K2 * (np.mean(u * np.exp(-0.5 * u * u))) ** 2

def _residual(xi, xj):
    """Residual of xi after regressing on xj (both zero-mean)."""
    return xi - (np.dot(xi, xj) / np.dot(xj, xj)) * xj

def _std(u):
    return (u - u.mean()) / (u.std() + 1e-12)

def _diff_mi(xi, xj):
    """LiNGAM pairwise likelihood ratio T (Hyvarinen & Smith 2013). T>0 => xi -> xj (xi is the cause)."""
    ri_j = _residual(xi, xj); rj_i = _residual(xj, xi)
    return (_entropy(xj) + _entropy(_std(ri_j))) - (_entropy(xi) + _entropy(_std(rj_i)))

def direct_lingam(X):
    """DirectLiNGAM (Shimizu 2011), pwling entropy measure. Returns (causal_order, B_hat[i,j]=j->i)."""
    n, P = X.shape
    Xc = X - X.mean(0)
    Xs = Xc / (Xc.std(0) + 1e-12)                  # standardized copy for the independence search
    U = list(range(P)); order = []
    work = Xs.copy()
    for _ in range(P):
        if len(U) == 1:
            order.append(U[0]); break
        Ms = []
        for i in U:
            M = 0.0
            for j in U:
                if i == j:
                    continue
                t = _diff_mi(work[:, i], work[:, j])
                M += min(0.0, t) ** 2
            Ms.append(-M)
        root = U[int(np.argmax(Ms))]
        order.append(root)
        for i in U:                                 # regress the chosen root out of the remaining vars
            if i != root:
                work[:, i] = _std(_residual(work[:, i], work[:, root]))
        U.remove(root)
    # adjacency by regressing each var on its predecessors in the causal order (original scale)
    B = np.zeros((P, P))
    for k, c in enumerate(order):
        pa = order[:k]
        if pa:
            coef, *_ = np.linalg.lstsq(Xc[:, pa], Xc[:, c], rcond=None)
            B[c, pa] = coef
    return order, B

# ============================================================================
# 3. READ-OUTS   (each returns a directed adjacency estimate B_hat[i,j] = weight j->i)
# ============================================================================
def readout_partialcorr(Z, edges):
    """Pure 2nd-order: partial correlation from the precision matrix. Symmetric by construction ->
    identical for i->j and j->i -> can NEVER orient (this is exactly why G_rr = D A D^T is orientation-
    blind). Filled only on the true skeleton for a like-for-like comparison with the pairwise read-outs."""
    S = np.cov(Z, rowvar=False)
    Th = np.linalg.pinv(S)
    d = np.sqrt(np.clip(np.diag(Th), 1e-12, None))
    Pc = -Th / np.outer(d, d)                       # partial correlation (symmetric)
    B = np.zeros_like(Pc)
    for (c, p) in edges:
        B[c, p] = abs(Pc[c, p]); B[p, c] = abs(Pc[p, c])   # equal => tie
    return B

def readout_pairwise(Z, edges, score, rng):
    """Orientation-only on the KNOWN skeleton. For each true edge p->c decide p->c vs c->p by `score`:
      gaussian : residual variance (2nd-order Gaussian prior ||eps||^2)   -> EXACT tie (var identical)
      laplace  : mean |residual|   (Laplace prior ||eps||_1)              -> orients super-Gaussian noise
      entropy  : LiNGAM pairwise likelihood ratio (adaptive)             -> orients super/sub-Gaussian
    Stores the chosen-direction regression coefficient in B_hat."""
    B = np.zeros((Z.shape[1], Z.shape[1]))
    for (c, p) in edges:
        xc = _std(Z[:, c]); xp = _std(Z[:, p])
        r_cp = _residual(xc, xp)                     # model p->c  (child regressed on parent)
        r_pc = _residual(xp, xc)                     # model c->p  (parent regressed on child)
        b_cp = np.dot(xc, xp) / np.dot(xp, xp)       # coef of p when predicting c  (edge p->c)
        b_pc = np.dot(xp, xc) / np.dot(xc, xc)       # coef of c when predicting p  (edge c->p)
        if score == "gaussian":
            s_fwd, s_rev = np.var(r_cp), np.var(r_pc)          # identical up to fp -> tie
        elif score == "laplace":
            s_fwd, s_rev = np.mean(np.abs(r_cp)), np.mean(np.abs(r_pc))
        elif score == "entropy":
            t = _diff_mi(xp, xc)                     # T>0 => p->c correct ; pick smaller "cost" = -t
            s_fwd, s_rev = -t, t
        else:
            raise ValueError(score)
        if abs(s_fwd - s_rev) < 1e-12:               # genuine tie (Gaussian) -> random, i.e. chance
            pick_fwd = rng.random() < 0.5
        else:
            pick_fwd = s_fwd < s_rev
        if pick_fwd:
            B[c, p] = b_cp                            # oriented p->c
        else:
            B[p, c] = b_pc                            # oriented c->p
    return B

# ============================================================================
# 4. METRICS  (parallel to analysis/common/causal_eval_suite.py asymmetry + a pearDE analog)
# ============================================================================
def orient_metrics(Bhat, Btrue, edges):
    fwd = np.array([abs(Bhat[c, p]) for (c, p) in edges])   # recovered weight in TRUE direction p->c
    rev = np.array([abs(Bhat[p, c]) for (c, p) in edges])   # recovered weight in REVERSE
    acc = float(np.mean(np.where(fwd > rev, 1.0, np.where(fwd < rev, 0.0, 0.5))))
    ratio = float(np.median(fwd) / (np.median(rev) + 1e-12))
    try:
        _, mwp = mannwhitneyu(fwd, rev, alternative="greater")
    except Exception:
        mwp = 1.0
    s_hat = np.array([Bhat[c, p] - Bhat[p, c] for (c, p) in edges])   # signed directed asymmetry
    s_true = np.array([Btrue[c, p] for (c, p) in edges])             # true signed weight (reverse=0)
    if s_hat.std() < 1e-12 or s_true.std() < 1e-12:
        pde = 0.0
    else:
        pde = float(pearsonr(s_hat, s_true)[0])
    return dict(orient_acc=acc, asym_ratio=ratio, asym_mwu_p=float(mwp), pearDE=pde)

def order_accuracy(order, edges):
    """Fraction of true edges p->c whose discovered causal order places parent before child."""
    pos = {v: i for i, v in enumerate(order)}
    ok = [pos[p] < pos[c] for (c, p) in edges]
    return float(np.mean(ok)) if ok else float("nan")

# ============================================================================
# 5. SWEEP
# ============================================================================
def run_cell(family, param, n, P, G, parents, obs_noise, seed):
    rng = np.random.default_rng(seed)
    B = random_dag(P, parents, 0.3, 0.8, rng)
    eps, exk = gen_noise(family, param, n, P, rng)
    Z = simulate_latent(B, eps)
    Zhat = decode_recover(Z, G, obs_noise, rng)
    edges = edge_list(B)
    out = {"family": family, "param": param, "n": n, "P": P, "G": G, "seed": seed,
           "excess_kurtosis": exk, "n_edges": len(edges)}

    # --- read-outs on the LATENT z (identifiability theory lives here) ---
    out["secondorder_partialcorr"] = orient_metrics(readout_partialcorr(Z, edges), B, edges)
    out["pairwise_gaussian"] = orient_metrics(readout_pairwise(Z, edges, "gaussian", rng), B, edges)
    out["pairwise_laplace"]  = orient_metrics(readout_pairwise(Z, edges, "laplace",  rng), B, edges)
    out["pairwise_entropy"]  = orient_metrics(readout_pairwise(Z, edges, "entropy",  rng), B, edges)
    order, Bdl = direct_lingam(Z)
    m = orient_metrics(Bdl, B, edges); m["order_acc"] = order_accuracy(order, edges)
    out["directlingam"] = m

    # --- transfer: decode -> recover latent, does the decoder mixing D destroy identifiability? ---
    out["secondorder_recovered"] = orient_metrics(readout_partialcorr(Zhat, edges), B, edges)
    order_h, Bdl_h = direct_lingam(Zhat)
    mh = orient_metrics(Bdl_h, B, edges); mh["order_acc"] = order_accuracy(order_h, edges)
    out["directlingam_recovered"] = mh
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke sweep")
    args = ap.parse_args()

    if args.quick:
        families = [("gaussian", None), ("laplace", None), ("t", 8)]
        ns = [2000, 10000]; seeds = [0, 1]; P, G, parents = 24, 300, 2
        p64 = []
    else:
        families = [("uniform", None), ("gaussian", None), ("t", 30), ("t", 12),
                    ("t", 8), ("t", 6), ("laplace", None), ("exp", None)]
        ns = [2000, 10000, 50000]; seeds = [0, 1, 2]; P, G, parents = 32, 500, 2
        # scale robustness: does bigger latent / bigger decoder change the picture?
        p64 = [("gaussian", None, 10000, 64, 2000), ("laplace", None, 10000, 64, 2000),
               ("t", 8, 10000, 64, 2000)]
    obs_noise = 0.1

    records = []
    t0 = time.time()
    total = len(families) * len(ns) * len(seeds) + len(p64) * len(seeds)
    done = 0
    for (fam, par) in families:
        for n in ns:
            for sd in seeds:
                records.append(run_cell(fam, par, n, P, G, parents, obs_noise, 1000 * sd + n))
                done += 1
                print(f"[{done}/{total}] {fam}({par}) n={n} seed={sd}  "
                      f"kurt={records[-1]['excess_kurtosis']:.2f}  "
                      f"2nd-ord acc={records[-1]['secondorder_partialcorr']['orient_acc']:.2f}  "
                      f"LiNGAM acc={records[-1]['directlingam']['orient_acc']:.2f}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    for (fam, par, n, Pb, Gb) in p64:
        for sd in seeds[:2]:
            records.append(run_cell(fam, par, n, Pb, Gb, 2, obs_noise, 7000 + 1000 * sd + n))
            done += 1
            print(f"[{done}/{total}] (scale) {fam}({par}) n={n} P={Pb} G={Gb} seed={sd}  "
                  f"LiNGAM acc={records[-1]['directlingam']['orient_acc']:.2f}  [{time.time()-t0:.0f}s]",
                  flush=True)

    # ---------- aggregate + gate verdict ----------
    def cell_vals(recs, key, sub):
        return np.array([r[key][sub] for r in recs])

    # Group P=32 main sweep by (family, n) averaging seeds
    main = [r for r in records if r["P"] == P and r["G"] == G]
    keys_plot = ["secondorder_partialcorr", "pairwise_gaussian", "pairwise_laplace",
                 "pairwise_entropy", "directlingam", "directlingam_recovered"]
    curve = {}
    fam_order = [f"{f}:{p}" for (f, p) in families]
    for n in ns:
        curve[n] = {}
        for (f, p) in families:
            cell = [r for r in main if r["family"] == f and r["param"] == p and r["n"] == n]
            if not cell:
                continue
            entry = {"excess_kurtosis": float(np.mean([r["excess_kurtosis"] for r in cell]))}
            for k in keys_plot:
                for sub in ("orient_acc", "asym_ratio", "pearDE"):
                    vals = cell_vals(cell, k, sub)
                    entry[f"{k}.{sub}.mean"] = float(np.mean(vals))
                    entry[f"{k}.{sub}.sem"] = float(np.std(vals) / np.sqrt(len(vals)))
            curve[n][f"{f}:{p}"] = entry

    # decisive numbers for the verdict
    def pick(fam, par, n, key, sub):
        cell = [r for r in main if r["family"] == fam and r["param"] == par and r["n"] == n]
        return float(np.mean([r[key][sub] for r in cell])) if cell else float("nan")

    nmax = ns[-1]
    def kurt_of(fam, par, n):
        cell = [r for r in main if r["family"] == fam and r["param"] == par and r["n"] == n]
        return float(np.mean([r["excess_kurtosis"] for r in cell])) if cell else float("nan")
    verdict = {
        "gaussian_orient_acc": {
            "secondorder": pick("gaussian", None, nmax, "secondorder_partialcorr", "orient_acc"),
            "pairwise_gaussian": pick("gaussian", None, nmax, "pairwise_gaussian", "orient_acc"),
            "pairwise_laplace": pick("gaussian", None, nmax, "pairwise_laplace", "orient_acc"),
            "directlingam": pick("gaussian", None, nmax, "directlingam", "orient_acc"),
        },
        "laplace_orient_acc": {
            "secondorder": pick("laplace", None, nmax, "secondorder_partialcorr", "orient_acc"),
            "pairwise_gaussian": pick("laplace", None, nmax, "pairwise_gaussian", "orient_acc"),
            "pairwise_laplace": pick("laplace", None, nmax, "pairwise_laplace", "orient_acc"),
            "directlingam": pick("laplace", None, nmax, "directlingam", "orient_acc"),
            "directlingam_recovered": pick("laplace", None, nmax, "directlingam_recovered", "orient_acc"),
        },
        "uniform_subgaussian_orient_acc": {  # sub-Gaussian: Laplace prior assumes the WRONG tail
            "secondorder": pick("uniform", None, nmax, "secondorder_partialcorr", "orient_acc"),
            "pairwise_laplace": pick("uniform", None, nmax, "pairwise_laplace", "orient_acc"),
            "pairwise_entropy": pick("uniform", None, nmax, "pairwise_entropy", "orient_acc"),
            "directlingam": pick("uniform", None, nmax, "directlingam", "orient_acc"),
        },
        "gaussian_asym_ratio_secondorder": pick("gaussian", None, nmax, "secondorder_partialcorr", "asym_ratio"),
        "laplace_asym_ratio_directlingam": pick("laplace", None, nmax, "directlingam", "asym_ratio"),
        "n_needed_scan": {  # DirectLiNGAM orient_acc for laplace across n
            str(n): pick("laplace", None, n, "directlingam", "orient_acc") for n in ns},
        "kurtosis_scan_directlingam_nmax": {  # orient acc vs kurtosis at largest n (the identifiability curve)
            f"{f}:{p}": {"kurt": kurt_of(f, p, nmax),
                         "lingam_acc": pick(f, p, nmax, "directlingam", "orient_acc"),
                         "secondorder_acc": pick(f, p, nmax, "secondorder_partialcorr", "orient_acc")}
            for (f, p) in families},
    }

    payload = {
        "description": "Synthetic identifiability probe for the CMC-VAE model class "
                       "(z=(I-A^T)^-1 eps, x=D z, read-out G_rr=D A D^T). Gate for a real-data "
                       "non-Gaussian training sweep (G4 Part 1).",
        "conventions": {"B[i,j]": "weight of edge j->i (LiNGAM x=Bx+e); model A_full=B^T",
                        "orient_acc_chance": 0.5, "asym_ratio_unidentifiable": 1.0, "pearDE_null": 0.0},
        "grid": {"P": P, "G": G, "parents_per_node": parents, "obs_noise_frac": obs_noise,
                 "families": fam_order, "n": ns, "seeds": seeds,
                 "scale_robustness_P64_G2000": bool(p64)},
        "readouts": {
            "secondorder_partialcorr": "pure 2nd-order (partial correlation) -> our G_rr=D A D^T class",
            "pairwise_gaussian": "our class, GAUSSIAN eps prior (||eps||^2)",
            "pairwise_laplace": "our class, LAPLACE eps prior (||eps||_1)",
            "pairwise_entropy": "LiNGAM pairwise likelihood ratio (Hyvarinen entropy)",
            "directlingam": "full DirectLiNGAM (Shimizu 2011) on latent z",
            "directlingam_recovered": "DirectLiNGAM on decode->recover z_hat = D^+ x (transfer test)"},
        "curve": {str(k): v for k, v in curve.items()},
        "verdict_numbers": verdict,
        "records": records,
    }
    outjson = os.path.join(RES, "synthetic_identifiability.json")
    json.dump(payload, open(outjson, "w"), indent=2)
    print("WROTE", outjson)

    make_figure(curve, ns, families, verdict)
    return payload


# ============================================================================
# 6. FIGURE  (house style from eval/slides_figures.py)
# ============================================================================
def make_figure(curve, ns, families, verdict):
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
    LINES = [  # (json key, sub-label, color, linestyle, marker)
        ("secondorder_partialcorr", "2nd-order read-out (G=D·A·Dᵀ class)", "#9aa0a6", "-", "o"),
        ("pairwise_gaussian",       "our SCM, Gaussian ε prior",          "#7a9cc6", "-", "s"),
        ("pairwise_laplace",        "our SCM, Laplace ε prior",           "#b5651d", "-", "^"),
        ("directlingam",            "DirectLiNGAM (latent z)",            "#1f4e79", "-", "D"),
        ("directlingam_recovered",  "DirectLiNGAM (decode→recover)",      "#1f4e79", "--", "x"),
    ]
    # (json-sub, y-label, reference line, ylim, log10-transform?)
    metrics = [("orient_acc", "edge-orientation accuracy", 0.5, (0.35, 1.02), False),
               ("asym_ratio", "log₁₀ asymmetry  |fwd|/|rev|", 0.0, (-0.6, 3.6), True),
               ("pearDE", "pearDE analog (signed Pearson)", 0.0, (-0.2, 1.05), False)]
    fig, axs = plt.subplots(len(metrics), len(ns), figsize=(5.0 * len(ns), 4.2 * len(metrics)),
                            squeeze=False)
    fam_keys = [f"{f}:{p}" for (f, p) in families]
    for ri, (msub, mlabel, ref, ylim, logt) in enumerate(metrics):
        for ci, n in enumerate(ns):
            ax = axs[ri][ci]
            cn = curve.get(n, {})
            # x-axis = mean measured excess kurtosis for each family, sorted
            xk = np.array([cn[k]["excess_kurtosis"] for k in fam_keys if k in cn])
            order = np.argsort(xk)
            xs = xk[order]
            present = [fam_keys[i] for i in range(len(fam_keys)) if fam_keys[i] in cn]
            present = [present[i] for i in order]
            for (key, lab, col, ls, mk) in LINES:
                ys = np.array([cn[k][f"{key}.{msub}.mean"] for k in present])
                es = np.array([cn[k][f"{key}.{msub}.sem"] for k in present])
                if logt:  # log10 of the ratio; propagate sem; clip to keep panel readable
                    es = np.clip((es / (np.abs(ys) + 1e-12)) / np.log(10.0), 0.0, 0.6)
                    ys = np.clip(np.log10(np.clip(ys, 1e-3, None)), -0.6, 3.6)
                ax.errorbar(xs, ys, yerr=es, color=col, ls=ls, marker=mk, ms=5, lw=1.8,
                            capsize=2, label=lab if (ri == 0 and ci == 0) else None)
            ax.axhline(ref, color="k", ls=":", lw=0.9)
            if ylim:
                ax.set_ylim(*ylim)
            if ri == 0:
                ax.set_title(f"n = {n:,}")
            if ci == 0:
                ax.set_ylabel(mlabel)
            if ri == len(metrics) - 1:
                ax.set_xlabel("excess kurtosis of ε  (Gaussian = 0)")
            ax.axvspan(-2.0, 0.05, color="#f2c4c4", alpha=0.25, lw=0)   # Gaussian/sub-Gaussian region
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    g2 = verdict["gaussian_orient_acc"]; l2 = verdict["laplace_orient_acc"]
    fig.suptitle(
        "Synthetic identifiability boundary for the CMC-VAE class  (z=(I−Aᵀ)⁻¹ε, x=Dz)\n"
        f"Gaussian ε → all read-outs at chance (2nd-ord acc={g2['secondorder']:.2f}, "
        f"LiNGAM acc={g2['directlingam']:.2f});   Laplace ε → LiNGAM orients "
        f"(acc={l2['directlingam']:.2f}) but the 2nd-order read-out stays blind "
        f"(acc={l2['secondorder']:.2f})",
        fontsize=12.5, y=1.055)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    outpng = os.path.join(OUTFIG, "fig_synthetic_identifiability.png")
    fig.savefig(outpng, bbox_inches="tight"); plt.close(fig)
    print("WROTE", outpng)


if __name__ == "__main__":
    main()
