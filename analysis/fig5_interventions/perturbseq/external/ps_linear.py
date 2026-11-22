"""Simple field baselines for single-gene Perturb-seq prediction (Ahlmann-Eltze, Huber & Anders, Nat Methods 2025),
on the SAME features / DE responses / by-gene 70/30 split as analysis/fig5_interventions/perturbseq/ft_eval.py.

Training data = measured responses (lfc = mean KO - mean NT, log-normalised units) of the evaluable TRAIN-split KOs of
the line (same evaluability rule as ft_eval.build_ctx). All predictions are in log units; scored by
ft_eval_bench.py part 'ext' (which divides by the model-line control sd, as for every other arm).

  no_change       : predicted response 0 (Ahlmann-Eltze 'no change'; the 'additive' baseline is defined only for
                    combinations and reduces to no-change for single perturbations).
  train_mean      : mean training response (Ahlmann-Eltze 'mean' baseline / Systema 'perturbed mean').
  linear_K10      : Ahlmann-Eltze linear model, exactly as in their linear_perturbation_prediction:
                    Y_train (features x training perts, centred by b = row means), G = top-K PCA scores of the features
                    (prcomp(Y_c)$x analogue: U_K S_K, features x K), P = G[perturbed gene] (the perturbed gene is a
                    read-out feature), W = (G'G + l I)^-1 G' Y_c P (P'P + l I)^-1, pred_g = b + G W G[g]'; K=10, l=0.1.
  linear_tuned    : same, (K, l) chosen on an inner 70/30 split of the training genes (criterion: mean per-KO
                    centred Pearson over non-target features on the inner-validation genes).
  linear_ctrlpca_K10 / _tuned : G from PCA of CONTROL-cell expression instead of the perturbation responses (gene
                    loadings x sd of the top-K PCs of the z-scored NT-cell matrix) -- information-matched to the
                    control-only CMC-VAE (gene embedding from controls; only W uses the training responses).
usage: python ps_linear.py LINE SPLITSEED
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
import os, sys, json
import numpy as np, pandas as pd
LINE, SPLIT = sys.argv[1], int(sys.argv[2])
REPO = f"{_CMC_ROOT}"; D = f"{_CMC_RESULTS}/replogle/data"; B = f"{_CMC_RESULTS}/paper/analyses/benchmark"
FEATS = json.load(open(f"{D}/features.json"))["features"]; G_ = len(FEATS); FI = {g: i for i, g in enumerate(FEATS)}
SPL = json.load(open(f"{B}/splits/ps_split_s{SPLIT}.json")); TRS, TES = set(SPL["train_genes"]), set(SPL["test_genes"])
de = np.load(f"{D}/{LINE}_de.npz"); LFC = {}
for g, lfc, pv in zip(de["genes"].astype(str), de["lfc"].astype(float), de["pvals"].astype(float)):
    keep = np.array(FEATS) != g
    if (((pv < 0.05) & (np.abs(lfc) >= 0.10)) & keep).sum() >= 3: LFC[g] = lfc
TRAIN = sorted(g for g in LFC if g in TRS); TEST = sorted(g for g in LFC if g in TES); ALL = TRAIN + TEST
TGT = np.array([f in (TRS | TES) for f in FEATS])

def fit_linear(Gm, genes, K, lam):
    Y = np.stack([LFC[g] for g in genes]).T                         # features x perts
    b = Y.mean(1, keepdims=True); Yc = Y - b
    Ge = Gm[:, :K]; P = Ge[[FI[g] for g in genes]]                     # perts x K
    W = np.linalg.solve(Ge.T @ Ge + lam * np.eye(K), Ge.T @ Yc @ P) @ np.linalg.inv(P.T @ P + lam * np.eye(K))
    return lambda gs: (b + Ge @ W @ Ge[[FI[g] for g in gs]].T).T     # perts x features

def resp_pca(genes, kmax=200):
    Y = np.stack([LFC[g] for g in genes]).T; Yc = Y - Y.mean(1, keepdims=True)
    U, S, _ = np.linalg.svd(Yc, full_matrices=False); k = min(kmax, len(S)); return U[:, :k] * S[:k]

obs = pd.read_csv(f"{D}/{LINE}_obs.csv"); nt = np.sort(np.where(obs.is_nt.values)[0])
Xn = np.load(f"{D}/{LINE}_X.npy", mmap_mode="r")[nt].astype(np.float64); Zn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-8)
_, Sc, Vt = np.linalg.svd(Zn, full_matrices=False); GCTRL = (Vt[:200].T * Sc[:200] / np.sqrt(len(Zn)))   # features x 200
del Xn, Zn

def inner_split(genes):   # == ft_eval.inner_split
    g = np.array(sorted(genes)); prm = np.random.RandomState(7).permutation(len(g)); nv = max(1, round(0.3 * len(g)))
    return sorted(g[prm[nv:]].tolist()), sorted(g[prm[:nv]].tolist())

def crit(pred_fn, itr, iva):
    P_tr, P_va = pred_fn(itr), pred_fn(iva); mp = P_tr.mean(0); mo = np.stack([LFC[g] for g in itr]).mean(0); r = []
    for i, g in enumerate(iva):
        m = ~TGT; p = (P_va[i] - mp)[m]; o = (LFC[g] - mo)[m]
        if p.std() > 0: r.append(np.corrcoef(p, o)[0, 1])
    return float(np.mean(r)) if r else -np.inf

OUT = {}; INFO = {}
OUT["no_change"] = np.zeros((len(ALL), G_))
OUT["train_mean"] = np.tile(np.stack([LFC[g] for g in TRAIN]).mean(0), (len(ALL), 1))
GRID_K, GRID_L = [5, 10, 20, 50, 100], [0.01, 0.1, 1, 10, 100]
for nm, emb_full, emb_inner in [("linear", lambda: resp_pca(TRAIN), lambda itr: resp_pca(itr)),
                                ("linear_ctrlpca", lambda: GCTRL, lambda itr: GCTRL)]:
    Gm = emb_full(); OUT[f"{nm}_K10"] = fit_linear(Gm, TRAIN, 10, 0.1)(ALL)
    itr, iva = inner_split(TRAIN); Gi = emb_inner(itr); grid = {}
    for K in GRID_K:
        for lam in GRID_L:
            if K > min(Gi.shape[1], len(itr)): continue
            grid[f"{K}_{lam}"] = crit(fit_linear(Gi, itr, K, lam), itr, iva)
    best = max(grid, key=grid.get); K, lam = int(best.split("_")[0]), float(best.split("_")[1])
    OUT[f"{nm}_tuned"] = fit_linear(Gm, TRAIN, K, lam)(ALL); INFO[f"{nm}_tuned"] = dict(K=K, lam=lam, grid=grid)
os.makedirs(f"{B}/ext", exist_ok=True); o = f"{B}/ext/linear_{LINE}_split{SPLIT}.npz"
np.savez_compressed(o, names=np.array(list(OUT)), **{f"{k}|genes": np.array(ALL) for k in OUT}, **{f"{k}|lfc": v.astype(np.float32) for k, v in OUT.items()})
json.dump(dict(n_train=len(TRAIN), n_test=len(TEST), info=INFO), open(o.replace(".npz", ".json"), "w"), indent=1)
print("DONE", o, {k: v.shape for k, v in OUT.items()}, INFO and {k: (v["K"], v["lam"]) for k, v in INFO.items()})
