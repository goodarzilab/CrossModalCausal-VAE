"""Fully specified ORACLE-ACCESS synthetic orientation study (replaces Supp Table 3).

Latent linear SCM z = (I - A^T)^{-1} eps on a random DAG with P nodes (each node <= 2 parents drawn from earlier
nodes in a random topological order; |w| ~ U(0.3, 0.8), random sign), eps iid unit variance from
Gaussian / Laplace / Student-t(8). Observations x = D z + N(0, (0.1 sd)^2) with a random Gaussian decoder
D (G = 10 P features). n = 10,000 samples, 10 independent replicates (new graph, decoder, noise) per cell.

Read-outs (orientation accuracy on the TRUE edges, ties = 0.5, chance 0.5):
  second_order     : any covariance-based score (partial correlation; the class of G = D A D^T computed from
                     second moments) -- symmetric by construction, ties on every edge.
  lingam_oracle_z  : DirectLiNGAM (lingam package) on the true latent z (oracle latent access).
  lingam_oracle_D  : DirectLiNGAM on D^+ x with the TRUE decoder (oracle decoder access).
  lingam_learned_D : DirectLiNGAM on latents from a decoder estimated from x alone (top-P principal components;
                     equal in likelihood to the truth up to an invertible mixing R), Hungarian-matched to z.
                     PCA scores are exactly uncorrelated, so the pruned adjacency is empty (all ties -> 0.5);
                     *_order = causal-order agreement on true edges (parent before child), never tied.
Output: analyses/identifiability/synthetic/oracle_table.json (per replicate + mean, bootstrap 95% CI).
"""
import json, os, sys, time
import numpy as np
import lingam
from scipy.optimize import linear_sum_assignment

OUT = sys.argv[1]
FAMS = ["gaussian", "laplace", "t8"]; PS = [8, 16, 32]; N = 10000; REPS = 10


def dag(P, rng):
    order = rng.permutation(P); B = np.zeros((P, P))      # B[c, p] = weight p -> c
    for k, c in enumerate(order[1:], 1):
        pa = rng.choice(order[:k], size=min(2, k), replace=False)
        B[c, pa] = rng.uniform(0.3, 0.8, len(pa)) * rng.choice([-1, 1], len(pa))
    return B


def noise(f, n, P, rng):
    if f == "gaussian": return rng.standard_normal((n, P))
    if f == "laplace": return rng.laplace(0, 1 / np.sqrt(2), (n, P))
    return rng.standard_t(8, (n, P)) / np.sqrt(8 / 6)


def orient(Bhat, edges, pi=None):
    pi = np.arange(Bhat.shape[0]) if pi is None else pi
    f = np.array([abs(Bhat[pi[c], pi[p]]) for c, p in edges]); r = np.array([abs(Bhat[pi[p], pi[c]]) for c, p in edges])
    return float(np.mean(np.where(f > r, 1, np.where(f < r, 0, 0.5))))


def lingam_fit(Z):
    m = lingam.DirectLiNGAM(); m.fit(Z); return m


def order_acc(mdl, edges, pi=None):
    pi = np.arange(len(mdl.causal_order_)) if pi is None else pi
    pos = {v: i for i, v in enumerate(mdl.causal_order_)}
    return float(np.mean([pos[pi[p]] < pos[pi[c]] for c, p in edges]))


def ci(v, rng):
    v = np.asarray(v); bs = [rng.choice(v, len(v)).mean() for _ in range(5000)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


recs = []; T0 = time.time()
for P in PS:
    for f in FAMS:
        for rep in range(REPS):
            rng = np.random.default_rng(100000 * P + 1000 * FAMS.index(f) + rep)
            B = dag(P, rng); E = noise(f, N, P, rng)
            Z = E @ np.linalg.inv(np.eye(P) - B).T
            G = 10 * P; D = rng.standard_normal((G, P))
            X = Z @ D.T; X = X + rng.standard_normal(X.shape) * 0.1 * X.std(0)
            edges = list(zip(*np.where(np.abs(B) > 0)))
            r = dict(P=P, family=f, rep=rep, n=N, G=G, n_edges=len(edges), second_order=0.5)
            mz = lingam_fit(Z); md = lingam_fit(X @ np.linalg.pinv(D).T)
            r["lingam_oracle_z"] = orient(mz.adjacency_matrix_, edges); r["lingam_oracle_z_order"] = order_acc(mz, edges)
            r["lingam_oracle_D"] = orient(md.adjacency_matrix_, edges); r["lingam_oracle_D_order"] = order_acc(md, edges)
            U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False); Zp = U[:, :P] * S[:P]
            C = np.abs(np.corrcoef(Zp.T, Z.T)[:P, P:]); ri, cj = linear_sum_assignment(-C)
            pi = np.empty(P, int); pi[cj] = ri
            mp = lingam_fit(Zp)
            r["lingam_learned_D"] = orient(mp.adjacency_matrix_, edges, pi); r["lingam_learned_D_order"] = order_acc(mp, edges, pi)
            r["pca_match_absr"] = float(C[ri, cj].mean())
            recs.append(r)
            print(f"[{time.time()-T0:5.0f}s] P={P} {f} rep{rep}: z {r['lingam_oracle_z']:.2f} D {r['lingam_oracle_D']:.2f} "
                  f"learnedD {r['lingam_learned_D']:.2f} (order {r['lingam_learned_D_order']:.2f})", flush=True)
            json.dump(dict(records=recs), open(OUT, "w"), indent=1)

rng = np.random.default_rng(0); summ = {}
for P in PS:
    for f in FAMS:
        rs = [r for r in recs if r["P"] == P and r["family"] == f]
        summ[f"P{P}_{f}"] = {k: dict(mean=float(np.mean([r[k] for r in rs])), ci95=ci([r[k] for r in rs], rng), n_reps=len(rs))
                             for k in ["second_order", "lingam_oracle_z", "lingam_oracle_D", "lingam_learned_D", "lingam_oracle_z_order", "lingam_oracle_D_order", "lingam_learned_D_order"]}
        summ[f"P{P}_{f}"]["n_edges_mean"] = float(np.mean([r["n_edges"] for r in rs]))
json.dump(dict(spec=__doc__, summary=summ, records=recs), open(OUT, "w"), indent=1)
print("done")
