"""Cognate-enrichment fold + permutation nulls computed on the SAVED projected translation map
(Transcription_RNA_to_Prot.csv) so it is EXACTLY consistent with fig2 C / fig_crossmodal ECDFs
(which read the same CSV). fold = median(|matched cognate|)/median(|non-cognate in cognate columns|).
Three nulls (N=1000): permute all entries; permute protein-column assignment; matched-degree
(permute genes within each cognate column). Writes <run>/cognate_nulls_csv.json.
Usage: python analysis/fig2_networks/cognate_nulls_csv.py results/<run>"""
import sys, os, json, numpy as np, pandas as pd
rd = sys.argv[1].rstrip("/")
rng = np.random.RandomState(0)
Tr = pd.read_csv(f"{rd}/Transcription_RNA_to_Prot.csv", index_col=0)
def norm(s): return str(s).upper().replace("-", "").replace("_", "").replace(".", "")
genes = [norm(g) for g in Tr.index]; prots = [norm(p) for p in Tr.columns]; gup = {g: i for i, g in enumerate(genes)}
V = np.abs(Tr.values); ng, npr = V.shape
cog = []
for j, pr in enumerate(prots):
    for cand in [pr, pr.replace("CD", ""), pr.split("/")[0]]:
        if cand in gup: cog.append((gup[cand], j)); break
cbp = {}
for g, p in cog: cbp.setdefault(p, set()).add(g)

def fold_of(M):
    matched = np.array([M[g, p] for g, p in cog])
    bg = np.concatenate([np.delete(M[:, p], list(cbp[p])) for g, p in cog])
    return float(np.median(matched) / (np.median(bg) + 1e-12))

obs = fold_of(V)
N = 1000
# A: permute all entries
foldsA = []
for _ in range(N):
    Mp = V.ravel().copy(); rng.shuffle(Mp); foldsA.append(fold_of(Mp.reshape(V.shape)))
# B: permute protein-column assignment (shuffle columns)
foldsB = []
for _ in range(N):
    perm = rng.permutation(npr); foldsB.append(fold_of(V[:, perm]))
# C: matched-degree — permute gene rows within each column (exact col-norm)
foldsC = []
for _ in range(N):
    Mc = V.copy()
    for j in range(npr): Mc[:, j] = V[rng.permutation(ng), j]
    foldsC.append(fold_of(Mc))

def zstat(folds):
    folds = np.asarray(folds); m, s = float(folds.mean()), float(folds.std() + 1e-12)
    return dict(null_mean=round(m, 3), null_std=round(s, 3), z=round((obs - m) / s, 1),
                p_emp=round((np.sum(folds >= obs) + 1) / (N + 1), 4))
out = {"observed_fold": round(obs, 2), "n_cognate": len(cog), "N_perm": N,
       "nullA_permute_entries": zstat(foldsA), "nullB_permute_columns": zstat(foldsB),
       "nullC_matched_degree": zstat(foldsC)}
json.dump(out, open(f"{rd}/cognate_nulls_csv.json", "w"), indent=2)
print(json.dumps(out, indent=2))
