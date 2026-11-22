"""Does plain training-data RNA-protein correlation already give cognate enrichment? Same 74-pair curated
set and the same fold statistic (median |cognate| / median |non-cognate in the same protein column|) as Fig 2c."""
import json, numpy as np, pandas as pd, scanpy as sc
from scipy.stats import mannwhitneyu
RD = "results/sweep_sparse/sp_dec_gl1000"; cfg = json.load(open(f"{RD}/config.json"))
def norm(s): return str(s).upper().replace("-", "").replace("_", "").replace(".", "")
cur = json.load(open("results/baselines/_cmcvae_cognate.json"))
ar = sc.read_h5ad(cfg["rna_path"]); ap = sc.read_h5ad(cfg["prot_path"])
genes = list(ar.var_names.astype(str)); prots = list(ap.var_names.astype(str))
X = ar.X.toarray() if hasattr(ar.X, "toarray") else np.asarray(ar.X); Y = ap.X.toarray() if hasattr(ap.X, "toarray") else np.asarray(ap.X)
X = X.astype(np.float32); Y = Y.astype(np.float32)
def zc(M):
    M = M - M.mean(0, keepdims=True); M /= (M.std(0, keepdims=True) + 1e-8); return M
C = (zc(X).T @ zc(Y)) / X.shape[0]          # [genes x proteins] Pearson
np.save("results/reviewer_controls/train_rna_prot_corr.npy", C)
def fold74(V, gnames, pnames):
    gup = {norm(g): i for i, g in enumerate(gnames)}; ppos = {norm(p): j for j, p in enumerate(pnames)}
    cog = [(gup[norm(g)], ppos[norm(p)]) for p, g in zip(cur["protein_names"], cur["cognate_gene"]) if g and norm(g) in gup and norm(p) in ppos]
    cbp = {}; [cbp.setdefault(p, set()).add(g) for g, p in cog]
    m = np.abs([V[g, p] for g, p in cog]); bg = np.abs(np.concatenate([np.delete(V[:, p], list(cbp[p])) for g, p in cog]))
    ranks = [float((np.abs(V[:, p]) > abs(V[g, p])).sum()) / V.shape[0] for g, p in cog]
    return dict(n=len(cog), fold=float(np.median(m) / np.median(bg)), mwu_p=float(mannwhitneyu(m, bg, alternative="greater").pvalue),
                median_rank_frac=float(np.median(ranks)), top10pct=float(np.mean(np.array(ranks) < 0.1)))
out = {"raw_corr": fold74(C, genes, prots)}
Tr = pd.read_csv(f"{RD}/Transcription_RNA_to_Prot.csv", index_col=0)
out["cmc_vae"] = fold74(Tr.values, list(Tr.index), list(Tr.columns))
tv = np.load("results/baselines/totalvi_raw.npz", allow_pickle=True)
out["totalvi"] = fold74(np.asarray(tv["G_rp_tv"]), list(np.asarray(tv["genes"]).astype(str)), prots)
# per-protein: is CMC's cognate rank better than correlation's?
print(json.dumps(out, indent=1)); json.dump(out, open("results/reviewer_controls/cognate_vs_corr.json", "w"), indent=1)
