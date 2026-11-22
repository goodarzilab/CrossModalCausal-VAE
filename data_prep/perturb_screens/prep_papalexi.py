#!/usr/bin/env python
"""Prepare the Papalexi 2021 THP-1 ECCITE-seq screen (GSE153056; pertpy papalexi_2021 h5mu) in the SAME format
the perturbation harness expects (like arce_rna.h5ad / iccite_prot.h5ad):
  papalexi_rna.h5ad  : log1p(library-size 1e4) RNA, obs['sgRNA'] = target gene, 'NON_TARGET' for NT (-> control)
  papalexi_prot.h5ad : muon CLR ADT (axis 0, as training corpus / Chen), var renamed to the 144-panel names
Also writes a mixscape-style escape-filtered pair (papalexi_mix_*.h5ad) for the pre-registered sensitivity analysis.
Usage: python -I prep_papalexi.py <in.h5mu> <out_dir>
"""
import sys, os, json, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
warnings.filterwarnings("ignore")
import anndata as ad, mudata as md, scanpy as sc, muon
from scipy.stats import rankdata, norm
from sklearn.neighbors import NearestNeighbors

IN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
m = md.read_h5mu(IN)
rna = m["rna"].copy(); adt = m["adt"].copy()
assert (rna.obs_names == adt.obs_names).all()
obs = rna.obs
tgt = obs["gene_target"].astype(str).values
sg = np.where(tgt == "NT", "NON_TARGET", tgt)
info = dict(source=IN, n_cells=int(rna.n_obs), n_genes=int(rna.n_vars), adt=list(map(str, adt.var_names)),
            hto_classes=obs["HTO_classification"].astype(str).value_counts().to_dict(),
            replicate=obs["replicate"].astype(str).value_counts().to_dict(),
            per_target=pd.Series(tgt).value_counts().to_dict())

# ---- RNA: counts -> normalize_total 1e4 -> log1p
rna.X = sp.csr_matrix(rna.X).astype(np.float32)
rna.layers["counts"] = rna.X.copy()
sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
rna.obs = pd.DataFrame(dict(sgRNA=sg, gene_target=tgt, guide_ID=obs["guide_ID"].astype(str).values,
                            replicate=obs["replicate"].astype(str).values), index=rna.obs_names)
# ---- ADT: muon CLR (axis 0), rename to panel names
ALIAS = {"CD86": "CD86", "PDL1": "CD274", "PDL2": "CD273", "CD366": "CD366"}
adt.X = sp.csc_matrix(adt.X).astype(np.float32)
adt.layers["counts"] = adt.X.copy()
muon.prot.pp.clr(adt)
adt.X = np.asarray(adt.X.toarray() if sp.issparse(adt.X) else adt.X, dtype=np.float32)
adt.var_names = [ALIAS.get(v, v) for v in adt.var_names.astype(str)]
adt.obs = rna.obs.copy()
rna.write(f"{OUT}/papalexi_rna.h5ad"); adt.write(f"{OUT}/papalexi_prot.h5ad")

# ---- mixscape-style escape filtering (pre-registered sensitivity)
def mwp(Xp, Xc):
    n1, n2 = len(Xp), len(Xc); R = np.apply_along_axis(rankdata, 0, np.vstack([Xp, Xc]))
    U = R[:n1].sum(0) - n1 * (n1 + 1) / 2; z = (U - n1 * n2 / 2) / (np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) + 1e-10)
    return 2 * norm.sf(np.abs(z))

h = rna.copy(); sc.pp.highly_variable_genes(h, n_top_genes=2000, flavor="seurat")
h = h[:, h.var["highly_variable"]].copy(); X = h.X.toarray()
Xs = (X - X.mean(0)) / (X.std(0) + 1e-8); Xs = np.clip(Xs, -10, 10)
U_, S_, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False); PC = U_[:, :40] * S_[:40]
nt = np.where(tgt == "NT")[0]
nn = NearestNeighbors(n_neighbors=20).fit(PC[nt]); _, idx = nn.kneighbors(PC)
SIG = X - X[nt][idx].mean(1)                       # perturbation signature (cell minus mean of 20 NT neighbours)
keep_ko = np.zeros(len(tgt), bool); mix = {}
for g in sorted(set(tgt) - {"NT"}):
    gi = np.where(tgt == g)[0]
    lab = np.ones(len(gi), bool); status = "KO"
    for it in range(10):
        if lab.sum() < 5: status = "NP(<5 KO cells)"; lab[:] = False; break
        p = mwp(SIG[gi[lab]], SIG[nt]); lfc = SIG[gi[lab]].mean(0) - SIG[nt].mean(0)
        de = (p < 0.05) & (np.abs(lfc) >= 0.25)
        if de.sum() < 5: status = "NP(<5 DE genes)"; lab[:] = False; break
        v = lfc[de]; proj = lambda Z: (Z[:, de] - SIG[nt][:, de].mean(0)) @ v / (v @ v)
        pg, pn = proj(SIG[gi]), proj(SIG[nt])
        mu0, sd0 = pn.mean(), pn.std() + 1e-8; mu1, sd1, w = pg[lab].mean(), pg[lab].std() + 1e-8, 0.5
        for _ in range(200):                           # EM: NT component fixed, KO component free
            l1 = w * norm.pdf(pg, mu1, sd1); l0 = (1 - w) * norm.pdf(pg, mu0, sd0); r = l1 / (l1 + l0 + 1e-300)
            w = r.mean(); mu1 = (r * pg).sum() / (r.sum() + 1e-12); sd1 = np.sqrt((r * (pg - mu1) ** 2).sum() / (r.sum() + 1e-12)) + 1e-8
        new = r > 0.5
        if (new == lab).all(): break
        lab = new
    keep_ko[gi[lab]] = True
    mix[g] = dict(n=int(len(gi)), n_ko=int(lab.sum()), status=status if lab.sum() else status)
info["mixscape"] = mix
sel = keep_ko | (tgt == "NT")
rna[sel].write(f"{OUT}/papalexi_mix_rna.h5ad"); adt[sel].write(f"{OUT}/papalexi_mix_prot.h5ad")
json.dump(info, open(f"{OUT}/prep_info.json", "w"), indent=1)
print(json.dumps(mix, indent=0)); print("PREP_DONE", rna.shape, adt.shape, int(sel.sum()))
