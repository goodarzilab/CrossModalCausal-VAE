"""PCA + UMAP of the 2c and 2c+COMBAT training data (v4 vocab), colored by dataset and donor, to
visualize batch structure (is there a dataset axis? does adding the COMBAT dataset create one?).
NOTE: no ComBat/batch-correction is applied anywhere in the pipeline; 'combat' = the COMBAT-CITEseq
DATASET (3rd cohort), and the only batch handling is batch-aware HVG selection. Writes
results/figures/pca_umap_2c_vs_combat.png."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, numpy as np, scanpy as sc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = f"{_CMC_DATA}"
COH = {"2c (Hao+Stephenson)": f"{DATA}/combined_v4", "2c+COMBAT (3 datasets)": f"{DATA}/combined_combat_v4"}
N = 20000
rng = np.random.RandomState(42)
fig, axs = plt.subplots(2, 3, figsize=(19, 12))
for ri, (label, d) in enumerate(COH.items()):
    a = sc.read_h5ad(f"{d}/rna_combined.h5ad", backed="r")
    idx = np.sort(rng.choice(a.n_obs, min(N, a.n_obs), replace=False))
    sub = a[idx].to_memory(); a.file.close()
    ds = sub.obs["dataset"].astype(str).values
    dn = sub.obs["donor"].astype(str).values
    sc.pp.scale(sub, max_value=10); sc.tl.pca(sub, n_comps=30)
    sil_ds = silhouette_score(sub.obsm["X_pca"][:, :10], ds)
    sc.pp.neighbors(sub, n_pcs=30); sc.tl.umap(sub)
    P = sub.obsm["X_pca"][:, :2]; U = sub.obsm["X_umap"]
    # col 0: PCA by dataset
    for g in np.unique(ds):
        m = ds == g; axs[ri, 0].scatter(P[m, 0], P[m, 1], s=3, alpha=0.4, label=g)
    axs[ri, 0].set_title(f"{label}\nPCA by DATASET (silhouette={sil_ds:.2f})", fontweight="bold", fontsize=10)
    axs[ri, 0].legend(markerscale=3, fontsize=8); axs[ri, 0].set_xlabel("PC1"); axs[ri, 0].set_ylabel("PC2")
    # col 1: UMAP by dataset
    for g in np.unique(ds):
        m = ds == g; axs[ri, 1].scatter(U[m, 0], U[m, 1], s=3, alpha=0.4, label=g)
    axs[ri, 1].set_title(f"{label}\nUMAP by DATASET", fontweight="bold", fontsize=10)
    axs[ri, 1].legend(markerscale=3, fontsize=8); axs[ri, 1].set_xlabel("UMAP1"); axs[ri, 1].set_ylabel("UMAP2")
    # col 2: UMAP by donor (top 8)
    top = [x for x, _ in sorted(zip(*np.unique(dn, return_counts=True)), key=lambda t: -t[1])][:8]
    for g in top:
        m = dn == g; axs[ri, 2].scatter(U[m, 0], U[m, 1], s=3, alpha=0.4, label=g)
    m = ~np.isin(dn, top); axs[ri, 2].scatter(U[m, 0], U[m, 1], s=2, alpha=0.2, c="lightgray", label="other")
    sil_dn = silhouette_score(sub.obsm["X_pca"][:, :10], dn)
    axs[ri, 2].set_title(f"{label}\nUMAP by DONOR (top8, donor-silhouette={sil_dn:.2f})", fontweight="bold", fontsize=10)
    axs[ri, 2].legend(markerscale=3, fontsize=7, ncol=2); axs[ri, 2].set_xlabel("UMAP1"); axs[ri, 2].set_ylabel("UMAP2")
    print(f"{label}: n={len(idx)} dataset-silhouette={sil_ds:.3f} donor-silhouette={sil_dn:.3f}", flush=True)
fig.suptitle("Training-data batch structure (v4 vocab) — NO ComBat applied; only batch-aware HVG. "
             "silhouette>0 = datasets separate (batch axis present)", fontweight="bold", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs(f"{_CMC_RESULTS}/figures", exist_ok=True)
fig.savefig(f"{_CMC_RESULTS}/figures/pca_umap_2c_vs_combat.png", dpi=130)
print("WROTE results/figures/pca_umap_2c_vs_combat.png")
