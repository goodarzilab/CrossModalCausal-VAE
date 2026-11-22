#!/usr/bin/env python
"""
Preprocess Hao et al. 2021 PBMC CITE-seq dataset.

Source: hao_citeseq.h5seurat (Seurat v4 reference, 161,764 cells, 8 donors)
Output: rna_processed.h5ad, citeseq_processed.h5ad

NOTE: The h5seurat only contains the SCT assay (SCTransform-corrected counts),
not the original raw RNA UMI counts. SCT corrected counts are integer-valued
and count-like, so we apply the standard preprocessing pipeline on them.
If raw counts are needed, download the original data from:
  https://atlas.fredhutch.org/data/nygc/multimodal/pbmc_multimodal.h5seurat

Preprocessing:
  RNA:  normalize_total(target_sum=1e4) -> log1p -> 2000 HVGs
  Protein: CLR transform (muon.prot.pp.clr)
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
import h5py
import numpy as np
import scanpy as sc
import muon
import anndata as ad
from scipy.sparse import csc_matrix, csr_matrix
import os

SCRIPT_DIR = os.path.join(_CMC_DATA, "hao_citeseq")  # [public release] originally the data directory holding this script
H5SEURAT_PATH = os.path.join(SCRIPT_DIR, "hao_citeseq.h5seurat")
RNA_OUT = os.path.join(SCRIPT_DIR, "rna_processed.h5ad")
PROT_OUT = os.path.join(SCRIPT_DIR, "citeseq_processed.h5ad")


def read_sparse_assay(f, assay_name):
    """Read a sparse assay from h5seurat into a dense numpy array."""
    grp = f["assays"][assay_name]["counts"]
    data = grp["data"][:]
    indices = grp["indices"][:]
    indptr = grp["indptr"][:]
    n_features = len(f["assays"][assay_name]["features"][:])
    n_cells = len(indptr) - 1
    # h5seurat stores features x cells (CSC), transpose to cells x features
    mat = csc_matrix((data, indices, indptr), shape=(n_features, n_cells)).T
    return csr_matrix(mat)


def main():
    print("Reading h5seurat file...")
    f = h5py.File(H5SEURAT_PATH, "r")

    # --- Cell barcodes ---
    cell_names = np.array([x.decode() for x in f["cell.names"][:]])
    print(f"  Total cells: {len(cell_names)}")

    # --- RNA (SCT corrected counts) ---
    print("Extracting RNA (SCT corrected counts)...")
    rna_mat = read_sparse_assay(f, "SCT")
    rna_features = np.array([x.decode() for x in f["assays"]["SCT"]["features"][:]])
    print(f"  RNA shape: {rna_mat.shape} ({len(rna_features)} genes)")

    # --- ADT (raw counts) ---
    print("Extracting ADT (raw counts)...")
    adt_mat = read_sparse_assay(f, "ADT")
    adt_features = np.array([x.decode() for x in f["assays"]["ADT"]["features"][:]])
    print(f"  ADT shape: {adt_mat.shape} ({len(adt_features)} proteins)")

    # --- Metadata ---
    print("Extracting metadata...")
    donor = np.array([x.decode() for x in f["meta.data"]["donor"][:]])
    celltype_l1 = np.array([x.decode() for x in f["meta.data"]["celltype.l1"][:]])
    celltype_l2 = np.array([x.decode() for x in f["meta.data"]["celltype.l2"][:]])
    celltype_l3 = np.array([x.decode() for x in f["meta.data"]["celltype.l3"][:]])
    time = np.array([x.decode() for x in f["meta.data"]["time"][:]])
    lane = np.array([x.decode() for x in f["meta.data"]["lane"][:]])

    f.close()

    import pandas as pd
    obs_df = pd.DataFrame(
        {
            "donor": pd.Categorical(donor),
            "celltype_l1": pd.Categorical(celltype_l1),
            "celltype_l2": pd.Categorical(celltype_l2),
            "celltype_l3": pd.Categorical(celltype_l3),
            "time": pd.Categorical(time),
            "lane": pd.Categorical(lane),
            "dataset": "hao_2021",
        },
        index=cell_names,
    )

    # =====================
    # RNA Preprocessing
    # =====================
    print("\n--- RNA Preprocessing ---")
    rna_adata = ad.AnnData(X=rna_mat, obs=obs_df.copy(), var=pd.DataFrame(index=rna_features))
    print(f"  Starting shape: {rna_adata.shape}")

    # Library-size normalize to 10^4
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    print("  Normalized to 10^4")

    # Log transform
    sc.pp.log1p(rna_adata)
    print("  Log1p transformed")

    # Select 2000 HVGs
    sc.pp.highly_variable_genes(rna_adata, n_top_genes=2000)
    n_hvg = rna_adata.var["highly_variable"].sum()
    print(f"  HVGs selected: {n_hvg}")
    rna_adata = rna_adata[:, rna_adata.var["highly_variable"]].copy()
    print(f"  Final RNA shape: {rna_adata.shape}")

    # Save
    rna_adata.write_h5ad(RNA_OUT)
    print(f"  Saved: {RNA_OUT}")

    # =====================
    # Protein Preprocessing
    # =====================
    print("\n--- Protein Preprocessing ---")
    prot_adata = ad.AnnData(
        X=adt_mat.toarray().astype(np.float32), obs=obs_df.copy(), var=pd.DataFrame(index=adt_features)
    )
    print(f"  Starting shape: {prot_adata.shape}")

    # CLR transform
    muon.prot.pp.clr(prot_adata)
    print("  CLR transformed")
    print(f"  Value range: [{prot_adata.X.min():.3f}, {prot_adata.X.max():.3f}]")
    print(f"  Final protein shape: {prot_adata.shape}")

    # Save
    prot_adata.write_h5ad(PROT_OUT)
    print(f"  Saved: {PROT_OUT}")

    # Summary
    print("\n=== Summary ===")
    print(f"  Cells: {rna_adata.n_obs}")
    print(f"  RNA genes (HVGs): {rna_adata.n_vars}")
    print(f"  Proteins: {prot_adata.n_vars}")
    print(f"  Donors: {obs_df['donor'].nunique()} ({sorted(obs_df['donor'].unique())})")
    print("  Done!")


if __name__ == "__main__":
    main()
