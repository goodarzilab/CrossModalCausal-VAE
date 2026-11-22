#!/usr/bin/env python
"""
Preprocess Stephenson/Haniffa et al. 2021 COVID-19 PBMC CITE-seq dataset.

Source: haniffa21.processed.h5ad (647,366 cells, 130 donors, 3 sites)
  - 24,737 RNA genes + 192 ADT proteins in a single matrix
  - Raw counts in layers['raw']
  - 4 isotype controls among the 192 proteins (excluded)

Output: rna_processed.h5ad, citeseq_processed.h5ad

Preprocessing (from raw counts):
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
import numpy as np
import scanpy as sc
import muon
import anndata as ad
import pandas as pd
import os

SCRIPT_DIR = os.path.join(_CMC_DATA, "stephenson_citeseq")  # [public release] originally the data directory holding this script
H5AD_PATH = os.path.join(SCRIPT_DIR, "haniffa21.processed.h5ad")
RNA_OUT = os.path.join(SCRIPT_DIR, "rna_processed.h5ad")
PROT_OUT = os.path.join(SCRIPT_DIR, "citeseq_processed.h5ad")

# Isotype controls to exclude from protein panel
ISOTYPE_CONTROLS = [
    "AB_Mouse IgG1_K_Iso",
    "AB_Mouse_IgG2a_K_Iso",
    "AB_Mouse_IgG2b_K_Iso",
    "AB_Rat_IgG2b_K_Iso",
]


def main():
    print("Reading h5ad file (this may take a while for 6.7GB)...")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"  Total shape: {adata.shape}")
    print(f"  Feature types: {adata.var['feature_types'].value_counts().to_dict()}")

    # --- Separate RNA and protein by feature_types ---
    rna_mask = adata.var["feature_types"] == "Gene Expression"
    prot_mask = adata.var["feature_types"] == "Antibody Capture"
    print(f"  RNA features: {rna_mask.sum()}")
    print(f"  Protein features: {prot_mask.sum()}")

    # --- Build metadata ---
    obs_df = pd.DataFrame(
        {
            "donor": pd.Categorical(adata.obs["patient_id"].values),
            "sample_id": pd.Categorical(adata.obs["sample_id"].values),
            "site": pd.Categorical(adata.obs["Site"].values),
            "status": pd.Categorical(adata.obs["Status"].values),
            "severity": pd.Categorical(adata.obs["Status_on_day_collection_summary"].values),
            "celltype_coarse": pd.Categorical(adata.obs["initial_clustering"].values),
            "celltype_fine": pd.Categorical(adata.obs["full_clustering"].values),
            "sex": pd.Categorical(adata.obs["Sex"].values),
            "age": adata.obs["Age_interval"].values,
            "dataset": "stephenson_2021",
        },
        index=adata.obs_names,
    )

    # =====================
    # RNA Preprocessing (from raw counts)
    # =====================
    print("\n--- RNA Preprocessing ---")
    rna_raw = adata[:, rna_mask].layers["raw"].copy()
    rna_var = pd.DataFrame(index=adata.var_names[rna_mask])
    rna_adata = ad.AnnData(X=rna_raw, obs=obs_df.copy(), var=rna_var)
    print(f"  Starting shape: {rna_adata.shape}")
    print(f"  Raw counts range: [{rna_adata.X.min():.0f}, {rna_adata.X.max():.0f}]")

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

    # Free memory
    del rna_raw, rna_adata

    # =====================
    # Protein Preprocessing (from raw counts)
    # =====================
    print("\n--- Protein Preprocessing ---")
    prot_raw = adata[:, prot_mask].layers["raw"].copy()
    prot_var_names = adata.var_names[prot_mask]

    # Remove isotype controls
    keep_mask = ~prot_var_names.isin(ISOTYPE_CONTROLS)
    n_removed = (~keep_mask).sum()
    keep_idx = np.where(keep_mask)[0]
    if hasattr(prot_raw, "toarray"):
        prot_raw = prot_raw[:, keep_idx]
    else:
        prot_raw = prot_raw[:, keep_idx]
    prot_var_names = prot_var_names[keep_mask]
    print(f"  Removed {n_removed} isotype controls")

    prot_var = pd.DataFrame(index=prot_var_names)
    # Convert to dense for CLR
    if hasattr(prot_raw, "toarray"):
        prot_raw = prot_raw.toarray()
    prot_adata = ad.AnnData(X=prot_raw.astype(np.float32), obs=obs_df.copy(), var=prot_var)
    print(f"  Starting shape: {prot_adata.shape}")
    print(f"  Raw counts range: [{prot_adata.X.min():.0f}, {prot_adata.X.max():.0f}]")

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
    print(f"  Cells: {prot_adata.n_obs}")
    print(f"  RNA genes (HVGs): 2000")
    print(f"  Proteins: {prot_adata.n_vars}")
    print(f"  Donors: {obs_df['donor'].nunique()}")
    print(f"  Sites: {obs_df['site'].nunique()}")
    status_counts = obs_df["status"].value_counts()
    for s, c in status_counts.items():
        print(f"    {s}: {c}")
    print("  Done!")


if __name__ == "__main__":
    main()
