#!/usr/bin/env python
"""
Preprocess Frangieh et al. 2021 Perturb-CITE-seq data into h5ad format.

Source CSVs are genes-by-cells (transposed from AnnData convention).
RNA data is already log-normalized. Protein data is already log1p-normalized.
Since raw RNA counts are not available, we use the pre-normalized values as-is.

Memory-efficient: reads each RNA subset one at a time, transposes to a
numpy chunk (float32), then stacks them vertically — never holding more
than one subset DataFrame in memory at once.

Output:
  perturb_cite_seq_rna_processed.h5ad   (218,331 cells x 23,712 genes)
  perturb_cite_seq_protein_processed.h5ad (218,331 cells x 20 proteins)
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
import pandas as pd
import anndata as ad
import os
import gc

SCRIPT_DIR = os.path.join(_CMC_DATA, "perturb_citeseq")  # [public release] originally the data directory holding this script

RNA_SUBSETS = [
    "RNA_expression_subset1a.csv.gz",
    "RNA_expression_subset1b_alpha.csv.gz",
    "RNA_expression_subset1b_beta.csv.gz",
    "RNA_expression_subset2.csv.gz",
    "RNA_expression_subset3.csv.gz",
    "RNA_expression_subset4.csv.gz",
]

PROTEIN_PATH = os.path.join(SCRIPT_DIR, "Protein_expression.csv.gz")
METADATA_PATH = os.path.join(SCRIPT_DIR, "RNA_metadata.csv")
SGRNA_PATH = os.path.join(SCRIPT_DIR, "all_sgRNA_assignments.txt")

RNA_OUT = os.path.join(SCRIPT_DIR, "perturb_cite_seq_rna_processed.h5ad")
PROT_OUT = os.path.join(SCRIPT_DIR, "perturb_cite_seq_protein_processed.h5ad")


def load_metadata():
    """Load cell metadata and sgRNA assignments."""
    print("Loading metadata...")
    meta = pd.read_csv(METADATA_PATH)

    # Skip the TYPE annotation row
    if meta.iloc[0]["NAME"] == "TYPE":
        meta = meta.iloc[1:].reset_index(drop=True)

    meta = meta.set_index("NAME")
    meta.index.name = None
    print(f"  Metadata: {meta.shape[0]} cells, columns: {list(meta.columns)}")

    # Load authoritative sgRNA assignments
    print("Loading sgRNA assignments...")
    sgrna = pd.read_csv(SGRNA_PATH)
    sgrna = sgrna.set_index("Cell")
    sgrna.index.name = None
    sgrna.columns = ["sgRNA"]
    sgrna["sgRNA"] = sgrna["sgRNA"].fillna("")

    meta = meta.drop(columns=["sgRNA"], errors="ignore")
    meta = meta.join(sgrna, how="left")
    meta["sgRNA"] = meta["sgRNA"].fillna("")

    print(f"  Cells with sgRNA: {(meta['sgRNA'] != '').sum()}")
    print(f"  Conditions: {meta['condition'].value_counts().to_dict()}")
    return meta


def load_rna_subsets():
    """
    Load RNA subsets one at a time, transpose each to float32 numpy,
    then vstack. Peak memory ≈ 1 subset DataFrame + accumulated numpy array.
    """
    print("\nLoading RNA expression from subsets (memory-efficient)...")

    gene_names = None
    chunks = []       # list of (cell_names, np_array) per subset
    cell_names_all = []

    for fname in RNA_SUBSETS:
        path = os.path.join(SCRIPT_DIR, fname)
        print(f"  Reading {fname}...")
        df = pd.read_csv(path, index_col=0)
        print(f"    {df.shape[0]} genes x {df.shape[1]} cells")

        # Capture gene names from first subset
        if gene_names is None:
            gene_names = df.index.tolist()

        # Transpose to cells x genes, convert to float32 numpy immediately
        arr = df.values.T.astype(np.float32)  # (n_cells, n_genes)
        cell_ids = df.columns.tolist()
        cell_names_all.extend(cell_ids)
        chunks.append(arr)

        del df
        gc.collect()
        print(f"    -> chunk: {arr.shape}, cumulative cells: {len(cell_names_all)}")

    print("  Stacking all chunks...")
    X = np.vstack(chunks)
    del chunks
    gc.collect()
    print(f"  Final RNA matrix: {X.shape} (cells x genes), dtype={X.dtype}")

    return X, cell_names_all, gene_names


def load_protein():
    """Load protein expression (proteins x cells), transpose to cells x proteins."""
    print("\nLoading protein expression...")
    prot = pd.read_csv(PROTEIN_PATH, index_col=0)
    print(f"  Shape: {prot.shape} (proteins x cells)")

    # Clean protein names: remove " protein" suffix
    prot.index = prot.index.str.replace(" protein", "", regex=False)
    print(f"  Protein names: {list(prot.index)}")

    prot_names = prot.index.tolist()
    cell_names = prot.columns.tolist()
    X = prot.values.T.astype(np.float32)

    del prot
    gc.collect()
    print(f"  Transposed: {X.shape} (cells x proteins)")
    return X, cell_names, prot_names


def main():
    meta = load_metadata()

    # --- RNA ---
    rna_X, rna_cells, gene_names = load_rna_subsets()

    # --- Protein ---
    prot_X, prot_cells, prot_names = load_protein()

    # Align cells across all three sources
    rna_cell_set = set(rna_cells)
    prot_cell_set = set(prot_cells)
    meta_cell_set = set(meta.index)
    common_cells = sorted(rna_cell_set & prot_cell_set & meta_cell_set)
    print(f"\nCommon cells: {len(common_cells)}")

    # Build index maps for reordering
    rna_idx_map = {c: i for i, c in enumerate(rna_cells)}
    prot_idx_map = {c: i for i, c in enumerate(prot_cells)}

    rna_order = np.array([rna_idx_map[c] for c in common_cells])
    prot_order = np.array([prot_idx_map[c] for c in common_cells])

    rna_X = rna_X[rna_order]
    prot_X = prot_X[prot_order]
    meta = meta.loc[common_cells]

    # Build obs
    obs_df = pd.DataFrame({
        "condition": pd.Categorical(meta["condition"].values),
        "MOI": meta["MOI"].values.astype(int),
        "sgRNA": meta["sgRNA"].values,
        "UMI_count": meta["UMI_count"].values.astype(float),
    }, index=common_cells)

    # === Save RNA h5ad ===
    print("\nCreating RNA h5ad...")
    rna_adata = ad.AnnData(
        X=rna_X,
        obs=obs_df.copy(),
        var=pd.DataFrame(index=gene_names),
    )
    print(f"  Shape: {rna_adata.shape}")
    print(f"  X range: [{rna_adata.X.min():.3f}, {rna_adata.X.max():.3f}]")
    rna_adata.write_h5ad(RNA_OUT)
    print(f"  Saved: {RNA_OUT}")

    del rna_X, rna_adata
    gc.collect()

    # === Save Protein h5ad ===
    print("\nCreating protein h5ad...")
    prot_adata = ad.AnnData(
        X=prot_X,
        obs=obs_df.copy(),
        var=pd.DataFrame(index=prot_names),
    )
    print(f"  Shape: {prot_adata.shape}")
    print(f"  X range: [{prot_adata.X.min():.3f}, {prot_adata.X.max():.3f}]")
    print(f"  Protein names: {list(prot_adata.var_names)}")
    prot_adata.write_h5ad(PROT_OUT)
    print(f"  Saved: {PROT_OUT}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"  Cells: {len(common_cells)}")
    print(f"  RNA genes: {len(gene_names)}")
    print(f"  Proteins: {len(prot_names)}")
    print(f"  Conditions: {obs_df['condition'].value_counts().to_dict()}")
    print(f"  Cells with sgRNA: {(obs_df['sgRNA'] != '').sum()}")
    print("  Done!")


if __name__ == "__main__":
    main()
