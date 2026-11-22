#!/usr/bin/env python
"""
Combine Hao 2021 and Stephenson 2021 PBMC CITE-seq datasets into unified
rna_combined.h5ad and citeseq_combined.h5ad for VAE training.

Memory-efficient: processes datasets sequentially, keeps sparse format,
and subsets to common genes/proteins before concatenation.

RNA pipeline:
  1. Discover common genes (name-only pass)
  2. Per dataset: load raw → normalize_total(1e4) → log1p → subset to common genes → save temp
  3. Concatenate temps → compute 2000 HVGs → save final

Protein pipeline:
  1. Map Stephenson AB_ names to Hao CD nomenclature
  2. Per dataset: load raw counts for common proteins → CLR transform → save temp
  3. Concatenate → save final

Output:
  data/combined/rna_combined.h5ad
  data/combined/citeseq_combined.h5ad
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
import gc
import h5py
import numpy as np
import scanpy as sc
import muon
import anndata as ad
import pandas as pd
import os
from scipy.sparse import csc_matrix, csr_matrix

SCRIPT_DIR = _CMC_DATA  # [public release] originally the data directory holding this script
HAO_H5SEURAT = os.path.join(SCRIPT_DIR, "hao_citeseq", "hao_citeseq.h5seurat")
STEPH_H5AD = os.path.join(SCRIPT_DIR, "stephenson_citeseq", "haniffa21.processed.h5ad")
OUT_DIR = os.path.join(SCRIPT_DIR, "combined")
RNA_OUT = os.path.join(OUT_DIR, "rna_combined.h5ad")
PROT_OUT = os.path.join(OUT_DIR, "citeseq_combined.h5ad")
HAO_RNA_TEMP = os.path.join(OUT_DIR, "_hao_rna_temp.h5ad")
STEPH_RNA_TEMP = os.path.join(OUT_DIR, "_steph_rna_temp.h5ad")

# ================================================================
# Protein name mapping: Stephenson (AB_ prefix) → Hao (CD names)
# For Hao duplicates (e.g. CD3-1/CD3-2), we use the -1 variant.
# ================================================================
STEPH_TO_HAO = {
    # Direct/near-direct matches
    "AB_B7-H4": "B7-H4",
    "AB_CD123": "CD123",
    "AB_CD14": "CD14",
    "AB_CD141": "CD141",
    "AB_CD15": "CD15",
    "AB_CD158": "CD158",
    "AB_CD158b": "CD158b",
    "AB_CD16": "CD16",
    "AB_CD161": "CD161",
    "AB_CD163": "CD163",
    "AB_CD19": "CD19",
    "AB_CD1a": "CD1a",
    "AB_CD1C": "CD1c",
    "AB_CD1d": "CD1d",
    "AB_CD2": "CD2",
    "AB_CD20": "CD20",
    "AB_CD209": "CD209",
    "AB_CD21": "CD21",
    "AB_CD22": "CD22",
    "AB_CD226": "CD226",
    "AB_CD235ab": "CD235ab",
    "AB_CD24": "CD24",
    "AB_CD244": "CD244",
    "AB_CD25": "CD25",
    "AB_CD27": "CD27",
    "AB_CD274": "CD274",
    "AB_CD28": "CD28",
    "AB_CD303": "CD303",
    "AB_CD304": "CD304",
    "AB_CD34": "CD34",
    "AB_CD36": "CD36",
    "AB_CD3": "CD3-1",
    "AB_CD4": "CD4-1",
    "AB_CD40": "CD40",
    "AB_CD44": "CD44-1",
    "AB_CD45": "CD45-1",
    "AB_CD45RA": "CD45RA",
    "AB_CD45RO": "CD45RO",
    "AB_CD47": "CD47",
    "AB_CD52": "CD52",
    "AB_CD56": "CD56-1",
    "AB_CD64": "CD64",
    "AB_CD69": "CD69",
    "AB_CD70": "CD70",
    "AB_CD71": "CD71",
    "AB_CD79b": "CD79b",
    "AB_CD8": "CD8",
    "AB_CD80": "CD80",
    "AB_CD81": "CD81",
    "AB_CD83": "CD83",
    "AB_CD86": "CD86",
    "AB_CD96": "CD96",
    "AB_CD99": "CD99",
    "AB_CLEC12A": "CLEC12A",
    "AB_CX3CR1": "CX3CR1",
    "AB_HLA-DR": "HLA-DR",
    "AB_IgD": "IgD",
    "AB_IgM": "IgM",
    "AB_TIGIT": "TIGIT",
    "AB_TSLPR": "TSLPR",
    "AB_XCR1": "XCR1",
    "AB_podoplanin": "Podoplanin",
    # CD → gene symbol mappings
    "AB_ITGAE": "CD103",
    "AB_VCAM1": "CD106",
    "AB_LAMP1": "CD107a",
    "AB_NECTIN2": "CD112",
    "AB_KIT": "CD117",
    "AB_ITGAL": "CD11a/CD18",
    "AB_ITGAM": "CD11b-1",
    "AB_ITGAX": "CD11c",
    "AB_IL2RB": "CD122",
    "AB_IL4R": "CD124",
    "AB_IL7R": "CD127",
    "AB_PROM1": "CD133-1",
    "AB_TNFRSF9": "CD137",
    "AB_SDC1": "CD138-1",
    "AB_CDH5": "CD144",
    "AB_MCAM": "CD146",
    "AB_CTLA4": "CD152",
    "AB_CD40LG": "CD154",
    "AB_PVR": "CD155",
    "AB_KIR3DL1": "CD158e1",
    "AB_SIGLEC1": "CD169",
    "AB_FASLG": "CD178",
    "AB_ITGB2": "CD18",
    "AB_CXCR4": "CD184",
    "AB_CXCR5": "CD185",
    "AB_CCR3": "CD193",
    "AB_CCR4": "CD194",
    "AB_CCR5": "CD195",
    "AB_CCR6": "CD196",
    "AB_MSR1": "CD204",
    "AB_MMR": "CD206",
    "AB_langerin": "CD207",
    "AB_LAG3": "CD223",
    "AB_OX40L": "CD252",
    "AB_DPP4": "CD26-1",
    "AB_TACI": "CD267",
    "AB_BAFFR": "CD268",
    "AB_TNFRSF17": "CD269",
    "AB_BTLA": "CD272",
    "AB_PDCD1LG2": "CD273",
    "AB_ICOSLG": "CD275-1",
    "AB_ICOS": "CD278",
    "AB_PD1": "CD279",
    "AB_ITGB1": "CD29",
    "AB_PTGDR2": "CD294",
    "AB_TNFRSF8": "CD30",
    "AB_LAIR1": "CD305",
    "AB_FCRL4": "CD307d",
    "AB_FCRL5": "CD307e",
    "AB_KDR": "CD309",
    "AB_PECAM1": "CD31",
    "AB_KLRK1": "CD314",
    "AB_SLAMF7": "CD319",
    "AB_CDH1": "CD324",
    "AB_NCR1": "CD335",
    "AB_NCR3": "CD337",
    "AB_CR1": "CD35",
    "AB_GITR": "CD357",
    "AB_HAVCR2": "CD366",
    "AB_CLEC9A": "CD370",
    "AB_CD38": "CD38-1",
    "AB_ENTPD1": "CD39",
    "AB_ITGA2B": "CD41",
    "AB_ITGA2": "CD49b",
    "AB_ITGA4": "CD49d",
    "AB_ICAM1": "CD54",
    "AB_B3GAT1": "CD57",
    "AB_SELP": "CD62P",
    "AB_CEACAM1/5/6": "CD66a/c/e",
    "AB_CEACAM8": "CD66b",
    "AB_NT5E": "CD73",
    "AB_THY1": "CD90",
    "AB_FAS": "CD95",
    "AB_SLC3A2": "CD98",
    "AB_ITGB7": "Integrin-7",
    "AB_OLR1": "LOX-1",
    "AB_TCR": "TCR-1",
    "AB_TCRg_d": "TCR-2",
    "AB_TCR_Vg2": "TCR-V-2",
    "AB_TCR_Va24-Ja18": "TCR-V-24-J-18",
    "AB_TCR_Va7.2": "TCR-V-7.2",
    "AB_TCR_Vg9": "TCR-V-9",
}

HAO_ISOTYPES = {"Rag-IgG2c", "Rat-IgG1-1", "Rat-IgG1-2", "Rat-IgG2b"}


def read_hao_sparse(f, assay_name):
    """Read sparse assay from h5seurat → cells x features CSR matrix."""
    grp = f["assays"][assay_name]["counts"]
    data = grp["data"][:]
    indices = grp["indices"][:]
    indptr = grp["indptr"][:]
    n_feats = len(f["assays"][assay_name]["features"][:])
    n_cells = len(indptr) - 1
    return csc_matrix((data, indices, indptr), shape=(n_feats, n_cells)).T.tocsr()


def get_common_genes():
    """Get common gene names between Hao and Stephenson (names only, low memory)."""
    print("Discovering common genes...")
    f = h5py.File(HAO_H5SEURAT, "r")
    hao_genes = set(x.decode() for x in f["assays"]["SCT"]["features"][:])
    f.close()

    steph = ad.read_h5ad(STEPH_H5AD, backed="r")
    steph_rna_mask = steph.var["feature_types"] == "Gene Expression"
    steph_genes = set(steph.var_names[steph_rna_mask])
    steph.file.close()

    common = sorted(hao_genes & steph_genes)
    print(f"  Hao: {len(hao_genes)}, Stephenson: {len(steph_genes)}, Common: {len(common)}")
    return common


def get_common_proteins():
    """Determine common protein set using the mapping."""
    f = h5py.File(HAO_H5SEURAT, "r")
    hao_prots = set(x.decode() for x in f["assays"]["ADT"]["features"][:])
    f.close()
    hao_prots -= HAO_ISOTYPES

    steph = ad.read_h5ad(STEPH_H5AD, backed="r")
    steph_prots = set(steph.var_names[steph.var["feature_types"] == "Antibody Capture"])
    steph.file.close()

    # Find valid mappings
    valid_pairs = []
    for steph_name, hao_name in STEPH_TO_HAO.items():
        if steph_name in steph_prots and hao_name in hao_prots:
            valid_pairs.append((steph_name, hao_name))
    print(f"  Common proteins (mapped): {len(valid_pairs)}")
    return valid_pairs


def process_hao_rna(common_genes):
    """Load Hao RNA, normalize, subset to common genes, save temp."""
    print("\n[1/4] Processing Hao RNA...")
    f = h5py.File(HAO_H5SEURAT, "r")
    cell_names = np.array([x.decode() for x in f["cell.names"][:]])
    rna_mat = read_hao_sparse(f, "SCT")
    rna_feats = np.array([x.decode() for x in f["assays"]["SCT"]["features"][:]])
    donor = np.array([x.decode() for x in f["meta.data"]["donor"][:]])
    celltype_l1 = np.array([x.decode() for x in f["meta.data"]["celltype.l1"][:]])
    celltype_l2 = np.array([x.decode() for x in f["meta.data"]["celltype.l2"][:]])
    f.close()

    obs_df = pd.DataFrame({
        "donor": pd.Categorical(["hao_" + d for d in donor]),
        "celltype": pd.Categorical(celltype_l1),
        "celltype_fine": pd.Categorical(celltype_l2),
        "dataset": "hao",
    }, index=cell_names)

    adata = ad.AnnData(X=rna_mat, obs=obs_df, var=pd.DataFrame(index=rna_feats))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Subset to common genes
    adata = adata[:, common_genes].copy()
    print(f"  Hao RNA (common genes): {adata.shape}")

    adata.write_h5ad(HAO_RNA_TEMP)
    del adata, rna_mat
    gc.collect()
    return obs_df


def process_steph_rna(common_genes):
    """Load Stephenson RNA raw counts, normalize, subset to common genes, save temp."""
    print("\n[2/4] Processing Stephenson RNA...")
    adata_full = sc.read_h5ad(STEPH_H5AD)
    rna_mask = adata_full.var["feature_types"] == "Gene Expression"

    obs_df = pd.DataFrame({
        "donor": pd.Categorical(["steph_" + str(d) for d in adata_full.obs["patient_id"].values]),
        "celltype": pd.Categorical(adata_full.obs["initial_clustering"].values),
        "celltype_fine": pd.Categorical(adata_full.obs["full_clustering"].values),
        "dataset": "stephenson",
    }, index=adata_full.obs_names)

    # Extract raw RNA for common genes only (memory efficient)
    rna_genes = adata_full.var_names[rna_mask]
    common_set = set(common_genes)
    gene_idx = [i for i, g in enumerate(rna_genes) if g in common_set]
    gene_names = rna_genes[gene_idx]

    rna_raw = adata_full[:, adata_full.var_names[rna_mask][gene_idx]].layers["raw"].copy()
    del adata_full
    gc.collect()

    adata = ad.AnnData(X=rna_raw, obs=obs_df, var=pd.DataFrame(index=gene_names))
    del rna_raw
    gc.collect()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Reorder to match common_genes order
    adata = adata[:, common_genes].copy()
    print(f"  Stephenson RNA (common genes): {adata.shape}")

    adata.write_h5ad(STEPH_RNA_TEMP)
    del adata
    gc.collect()
    return obs_df


def combine_rna_from_temps():
    """Load temp files, concatenate, compute HVGs, save final."""
    print("\n[3/4] Combining RNA and computing HVGs...")
    hao = sc.read_h5ad(HAO_RNA_TEMP)
    steph = sc.read_h5ad(STEPH_RNA_TEMP)

    combined = ad.concat([hao, steph], merge="same")
    del hao, steph
    gc.collect()
    print(f"  Combined: {combined.shape}")

    sc.pp.highly_variable_genes(combined, n_top_genes=2000)
    n_hvg = combined.var["highly_variable"].sum()
    print(f"  HVGs: {n_hvg}")
    combined = combined[:, combined.var["highly_variable"]].copy()
    print(f"  Final RNA: {combined.shape}")

    combined.write_h5ad(RNA_OUT)
    print(f"  Saved: {RNA_OUT}")

    # Cleanup temps
    os.remove(HAO_RNA_TEMP)
    os.remove(STEPH_RNA_TEMP)

    return combined


def process_protein(protein_pairs):
    """Process protein for both datasets and combine."""
    print("\n[4/4] Processing and combining protein...")
    steph_names = [s for s, h in protein_pairs]
    hao_names = [h for s, h in protein_pairs]

    # --- Hao protein ---
    print("  Loading Hao protein...")
    f = h5py.File(HAO_H5SEURAT, "r")
    cell_names = np.array([x.decode() for x in f["cell.names"][:]])
    adt_mat = read_hao_sparse(f, "ADT")
    adt_feats = np.array([x.decode() for x in f["assays"]["ADT"]["features"][:]])
    donor = np.array([x.decode() for x in f["meta.data"]["donor"][:]])
    celltype_l1 = np.array([x.decode() for x in f["meta.data"]["celltype.l1"][:]])
    celltype_l2 = np.array([x.decode() for x in f["meta.data"]["celltype.l2"][:]])
    f.close()

    obs_hao = pd.DataFrame({
        "donor": pd.Categorical(["hao_" + d for d in donor]),
        "celltype": pd.Categorical(celltype_l1),
        "celltype_fine": pd.Categorical(celltype_l2),
        "dataset": "hao",
    }, index=cell_names)

    # Subset to mapped proteins
    feat_idx = [list(adt_feats).index(h) for h in hao_names]
    hao_prot = ad.AnnData(
        X=adt_mat[:, feat_idx].toarray().astype(np.float32),
        obs=obs_hao,
        var=pd.DataFrame(index=hao_names),
    )
    del adt_mat
    gc.collect()

    muon.prot.pp.clr(hao_prot)
    print(f"  Hao protein: {hao_prot.shape}, CLR range: [{hao_prot.X.min():.3f}, {hao_prot.X.max():.3f}]")

    # --- Stephenson protein ---
    print("  Loading Stephenson protein...")
    adata_full = sc.read_h5ad(STEPH_H5AD)
    prot_mask = adata_full.var["feature_types"] == "Antibody Capture"

    obs_steph = pd.DataFrame({
        "donor": pd.Categorical(["steph_" + str(d) for d in adata_full.obs["patient_id"].values]),
        "celltype": pd.Categorical(adata_full.obs["initial_clustering"].values),
        "celltype_fine": pd.Categorical(adata_full.obs["full_clustering"].values),
        "dataset": "stephenson",
    }, index=adata_full.obs_names)

    prot_raw = adata_full[:, steph_names].layers["raw"].copy()
    del adata_full
    gc.collect()

    if hasattr(prot_raw, "toarray"):
        prot_raw = prot_raw.toarray()
    steph_prot = ad.AnnData(
        X=prot_raw.astype(np.float32),
        obs=obs_steph,
        var=pd.DataFrame(index=hao_names),  # use Hao names as canonical
    )
    del prot_raw
    gc.collect()

    muon.prot.pp.clr(steph_prot)
    print(f"  Steph protein: {steph_prot.shape}, CLR range: [{steph_prot.X.min():.3f}, {steph_prot.X.max():.3f}]")

    # Concatenate
    combined = ad.concat([hao_prot, steph_prot], merge="same")
    del hao_prot, steph_prot
    gc.collect()
    print(f"  Final protein: {combined.shape}")

    combined.write_h5ad(PROT_OUT)
    print(f"  Saved: {PROT_OUT}")
    return combined


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Phase 1: Discover common genes and proteins (low memory)
    common_genes = get_common_genes()
    protein_pairs = get_common_proteins()

    # Phase 2: Process RNA sequentially
    process_hao_rna(common_genes)
    gc.collect()
    process_steph_rna(common_genes)
    gc.collect()

    # Phase 3: Combine RNA
    rna_combined = combine_rna_from_temps()
    gc.collect()

    # Phase 4: Process and combine protein
    prot_combined = process_protein(protein_pairs)

    # Verify cell alignment
    assert list(rna_combined.obs_names) == list(prot_combined.obs_names), "Cell name mismatch!"

    # Summary
    print("\n=== SUMMARY ===")
    print(f"  Total cells: {rna_combined.n_obs:,}")
    hao_n = (rna_combined.obs["dataset"] == "hao").sum()
    steph_n = (rna_combined.obs["dataset"] == "stephenson").sum()
    print(f"    Hao: {hao_n:,}")
    print(f"    Stephenson: {steph_n:,}")
    print(f"  RNA genes (HVGs): {rna_combined.n_vars}")
    print(f"  Proteins: {prot_combined.n_vars}")
    print(f"  Donors: {rna_combined.obs['donor'].nunique()}")
    print("  Done!")


if __name__ == "__main__":
    main()
