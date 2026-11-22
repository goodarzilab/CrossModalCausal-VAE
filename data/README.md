# Data

No data are distributed with this repository. All datasets are public; download them from the
accessions below and arrange the processed files as described under [Expected layout](#expected-layout).

By default every script looks for data under `<repo>/data`. To keep data elsewhere, set
`CMCVAE_DATA=/path/to/data` (see `docs/REPRODUCING.md` for the other path variables).

## Datasets

### Observational CITE-seq (training corpus)

| Dataset | Role in the paper | Accession / source |
|---|---|---|
| Hao et al. 2021, healthy PBMC reference (161,764 cells, 8 donors) | Training corpus (final model) | GEO **GSE164378** (Seurat `.h5seurat`, SCT assay counts) |
| Stephenson et al. 2021, COVID-19 PBMC atlas (647,366 cells, 130 donors) | Training corpus (final model) | ArrayExpress **E-MTAB-10026** (`haniffa21.processed.h5ad`, `raw` layer) |
| COMBAT Consortium 2022, COVID-19 blood atlas | Training-cohort sweep (Supplementary Note 2, Table 1) | Zenodo **doi:10.5281/zenodo.6120249** (`COMBAT-CITESeq-DATA.h5ad`) |
| Su et al. 2020, COVID-19 PBMC | Training-cohort sweep | ArrayExpress **E-MTAB-9357** (fetch with `data_prep/citeseq/fetch_emtab9357.sh`) |

### CRISPR perturbation screens (evaluation only; never used for observational training)

| Dataset | Role in the paper | Accession / source |
|---|---|---|
| Frangieh et al. 2021, Perturb-CITE-seq (melanoma, Cas9 KO) | Development screen (Figs 3-5) | Single Cell Portal **SCP1064** |
| Chen et al., Perturb-icCITE-seq (FOXP3 regulators, CD4 T cells, Cas9 KO) | Development screen | DDBJ SRA **PRJDB16517**; processed data Zenodo **doi:10.5281/zenodo.16020737** (`Perturb_icCITE_seq_FOXP3_regulators.rds`) |
| Arce et al., CRISPRi Perturb-CITE-seq (primary CD4 T cells) | Development screen | GEO **GSE278572** (SuperSeries GSE271090) |
| Papalexi et al. 2021, ECCITE-seq (IFN-γ-stimulated THP-1) | Untouched, pre-specified test screen (Fig. 3c) | GEO **GSE153056** (we used the `pertpy` `papalexi_2021` h5mu export of the same data) |
| Replogle et al. 2022, K562 essential-gene CRISPRi Perturb-seq | Interventional fine-tuning (Fig. 5c-f) | Figshare+ **doi:10.25452/figshare.plus.20029387** |
| Nadig et al. 2025, Jurkat essential-gene CRISPRi Perturb-seq | Interventional fine-tuning (Fig. 5c-f) | GEO **GSE264667** |

### Reference networks and gene sets

| Resource | Use | Access |
|---|---|---|
| CollecTRI | TF-regulon reference; TF list for GRNBoost2; optional training prior | `decoupler` (`dc.op.collectri`) |
| DoRothEA (A-C) | TF-regulon reference | `decoupler` (`dc.op.dorothea`) |
| STRING | Gene-gene reference | STRING v12 via `build_ref_networks.py` |
| OmniPath | Protein-protein reference (equal-latent-dimension comparison); protein prior | `omnipath` Python client |
| MSigDB Hallmark | Latent-program annotation (GSEA) | `gseapy` / `decoupler` |
| GO:0003723 (RNA binding) | Vocabulary construction | `data_prep/vocab/rbp_go0003723.txt` (copy to `$CMCVAE_DATA/`) |
| KnockTF | Signed-prior experiment (Supplementary Note 5) | knockTF portal, `differential_expression_genes_all_datasets(Human).txt` |

`build_ref_networks.py` and `build_signed_prior.py` need internet access the first time (they query
the resources above); pathway gene sets used for Fig. 2b are cached in
`analysis/fig2_networks/_pathway_genesets_cache.json` so that analysis runs offline.

## Expected layout

The scripts use the following layout under `$CMCVAE_DATA` (file names are those produced by the
`data_prep/` scripts; raw downloads go in the same folders).

```
$CMCVAE_DATA/
├── rbp_go0003723.txt                         # copy of data_prep/vocab/rbp_go0003723.txt
├── hao_citeseq/
│   ├── hao_citeseq.h5seurat                  # GSE164378 (raw download)
│   └── rna_processed.h5ad, citeseq_processed.h5ad        # data_prep/citeseq/preprocess_hao.py
├── stephenson_citeseq/
│   ├── haniffa21.processed.h5ad              # E-MTAB-10026 (raw download)
│   └── rna_processed.h5ad, citeseq_processed.h5ad        # data_prep/citeseq/preprocess_stephenson.py
├── COMBAT_citeseq/COMBAT-CITESeq-DATA.h5ad   # cohort sweep only
├── su_citeseq/                               # cohort sweep only (fetch_emtab9357.sh + assemble_su.py)
├── combined/citeseq_combined.h5ad            # 144-protein panel (data_prep/citeseq/combine_datasets.py)
├── combined_v3/                              # data_prep/vocab/build_vocab_v3.py 2c (protein panel link)
├── combined_v4/                              # FINAL training corpus (data_prep/vocab/build_vocab_v4.py 2c)
│   ├── rna_combined.h5ad                     #   809,130 cells x 2,692 genes, X = log1p(CP10k), layers['counts']
│   ├── citeseq_combined.h5ad                 #   809,130 cells x 144 proteins (CLR per cohort)
│   └── vocab_genes.csv                       #   = weights/vocab_genes.csv
├── combined_combo_<cohorts>/, combat_v4vocab/   # cohort-sweep corpora (build_vocab_multi.py, build_vocab_combat.py)
├── perturb_citeseq/                          # Frangieh (SCP1064 files: RNA_expression_subset*.csv.gz,
│   ├── perturb_cite_seq_rna_processed.h5ad   #   Protein_expression.csv.gz, RNA_metadata.csv,
│   └── perturb_cite_seq_protein_processed.h5ad   #   all_sgRNA_assignments.txt) -> preprocess_frangieh.py
├── chen_perturb_icciteseq/
│   ├── Perturb_icCITE_seq_FOXP3_regulators.rds
│   └── export/                               # {RNA,ADT,intra}_{counts.mtx,features.txt,barcodes.txt}, metadata.csv
│       ├── iccite_rna.h5ad                   #   build_iccite_h5ad.py (+ fix_iccite_clr.py)
│       └── iccite_prot.h5ad
├── arce_perturb_citeseq/                     # GSE278572_{matrix.mtx,features.tsv,barcodes.tsv}.gz,
│   ├── arce_rna.h5ad                         #   GSE278572_protospacer_calls_per_cell.csv.gz -> prep_arce_eval.py
│   └── arce_prot.h5ad
├── replogle/replogle.h5ad                    # K562 + Jurkat essential screens in one AnnData
│                                             #   (obs: gene, cell_line, gem_group, UMI_count; X = log1p(CP10k))
└── knockTF/                                  # signed-prior experiment only
```

The Chen `export/` folder holds plain-text exports of the count matrices and cell metadata of the
Seurat object in the `.rds` file (written with `Matrix::writeMM` / `write.csv` in R); the
export script itself is not part of this repository.

Derived inputs written by the analysis pipeline (Papalexi processed screen, pseudobulk DE, Perturb-seq
matrices, baseline networks, ...) go under `$CMCVAE_RESULTS` (default `<repo>/results`), not under
`$CMCVAE_DATA`; see `docs/REPRODUCING.md`.
