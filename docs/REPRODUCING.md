# Reproducing the paper

This document maps every figure, table and supplementary analysis of the paper to the scripts that
produce it, in pipeline order: **data → vocabulary → training → evaluation → figures**.

> Goudarzi S., Nandi V., Liu N. & Goodarzi H. *Multi-omics causal deep learning reveals regulatory
> structure while interventions refine perturbation predictions.* bioRxiv (2026).

The scripts are the ones used for the paper. The only changes made for this release are to file paths and imports;
the analysis logic is unchanged. Many scripts were written as cluster jobs, so they read settings from
environment variables and write JSON/CSV/NPZ outputs that later scripts pick up. Each script's docstring lists
its options.

---

## 0. Conventions

### Paths

Every script resolves paths from four environment variables. Each has a default, so with the defaults
you only need to place the data under `<repo>/data`.

| Variable | Default | Contents |
|---|---|---|
| `CMCVAE_ROOT` | repository root (computed from the script location) | code |
| `CMCVAE_DATA` | `$CMCVAE_ROOT/data` | raw and processed datasets ([data/README.md](../data/README.md)) |
| `CMCVAE_RESULTS` | `$CMCVAE_ROOT/results` | trained models, evaluation outputs; `results/paper/` holds the paper-analysis outputs (`analyses/`, `figures/`, `FINAL_FIGURES/`) |
| `CMCVAE_SCRATCH` | `$CMCVAE_RESULTS/scratch` | large intermediate files (encoded-cell caches, saved predictions, adapted models) |

Optional variables: `DEEPSEM_DIR` (a clone of the DeepSEM repository, default `external/DeepSEM`) and
`SCVI_PY` (the Python interpreter of an environment with scvi-tools, used by one script).

**Run every command from the repository root.** Some scripts add `src/` or `analysis/common/` to
`sys.path` relative to the working directory, and some config files use repo-relative paths
(`data/...`, `results/...`).

### Models

The released final models (seeds 42, 1, 2) are in `weights/`. Most analyses expect a model directory
`results/sweep_sparse/<run>/` that contains `config.json` and `final_model.pt`, plus the projected-network
CSVs that `src/train.py` writes at the end of training. To use the released weights without retraining:

```bash
for r in sp_dec_gl1000 sp_dec_gl1000_s1 sp_dec_gl1000_s2; do
  mkdir -p results/sweep_sparse/$r && cp weights/$r/* results/sweep_sparse/$r/
done
```

Analyses that read `GRN_inferred.csv`, `Transcription_RNA_to_Prot.csv` or the loading CSVs need the
artifacts that `src/train.py::save_artifacts` writes. Retraining (§3) produces them; otherwise call
`save_artifacts` on the loaded model.

### Terminology in the code

| Code | Paper |
|---|---|
| `frangieh` / `iccite` / `arce` / `papalexi` | Frangieh / Chen / Arce / Papalexi screens |
| `sp_dec_gl1000` (`_s1`, `_s2`) | final CMC-VAE (seed 42, 1, 2) |
| `rr_*`, `rp_*` | RNA→RNA, RNA→protein read-outs |
| `pearDE`, `pe`, `pe_cent` | signed pearDE, mean-centred pearDE |
| `pb_bh` | condition-matched pseudobulk DE (BH q < 0.05) used for the final figures |
| `orig` DE | cell-level Mann-Whitney DE used during model development |
| `clamp_zero` / `edgecut` / soft do | in-silico knockdown operators (Methods, "Decoder-guided perturbation prediction") |

---

## 1. Data preprocessing (`data_prep/`)

| Step | Script | Output (under `$CMCVAE_DATA` unless noted) |
|---|---|---|
| Hao PBMC → AnnData | `data_prep/citeseq/preprocess_hao.py` | `hao_citeseq/{rna,citeseq}_processed.h5ad` |
| Stephenson atlas → AnnData | `data_prep/citeseq/preprocess_stephenson.py` | `stephenson_citeseq/{rna,citeseq}_processed.h5ad` |
| Su (cohort sweep only) | `data_prep/citeseq/fetch_emtab9357.sh`, `assemble_su.py` | `su_citeseq/su_{rna,prot}_raw.h5ad` |
| Harmonised 144-protein panel (isotypes removed, CLR per cohort) | `data_prep/citeseq/combine_datasets.py` | `combined/citeseq_combined.h5ad` |
| Frangieh screen | `data_prep/perturb_screens/preprocess_frangieh.py` | `perturb_citeseq/perturb_cite_seq_{rna,protein}_processed.h5ad` |
| Chen screen | `data_prep/perturb_screens/build_iccite_h5ad.py`, then `fix_iccite_clr.py` | `chen_perturb_icciteseq/export/iccite_{rna,prot}.h5ad` |
| Arce screen | `data_prep/perturb_screens/prep_arce_eval.py` | `arce_perturb_citeseq/arce_{rna,prot}.h5ad` |
| Arce hashtag strata (for pseudobulk DE) | `bash data_prep/perturb_screens/arce_hto.sh` | `$CMCVAE_RESULTS/paper/analyses/stats/de/arce_hto_triplets.tsv` |
| Papalexi screen (untouched test) | `python -I data_prep/perturb_screens/prep_papalexi.py <papalexi_2021.h5mu> results/paper/analyses/untouched/data/papalexi_proc`; guide-clean control variant: `prep_papalexi_clean.py` | `results/paper/analyses/untouched/data/papalexi_proc/papalexi_{rna,prot}.h5ad` |
| Perturb-seq (K562 + Jurkat) | `data_prep/perturbseq/prep.py`; control-only statistics `prep_nt.py` | `$CMCVAE_RESULTS/replogle/data/` (shared 3,000 features, 70/30 gene split seed 42, cell-level DE), `results/paper/analyses/perturbseq/data/*_ntstats.npz` |
| **Condition-matched pseudobulk DE** (guide × stratum, limma-trend-like moderated t, BH) | `python -I data_prep/pseudobulk_de/pb_de.py {frangieh,iccite,arce}` (library: `pblimma.py`) | `results/paper/analyses/stats/de/pb_<screen>.pkl` |
| Reference networks aligned to the vocabulary (CollecTRI, DoRothEA, STRING, co-expression) | `python data_prep/reference_networks/build_ref_networks.py 2c` (needs internet) | `results/ref_nets_v3_2c.{npz,json}` |
| Signed KnockTF + CollecTRI prior (Supplementary Note 5) | `data_prep/reference_networks/build_signed_prior.py` | `priors/signed_prior_2c_v4.pt` |

## 2. Vocabulary and training corpora (`data_prep/vocab/`)

| Corpus | Script | Output |
|---|---|---|
| Protein panel link (v3, protein identical in v4) | `python data_prep/vocab/build_vocab_v3.py 2c` | `combined_v3/` |
| **Final corpus**: 809,130 cells, 2,692 genes (seurat_v3 HVG with cohort as batch key, mito/ribo removed, CollecTRI TFs HVG or ≥5 % detected, HVG RNA-binding proteins, perturbation targets), 144 proteins | `python data_prep/vocab/build_vocab_v4.py 2c 0.05` | `combined_v4/{rna_combined,citeseq_combined}.h5ad`, `vocab_genes.csv` (also in `weights/`) |
| Cohort-sweep corpora (Supplementary Table 1) | `OUT_DIR=$CMCVAE_DATA/combined_combo_<name> COHORTS=<hao,ste,com,su subset> python data_prep/vocab/build_vocab_multi.py pooled`; COMBAT only: `build_vocab_combat.py`, `make_combat_input.py` | `combined_combo_*/`, `combat_v4vocab/` |
| Healthy / COVID-19 split (Supplementary Fig. 2) | `python data_prep/vocab/hd_split_build.py` | `results/hd_split/{healthy,disease}/` |

Optional knowledge priors (not used by the final model, `prior_w_final = 0`). They are used by the prior-weight
experiments of Supplementary Note 5:

```bash
python src/generate_priors.py --rna_path data/combined_v4/rna_combined.h5ad \
    --prot_path data/combined_v4/citeseq_combined.h5ad --out_prefix vocab4_2c --out_dir priors
```

If the prior files are missing, `src/train.py` warns and uses empty masks. This gives the same result
for configs whose prior weight is 0.

## 3. Training (`src/`, `configs/`)

```bash
# final model (seed 42; trained on one H100 GPU); seeds 1 and 2: configs/final/sp_dec_gl1000_s{1,2}.json
python src/train.py --config configs/final/sp_dec_gl1000.json --no_wandb
```

| Paper item | Configs | Launcher |
|---|---|---|
| Final model, 3 seeds (Supplementary Table 8, Note 12) | `configs/final/` | `src/train.py --config ...` |
| Latent / projected / decoder sparsity sweep that selected the group-lasso decoder (Supplementary Note 1, Table 11 "archived L1 runs") | `configs/sweep_sparse/` (generated by `src/grid_search/gen_sparsity_sweep.py` from `configs/base/linfb_2c_nodag_v4.json`) | `scripts/slurm/sweep_eval.sh <run>` evaluates each |
| Training-cohort sweep (Supplementary Note 2, Table 1) | `configs/combo_gl/` | `scripts/slurm/cohort_sweep_train_eval.sh <name>` → `analysis/supplementary/cohort_sweep/collect_combo_gl.py` |
| DAGMA acyclicity comparison (Supplementary Note 1) | `configs/acyclicity/` | `src/train.py` |
| 27-configuration architecture/regularisation sweep (Supplementary Note 5, Table 4, Supplementary Fig. 4b-e): KL / free bits, ε-bottleneck, nonlinear SCM, directed-prior weight, equal latent dims (Note 8), seeds, KnockTF signed prior | drivers in `src/grid_search/` (`run_batch_starve_eps.py`, `run_batch_epsbottleneck*.py`, `run_batch_nl_sweep.py`, `run_batch_priorwsweep.py`, `run_batch_linfb_variants.py`, `run_batch_linfb_p128.py`, `run_batch_linfb_seeds.py`, `run_signedprior.py`) | each driver rebuilds its configs from the base run `results/linfb_2c_fb10_b20_v4/config.json` (copy `configs/base/linfb_2c_fb10_b20_v4.json` there) and selects one experiment with `EXPERIMENT_OFFSET` |
| Additional objectives (Supplementary Note 7, Tables 5-6, Figs 5-6): conditional (iVAE-style) priors, Laplace ε prior, total-correlation / HSIC penalties, reversal loss | `configs/cond_prior/`, `configs/eps_prior/`, `configs/resid_indep/`, `configs/rev_loss/` | `scripts/slurm/{cond_prior,eps_prior,resid_indep,rev_loss}_train.sh <name>` |
| Healthy vs COVID-19 models (Supplementary Fig. 2) | `configs/hd_split/` | `scripts/slurm/hd_split_train.sh {healthy,disease}` |
| A_rr fixed to 0 during training (Fig. 3d retrain control) | `configs/arr0/` (`fix_A_rr_zero: true`) | `src/train.py --config configs/arr0/review2_arr0_s42.json --no_wandb` |
| RNA-only CMC-VAE on the same cells (pairing control used in Supplementary Fig. 9) | — | `analysis/fig3_zero_shot/ko_invariant_refs/train_rnaonly.py <seed> <outdir>` |

`src/train.py` writes `best_model.pt`, `final_model.pt` (last epoch, **used for all evaluations**),
`config.json`, `metrics.json` and the projected matrices (G = D A Dᵀ blocks, decoder loadings) to `results_dir`.

## 4. Baselines (`analysis/baselines/`)

These outputs feed Fig. 2, Fig. 3 and Supplementary Note 4.

| Baseline | Scripts | Output |
|---|---|---|
| totalVI (same cells, batch = donor, 50 epochs) | `prep_totalvi_adata.py` → `run_totalvi.py` → `score_totalvi.py`; KO read-out `perturb_totalvi_ko.py` | `results/baselines/totalvi_raw.npz`, scores |
| GRNBoost2 (CollecTRI TF list `collectri_tfs_v4_2c.json`) | `run_genie3.py --method grnboost2 --mode all` (separate env with arboreto; `_arboreto_patch.py` makes arboreto work with current dask) → `score_genie3.py` | `results/baselines/genie3_matrix_grnboost2_all_809k.npz` |
| DeepSEM (L1 α = 100, spectral clamp ρ ≤ 0.9) | `run_deepsem.py` (needs `DEEPSEM_DIR`) → `ko_genegraph.py` / `perturb_genegraph_ko.py`, `score_genegraph_recovery.py`; observed responses for matched fine-tuning: `dump_observed_de.py` | `results/baselines/deepsem_W_809k_clamp09.npz` |
| Protein imputation and the 74-pair cognate map | `cmcvae_cognate_impute.py` | `results/baselines/_cmcvae_cognate.json` (cognate map used by many later scripts) |
| Correlation / partial / within-cell-type correlation on the 809,130 training cells | `analysis/fig2_networks/prep_train_stats.py` | `results/paper/analyses/baselines/train_stats/joint_stats.npz` |

## 5. Figure 2 and Supplementary Notes 3, 4, 9, 10, 12: learned networks (`analysis/fig2_networks/`)

| Panel / item | Script(s) |
|---|---|
| Fig. 2a reference-network recovery (co-expression, STRING, DoRothEA, CollecTRI), Fig. 2b within-pathway AUROC, Fig. 2c cognate map, Fig. 2d cognate nulls (1,000 permutations), Fig. 2e sparsity / Gini / top-1 % mass, Fig. 2f surface complexes; all with information-matched baselines and paired tests | `fig2_baselines.py` (needs §4 outputs and the three seeds) → `results/paper/analyses/baselines/fig2_baselines.json`; panels: `make_baseline_figures.py`, `analysis/figures/original/fig2_observational.py` |
| Gene-block (node) bootstrap / within-gene permutation for the Fig. 2a,b differences | `gene_block_bootstrap/task9_prep.py` → `task9_gbboot.py {boot,perm}` → `task9_aggregate.py` |
| Projected-network recovery and pathways (inputs to Fig. 2a,b) | `grn_recovery.py`, `recovery_all_models.py`, `pathway_recovery.py`, `causal_eval_suite.py` (in `analysis/common/`) |
| Cognate enrichment and nulls | `translation_nulls.py`, `cognate_nulls_csv.py` |
| Cognate map vs raw RNA-protein correlation (Supplementary Note 15, 8.22× vs 8.20×) | `cognate_corr_baseline.py` |
| Sparsity of A and of G (Supplementary Fig. 3d,e) | `dag_sparsity.py`, `grn_sparsity_gini.py` |
| Surface complexes and intracellular PPI (Supplementary Note 4, Fig. 3c) | `ppi_surface_complexes.py`, `ppi_feedback_eval.py`; figure `analysis/figures/original/figS7_ppi_surface.py` |
| Protein imputation vs totalVI (Supplementary Fig. 3a, Table 2) | `crossmodal_imputation.py`, `analysis/baselines/score_totalvi.py`; figure `analysis/figures/original/figS_baselines_ppi.py` |
| Same-cell CMC-VAE vs correlation across the cohort-sweep corpora | `corpora_compare.py` |
| Latent immune programs (GSEA on decoder loadings, program graph, hubs; Supplementary Note 3, Supplementary Fig. 1) | `gsea_latents.py`, `graph_topology.py`; figure `analysis/figures/original/fig_latent_graph_new.py` |
| Healthy vs COVID-19 programs (Supplementary Fig. 2) | `hd_compare.py`, `healthy_vs_disease_grn.py`; figure `analysis/figures/original/fig_latent_graph_hd.py` |
| Protein→RNA map (transpose component, receptor target sets; Supplementary Note 10, Fig. 7) | `feedback_biology.py`; figure `analysis/figures/original/figS8_feedback_biology.py` |
| Seed stability (Supplementary Note 12, Table 8) | `seed_stability.py`, `seed_collect.py` |

## 6. Figure 3: zero-shot knockout-response ranking (`analysis/fig3_zero_shot/`)

Prerequisites: §1 pseudobulk DE (`pb_de.py` for all three screens), §4 baselines, and the fine-tuning
gene split. Copy the split so that it matches the paper exactly:
`cp configs/splits/finetune_split_genes.json results/` (the file `finetune_combined.py` writes).

| Panel / item | Pipeline |
|---|---|
| **Fig. 3b,d,e, Supplementary Fig. 8** (CMC-VAE vs totalVI / correlation / DeepSEM; A_rr shuffled / zeroed; per-knockout) | (1) `ST_DE=pb_bh RC_BASELINES=1 python analysis/fig3_zero_shot/stats/rc_stats.py results/sweep_sparse/<run>` for the three seeds. This writes per-knockout predictions and scores to `results/paper/analyses/stats/runs/`. (2) Joint knockout × control-cell bootstrap: `python -I analysis/fig3_zero_shot/stats/boot.py <task> <nrep>` (array job: `scripts/slurm/stats_boot.sbatch`; 999 replicates). (3) `python -I analysis/fig3_zero_shot/stats/analyze.py pb_bh` → `stats_summary_pb_bh.json`. (4) `python analysis/fig3_zero_shot/stats/figs.py` → `results/paper/figures/stats/source_data_fig3_4_5_pb.csv`. Scoring library: `stats/score_lib.py` |
| Information-matched zero-shot comparators (training-corpus correlation etc.) | `zeroshot_baselines/zeroshot_fig3.py <run>` (uses `_rc_header.py`) → `aggregate_zeroshot.py` |
| **Fig. 3c**, untouched Papalexi screen (pre-specified; 3 models per method) | `analysis/fig3_zero_shot/papalexi_untouched/rc_untouched.py <run> results/paper/analyses/untouched/rc_<run>.json` (`scripts/slurm/papalexi_eval.sbatch`) → `stats_untouched.py papalexi 200` → `stats_papalexi.json`; panel `fig_untouched.py` |
| Fig. 3d retrain control (A_rr = 0 during training) | `arr_ablation_retrain/rc_stats_arr0.py` (on the `configs/arr0` models) → `compare.py` |
| Seed replication of Fig. 3 (Supplementary Note 15, "Seed replication") | `seed_replication/point.py` → `boot3.py <task> <nrep>` → `collect.py` |
| Knockdown operators and sign of the in-silico knockout (Supplementary Fig. 12; Methods operator statements) | `operator/prep_ctrl_means.py`, `operator/op_controls.py <run> <out.json>` (env `OP_PARTS`, `OP_DE=pb_bh`) → `operator/aggregate.py` |
| **Supplementary Fig. 9**, knockout-invariant control-cell mean / s.d. rankings | `ko_invariant_refs/`: `r2_build_cache.py` → `r2_dump.py <run> <seedtag> <outdir>` (runs `op_controls_r2.py`) → `r2_papalexi.py`, `r2_papalexi_cmp.py` → `r2_boot.py <task> <nrep> <outdir>` → `r2_analyze.py` → `r2_figs.py` (→ `results/paper/figures/round2/R2_Fig3b_koinvariant_stratified_source_data.json`) |
| DE-threshold sensitivity (Supplementary Note 11) | `analysis/supplementary/de_sensitivity/{de_filter_diag,de_counts,bhfdr_de_sensitivity}.py` |
| Development-time evaluation (cell-level DE; cohort sweep, sparsity sweep, Supplementary Notes 2, 5-8) | `analysis/common/full_perturb_eval.py`, `block_isolated_ablation.py`, `edge_reversal_ablation.py`, `dag_shuffle_ablation.py`, `sweep_eval_model.py`, `sweep_collect.py` (see `scripts/slurm/sweep_eval.sh` for the environment variables) |

## 7. Figure 4: cross-modal translation and control-cell adaptation (`analysis/fig4_cross_modal/`)

| Panel / item | Pipeline |
|---|---|
| **Fig. 4a**, cognate edge strength vs decodability (ρ = 0.90, n = 74; Supplementary Note 9) | `python analysis/fig4_cross_modal/translation_decodability.py results/sweep_sparse/sp_dec_gl1000` → `results/translation_decodability.json` |
| **Fig. 4b**, control-cell RNA→protein imputation before / after unsupervised adaptation (no perturbation labels; 90/10 control-cell split, early stopping) | `adapt/prep_ctrlinfo.py`; `python -I analysis/fig4_cross_modal/adapt/adapt.py sp <s42\|s1\|s2> frangieh,iccite,arce i` (variant `i` = the adapted model of the paper) → `adapt/build_cache.py` (encoded-cell caches) → `adapt/mm/mm_transfer.py` (env `MM_RD`, `MM_OUT`, `MM_OP`, `MM_TAGS`) → `adapt/aggregate.py` → `adapt/fig_adapt.py` → `results/paper/figures/adapt/source_data_adapt.json` |
| **Fig. 4c,d**, translating measured knockout RNA responses into protein responses (204 knockouts, 3 screens, 3 models; centred pearDE, retrieval) | `XM_RUNS='<json: seed → screen → model dir>' python -I analysis/fig4_cross_modal/adapt/mm/mm_translate.py <out.json>`, run for the unadapted (`sp_none`) and adapted (`sp_i`) models → `python -I analysis/fig4_cross_modal/rf_fig4d.py results/paper/figures/reviewfix/fig4d_ci.json` |
| Same architecture trained from scratch on the control cells (Supplementary Note 15, "adaptation control") | `scratch_control/scratch_adapt.py <seedtag> frangieh,iccite,arce <arms>` → `aggregate_scratch.py` |

Fig. 4c significance (adapted vs PBMC-model imputation; two-sided Wilcoxon signed-rank test over proteins,
Benjamini–Hochberg across screens): `python analysis/fig4_cross_modal/imputation_tests.py` →
`results/paper/figures/adapt/fig4b_imputation_tests.json`, which `analysis/figures/fig4.py` reads.

## 8. Figure 5: interventional fine-tuning (`analysis/fig5_interventions/`)

| Panel / item | Pipeline |
|---|---|
| **Fig. 5b**, synthetic latent graph, interventional recovery with known targets (8/16 RNA latents, 15 replicates each) | `analysis/fig5_interventions/synthetic/synthetic_cmcvae.py --families <f> --seeds <s> --dr <8\|16> --dp <4\|8> --genes <G> --proteins <P> --out <json>` (one job per fit: `scripts/slurm/synthetic_cmcvae.sbatch`) → `aggregate_synthetic.py` → `results/paper/figures/identifiability/SuppFig_synthetic_interventional_recovery_source.csv` |
| **Fig. 5c-f, Supplementary Fig. 11, Supplementary Table 10**: Perturb-seq K562 / Jurkat | (1) `data_prep/perturbseq/prep.py`, `prep_nt.py`. (2) RNA-only CMC-VAE: `perturbseq/train_rna.py <jurkat\|k562> <seed> <obs\|ctrl>` (`scripts/slurm/perturbseq_train.sbatch`). (3) DeepSEM on the same cells: `perturbseq/run_deepsem.py --line <l> --cells <obs\|ctrl> --out results/paper/analyses/perturbseq/deepsem/<l>_<cells>_W_seed<s>.npz --seed_list <s> --n_epochs 40 --spectral_clamp 0.9`. (4) Fine-tuning arms, controls and baselines: `PS_PRE=<obs\|ctrl> RP_PARTS=obs,ft,nb,xctx2 RP_OUT=results/paper/analyses/perturbseq/runs/<line>_<pre>_s<seed>.json python -I analysis/fig5_interventions/perturbseq/ft_eval.py <line> <seed> shift`. The learning curve uses `RP_PARTS=obs,lc`; matched DeepSEM fine-tuning uses `RP_PARTS=obs,ds PS_DS_W=<deepsem W.npz>` (`scripts/slurm/perturbseq_ft.sbatch`). (5) `perturbseq/aggregate_rev.py` → `results/paper/analyses/perturbseq/{summary.json,tables.md}`; panels `fig_perturbseq.py` |
| Training-set-size curve and control-only arms (Fig. 5d,f) | `perturbseq/review2/ft_eval_r2.py` → `review2/aggregate_r2.py` |
| External Perturb-seq predictors (GEARS, Ahlmann-Eltze linear baseline, scGPT-embedding ridge; Supplementary Fig. 11) | `perturbseq/external/ps_gears.py <line> 42` (separate env with cell-gears and torch_geometric), `ps_linear.py <line> 42`, `ps_scgpt_emb.py <line>` (needs the scGPT whole-human checkpoint) |
| **Supplementary Fig. 10, Table 9**: CITE-seq fine-tuning with label-shuffled, screen-mean and sign-flipped targets and InfoNCE (3 models) | `analysis/fig3_zero_shot/stats/rc_stats.py` (the fine-tuning arms are run by default in the same job as Fig. 3) and `analysis/common/reviewer_controls.py` → `reviewer_collect.py`. The original fine-tuning harness (gene split, loss of eq. 5): `citeseq_finetune/finetune_combined.py results/sweep_sparse/<run>` |
| Leave-one-cohort-out fine-tuning (Supplementary Note 14) | `citeseq_finetune/finetune_loto.py`; with shared-response controls: `citeseq_finetune/loco_cohort.py` (env `LOCO_HOLD`, `LOCO_MODE`) → `loco_aggregate.py` |
| Direction penalties during fine-tuning (DAGMA / asymmetry; Supplementary Note 15) | `citeseq_finetune/dir_orient.py` → `dir_orient_post.py` → `dir_collect.py` |
| Robustness to the DE definition for Perturb-seq and Papalexi (Supplementary Note 16) | `analysis/supplementary/de_robustness/`: `de_perturbseq.py <line>`, `ft_eval_de.py`, `rescore_perturbseq.py`, `papalexi_de_rescore.py`, `make_report.py` |

## 9. Other supplementary analyses (`analysis/supplementary/`)

| Item | Script(s) |
|---|---|
| Synthetic orientation study, oracle latent access vs decoder inversion vs our read-out (Supplementary Note 6, Table 3, Fig. 4a) | `identifiability/synthetic_oracle.py`, `identifiability/synthetic_identifiability.py` |
| DirectLiNGAM on the learned ε, non-Gaussianity of ε (Supplementary Note 6) | `identifiability/lingam_realdata_probe.py` |
| Proposition 1 numerical check (reparameterisation) | `identifiability/reparam_check.py` |
| Conditional-prior (iVAE) seed replicates with shared-response controls (Supplementary Tables 5-6) | `identifiability/reviewer_controls_ivae.py <run>` → `aggregate_ivae.py`, `fig_ivae_controls.py` |
| Elliptical ε test (random-direction / coordinate kurtosis ratio, Supplementary Note 7) | `identifiability/eps_elliptical_test.py` |
| ε vs z structure (ε-bottleneck results, Supplementary Note 5) | `objectives/eps_vs_z_structure.py` |
| KL-pressure sweep summary | `objectives/agg_starve_eps.py` |
| Laplace ε prior, TC / HSIC penalties, reversal loss: symmetry and pearDE differences (Supplementary Table 5, Fig. 5) | `objectives/eps_prior_collect.py`, `objectives/eps_prior_symkurt.py`, `objectives/resid_indep_metrics.py`, `objectives/collect_resid_indep.py`, `objectives/collect_rev_loss.py`; figure `analysis/figures/original/figS_orientation_battery.py` |
| Conditional priors and fitted-operator distance (Supplementary Fig. 6, Table 6) | figure `analysis/figures/original/fig_scm_depletion.py` |
| KnockTF signed prior on held-out TFs (Supplementary Note 5) | `objectives/signed_prior_heldout_eval.py <baseline_dir> <signedprior_dir>` |
| Latent-graph sparsity creates asymmetry, not orientation (Supplementary Table 11) | `objectives/latent_sparsity_orientation.py` |
| Architecture sweep figure (Supplementary Fig. 4) | `analysis/figures/original/fig_identifiability_sweep.py` |
| Data QC: kurtosis with and without ComBat, cohort / donor mixing, standardisation leakage (Methods; Supplementary Note 13) | `qc/data_skew_diag.py`, `qc/latent_batch_diag.py`, `qc/pca_umap_cohorts.py`, `qc/std_leakage_check.py` |

## 10. Figures (`analysis/figures/`)

| Figure | Script | Inputs |
|---|---|---|
| Fig. 1 | schematic (not code-generated) | — |
| Fig. 2 | `analysis/figures/original/fig2_observational.py` (and `analysis/fig2_networks/make_baseline_figures.py`) | §5 outputs |
| Fig. 3 | `analysis/figures/fig3.py` | `source_data_fig3_4_5_pb.csv`, `stats_summary_pb_bh.json`, `stats_papalexi.json` |
| Fig. 4 | `analysis/figures/fig4.py` | `translation_decodability.json`, `source_data_adapt.json`, `fig4b_imputation_tests.json`, `fig4d_ci.json` |
| Fig. 5 | `analysis/figures/fig5.py` | synthetic source CSV, `perturbseq/summary.json`, `tables.md` |
| Supplementary Figs 8-12 (new in the revision) | `analysis/figures/figS.py` | as above + `R2_Fig3b_koinvariant_stratified_source_data.json` |
| Supplementary Figs 1-7 | `analysis/figures/original/*.py` (see §5 and §9) | |

All final figures go to `$CMCVAE_RESULTS/paper/FINAL_FIGURES/`. The final-figure style uses the Carlito font,
which must be installed in `~/.fonts`. `fig3.py` and `fig5.py` paste the schematic panels (3a, 5a) from the
original vector figure files. They expect `Figures_Main/Figure3.pdf` and `Figure5.pdf` under
`$CMCVAE_ROOT`. These artwork files are not distributed. Without them, the two scripts write the data panels to
`FINAL_FIGURES/_parts/` and then fail at the compositing step.

## Statistics

Statistical procedures are described in Methods ("Statistics and reproducibility"). They are implemented in
`analysis/fig3_zero_shot/stats/{boot,analyze}.py` (Fig. 3 joint bootstrap, BH), `papalexi_untouched/stats_untouched.py`
(Fig. 3c), `fig4_cross_modal/rf_fig4d.py` and `adapt/aggregate.py` (Fig. 4), and
`fig5_interventions/perturbseq/aggregate_rev.py` and `synthetic/aggregate_synthetic.py` (Fig. 5).
