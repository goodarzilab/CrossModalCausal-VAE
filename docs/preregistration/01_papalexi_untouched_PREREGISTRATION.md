# Pre-registration: frozen-pipeline evaluation of CMC-VAE on an untouched multimodal CRISPR screen

Written 2026-11-03, BEFORE any CMC-VAE prediction, baseline prediction or metric was computed on the
Papalexi (or Mimitou) data. At the time of writing only the data structure had been inspected:
cell counts per guide target, ADT/HTO/GDO feature names, and symbol overlap with the model vocabulary.
No differential expression, prediction or score had been computed. Addresses review point M2
("freeze the full pipeline and evaluate it once on an untouched multimodal screen").

We commit to reporting every result listed here, including null, negative and undefined results,
and to labelling clearly any analysis added after this file was written as post hoc.

## 1. Untouched screen

Papalexi et al., Nat. Genet. 53, 322–331 (2021). ECCITE-seq in THP-1 monocytic cells stimulated
with IFN-γ, with 25 targeted genes plus non-targeting (NT) guides and 4 surface ADTs
(CD86, PD-L1, PD-L2, CD366/TIM-3). GEO GSE153056. Processed object used: the pertpy `papalexi_2021`
MuData (figshare file 36509460, the Seurat/SeuratData `thp1.eccite` object converted to h5mu), downloaded
to `data/papalexi_pertpy/papalexi_2021.h5mu` (sha256 f868ceb4c92d572b9b006309c0d0314991ae766dee1a7291a21e13600fbc0725).
None of the CMC-VAE development steps (cohort choice, vocabulary, architecture sweep, fine-tuning weights,
operator choice, DE thresholds) used this screen.

Disclosed overlap (not data leakage, but not full independence): 9 of the 18 Papalexi targets present in
the model vocabulary were also targets in a development screen (Frangieh: IFNGR1, IFNGR2, JAK2, MYC, SMAD4,
STAT1, STAT3; Chen: SMAD4, STAT5A; Arce: IRF1, MYC). Perturbation-target genes of the development
screens were added to the vocabulary, so these genes are in the model partly because of the development
screens. The other 9 (ATF2, BRD4, CAV1, CD86, IRF7, NFKBIA, POU2F2, SPI1, STAT2) are novel targets.
A secondary breakdown by this overlap is pre-specified (Section 7).

## 2. Frozen model and operator

- Models: `results/sweep_sparse/sp_dec_gl1000{,_s1,_s2}/final_model.pt` (seeds 42, 1, 2);
  sha256 prefixes df69facdfd547287, 7c99706095a76775, 98c30e1a40273161. Not `best_model.pt`. No retraining.
- Operator: the decoder-guided edge-cut do-operation exactly as in `eval/reviewer_controls.py`
  (`predict_target`, RC_MUT=1, RC_SHOCK=orig; Methods eq. 4): incoming RNA-latent edges scaled by (1 − s),
  s = |D_r[g,:]|/max|D_r[g,:]|, plus the exogenous shift; encoder posterior means of up to 4,000 control
  cells (fixed seed 42 subsample); prediction = mean decoded difference; knocked-out gene excluded from scoring.
- Code: frozen copy of `eval/reviewer_controls.py` (git 947189200b94, md5 1c890c3e) and
  `eval/perturb_validation_analysis.py` (md5 13fd1018); the copy only adds a `papalexi` context, redirects
  caches away from `results/`, and adds the baselines below. The scoring functions (`pe`, `au`, `score`,
  `ref_means`) are used unchanged.

## 3. Data preparation (frozen to match the development screens)

- Cells: all guide-assigned cells in the provided object (all are HTO-classified IFN-γ-treated "tx" singlets,
  replicates rep1–rep3). Control = NT guide cells (`gene_target == "NT"`), pooled across replicates.
  Condition matching is automatic because every cell in the object is IFN-γ stimulated.
- Primary analysis uses all guide-assigned cells (as for Frangieh/Chen/Arce, which did not use escape filtering).
- RNA: raw counts → library-size normalized to 10,000 → log1p. Gene symbols matched exactly to the 2,692-gene
  vocabulary (as for the development screens; no alias remapping). Vocabulary genes missing from the screen are
  set to 0 in standardized input space (= training mean), the existing behaviour of `encode_control_cells`.
- Protein: raw ADT counts → CLR with `muon.prot.pp.clr` (the normalization used for the training corpus and the
  Chen screen). Names mapped to the 144-marker panel: CD86→CD86, PDL1→CD274, PDL2→CD273, CD366→CD366.
  The 140 panel proteins not measured are set to 0 in standardized input space (training mean), exactly
  as for Frangieh (7 proteins), Chen (78) and Arce (84).
- Units: target genes in the model vocabulary with ≥30 cells (expected: 18 targets).

## 4. DE ground truth (Methods, unchanged)

Per-feature LFC = difference of mean log-normalized (RNA) / CLR (protein) expression, perturbed vs all control
cells. DE = raw two-sided Mann–Whitney P < 0.05 and |LFC| ≥ 0.10 (RNA) / ≥ 0.05 (protein). A unit enters
RNA scoring with ≥3 DE RNA features (target gene excluded); pearDE requires ≥5 DE features; protein AUROC
requires ≥3 DE proteins and at least one non-DE protein. With only 4 measured proteins, protein pearDE
(≥5 DE proteins) is undefined by construction, and protein AUROC is defined only when exactly 3 of 4 proteins
are DE. We pre-commit to reporting protein metrics as undefined/sparse rather than relaxing thresholds; a
descriptive post-hoc look at the 4-protein responses (sign agreement of predicted vs measured change for the
four proteins) will be labelled descriptive.

## 5. Metrics (all from `score()`)

Primary endpoints:
  (P1) RNA mean-centred pearDE on the common (log) scale (`rr_pe_cent`), centring by the leave-one-out mean
       response of the other Papalexi knockouts (applied to both predicted and measured responses);
  (P2) held-out-gene retrieval (`rr_retrieval`, chance 0.5), on features DE in ≥1 Papalexi knockout,
       excluding all target genes (zero-shot: all evaluable units define the feature set; CV: training folds).
Secondary: RNA AUROC (`rr_auroc`), signed RNA pearDE (`rr_pe`, and `rr_pe_log`), mean-response reference
(`rr_pe_meanresp`, `rr_auroc_meanresp`), target-shuffled pearDE (`rr_pe_shuf`), protein AUROC/pearDE/retrieval
where defined.

## 6. Comparators (all scored with the knockdown sign)

- Correlation baseline: negated row of the training-corpus RNA–RNA (and RNA–protein) Pearson correlation.
- totalVI: negated rows of `results/baselines/totalvi_raw.npz` (G_rr_tv, G_rp_tv).
- DeepSEM: negated column of (I − Wᵀ)⁻¹ from `results/baselines/deepsem_W_809k_clamp09.npz` (RNA only).
- Zero-shot rankings (knockout-invariant; AUROC is their intended use; signed metrics reported with the
  "predicted decrease ∝ score" convention; centred pearDE and retrieval are undefined/chance by construction):
  (a) mean control-cell expression of each feature, (b) control-cell variance, (c) CMC-VAE decoder-row norm ‖D_r[j,:]‖.
- Mean-response reference (model-free, leave-one-out over the other Papalexi knockouts).
- Cross-context copy reference (post-specified here, before computing): the same gene's measured response in a
  development screen (Frangieh/Chen/Arce), for the 9 overlapping targets only.
- Ridge baselines (requested in the review): in the secondary designs only, because ridge needs training
  perturbations (Section 8): ridge from the target's CMC-VAE decoder loadings D_r[g,:] and from its
  training-corpus co-expression profile.

## 7. Statistics

- Each seed is evaluated separately; the headline per-unit CMC-VAE value is the mean over the three seeds.
- CMC-VAE vs each comparator: two-sided paired t-test over units (Wilcoxon signed-rank as a check);
  BH correction within each metric panel (family = comparators for one metric); 95% CIs of the mean paired
  difference by bootstrap over units (10,000 resamples). One-sample two-sided tests of P1 against 0 and P2
  against 0.5. A joint unit + cell bootstrap (perturbed and control cells resampled, DE recomputed, B = 200)
  for the CMC-VAE P1/P2 CIs.
- Decision rule: CMC-VAE "beats" a comparator on an endpoint only if the seed-mean paired difference is > 0
  with BH q < 0.05. A non-significant difference is reported as "no evidence of a difference" (n ≈ 18, low power),
  not as parity.
- Pre-specified breakdowns (descriptive): novel targets vs targets shared with a development screen; per seed.
- Sensitivity (pre-specified): mixscape-style escape filtering — perturbation signatures (each cell minus the
  mean of its 20 nearest NT neighbours in 40-PC space of the screen), projection on the mean perturbed-minus-NT
  signature direction per target, 2-component Gaussian mixture with the NT component fixed; cells with posterior
  P(KO) > 0.5 retained; targets with <30 retained cells drop out. All primary metrics re-reported on this set.

## 8. Secondary designs (also pre-specified)

(S1) Within-Papalexi fine-tuning. Frozen fine-tuning hyperparameters from the paper (Methods eq. 5:
w_dir = 1, w_mag = 1, w_rp = 0.5, w_ρ = 0.1, ρ₀ = 0.9, guard 0.92, lr 5×10⁻⁴, 150 steps, patience 30,
weight decay 1×10⁻⁴, 3,000 control cells; GPU). Because only ~18 units exist, a single split is too small;
we use 3-fold cross-validation over target genes (fold assignment by `RandomState(42)` permutation of sorted
units). For each fold and seed: observational baseline, fine-tuning on real targets, label-shuffled targets
(3 draws) and screen-mean targets; ridge(decoder loadings) and ridge(co-expression) fitted on the training fold
(RidgeCV, alphas 10^[-2..4]). Held-out scoring uses training-fold references for centring/mean response and
retrieval among the held-out fold. Per-unit held-out values are pooled across folds.
Tests: fine-tuned vs observational and fine-tuned vs label-shuffled / screen-mean (two-sided paired, BH within panel).

(S2) Cross-screen transfer. The paper's fine-tuned graph (Frangieh+Chen+Arce training genes, the Fig. 5
`ft_standard` protocol, reproduced in the same run and checked against the paper's held-out values) applied
zero-shot to Papalexi; controls: the same transfer from label-shuffled (3 draws) and screen-mean fine-tunes.
Ridge transfer comparators: ridge from decoder loadings and from co-expression fitted on all development-screen
units (responses centred within screen) and applied to Papalexi. Primary endpoints P1/P2 as above.

## 9. Optional second untouched screen

Mimitou et al. 2019 ECCITE-seq (GSE126310) only if it yields ≥5 evaluable units with the same rules; otherwise
reported as not evaluable, with the reason.
