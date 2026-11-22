# Workstream `adapt` — PRE-REGISTRATION (control-cell domain adaptation)
Written 2026-11-06, BEFORE any adapted model was trained to completion and BEFORE any perturbation metric (or the
control-cell imputation diagnostic) was computed for an adapted model. Only mechanics smoke tests were run: 60 optimisation steps
on Chen and gw48, used only to check that the validation loss decreases. They showed one bug: BatchNorm batch statistics were
corrupted by single-modality minibatches. Fix: BN running statistics are frozen in every variant. Smoke outputs were deleted.
Code: `paper_nov_revisions/code/adapt/` (adapt_lib.py holds every hyperparameter below verbatim).

## Hypothesis
The cross-modal / perturbation bottleneck is domain transfer: the PBMC-trained model does not represent the screens' cell systems.
Adapting it to each screen's UNPERTURBED control cells is unlabelled and observational (no perturbation label, no KO cell),
so it cannot leak labels. It should restore the RNA→protein map, and possibly perturbation prediction.

## Base models
- PRIMARY base: sp_dec_gl1000, seeds 42/1/2 (`final_model.pt`).
- SECONDARY base: xm_l0.03 (routing-loss model, nov_xmodal), seeds 42/1/2.

## Domains and data (one adapted model per base × seed × variant × domain)
- **frangieh / iccite (Chen) / arce:** the screen's control cells from `analyses/stats/de/pb_<t>.pkl` (ct.cells_r/cells_p:
  16,319 / 2,938 / 20,420 cells). These are the same cells as the control-cell ridge in the multimodal workstream.
  - Inputs are standardised with the training-corpus mean/sd.
  - Vocabulary features not measured in the screen are set to 0 (= training mean) and masked out of every loss.
  - Their decoder rows and biases receive no gradient, so they stay at pretrained values.
- **gw48 / gw8:**
  - RNA side: GWCD4i non-targeting pseudobulks (keep_min_cells) of Stim48hr (3,441) or Stim8hr (3,475). These are log-normalised,
    standardised with training stats, and 419 unmeasured vocab genes are masked.
  - Protein side: 20,000 random SCITO non-targeting cells (RandomState(0)), on the 109 panel proteins.
  - **SCITO scale:** values are the authors' per-batch deviance residuals, z-scored per protein (not CLR). They are used directly
    as the standardised protein input: both are unit-variance per protein, and per-protein offsets are absorbed by the adapted
    decoder bias and encoder. No rescaling is fitted.
  - The two modalities are unpaired. Each step sums an RNA-only loss on a GWCD4i minibatch and a protein-only loss on a SCITO
    minibatch. There is no routing loss, because no unit has both modalities.
  - gw48 serves the pre-registered GWCD4i→SCITO headline (Stim48hr) and the fixpert Stim48hr arm. gw8 serves the Stim8hr
    sensitivity analysis and the fixpert Stim8hr learning curves.
- **Split:** 10% of the domain's control units are held out (RandomState(12345), seed-independent) for early stopping and the
  diagnostic. The information-matched control-cell ridge baselines are fitted on the same 90%.

## Variants (all pre-specified, no tuning)
- **(i) PRIMARY.** Fine-tune encoders, decoders (measured-feature rows and biases) and A (all SCM gate logits and weights).
- **(ii)** Same as (i) with A frozen.
- **(iii) Minimal.** Feature-wise affine only: BN affine (γ, β) of both encoders (hidden-unit scale-shift; running statistics
  frozen) plus a per-feature decoder scale and bias. The scale is folded into the decoder rows when the model is saved. lr 1e-3.
- **(i_xm1) Sensitivity.** Variant (i) with λ_xm = 1.0. CITE-seq screens only.
- **Secondary base:** xm_l0.03 is adapted with variant (i) only.

**Objective** (the final-epoch observational objective of sp_dec_gl1000 plus the routing loss):
- Reconstruction: rec_weight 100 × (masked MSE_rna + masked MSE_prot), where each MSE is the mean over measured elements.
- Routing: + 100 × λ_xm × masked MSE(x_p, D_p z_p(ε_r, ε_p = 0) + b_p), with λ_xm = 0.1 (paired screens only).
- KL: + 2.0·KL_r + 2.0·KL_p, closed form, with free bits 1.0 / 0.5 per dimension.
- If A is trainable: + 20 × relu(σ_max(A) − 0.9).
- If the decoders are trainable ((i)/(ii)): + 5 × (ortho(D_r) + ortho(D_p)) + 50 × grouplasso16(D_r) + 0.9 × grouplasso16(D_p).
- Forward pass: α = 1, deterministic gates at τ = 0.5 (exactly as every eval harness), RNA input feature dropout 0.1, and the
  reparameterised ε sample during training.

**Optimisation:**
- Adam, lr 1e-4 ((i)/(ii)/(i_xm1)) or 1e-3 ((iii)), batch 256, grad-clip 1.0, max 4,000 steps.
- **Early-stop rule:** every 50 steps, compute the deterministic held-out-10% criterion: the data part of the objective
  (reconstruction + routing + β·KL) with posterior-mean ε, summed over sources. Stop after 10 evaluations with no improvement and
  restore the best state. If no evaluation beats step 0, the model stays unadapted.
- Seeds: the model seed is also the adaptation RNG seed.
- Hardware: SLURM; CPU if no GPU is free (the problem is small).

## Endpoints
**(a) Mechanism check (diagnostic, not a test of the hypothesis' payoff).** In the held-out 10% of control cells:
- median per-protein Pearson r of protein imputed from RNA only (protein input = training mean = 0; encoder means → SCM → D_p);
- the same with the protein input given;
- the RNA-path share of decoded protein variance;
- compared against the control-cell ridge fitted on the same 90% (RidgeCV per protein, α ∈ logspace(0, 8, 17)) and against the
  unadapted model.

Pre-specified "mechanism restored" = the primary variant reaches ≥ 0.5 × the ridge r in both Chen and Arce (seed-mean).

**(b) Held-out perturbation metrics.** Existing harnesses, unchanged scoring, signed soft operators, condition-matched pseudobulk
DE ground truth (BH q < 0.05, |LFC| ≥ 0.10 RNA / 0.05 protein).
- **b1 Translation (measured RNA → protein; `code/adapt/mm/mm_translate.py`).**
  - Each screen is encoded and decoded by its own adapted model: CMC-enc route, seed-mean prediction.
  - Units: KOs with ≥ 3 DE proteins (204).
  - Metrics: centred pearDE, retrieval, pearDE, AUROC, per-protein r across KOs.
- **b2 CITE-seq transfer (`code/adapt/mm/mm_transfer.py`, main part).**
  - Same 5-fold gene CV, arms (obs / ftR / ftR_shuf / ftP / ftP_shuf / ftRP), inner-split step selection and scorer as the
    multimodal workstream.
  - Run PER SCREEN (MM_TAGS = one screen), because each screen has its own adapted model and A. The unadapted base is re-run
    under the identical per-screen protocol and is the matched comparator.
  - Protein metrics: rp_pe_cent, rp_retrieval, rp_auroc, rp_pe. RNA metrics: rr_pe_cent, rr_retrieval, rr_auroc.
  - Operators: signed clamp_zero (primary) and cutclamp_zero (secondary; base sp, variants none and (i) only).
- **b3 GWCD4i → SCITO.**
  - `code/adapt/mm/mm_gwscito.py` + `gwscito_score.py`, unchanged from `analyses/multimodal/gwscito/PREREGISTRATION.md`
    (incl. amendments 1–2).
  - Model = gw48-adapted (Stim48hr headline) or gw8-adapted (Stim8hr sensitivity). Clamp only.
  - Primary metric M1 pair AUROC; also M2, M4, M3, M5.
- **b4 GWCD4i learning curves (fixpert protocol; `code/adapt/fx/`, copied verbatim with only the model path changed).**
  - gw_Stim8hr, n ∈ {5, 10, 25, 50, 100, 200, all} with fixpert's subsets.
  - Methods: base_shift (zero-shot graph), F_sym_shift_short_pm (harness SCM FT), F_sym_global_lbfgs_ls (corrected protocol),
    cutclamp, rrr_dec, ridge_dec, ridge_ntcorr, linear_ae_ctrlpca, corr_zeroshot, mean_train.
  - Metrics: centred pearDE (canonical) and retrieval.
  - gw_Stim48hr: full training set only.

**(c) Comparators** (same KOs, same scorer):
- the unadapted model under the identical protocol;
- **information-matched control-cell baselines (the fair comparators):**
  - control-cell ridge RNA→protein (ctrlridge90; map 'ctrl');
  - ridge on RNA labels with control-cell co-expression gene features (ridgeR_ctrlcx), alone and followed by the control-cell
    map (ridgeR_ctrlcx+ctrl);
  - zero-shot control co-expression (zs_ctrlcorr+ctrl);
  - GWCD4i: ridge_ntcorr / linear_ae_ctrlpca (NTC-pseudobulk co-expression / PCA);
- corpus-level baselines as before (corr / cridge / totalVI maps, ridgeR_cx, ridge on decoder loadings);
- protein-label ridge learning curves (ridgeP_cx / ridgeP_ctrlcx, n ∈ {3, 6, 12, 25, 50, all}) as references;
- measured-RNA references (measRNA + map). These use the held-out RNA response.

## Tests
**Statistics:**
- Per-KO metrics are averaged over the 3 seeds before testing.
- Two-sided Wilcoxon signed-rank over KOs (GWCD4i M1: batch-stratified bootstrap over targets, 1,000 reps, as pre-registered
  there).
- 95% CIs from 2,000 bootstrap resamples over KOs.
- BH within each panel below.
- Per-seed signs are reported.

**Primary tests** (base sp, variant (i), clamp, pooled over screens unless stated):
- **Panel TR (b1):**
  - T1: adapted CMC-enc − unadapted CMC-enc;
  - T2: adapted CMC-enc − ctrlridge90;
  - metrics: centred pearDE, retrieval (4 tests).
- **Panel XM (b2 protein):**
  - T3: adapted ftR − adapted ftR_shuf;
  - T4: adapted ftR − unadapted ftR;
  - T5: adapted ftR − ridgeR_ctrlcx+ctrl;
  - metrics: rp_pe_cent, rp_retrieval (6 tests).
- **Panel GW (b3, Stim48hr):**
  - T6: adapted FT − unadapted FT, and adapted FT − adapted obs;
  - metrics: M1, M4 (4 tests).
- **Panel RNA (b2 RNA):**
  - T7: adapted ftR − unadapted ftR, and adapted ftR − ridgeR_ctrlcx;
  - metrics: rr_pe_cent, rr_retrieval (4 tests).
- **Panel LC (b4, Stim8hr):**
  - T8: adapted base_shift and adapted F_sym_shift_short_pm − ridge_ntcorr at n = 5, 10, 25;
  - metric: retrieval (6 tests). The adapted − unadapted versions are reported alongside.

**Secondary (exploratory):** every other variant, base, operator, screen and metric, each BH within its own panel.

**Decision rules** (q < 0.05 after BH within panel, and the effect sign agrees in ≥ 2 of 3 seeds):
- **"Working cross-modal transfer"** requires both of:
  - T3 > 0 and T4 > 0 on the same protein metric;
  - or, for GWCD4i→SCITO, T6 adapted FT > unadapted FT and > adapted obs on M1.
- **"Adaptation repairs perturbation-state translation"** = T1 > 0.
- **"Advantage over information-matched linear baselines"** = any of T2, T5, T7 (vs ridgeR_ctrlcx) or T8 > 0.
- Otherwise the answer is no, and it is reported as such. Positive secondary results without a primary positive are labelled
  exploratory.

## Not done / limitations declared in advance
- No joint KO + control-cell bootstrap: it would need re-adaptation per resample.
- Arce has few protein-evaluable KOs (≈ 19), so per-screen Arce results are underpowered.
- The control cells used for adaptation also serve as the reference in the pseudobulk DE ground truth. This shares no
  perturbation information, and the ridge baselines use the same cells.
