import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import scanpy as sc
import os
import json
import argparse
from dataclasses import dataclass, field, asdict
from sklearn.model_selection import train_test_split
import math

from model import CrossModalCausalLDVAE


# ==========================================
# EXPERIMENT CONFIG
# ==========================================
@dataclass
class ExperimentConfig:
    # Naming
    name: str = "default"
    results_dir: str = "latent_dag_vae_results_v16_test"

    # Data
    rna_path: str = "data/hao_citeseq/rna_processed.h5ad"
    prot_path: str = "data/hao_citeseq/citeseq_processed.h5ad"
    prior_rna_path: str = "rna_hvg_priors_collectri_sc.pt"
    prior_prot_path: str = "protein_priors_sc.pt"
    batch_col: str = "donor"

    # Architecture
    latent_dim_rna: int = 128
    latent_dim_prot: int = 64
    encoder_hidden_rna: int = 512
    encoder_hidden_prot: int = 128

    # Training
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    grad_clip: float = 1.0
    patience: int = 200
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    test_size: float = 0.2
    seed: int = 42
    mini: bool = False          # Use 20% of data for fast iteration
    baseline: bool = False      # Baseline mode: alpha=0 always (SCM bypassed), no structural losses

    # Loss weights (static) — defaults match gs17_rna1000_prot18 (best perturbation model)
    rec_weight: float = 100.0
    lambda_stable: float = 1.0
    lambda_sparse_intra: float = 0.01
    lambda_sparse_inter: float = 0.01
    lambda_ortho: float = 5.0
    lambda_dec_sparse: float = 0.001
    lambda_dec_sparse_rna: float = 1000    # Feature-norm: grad/weight = 1000*0.05/2000 = 2.5e-2
    lambda_dec_sparse_prot: float = 18     # Feature-norm: grad/weight = 18*0.05/228 = 3.9e-3
    prior_margin_grn: float = 0.1           # Margin for GRN MarginRankingLoss
    prior_margin_ppi: float = 0.1           # Margin for PPI MarginRankingLoss
    dec_sparse_mode: str = "feature_norm"  # "feature_norm": feat-norm both modalities
                                           # "unnorm_rna": unnorm L1 RNA + feat-norm prot
                                           # "unnorm_both": unnorm L1 both (GS3 style)
    prior_abs: bool = False                # Apply abs() to projected graphs before margin loss
    prior_norm: bool = False               # z-score normalize projected graphs before margin loss

    # SCM structure-sparsity target (lambda_sparse_intra/inter set the coefficient):
    #   "weight" : L1 on the product adjacency A=gate*weight  (original behaviour)
    #   "gate"   : L1 on the gate open-prob sigmoid(logits)   (sparsify structure, keep weight magnitudes)
    #   "both"   : penalize both
    scm_sparse_target: str = "weight"
    # reduction for the SCM structure-sparsity + projected-graph L1: "mean" (L1/numel, original,
    # very weak on a 128^2 / 2692^2 matrix) or "sum" (unnormalized L1, GS3-style, much stronger).
    scm_sparse_reduction: str = "mean"

    # Reconstruction likelihood: "mse" (z-scored Gaussian, original) or "nb" (negative binomial; needs raw counts)
    recon_likelihood: str = "mse"

    # Restrict training to a subset of cohorts by obs['dataset'] (comma-sep, e.g. "hao,stephenson"
    # for leave-Liu-out generalization). None = use all cells.
    train_datasets: str = None

    # Prior-leakage control: randomly relocate the prior-mask 1s (preserve edge count) so the margin
    # loss rewards random edges. If the model still recovers true biology -> structure is data-driven.
    prior_shuffle: bool = False
    # Memorization test: hold out a random fraction of GRN prior edges from the margin loss.
    # The held-out edges are saved to prior_holdout_edges.pt; eval scores recovery on them
    # (generalization) vs the kept edges (fit). kept>>held => memorization; kept~=held => generalizes.
    prior_holdout_frac: float = 0.0
    # Holdout granularity: "edge" (random edges) or "tf" (whole transcription factors — all targets of
    # a held-out TF). "tf" is the stringent test: it removes within-regulon interpolation, so held-out
    # recovery measures discovery of regulons the loss never saw.
    prior_holdout_mode: str = "edge"

    # --- SIGNED DIRECTED prior (KnockTF + signed CollecTRI/DoRothEA), gated + OFF by default ---
    # Path to a signed prior .pt {gene_names, sign[G,G] in {-1,0,+1}, weight[G,G], train_tf_idx, ...}
    # built by data_prep/reference_networks/build_signed_prior.py. When set with lambda_signed_prior>0, adds a hinge that pushes
    # the projected edge G_rr[i,j]=D_r A_rr D_rᵀ toward the known sign s (mode-of-regulation) with a
    # margin: L = mean_edges w * relu(margin - s * z(G_rr[i,j])), z = std-normalized (scale-free).
    # Only edges of PRIOR-TRAIN TFs (train_tf_idx) enter the loss -> held-out TFs test generalization.
    signed_prior_path: str = None
    lambda_signed_prior: float = 0.0        # loss coefficient (ramped post-warmup like the other priors)
    signed_prior_margin: float = 0.5        # hinge margin in std units of the projected graph
    signed_prior_use_train_tfs_only: bool = True  # filter edges to train_tf_idx (leave held-out TFs out)

    # SCM gate sampler: "gumbel" (Gumbel-Sigmoid, default) or "hardconcrete" (Louizos 2018 L0 gates).
    gate_mode: str = "gumbel"

    # cross_modal=False zeros RNA<->Protein blocks -> intra-modal-only (multimodal-vs-unimodal ablation)
    cross_modal: bool = True
    # fix_A_rr_zero=True fixes the latent RNA->RNA block A_rr to exactly 0 for the whole run (gates and
    # weights masked + frozen, no gradient); encoders/decoders/other blocks train normally (review2 M4a).
    fix_A_rr_zero: bool = False

    # Covariance-preserving decoder term: align the RNA decoder Gram (D_r D_rᵀ) with the empirical
    # gene-gene correlation, so the decoder basis is a better coexpression estimator (closes the
    # decoder_DDt -> corr_rank gap from the diagnostic). 0 = off. Needs cov precompute at load.
    lambda_cov: float = 0.0
    # decoder-Gram off-diagonal penalty: push D_r D_r^T -> diagonal so the decoder can't encode
    # gene-gene coexpression, forcing that structure into the latent SCM A_rr (make A_rr load-bearing for RR).
    lambda_decoder_gram: float = 0.0
    # Spectral-stability bound: penalize spectral radius above this (was hardcoded 0.9). Raise toward
    # ~0.95 to let the SCM propagate more strongly (denser effective (I-A)^-1).
    stable_rho_bound: float = 0.9

    # --- Architectural-lever flags (Tier-1 ideas; all default off = backward compatible) ---
    # ε-variance regularizer: ||std(eps)-1||^2 per dim, keeps eps well-conditioned for (I-A)^-1
    lambda_eps_var: float = 0.0
    # project each decoder COLUMN (latent direction in gene space) to unit L2 after each step
    decoder_colnorm: bool = False
    # decoder sparsity mode "group_lasso": group lasso over n_decoder_groups latent groups
    # (pathway-latent -> gene-module topology) instead of element-wise feature_norm L1
    n_decoder_groups: int = 16
    # Tier-2 surgery flags
    # multi-scale A: A_intra = sparse local (DAGMA'd) + low-rank dense global block
    multiscale_A: bool = False
    global_rank: int = 8
    global_scale: float = 0.5
    # coupled-noise: shared low-rank factor injecting cross-modal correlation into (eps_r, eps_p)
    coupled_noise: bool = False
    coupled_rank: int = 8
    # nonlinear SCM: z = phi(A^T z) + eps via fixed-point (linear default). Linear decoder kept; the
    # origin-Jacobian projected GRN G = D_r A_rr D_r^T stays exact. Forces A load-bearing for nonlinear deps.
    scm_nonlinear: bool = False
    phi: str = "tanh"
    n_fp_iters: int = 20
    scm_init_scale: float = 1.0   # <1 shrinks initial A-edge weights (keeps nonlinear SCM out of saturation)
    nonlin_gain: float = 1.0      # pre-activation gain so phi engages its CURVED regime (>1 => genuinely nonlinear)
    fp_damping: float = 0.0       # damped (Mann) fixed-point iteration for convergence when gain*rho(A)->1
    scm_mechanism: str = "elementwise"  # elementwise | mlp (per-node MLP, parent interactions) | locscale (heteroscedastic)
    lambda_proj_sparse: float = 0.0  # L1 on the PROJECTED gene-space GRN G=D*A*D^T (not latent A)
    mlp_hidden: int = 8
    free_bits_r: float = 0.0      # KL free-bits per RNA latent dim (>0 prevents posterior collapse at high beta)
    free_bits_p: float = 0.0      # KL free-bits per protein latent dim
    dec_mask_rna_path: str = None # pathway-structured RNA decoder: path to {mask[G,P],genes,pathways}.pt
    dec_mask_lambda: float = 0.1  # SENA out-of-set residual weight (1 in-set, lambda out-of-set)
    dec_mask_freeze: bool = False # True = decoder fixed to membership mask (not learned)
    # --- Exogenous ε-bottleneck (make A load-bearing without interventions) ---
    # Global per-dim hard-concrete L0 gate on eps; closed dims get no exogenous input so their latent
    # can only be produced by A (breaks the eps->z bijection). eps_l0_lambda>0 = penalty per expected
    # active root (ramped over training); eps_gate_init sets the initial open logit (2.0 ~ open).
    eps_bottleneck: bool = False
    eps_l0_lambda: float = 0.0
    eps_gate_init: float = 2.0
    eps_bottleneck_target: str = "both"   # gate eps of "both" modalities, "rna" only, or "prot" only
    # --- NON-GAUSSIAN exogenous prior on eps (confirmatory / reviewer-proofing; default OFF) ---
    # "gaussian" (default) = standard closed-form Gaussian KL, BYTE-IDENTICAL to the original model.
    # "laplace"/"studentt" = reparameterized non-Gaussian posterior q(eps|x) with a matching standardized
    # non-Gaussian prior p(eps); the closed-form KL is replaced by a single-sample MC estimate
    # E_q[log q - log p] (per-dim, same free-bits floor). All non-gaussian code in model.py + loss_fn is
    # guarded behind eps_prior != "gaussian" and never runs when off.
    eps_prior: str = "gaussian"
    eps_df: float = 3.0                    # Student-t degrees of freedom (only used when eps_prior="studentt")
    # --- CONDITIONAL exogenous prior p(eps|u)=N(mu(u),diag(exp(lv(u)))) (iVAE, Khemakhem 2020; default OFF) ---
    # None = off (standard N(0,I) prior, BYTE-IDENTICAL to the original model). Otherwise the auxiliary
    # variable u is the per-cell "celltype" (26 cats) | "donor" (138 cats) | "disease" (healthy vs COVID/other,
    # via the Stephenson Status map). A per-class embedding maps one-hot u -> (prior_mu, prior_logvar) and the
    # KL becomes the closed-form Gaussian-Gaussian KL(q(eps|x) || p(eps|u)). All new code is guarded behind
    # this flag; cond_prior_on=None never creates params and never changes the loss.
    cond_prior_on: str = None
    cond_prior_target: str = "both"   # which exogenous the aux prior conditions: "both" (joint) | "rna" | "prot"
    # context_adapt: None | "latent" (per-context affine on eps, #6/#8) | "decoder" (per-context FiLM, #10).
    # Resolves context per-cell from batch (donor=='iccite_ctrl' -> ctx 1, else 0).
    context_adapt: str = None
    # context_cross_modal: per-context cross-modal coupling (separate W_rp/W_pr per context). The
    # do-op RP prediction flows RNA->inv_pr->decoder, so an iccite-specific cross-modal block is the
    # primary lever for iccite RP transfer. Resolves context per-cell like context_adapt.
    context_cross_modal: bool = False
    # balanced_sampler: WeightedRandomSampler over the per-cell context (inverse-frequency) so the
    # 88k iccite-ctrl cells are not drowned by 809k PBMC when learning the shared/per-context blocks.
    balanced_sampler: bool = False
    # finetune: load a pretrained checkpoint and (optionally) freeze all but the SCM cross-modal block
    # + protein decoder. finetune_mode in {None, "cross_modal"}.
    pretrained_checkpoint: str = None
    finetune_mode: str = None
    # modality / feature dropout (cross-modal prediction). rna_feature_dropout default 0.1 = current behavior.
    # modality_dropout_p>0 randomly nulls a whole modality's latent so the cross-modal block must predict it.
    rna_feature_dropout: float = 0.1
    prot_feature_dropout: float = 0.0
    modality_dropout_p: float = 0.0
    modality_dropout_mode: str = "both"   # both | protein | rna
    modality_dropout_null: str = "zero"   # zero (fix) | learned

    # --- Reversal-discrepancy causality loss (intra-modal blocks A_rr, A_pp) ---
    # Re-solve the SCM with the two intra-modal DIAGONAL blocks TRANSPOSED (A_rr->A_rr^T, A_pp->A_pp^T;
    # cross-modal W_rp/W_pr left UNCHANGED) on the SAME encoded eps, decode, and penalize unless the
    # reversed intra-graph reconstructs STRICTLY worse than the forward by reversal_margin:
    #   L = lambda_reversal * relu(reversal_margin - (recon_rev - recon_fwd))
    # -> pressures A_rr/A_pp to be directionally asymmetric. 0 = OFF (byte-identical to a run without it;
    # all new code is guarded behind lambda_reversal>0 and never runs when 0).
    lambda_reversal: float = 0.0
    reversal_margin: float = 0.0

    # --- RESIDUAL-INDEPENDENCE penalty on the exogenous structural residual eps=[eps_r;eps_p] --------
    # z=(I-A^T)^{-1}eps, so the encoder posterior mean/draw eps IS the structural residual r=(I-A^T)z.
    # LiNGAM/ICA identifiability: the CORRECT causal orientation is the one whose structural residuals
    # are mutually INDEPENDENT (a higher-order property). Marginal non-Gaussianity (free-bits + heavy
    # tails) does NOT enforce between-component independence, so it can't implement the asymmetry; an
    # explicit Total-Correlation (beta-TCVAE) or HSIC penalty on the joint residual DOES. This tests
    # whether residual-independence pressure ORIENTS A_rr (baseline: corr(A_rr,A_rr^T)=0.9945, edge
    # reversal inert). resid_indep in {None,"tc","hsic"}; None = OFF (byte-identical to a run without it:
    # the term is only computed when resid_indep is set AND lambda_resid_indep>0, and it enters `total`
    # exactly like the other guarded 0.0*0.0 terms). Distinct flag / results dir from eps_prior +
    # cond_prior_on (all coexist cleanly). "tc" uses the Gaussian posterior (default eps_prior).
    resid_indep: str = None
    lambda_resid_indep: float = 0.0
    resid_indep_hsic_pairs: int = 256   # #random latent-dim pairs penalized per step (hsic only)

    # Artifact / logging control
    save_latents: bool = True              # dump per-cell latent CSVs (~GBs each); set False for grid runs
    use_wandb: bool = True
    wandb_project: str = "CrossModal-CausalVAE"
    wandb_entity: str = None
    wandb_group: str = None                # e.g. "phase05_gate_sparsity"

    # Warmup / annealing
    warmup_epochs: int = 20
    alpha_ramp_epochs: int = 30  # epochs after warmup to reach alpha=1.0
    tau_switch_epoch: int = 30
    tau_high: float = 1.0
    tau_low: float = 0.5

    # Annealing targets (value at end of training) — defaults match gs17
    beta_rna_init: float = 1e-4
    beta_rna_final: float = 1.0
    beta_protein_init: float = 1e-4
    beta_protein_final: float = 1.0
    dag_w_final: float = 5.0
    acyc_mode: str = "dagma"          # "dagma" | "notears" (scale-invariant; applied to A_rr, A_pp)
    prior_w_final: float = 500.0
    dec_sparse_w_init: float = 0.05
    dec_sparse_w_final: float = 0.05

    def __post_init__(self):
        if not self.results_dir:
            self.results_dir = f"results_{self.name}"

    def get_params(self, epoch):
        """Compute all loss weights for a given epoch."""
        if self.baseline:
            # Baseline: SCM bypassed (alpha=0), no structural losses, only reconstruction + KL + decoder reg
            alpha = 0.0
            if epoch < self.warmup_epochs:
                beta_rna = self.beta_rna_init
                beta_protein = self.beta_protein_init
            else:
                progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
                beta_rna = self.beta_rna_init + (self.beta_rna_final - self.beta_rna_init) * progress
                beta_protein = self.beta_protein_init + (self.beta_protein_final - self.beta_protein_init) * progress
            dag_w = 0.0
            prior_w = 0.0
            dec_sparse_w = self.dec_sparse_w_init + (self.dec_sparse_w_final - self.dec_sparse_w_init) * max(0, (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1))
        elif epoch < self.warmup_epochs:
            alpha = 0.0
            beta_rna = self.beta_rna_init
            beta_protein = self.beta_protein_init
            dag_w = 0.0
            prior_w = 0.0
            dec_sparse_w = self.dec_sparse_w_init
        else:
            ramp = self.epochs - self.warmup_epochs
            alpha = min(1.0, (epoch - self.warmup_epochs) / self.alpha_ramp_epochs)
            progress = (epoch - self.warmup_epochs) / max(ramp, 1)
            beta_rna = self.beta_rna_init + (self.beta_rna_final - self.beta_rna_init) * progress
            beta_protein = self.beta_protein_init + (self.beta_protein_final - self.beta_protein_init) * progress
            dag_w = self.dag_w_final * progress
            prior_w = self.prior_w_final * progress
            dec_sparse_w = self.dec_sparse_w_init + (self.dec_sparse_w_final - self.dec_sparse_w_init) * progress

        tau = self.tau_low if epoch > self.tau_switch_epoch else self.tau_high
        # tau = max(self.tau_low , self.tau_high * math.exp(-0.05 * epoch))

        # ε-bottleneck L0 weight: 0 during warmup/baseline, ramps 0->eps_l0_lambda post-warmup
        _prog = max(0.0, (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1))
        eps_l0_w = 0.0 if (self.baseline or epoch < self.warmup_epochs) else self.eps_l0_lambda * _prog

        # signed-prior weight: 0 during warmup/baseline, ramps 0->lambda_signed_prior post-warmup
        # (train content first, then impose signs — same schedule shape as the unsigned prior)
        signed_prior_w = 0.0 if (self.baseline or epoch < self.warmup_epochs) else self.lambda_signed_prior * _prog

        # Per-modality decoder sparsity: fall back to shared lambda_dec_sparse if not set
        lds_rna = (self.lambda_dec_sparse_rna if self.lambda_dec_sparse_rna is not None
                   else self.lambda_dec_sparse)
        lds_prot = (self.lambda_dec_sparse_prot if self.lambda_dec_sparse_prot is not None
                    else self.lambda_dec_sparse)

        params = {
            'beta_rna': beta_rna,
            'beta_protein': beta_protein,
            'free_bits_r': self.free_bits_r,
            'free_bits_p': self.free_bits_p,
            'lambda_dag': dag_w,
            'acyc_mode': self.acyc_mode,
            'lambda_stable': 0.0 if self.baseline else self.lambda_stable,
            'lambda_prior': prior_w,
            'lambda_sparse_intra': 0.0 if self.baseline else self.lambda_sparse_intra,
            'lambda_proj_sparse': 0.0 if self.baseline else self.lambda_proj_sparse,
            'lambda_sparse_inter': 0.0 if self.baseline else self.lambda_sparse_inter,
            'lambda_ortho': self.lambda_ortho,
            'lambda_dec_sparse': self.lambda_dec_sparse * dec_sparse_w,  # kept for backward compat
            'lambda_dec_sparse_rna': lds_rna * dec_sparse_w,
            'lambda_dec_sparse_prot': lds_prot * dec_sparse_w,
            'rec_weight': self.rec_weight,
            'prior_margin_grn': self.prior_margin_grn,
            'prior_margin_ppi': self.prior_margin_ppi,
            'dec_sparse_mode': self.dec_sparse_mode,
            'lambda_eps_var': self.lambda_eps_var,
            'n_decoder_groups': self.n_decoder_groups,
            'decoder_colnorm': self.decoder_colnorm,
            'prior_abs': self.prior_abs,
            'prior_norm': self.prior_norm,
            'scm_sparse_target': self.scm_sparse_target,
            'scm_sparse_reduction': self.scm_sparse_reduction,
            'recon_likelihood': self.recon_likelihood,
            'lambda_cov': self.lambda_cov,
            'lambda_decoder_gram': self.lambda_decoder_gram,
            'stable_rho_bound': self.stable_rho_bound,
            # ε-bottleneck L0 penalty, ramped 0->lambda over the post-warmup schedule (train first, then squeeze)
            'eps_l0_lambda': eps_l0_w,
            # signed-prior hinge, ramped 0->lambda_signed_prior post-warmup
            'lambda_signed_prior': signed_prior_w,
            'signed_prior_margin': self.signed_prior_margin,
            # reversal-discrepancy causality loss (intra-modal); constant pass-through (OFF in baseline).
            # The alpha ramp naturally gates it: during warmup alpha=0 -> z=eps -> fwd==rev -> loss=0.
            'lambda_reversal': 0.0 if self.baseline else self.lambda_reversal,
            'reversal_margin': self.reversal_margin,
            # residual-independence penalty (TC/HSIC on eps): OFF in baseline / when resid_indep is None;
            # otherwise ramped 0->lambda over the post-warmup schedule (train content first, then impose
            # independence — same shape as the priors). resid_indep is passed through so loss_fn can branch.
            'resid_indep': None if self.baseline else self.resid_indep,
            'lambda_resid_indep': (0.0 if (self.baseline or epoch < self.warmup_epochs
                                           or self.resid_indep is None)
                                   else self.lambda_resid_indep * _prog),
            'resid_indep_hsic_pairs': self.resid_indep_hsic_pairs,
        }
        return params, alpha, tau

    def save(self, path=None):
        path = path or os.path.join(self.results_dir, "config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


# ==========================================
# PRESET CONFIGS
# ==========================================
PRESETS = {
    "v14_baseline": ExperimentConfig(
        name="v14_baseline",
        latent_dim_rna=256, latent_dim_prot=128,
        rec_weight=100.0,
        lambda_ortho=1.0,
        lambda_dec_sparse=1e-3,
        lambda_sparse_intra=1e-6,
        lambda_sparse_inter=1e-6,
        dec_sparse_w_init=0.1, dec_sparse_w_final=0.5,
    ),
    "v15_dec_sparse": ExperimentConfig(
        name="v15_dec_sparse",
        latent_dim_rna=128, latent_dim_prot=64,
        rec_weight=50.0,
        lambda_ortho=1.0,
        lambda_dec_sparse=0.1,
        lambda_sparse_intra=1e-4,
        lambda_sparse_inter=1e-4,
        dec_sparse_w_init=0.05, dec_sparse_w_final=0.5,
    ),
    "v16_high_sparse": ExperimentConfig(
        name="v16_high_sparse",
        latent_dim_rna=128, latent_dim_prot=64,
        rec_weight=20.0,
        lambda_ortho=5.0,
        lambda_dec_sparse=0.25,
        lambda_sparse_intra=1e-3,
        lambda_sparse_inter=1e-3,
        dec_sparse_w_init=0.1, dec_sparse_w_final=1.0,
    ),
    "v17_small_latent": ExperimentConfig(
        name="v17_small_latent",
        latent_dim_rna=64, latent_dim_prot=32,
        rec_weight=50.0,
        lambda_ortho=5.0,
        lambda_dec_sparse=0.1,
        lambda_sparse_intra=1e-4,
        lambda_sparse_inter=1e-4,
        dec_sparse_w_init=0.05, dec_sparse_w_final=0.5,
    ),
}


# ==========================================
# 1. ROBUST LOSS FUNCTIONS
# ==========================================
def compute_spectral_radius(A, num_iters=None):
    """Spectral NORM (largest singular value) of A = exact, differentiable (full gradient) via svdvals.
    sigma_max >= |eigmax|, so penalizing it < b GUARANTEES true rho < b (well-conditioned (I-A)^-1
    for linear; unsaturated nonlinear SCM). d~192 so the SVD is cheap.
    FIX: the old code used a Rayleigh quotient |v^T A v| with 10 power iters, which for a non-symmetric
    A severely UNDERESTIMATED (e.g. 0.115 vs true 0.914) -> the stability penalty never bound and rho
    drifted to ~0.9 (nonlinear saturation) / ~2.6 (linear ill-conditioned inverse)."""
    return torch.linalg.svdvals(A)[0]

def _spectral_normalize(adj_matrix, target=1.0):
    """Scale A so its largest singular value == target (detached scale). Makes the acyclicity penalty
    SCALE-INVARIANT: the model can no longer trivially satisfy it by shrinking the (reciprocal) edge
    weights -> A**A -> 0 -> h -> 0. It must instead break the cyclic STRUCTURE. Also bounds the
    spectral radius fed to the matrix-exp so NOTEARS cannot explode. Zero-graph -> returned unchanged."""
    with torch.no_grad():
        s = compute_spectral_radius(adj_matrix.detach())
    if not torch.isfinite(s) or s < 1e-4:
        return adj_matrix
    return adj_matrix * (target / s)

def compute_dagma_loss(adj_matrix, scale_invariant=True):
    """DAGMA acyclicity h(A) = -logdet(sI - A*A) + d log s  (0 iff A*A nilpotent, i.e. acyclic).
    Applied to a spectrally-normalized A so it measures cyclic STRUCTURE, not magnitude."""
    A = _spectral_normalize(adj_matrix) if scale_invariant else adj_matrix
    d = A.shape[0]
    adj_sq = A * A
    with torch.no_grad():
        rho = compute_spectral_radius(adj_sq.detach())
    I = torch.eye(d, device=A.device)
    s = rho + 1.0
    for _ in range(6):                                  # guarantee s I - A*A is PD (grow s if not)
        M = s * I - adj_sq
        sign, logdet = torch.linalg.slogdet(M)
        if (sign > 0).all():
            return torch.relu(-logdet + d * torch.log(s))
        s = s * 2.0
    return torch.relu(-logdet + d * torch.log(s))        # gradient still flows (no dead constant)

def compute_notears_loss(adj_matrix, scale_invariant=True):
    """NOTEARS acyclicity h(A) = tr(exp(A*A)) - d. Applied to a spectrally-normalized A so the matrix
    exponential stays bounded (rho(A*A) <= 1 => tr(exp) <= d*e) -> never explodes to inf."""
    A = _spectral_normalize(adj_matrix) if scale_invariant else adj_matrix
    B = A * A
    E = torch.linalg.matrix_exp(B)
    return torch.trace(E) - A.shape[0]

def compute_acyc_loss(adj_matrix, mode="dagma"):
    if mode == "notears":
        return compute_notears_loss(adj_matrix)
    return compute_dagma_loss(adj_matrix)

# def compute_ortho_loss(W):
#     W_n = F.normalize(W, dim=0)
#     gram = W_n.T @ W_n
#     I = torch.eye(gram.shape[0], device=W.device)
#     return F.mse_loss(gram, I)

def compute_ortho_loss(W):
    gram = W.T @ W
    I = torch.eye(gram.shape[0], device=W.device)
    return torch.norm(gram - I, p='fro') ** 2 / gram.numel()

def nb_nll(x, mu, theta, eps=1e-8):
    """Negative-binomial negative log-likelihood (scVI parameterization), mean over features
    and cells. x=counts [B,F], mu=mean>0 [B,F], theta=dispersion>0 [F]."""
    theta = theta + eps
    log_theta_mu = torch.log(theta + mu + eps)
    ll = (theta * (torch.log(theta + eps) - log_theta_mu)
          + x * (torch.log(mu + eps) - log_theta_mu)
          + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0))
    return -ll.mean()  # mean over features & cells -> ~O(1), comparable to MSE scale


# ==========================================
# RESIDUAL-INDEPENDENCE penalties on the exogenous residual eps (beta-TCVAE TC / pairwise HSIC)
# ==========================================
def _log_normal_diag(sample, mu, logvar):
    """Per-element log-density of a diagonal Gaussian N(mu, exp(logvar)) at `sample` (broadcasts)."""
    return -0.5 * (math.log(2 * math.pi) + logvar + (sample - mu) ** 2 / logvar.exp())


def _log_importance_weight_matrix(B, N, device):
    """Minibatch-STRATIFIED-sampling importance log-weights (Chen et al. 2018, beta-TCVAE, eq. S6).
    [B,B] matrix used to estimate the aggregate posterior log q(z_i) from a size-B minibatch drawn
    from a size-N dataset (corrects the naive uniform 1/B weighting)."""
    M = B - 1
    strat = (N - M) / (N * M)
    W = torch.full((B, B), 1.0 / M, device=device)
    W.view(-1)[::M + 1] = 1.0 / N              # diagonal  = 1/N
    W.view(-1)[1::M + 1] = strat               # sub-/super-diagonal = stratified weight
    W[M - 1, 0] = strat
    return W.log()


def total_correlation_mss(z, mu, logvar, dataset_size):
    """beta-TCVAE Total Correlation TC = E_q[log q(z) - sum_j log q(z_j)] via the minibatch-weighted
    (stratified) sampling estimator of Chen et al. 2018. z,mu,logvar are [B,D] for the JOINT exogenous
    residual eps and its Gaussian posterior. TC>=0; larger => residual components more DEPENDENT, so
    penalizing TC enforces residual independence (the LiNGAM identifiability signal)."""
    B, D = z.shape
    N = max(int(dataset_size), B)
    # mat[i,j,d] = log N(z_i,d | mu_j,d, logvar_j,d)  (density of z_i under posterior of x_j)
    mat = _log_normal_diag(z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0))    # [B,B,D]
    logiw = _log_importance_weight_matrix(B, N, z.device)                          # [B,B]
    logqz = torch.logsumexp(logiw + mat.sum(dim=2), dim=1)                          # log q(z_i)        [B]
    logqz_prod = torch.logsumexp(logiw.unsqueeze(2) + mat, dim=1).sum(dim=1)        # sum_d log q(z_i,d) [B]
    return (logqz - logqz_prod).mean()


def pairwise_hsic(z, n_pairs, generator=None):
    """Mean pairwise HSIC (biased empirical estimator, RBF kernel, median-heuristic bandwidth) over a
    RANDOM subset of n_pairs latent-dimension pairs -> cost O(B^2 * n_pairs) instead of O(B^2 D^2).
    HSIC(x,y)=0 iff x |= y in the RKHS; driving the mean pairwise HSIC to 0 enforces mutual residual
    independence. z=[B,D]. Returns the mean HSIC over the (deduplicated) sampled pairs."""
    B, D = z.shape
    ii = torch.randint(0, D, (n_pairs,), device=z.device, generator=generator)
    jj = torch.randint(0, D, (n_pairs,), device=z.device, generator=generator)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    if ii.numel() == 0:
        return z.new_zeros(())
    x = z[:, ii].t()                                  # [P, B]
    y = z[:, jj].t()                                  # [P, B]
    eye = torch.eye(B, device=z.device).unsqueeze(0)

    def _rbf(v):                                      # v:[P,B] -> RBF kernel [P,B,B], median-heuristic
        d2 = (v.unsqueeze(2) - v.unsqueeze(1)) ** 2                   # [P,B,B] squared dists
        med = (d2 + eye * 1e30).reshape(d2.shape[0], -1).median(dim=1).values.clamp_min(1e-8)  # [P]
        return torch.exp(-d2 / med.view(-1, 1, 1))    # sigma^2 = median(d2)/2 => exp(-d2/median(d2))

    Kx, Ky = _rbf(x), _rbf(y)
    Kxc = Kx - Kx.mean(1, keepdim=True) - Kx.mean(2, keepdim=True) + Kx.mean((1, 2), keepdim=True)
    hsic = (Kxc * Ky).sum(dim=(1, 2)) / (B - 1) ** 2  # tr(Hc Kx Hc Ky)/(B-1)^2 per pair  [P]
    return hsic.mean()


def loss_fn(out, targets, priors, params, nb_targets=None):
    x_r, x_p = targets
    rec_r, rec_p = out['rec']
    ((mu_r, lv_r), (mu_p, lv_p)) = out['kl']

    A_rr, A_pp, C_rp, C_pr = out['blocks']
    A_full = out['adj_full']
    dec_r, dec_p = out['dec_w']

    # 1. Reconstruction (Base Loss) — MSE on z-scored data (default) or NB on raw counts.
    if params.get('recon_likelihood', 'mse') == 'nb' and nb_targets is not None:
        counts_r, counts_p = nb_targets
        theta_r, theta_p = out['nb_theta']
        lib_r = counts_r.sum(dim=1, keepdim=True)
        lib_p = counts_p.sum(dim=1, keepdim=True)
        mu_rate_r = lib_r * torch.softmax(rec_r, dim=1) + 1e-8   # LDVAE-style softmax rate * library
        mu_rate_p = lib_p * torch.softmax(rec_p, dim=1) + 1e-8
        loss_rec = (nb_nll(counts_r, mu_rate_r, theta_r) + nb_nll(counts_p, mu_rate_p, theta_p)) * params['rec_weight']
    else:
        # Weighted higher to ensure we learn contents before structure
        loss_rec = (F.mse_loss(rec_r, x_r) + F.mse_loss(rec_p, x_p)) * params['rec_weight']

    # 1b. Reversal-discrepancy causality loss (intra-modal A_rr/A_pp). GUARDED: fully OFF (identical
    #     behaviour) unless lambda_reversal>0 AND the model produced a reversed-intra reconstruction.
    #     recon_fwd/recon_rev = the model's normal reconstruction likelihood on the SAME targets;
    #     L = relu(reversal_margin - (recon_rev - recon_fwd)) penalizes unless the reversed intra-graph
    #     reconstructs strictly worse than the forward by the margin.
    loss_reversal = torch.tensor(0.0, device=x_r.device)
    rev_gap = 0.0
    if params.get('lambda_reversal', 0.0) > 0 and out.get('rec_rev') is not None:
        rec_r_rev, rec_p_rev = out['rec_rev']
        if params.get('recon_likelihood', 'mse') == 'nb' and nb_targets is not None:
            counts_r, counts_p = nb_targets
            theta_r, theta_p = out['nb_theta']
            lib_r = counts_r.sum(dim=1, keepdim=True); lib_p = counts_p.sum(dim=1, keepdim=True)
            recon_fwd = (nb_nll(counts_r, lib_r * torch.softmax(rec_r, dim=1) + 1e-8, theta_r)
                         + nb_nll(counts_p, lib_p * torch.softmax(rec_p, dim=1) + 1e-8, theta_p))
            recon_rev = (nb_nll(counts_r, lib_r * torch.softmax(rec_r_rev, dim=1) + 1e-8, theta_r)
                         + nb_nll(counts_p, lib_p * torch.softmax(rec_p_rev, dim=1) + 1e-8, theta_p))
        else:
            recon_fwd = F.mse_loss(rec_r, x_r) + F.mse_loss(rec_p, x_p)
            recon_rev = F.mse_loss(rec_r_rev, x_r) + F.mse_loss(rec_p_rev, x_p)
        margin_rev = params.get('reversal_margin', 0.0)
        loss_reversal = F.relu(margin_rev - (recon_rev - recon_fwd))
        rev_gap = float((recon_rev - recon_fwd).item())

    # 2. KL Divergence (with optional free-bits: floor each latent dim so beta can't drive it to 0)
    fb_r = params.get('free_bits_r', 0.0); fb_p = params.get('free_bits_p', 0.0)
    kl_mc = out.get('kl_mc')
    cond_prior = out.get('cond_prior')
    if kl_mc is not None:
        # NON-GAUSSIAN prior: per-dim single-sample MC estimate E_q[log q - log p] (from model._mc_kl_vec).
        # Same reduction as the Gaussian branch: per-dim free-bits floor (clamp_min) then mean over dims.
        klvec_r, klvec_p = kl_mc
        kl_r = klvec_r.clamp_min(fb_r).mean() if fb_r > 0 else klvec_r.mean()
        kl_p = klvec_p.clamp_min(fb_p).mean() if fb_p > 0 else klvec_p.mean()
    elif cond_prior is not None:
        # CONDITIONAL Gaussian prior p(eps|u)=N(mu_p, diag(exp(lv_p))) (iVAE). Closed-form KL between two
        # diagonal Gaussians q=N(mu,exp(lv)) and p=N(mu_p,exp(lv_p)), per element then the SAME per-dim
        # free-bits floor + mean-over-dims reduction as the Gaussian branch. Reduces EXACTLY to the N(0,I)
        # KL when (mu_p,lv_p)=(0,0) (algebraically; only used when cond_prior_on is set).
        pmu_r, plv_r, pmu_p, plv_p = cond_prior
        kelem_r = 0.5 * (plv_r - lv_r + (lv_r.exp() + (mu_r - pmu_r).pow(2)) / plv_r.exp() - 1.0)
        kelem_p = 0.5 * (plv_p - lv_p + (lv_p.exp() + (mu_p - pmu_p).pow(2)) / plv_p.exp() - 1.0)
        kl_r = kelem_r.mean(dim=0).clamp_min(fb_r).mean() if fb_r > 0 else kelem_r.mean()
        kl_p = kelem_p.mean(dim=0).clamp_min(fb_p).mean() if fb_p > 0 else kelem_p.mean()
    else:
        # GAUSSIAN closed-form (unchanged / byte-identical when eps_prior='gaussian' and cond_prior off)
        if fb_r > 0:
            kl_r = (-0.5 * torch.mean(1 + lv_r - mu_r.pow(2) - lv_r.exp(), dim=0)).clamp_min(fb_r).mean()
        else:
            kl_r = -0.5 * torch.mean(1 + lv_r - mu_r.pow(2) - lv_r.exp())
        if fb_p > 0:
            kl_p = (-0.5 * torch.mean(1 + lv_p - mu_p.pow(2) - lv_p.exp(), dim=0)).clamp_min(fb_p).mean()
        else:
            kl_p = -0.5 * torch.mean(1 + lv_p - mu_p.pow(2) - lv_p.exp())
    loss_kl = kl_r + kl_p
    
    # 3. Structure / DAG Constraints
    if params['lambda_dag'] > 0:
        A_rr_dag, A_pp_dag = out.get('blocks_dag', (A_rr, A_pp))  # local intra blocks (multiscale: excludes global)
        _acyc = params.get('acyc_mode', 'dagma')
        loss_dag = compute_acyc_loss(A_rr_dag, _acyc) + compute_acyc_loss(A_pp_dag, _acyc)
    else:
        loss_dag = torch.tensor(0.0, device=x_r.device)
    
    # 4. Stability (Crucial for linalg.solve)
    rho = compute_spectral_radius(A_full)
    loss_stable = torch.relu(rho - params.get('stable_rho_bound', 0.9))
    
    # 5. SCM structure sparsity — penalize edge WEIGHTS (product adjacency), the GATES
    #    (sigmoid(logits) open-probabilities), or both, per params['scm_sparse_target'].
    sparse_target = params.get('scm_sparse_target', 'weight')
    _red = params.get('scm_sparse_reduction', 'mean')
    def _l1(M): return torch.norm(M, 1) if _red == 'sum' else torch.norm(M, 1) / M.numel()
    def _l1v(v): return v.sum() if _red == 'sum' else v.sum() / v.numel()
    norm_intra = torch.tensor(0.0, device=x_r.device)
    norm_inter = torch.tensor(0.0, device=x_r.device)
    if sparse_target in ('weight', 'both'):
        norm_intra = norm_intra + _l1(A_rr) + _l1(A_pp)
        norm_inter = norm_inter + _l1(C_rp) + _l1(C_pr)
    if sparse_target in ('gate', 'both'):
        g_rr, g_pp, g_rp, g_pr = out['gates']
        norm_intra = norm_intra + _l1v(g_rr) + _l1v(g_pp)
        norm_inter = norm_inter + _l1v(g_rp) + _l1v(g_pr)
    
    # 6. Decoder Orthogonality
    loss_ortho = compute_ortho_loss(dec_r) + compute_ortho_loss(dec_p)
    
    # Per-modality decoder sparsity (configurable via dec_sparse_mode)
    mode = params.get('dec_sparse_mode', 'unnorm_rna')
    if mode == 'feature_norm':
        # Feature-normalized L1 for both: norm(W,1) / num_features
        loss_dec_sparse_rna = torch.norm(dec_r, 1) / dec_r.shape[0]
        loss_dec_sparse_prot = torch.norm(dec_p, 1) / dec_p.shape[0]
    elif mode == 'group_lasso':
        # Group lasso over latent GROUPS: each gene loads onto few groups of latents
        # ("pathway-latent -> gene-module" topology). Sum over genes of sum over groups of
        # the L2 norm of that gene's weights within the group.
        def _group_lasso(W, n_groups):
            G, L = W.shape
            k = max(1, L // n_groups)
            Wg = W[:, :k * (L // k)].reshape(G, L // k, k)  # [genes, groups, per-group]
            return Wg.norm(dim=2).sum() / G
        ng = int(params.get('n_decoder_groups', 16))
        loss_dec_sparse_rna = _group_lasso(dec_r, ng)
        loss_dec_sparse_prot = _group_lasso(dec_p, ng)
    elif mode == 'unnorm_both':
        # Unnormalized L1 for both (exact GS3 formulation)
        loss_dec_sparse_rna = torch.norm(dec_r, 1)
        loss_dec_sparse_prot = torch.norm(dec_p, 1)
    else:  # 'unnorm_rna' (default)
        # Unnormalized L1 for RNA (GS3-style), feature-norm for protein
        loss_dec_sparse_rna = torch.norm(dec_r, 1)
        loss_dec_sparse_prot = torch.norm(dec_p, 1) / dec_p.shape[0]
    loss_dec_sparse = loss_dec_sparse_rna + loss_dec_sparse_prot  # backward compat
    
    
    # 7. Biological Priors
    loss_prior = torch.tensor(0.0, device=x_r.device)
    loss_prior_grn = torch.tensor(0.0, device=x_r.device)
    loss_prior_ppi = torch.tensor(0.0, device=x_r.device)
    if priors is not None and params['lambda_prior'] > 0:
        mask_grn, mask_ppi = priors[0], priors[1]

        proj_grn_raw = (dec_r @ A_rr @ dec_r.T).flatten()
        proj_ppi_raw = (dec_p @ A_pp @ dec_p.T).flatten()

        # Optional abs(): treats inhibitory edges same as activating
        if params.get('prior_abs', False):
            proj_grn_raw = proj_grn_raw.abs()
            proj_ppi_raw = proj_ppi_raw.abs()

        # Optional z-score normalization
        if params.get('prior_norm', False):
            proj_grn_flat = proj_grn_raw / (proj_grn_raw.std().detach() + 1e-8)
            proj_ppi_flat = proj_ppi_raw / (proj_ppi_raw.std().detach() + 1e-8)
        else:
            proj_grn_flat = proj_grn_raw
            proj_ppi_flat = proj_ppi_raw

        margin_grn = params.get('prior_margin_grn', 0.1)
        margin_ppi = params.get('prior_margin_ppi', 0.1)

        mask_grn_flat = mask_grn.flatten()
        mask_ppi_flat = mask_ppi.flatten()

        idx_g = mask_grn_flat != 0
        idx_p = mask_ppi_flat != 0

        if idx_g.any():
            signal_scores = proj_grn_flat[idx_g]
            bg_indices = torch.nonzero(~idx_g, as_tuple=True)[0]

            if len(bg_indices) > 0:
                num_samples = min(len(signal_scores), len(bg_indices))
                rand_bg_idx = bg_indices[torch.randperm(len(bg_indices), device=x_r.device)[:num_samples]]
                background_scores = proj_grn_flat[rand_bg_idx]

                criterion_grn = torch.nn.MarginRankingLoss(margin=margin_grn)
                target = torch.ones(num_samples, device=x_r.device)
                loss_prior_grn = criterion_grn(signal_scores[:num_samples], background_scores, target)

        if idx_p.any():
            signal_scores = proj_ppi_flat[idx_p]
            bg_indices = torch.nonzero(~idx_p, as_tuple=True)[0]

            if len(bg_indices) > 0:
                num_samples = min(len(signal_scores), len(bg_indices))
                rand_bg_idx = bg_indices[torch.randperm(len(bg_indices), device=x_r.device)[:num_samples]]
                background_scores = proj_ppi_flat[rand_bg_idx]

                criterion_ppi = torch.nn.MarginRankingLoss(margin=margin_ppi)
                target = torch.ones(num_samples, device=x_r.device)
                loss_prior_ppi = criterion_ppi(signal_scores[:num_samples], background_scores, target)

        loss_prior = loss_prior_grn + loss_prior_ppi

    # 7b. SIGNED DIRECTED prior (KnockTF + signed CollecTRI/DoRothEA): push the projected edge
    #     G_rr[i,j] = D_r A_rr D_rᵀ toward the known mode-of-regulation sign s (+act / -repr) with a
    #     margin. Scale-free: normalize G_rr edge values by the std of the projected off-diagonal edges
    #     (detached). L_signed = mean_edges w * relu(margin - s * z(G_rr[i,j])). Only PRIOR-TRAIN TF
    #     edges are supplied (held-out TFs filtered out at load) so eval can test generalization.
    loss_signed = torch.tensor(0.0, device=x_r.device)
    sp = priors[3] if (priors is not None and len(priors) > 3) else None
    if sp is not None and params.get('lambda_signed_prior', 0.0) > 0 and sp['row'].numel() > 0:
        ri, ci, ss, sw = sp['row'], sp['col'], sp['sign'], sp['weight']
        M = dec_r @ A_rr                                   # [G, L]
        vals = (M[ri] * dec_r[ci]).sum(dim=1)             # G_rr[i,j] for the signed edges
        scale = vals.detach().std() + 1e-8                # std-normalize -> margin in std units
        vals_n = vals / scale
        margin_s = params.get('signed_prior_margin', 0.5)
        loss_signed = (sw * F.relu(margin_s - ss * vals_n)).sum() / (sw.sum() + 1e-8)

    # 8. Covariance-preserving decoder term: align RNA decoder Gram (D_r D_rᵀ) with the empirical
    #    gene-gene correlation (cosine on off-diagonal) so the decoder basis is a better coexpression
    #    estimator (closes the diagnostic's decoder_DDt -> corr gap). Trades off against orthogonality.
    loss_cov = torch.tensor(0.0, device=x_r.device)
    cov_g = priors[2] if (priors is not None and len(priors) > 2) else None
    if params.get('lambda_cov', 0.0) > 0 and cov_g is not None:
        # cosine alignment of off-diagonal decoder-Gram (K) vs empirical gene-gene cov.
        # Computed via direct Frobenius products (subtracting the diagonal) to avoid
        # materializing a [G,G] eye + a flattened G^2 dot (the latter SIGFPE'd on GPU).
        K = dec_r @ dec_r.T
        dK = torch.diagonal(K); dC = torch.diagonal(cov_g)
        dot = (K * cov_g).sum() - (dK * dC).sum()
        nK = torch.sqrt(((K * K).sum() - (dK * dK).sum()).clamp_min(1e-12))
        nC = torch.sqrt(((cov_g * cov_g).sum() - (dC * dC).sum()).clamp_min(1e-12))
        cos = dot / (nK * nC + 1e-8)
        loss_cov = 1.0 - cos

    # 8b. Decoder-Gram off-diagonal penalty: push D_r D_r^T toward DIAGONAL so the decoder cannot
    #     encode gene-gene coexpression -> forces cross-gene structure into the latent SCM (A_rr),
    #     making A_rr load-bearing for the RNA->RNA do-op (today it's a decoder-Gram shortcut, 95%
    #     off-diagonal). ||offdiag(D D^T)||_F^2 = ||D^T D||_F^2 - sum_i ||row_i||^4 (no [G,G] matmul).
    loss_gram = torch.tensor(0.0, device=x_r.device)
    if params.get('lambda_decoder_gram', 0.0) > 0:
        DtD = dec_r.T @ dec_r
        row2 = (dec_r ** 2).sum(dim=1)
        loss_gram = ((DtD * DtD).sum() - (row2 * row2).sum()).clamp_min(0.0) / dec_r.shape[0]

    # 9. ε-variance regularizer: keep sampled eps at unit per-dim std so the (I-A)^-1 operator
    #    runs in a well-conditioned regime (KL collapses on RNA, leaving eps scale free).
    loss_eps_var = torch.tensor(0.0, device=x_r.device)
    if params.get('lambda_eps_var', 0.0) > 0 and out.get('eps') is not None:
        e_r, e_p = out['eps']
        loss_eps_var = ((e_r.std(dim=0) - 1.0) ** 2).mean() + ((e_p.std(dim=0) - 1.0) ** 2).mean()

    # 9b. ε-bottleneck L0 penalty: expected number of OPEN exogenous gates (Louizos 2018 expected-L0).
    #     Drives most eps dims closed -> few "root" latents get exogenous noise, the rest must be
    #     produced by A (makes A load-bearing). Global gate (shared across cells).
    loss_eps_l0 = torch.tensor(0.0, device=x_r.device)
    n_roots = float('nan')
    op = out.get('eps_gate_openprob')
    if op is not None:
        loss_eps_l0 = op.sum()
        n_roots = float((op > 0.5).sum().item())

    # 9c. RESIDUAL-INDEPENDENCE penalty on the exogenous residual eps=[eps_r;eps_p] (beta-TCVAE TC or
    #     pairwise HSIC). GUARDED: only computed when params['resid_indep'] in {tc,hsic} AND lambda>0;
    #     otherwise loss_resid_indep stays 0.0 and enters `total` as +0.0*0.0 (byte-identical no-op).
    loss_resid_indep = torch.tensor(0.0, device=x_r.device)
    ri_mode = params.get('resid_indep', None)
    lam_ri = params.get('lambda_resid_indep', 0.0)
    if ri_mode is not None and lam_ri > 0 and out.get('eps') is not None:
        e_r, e_p = out['eps']
        eps_joint = torch.cat([e_r, e_p], dim=1)
        if ri_mode == 'tc':
            mu_j = torch.cat([mu_r, mu_p], dim=1)
            lv_j = torch.cat([lv_r, lv_p], dim=1)
            loss_resid_indep = total_correlation_mss(eps_joint, mu_j, lv_j,
                                                     params.get('dataset_size', eps_joint.shape[0]))
        elif ri_mode == 'hsic':
            loss_resid_indep = pairwise_hsic(eps_joint, int(params.get('resid_indep_hsic_pairs', 256)))
        else:
            raise ValueError(f"resid_indep must be None|tc|hsic, got {ri_mode}")

    # TOTAL LOSS — uses per-modality decoder sparsity weights
    # PROJECTED-graph sparsity: L1 on the gene-space GRN G=D*A*D^T (the REPORTED graph), not latent A
    loss_proj_sparse = torch.tensor(0.0, device=x_r.device)
    if params.get('lambda_proj_sparse', 0.0) > 0:
        _redp = params.get('scm_sparse_reduction', 'mean')
        _rr = (dec_r @ A_rr @ dec_r.T).abs(); _pp = (dec_p @ A_pp @ dec_p.T).abs()
        loss_proj_sparse = (_rr.sum() + _pp.sum()) if _redp == 'sum' else (_rr.mean() + _pp.mean())

    total = loss_rec + \
            params.get('lambda_reversal', 0.0) * loss_reversal + \
            params.get('lambda_resid_indep', 0.0) * loss_resid_indep + \
            params.get('eps_l0_lambda', 0.0) * loss_eps_l0 + \
            params.get('lambda_eps_var', 0.0) * loss_eps_var + \
            params.get('lambda_cov', 0.0) * loss_cov + \
            params.get('lambda_decoder_gram', 0.0) * loss_gram + \
            params['beta_rna'] * kl_r + \
            params['beta_protein'] * kl_p + \
            params['lambda_dag'] * loss_dag + \
            params['lambda_stable'] * loss_stable + \
            params['lambda_sparse_intra'] * norm_intra + \
            params.get('lambda_proj_sparse', 0.0) * loss_proj_sparse + \
            params['lambda_sparse_inter'] * norm_inter + \
            params['lambda_dec_sparse_rna'] * loss_dec_sparse_rna + \
            params['lambda_dec_sparse_prot'] * loss_dec_sparse_prot + \
            params['lambda_prior'] * loss_prior + \
            params.get('lambda_signed_prior', 0.0) * loss_signed + \
            params['lambda_ortho'] * loss_ortho

    return total, {
        'rec': loss_rec.item(),
        'kl_r': kl_r.item(),
        'kl_p': kl_p.item(),
        'dag': loss_dag.item(),
        'intra_sparse': norm_intra.item(),
        'proj_sparse': loss_proj_sparse.item() if torch.is_tensor(loss_proj_sparse) else float(loss_proj_sparse),
        'inter_sparse': norm_inter.item(),
        'stable': loss_stable.item(),
        'rho': rho.item(),
        'ortho': loss_ortho.item(),
        'gram': loss_gram.item(),
        'dec_sparse': loss_dec_sparse.item(),
        'dec_sparse_rna': loss_dec_sparse_rna.item(),
        'dec_sparse_prot': loss_dec_sparse_prot.item(),
        'prior': loss_prior.item(),
        'prior_grn': loss_prior_grn.item(),
        'prior_ppi': loss_prior_ppi.item(),
        'signed_prior': loss_signed.item(),
        'cov': loss_cov.item(),
        'eps_l0': loss_eps_l0.item(),
        'n_roots': n_roots,
        'reversal': loss_reversal.item() if torch.is_tensor(loss_reversal) else float(loss_reversal),
        'rev_gap': rev_gap,   # recon_rev - recon_fwd (>0 means reversed intra-graph reconstructs worse)
        'resid_indep': loss_resid_indep.item() if torch.is_tensor(loss_resid_indep) else float(loss_resid_indep),
        'total': total.item()
    }


# ==========================================
# 2. LOAD DATA
# ==========================================
def load_data_and_priors(cfg, device):
    print("Loading Data...")
    rna_adata = sc.read_h5ad(cfg.rna_path)
    prot_adata = sc.read_h5ad(cfg.prot_path)
    
    common_obs = sorted(set(rna_adata.obs_names) & set(prot_adata.obs_names))

    # Optional cohort filter by obs['dataset'] (e.g. "hao,stephenson" for leave-Liu-out)
    if getattr(cfg, 'train_datasets', None):
        keep_ds = set(s.strip() for s in cfg.train_datasets.split(','))
        ds = rna_adata.obs['dataset'].astype(str)
        ds_obs = set(ds.index[ds.isin(keep_ds)])
        before = len(common_obs)
        common_obs = [o for o in common_obs if o in ds_obs]
        print(f"Cohort filter {sorted(keep_ds)}: {len(common_obs)}/{before} cells kept")

    if cfg.mini:
        np.random.seed(cfg.seed)
        n_keep = max(int(len(common_obs) * 0.2), 1000)
        common_obs = list(np.random.choice(common_obs, size=n_keep, replace=False))
        print(f"MINI MODE: {n_keep}/{len(set(rna_adata.obs_names) & set(prot_adata.obs_names))} cells (20%)")

    print(f"Aligning samples: {len(common_obs)} cells.")

    rna_adata = rna_adata[common_obs].copy()
    prot_adata = prot_adata[common_obs].copy()
    
    if cfg.batch_col not in rna_adata.obs.columns:
        raise ValueError(f"Column '{cfg.batch_col}' not found in rna_adata.obs. Available: {rna_adata.obs.columns.tolist()}")

    # Convert batch strings to integers (0, 1, 2...)
    batch_cat = rna_adata.obs[cfg.batch_col].astype('category')
    batch_codes = batch_cat.cat.codes.values
    num_batches = len(np.unique(batch_codes))
    # context-of-batch: 1 if the batch is an iccite-control donor, else 0 (pbmc). For context_adapt.
    context_of_batch = np.array([1 if str(c) == 'iccite_ctrl' else 0 for c in batch_cat.cat.categories], dtype=np.int64)

    # --- CONDITIONAL exogenous prior (iVAE) auxiliary variable u, one code per cell ---
    # OFF (cond_prior_on is None) => zeros / n_classes=1 (unused; model builds no cond-prior params).
    u_codes = np.zeros(len(common_obs), dtype=np.int64)
    u_n_classes = 1
    cpo = getattr(cfg, 'cond_prior_on', None)
    if cpo is not None:
        if cpo in ('celltype', 'donor'):
            if cpo not in rna_adata.obs.columns:
                raise ValueError(f"cond_prior_on='{cpo}' but obs has no '{cpo}' column. "
                                 f"Available: {rna_adata.obs.columns.tolist()}")
            u_cat = rna_adata.obs[cpo].astype('category')
            u_codes = u_cat.cat.codes.values.astype(np.int64)
            u_n_classes = int(len(u_cat.cat.categories))
            print(f"COND-PRIOR u='{cpo}': {u_n_classes} classes; "
                  f"counts (top) {rna_adata.obs[cpo].astype(str).value_counts().head(6).to_dict()}")
        elif cpo == 'disease':
            # healthy(0) vs disease(1): Hao cells + Stephenson 'Healthy' donors = healthy; other
            # Stephenson (Covid/Non_covid/LPS) = disease. Reuse the hd_split_build donor->Status map
            # (donor = 'steph_'+patient_id) from the raw Stephenson h5ad. Unmapped Steph cells -> healthy.
            data_root = os.path.dirname(os.path.dirname(cfg.rna_path))
            steph_raw = os.path.join(data_root, 'stephenson_citeseq', 'haniffa21.processed.h5ad')
            print(f"COND-PRIOR u='disease': building donor->Status map from {steph_raw}", flush=True)
            sa = sc.read_h5ad(steph_raw, backed='r')
            _donor = 'steph_' + sa.obs['patient_id'].astype(str)
            _status = sa.obs['Status'].astype(str)
            _df = pd.DataFrame({'donor': _donor.values, 'Status': _status.values})
            try:
                sa.file.close()
            except Exception:
                pass
            dmap = _df.groupby('donor')['Status'].agg(lambda s: s.value_counts().index[0]).to_dict()
            ds = rna_adata.obs['dataset'].astype(str).values
            dn = rna_adata.obs['donor'].astype(str).values
            codes = np.zeros(len(common_obs), dtype=np.int64)
            for i in range(len(common_obs)):
                if ds[i] == 'stephenson':
                    st = dmap.get(dn[i], 'Healthy')
                    codes[i] = 0 if st == 'Healthy' else 1
                # hao (or anything else) -> healthy (0)
            u_codes = codes
            u_n_classes = 2
            print(f"COND-PRIOR u='disease': healthy(0)={int((u_codes==0).sum())} "
                  f"disease(1)={int((u_codes==1).sum())}")
        else:
            raise ValueError(f"cond_prior_on must be None|celltype|donor|disease, got '{cpo}'")

    X_rna = rna_adata.X.toarray() if hasattr(rna_adata.X, 'toarray') else rna_adata.X
    X_prot = prot_adata.X.toarray() if hasattr(prot_adata.X, 'toarray') else prot_adata.X
    
    # Z-score normalization (encoder input)
    X_rna = (X_rna - X_rna.mean(axis=0)) / (X_rna.std(axis=0) + 1e-6)
    X_prot = (X_prot - X_prot.mean(axis=0)) / (X_prot.std(axis=0) + 1e-6)

    # Raw-count targets for the NB likelihood (from layers['counts']; combined3 only)
    counts_r = counts_p = None
    if getattr(cfg, 'recon_likelihood', 'mse') == 'nb':
        if 'counts' not in rna_adata.layers or 'counts' not in prot_adata.layers:
            raise ValueError("recon_likelihood='nb' needs layers['counts'] in RNA and protein h5ad (use combined3).")
        Cr, Cp = rna_adata.layers['counts'], prot_adata.layers['counts']
        Cr = Cr.toarray() if hasattr(Cr, 'toarray') else np.asarray(Cr)
        Cp = Cp.toarray() if hasattr(Cp, 'toarray') else np.asarray(Cp)
        counts_r = torch.FloatTensor(Cr)
        counts_p = torch.FloatTensor(Cp)
        print(f"  NB counts loaded: RNA {tuple(counts_r.shape)} (max {counts_r.max():.0f}), "
              f"prot {tuple(counts_p.shape)} (max {counts_p.max():.0f})")

    gene_names = rna_adata.var_names.tolist()
    prot_names = prot_adata.var_names.tolist()

    print("Loading Priors...")
    try:
        rna_priors_data = torch.load(cfg.prior_rna_path, map_location=device)
        prior_mask_g = rna_priors_data['prior_mask'].to(device)
    except:
        print("Warning: Could not load RNA priors. Using zeros.")
        prior_mask_g = torch.zeros(len(gene_names), len(gene_names), device=device)
        
    try:
        prot_priors_data = torch.load(cfg.prior_prot_path, map_location=device)
        prior_mask_p = prot_priors_data['prior_mask'].to(device)
    except:
        print("Warning: Could not load Protein priors. Using zeros.")
        prior_mask_p = torch.zeros(len(prot_names), len(prot_names), device=device)

    # Prior-leakage control: relocate the 1s to random positions (preserve edge count)
    if getattr(cfg, 'prior_shuffle', False):
        gen = torch.Generator(device='cpu').manual_seed(cfg.seed)
        for nm, mk in [('GRN', prior_mask_g), ('PPI', prior_mask_p)]:
            flat = mk.flatten().cpu()
            perm = torch.randperm(flat.numel(), generator=gen)
            mk.copy_(flat[perm].reshape(mk.shape).to(mk.device))
        print(f"PRIOR SHUFFLED (leakage control): GRN edges={int(prior_mask_g.sum())}, PPI edges={int(prior_mask_p.sum())}")

    # Memorization test: hold out a random fraction of GRN edges from the margin loss; save the split.
    if getattr(cfg, 'prior_holdout_frac', 0.0) > 0:
        gen = torch.Generator(device='cpu').manual_seed(cfg.seed + 12345)
        mode = getattr(cfg, 'prior_holdout_mode', 'edge')
        m_cpu = prior_mask_g.cpu()
        n_total_edges = int((m_cpu != 0).sum())
        extra = {'mode': mode}
        if mode == 'tf':
            # stringent: hold out whole TFs (rows with outgoing edges) -> ALL their target edges
            tfs = (m_cpu.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
            n_hold_tf = int(round(len(tfs) * cfg.prior_holdout_frac))
            hold_tfs = tfs[torch.randperm(len(tfs), generator=gen)[:n_hold_tf]]
            row_mask = torch.zeros(m_cpu.shape[0], dtype=torch.bool); row_mask[hold_tfs] = True
            held_2d = row_mask.unsqueeze(1) & (m_cpu != 0)
            hold_idx = held_2d.flatten().nonzero(as_tuple=True)[0]
            extra.update(n_total_tfs=int(len(tfs)), n_holdout_tfs=int(n_hold_tf), holdout_tf_idx=hold_tfs)
        else:
            edges = (m_cpu.flatten() != 0).nonzero(as_tuple=True)[0]
            n_hold = int(round(len(edges) * cfg.prior_holdout_frac))
            hold_idx = edges[torch.randperm(len(edges), generator=gen)[:n_hold]]
        flat = prior_mask_g.flatten()
        flat[hold_idx.to(flat.device)] = 0.0                       # remove held-out edges from the loss
        prior_mask_g = flat.reshape(prior_mask_g.shape)
        os.makedirs(cfg.results_dir, exist_ok=True)
        torch.save({'holdout_flat_idx': hold_idx.cpu(), 'n_total_edges': n_total_edges,
                    'n_holdout': int(len(hold_idx)), 'shape': tuple(prior_mask_g.shape),
                    'holdout_seed': cfg.seed + 12345, **extra},
                   os.path.join(cfg.results_dir, 'prior_holdout_edges.pt'))
        tf_msg = f" ({extra['n_holdout_tfs']}/{extra['n_total_tfs']} TFs)" if mode == 'tf' else ""
        print(f"PRIOR HOLDOUT ({mode}): held out {int(len(hold_idx))}/{n_total_edges} GRN edges{tf_msg}; "
              f"loss now uses {int(prior_mask_g.sum())} edges")

    g_tensor = torch.FloatTensor(X_rna)
    p_tensor = torch.FloatTensor(X_prot)
    if counts_r is None:  # MSE: counts placeholders (ignored by loss) keep the loader uniform
        counts_r, counts_p = g_tensor, p_tensor

    # empirical gene-gene correlation for the covariance-preserving decoder term (X is z-scored)
    cov_g = None
    if getattr(cfg, 'lambda_cov', 0.0) > 0:
        Xz = X_rna.astype(np.float32)
        cov_g = torch.FloatTensor((Xz.T @ Xz) / Xz.shape[0]).to(device)
        print(f"  Cov term: precomputed gene-gene correlation {tuple(cov_g.shape)}")

    # SIGNED DIRECTED prior (optional): load, align to training gene order, restrict to prior-train TFs,
    # and materialize edge index/sign/weight tensors on device for the hinge in loss_fn.
    signed_prior = None
    if getattr(cfg, 'signed_prior_path', None):
        spd = torch.load(cfg.signed_prior_path, map_location='cpu')
        sp_names = list(map(str, spd['gene_names']))
        S = spd['sign']; Wt = spd['weight']
        S = S.numpy() if hasattr(S, 'numpy') else np.asarray(S)
        Wt = Wt.numpy() if hasattr(Wt, 'numpy') else np.asarray(Wt)
        if sp_names != gene_names:                       # align to training gene order via names
            if not set(gene_names) <= set(sp_names):
                raise ValueError("signed_prior gene_names do not cover the training genes")
            idx = {g: i for i, g in enumerate(sp_names)}
            perm = [idx[g] for g in gene_names]
            S = S[np.ix_(perm, perm)]; Wt = Wt[np.ix_(perm, perm)]
        train_tf_names = set(map(str, spd.get('train_tf_names', [])))
        rows, cols = np.nonzero(S)
        use_train_only = getattr(cfg, 'signed_prior_use_train_tfs_only', True)
        if use_train_only and train_tf_names:
            keep = np.array([gene_names[r] in train_tf_names for r in rows], dtype=bool)
            rows, cols = rows[keep], cols[keep]
        signed_prior = {
            'row': torch.as_tensor(rows, dtype=torch.long, device=device),
            'col': torch.as_tensor(cols, dtype=torch.long, device=device),
            'sign': torch.as_tensor(S[rows, cols], dtype=torch.float32, device=device),
            'weight': torch.as_tensor(Wt[rows, cols], dtype=torch.float32, device=device),
        }
        n_tf = len(np.unique(rows))
        print(f"SIGNED PRIOR: {len(rows)} edges over {n_tf} prior-train TFs "
              f"(train_tfs_only={use_train_only}) from {cfg.signed_prior_path}; "
              f"held-out TFs ({len(spd.get('heldout_tf_names', []))}) excluded from loss")

    return (g_tensor, p_tensor, torch.LongTensor(batch_codes),
            common_obs, gene_names, prot_names, prior_mask_g, prior_mask_p, num_batches,
            counts_r, counts_p, cov_g, torch.LongTensor(context_of_batch), signed_prior,
            torch.LongTensor(u_codes), u_n_classes)


# ==========================================
# 3. TRAINING
# ==========================================
def train_epoch(model, loader, optimizer, priors, params, alpha, tau, device, grad_clip):
    model.train()
    stats = {}
    count = 0
    for b_rna, b_prot, b_batch, b_cr, b_cp, b_u in loader:
        b_rna, b_prot, b_batch = b_rna.to(device), b_prot.to(device), b_batch.to(device)
        b_cr, b_cp = b_cr.to(device), b_cp.to(device)
        b_u = b_u.to(device) if getattr(model, 'cond_prior_on', None) is not None else None
        optimizer.zero_grad()
        out = model(b_rna, b_prot, batch_idx=b_batch, current_tau=tau, alpha=alpha,
                    compute_reversal=params.get('lambda_reversal', 0.0) > 0, u=b_u)
        loss, metrics = loss_fn(out, (b_rna, b_prot), priors, params, nb_targets=(b_cr, b_cp))
        loss.backward()
        # grad-flow diagnostic (pre-clip): per-element gradient magnitude through the DAG (A_rr = logits_rr,
        # weights_rr) vs the RNA decoder — is reconstruction/structure loss actually driving A? RMS-normalized
        # so it's comparable across latent sizes (e.g. pathway-decoder latent=491 vs 128).
        try:
            with torch.no_grad():
                _gp=[p.grad for p in (model.scm.logits_rr, model.scm.weights_rr) if p.grad is not None]
                _ga=torch.cat([g.flatten() for g in _gp]) if _gp else None
                _gd=model.dec_rna.weight.grad
                metrics['gradrms_A']=float(_ga.norm()/max(_ga.numel(),1)**0.5) if _ga is not None else 0.0
                metrics['gradrms_dec']=float(_gd.norm()/max(_gd.numel(),1)**0.5) if _gd is not None else 0.0
        except Exception:
            pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if params.get('decoder_colnorm', False):   # project each decoder column to unit L2
            with torch.no_grad():
                for dec in (model.dec_rna, model.dec_prot):
                    dec.weight.div_(dec.weight.norm(dim=0, keepdim=True) + 1e-8)
        for k, v in metrics.items():
            stats[k] = stats.get(k, 0) + v
        count += 1
    return {k: v / count for k, v in stats.items()}


@torch.no_grad()
def eval_epoch(model, loader, priors, params, alpha, tau, device):
    model.eval()
    stats = {}
    count = 0
    for b_rna, b_prot, b_batch, b_cr, b_cp, b_u in loader:
        b_rna, b_prot, b_batch = b_rna.to(device), b_prot.to(device), b_batch.to(device)
        b_cr, b_cp = b_cr.to(device), b_cp.to(device)
        b_u = b_u.to(device) if getattr(model, 'cond_prior_on', None) is not None else None
        out = model(b_rna, b_prot, batch_idx=b_batch, current_tau=tau, alpha=alpha,
                    compute_reversal=params.get('lambda_reversal', 0.0) > 0, u=b_u)
        _, metrics = loss_fn(out, (b_rna, b_prot), priors, params, nb_targets=(b_cr, b_cp))
        for k, v in metrics.items():
            stats[k] = stats.get(k, 0) + v
        count += 1
    return {k: v / count for k, v in stats.items()}


def save_artifacts(model, g_tensor, p_tensor, batch_tensor, obs_ids, g_names, p_names, cfg, device):
    """Save all post-training artifacts."""
    print("Generating Analysis Artifacts...")
    D = cfg.results_dir

    with torch.no_grad():
        logits_rr = model.scm.logits_rr
        logits_pp = model.scm.logits_pp
        logits_rp = model.scm.logits_rp
        logits_pr = model.scm.logits_pr
        
        weights_rr = model.scm.weights_rr
        weights_pp = model.scm.weights_pp
        weights_rp = model.scm.weights_rp
        weights_pr = model.scm.weights_pr
        
        probs_rr = torch.sigmoid(logits_rr)
        probs_pp = torch.sigmoid(logits_pp)
        probs_rp = torch.sigmoid(logits_rp)
        probs_pr = torch.sigmoid(logits_pr)
        
        mask_rr_diag = model.scm.mask_rr
        mask_pp_diag = model.scm.mask_pp
        
        # SOFT adjacency (expected gate prob * weight) — NOT hard-thresholded at >0.5.
        # Gate-sparsity drives most gate probs below 0.5, so a >0.5 cutoff would zero out the
        # entire projected graph; the soft graph preserves the learned sparse structure and is
        # what the SCM/perturbation propagation and DAGMA actually use.
        A_rr_clean = (probs_rr * mask_rr_diag) * weights_rr
        A_pp_clean = (probs_pp * mask_pp_diag) * weights_pp
        A_rp_clean = probs_rp * weights_rp
        A_pr_clean = probs_pr * weights_pr
        
        # 5. Get Decoder Weights
        W_r = model.dec_rna.weight
        W_p = model.dec_prot.weight
        
        # Gene Regulatory Network
        grn_hard = W_r @ A_rr_clean @ W_r.T
        
        # Protein-Protein Interaction
        ppi_hard = W_p @ A_pp_clean @ W_p.T
        
        # Protein -> RNA (Regulation/Translation effects)
        transl_hard = W_r @ A_rp_clean @ W_p.T
        
        # RNA -> Protein (Transcription effects)
        transcr_hard = W_p @ A_pr_clean @ W_r.T

    # 7. Save Files
    pd.DataFrame(grn_hard.cpu().numpy(), index=g_names, columns=g_names).to_csv(f"{D}/GRN_inferred.csv")
    pd.DataFrame(ppi_hard.cpu().numpy(), index=p_names, columns=p_names).to_csv(f"{D}/PPI_inferred.csv")
    pd.DataFrame(transcr_hard.cpu().numpy(), index=p_names, columns=g_names).to_csv(f"{D}/Regulation_Prot_to_RNA.csv")
    pd.DataFrame(transl_hard.cpu().numpy(), index=g_names, columns=p_names).to_csv(f"{D}/Transcription_RNA_to_Prot.csv")
    
    # Save Factor Loadings
    W_r_np = model.dec_rna.weight.detach().cpu().numpy()
    pd.DataFrame(W_r_np, index=g_names, columns=[f"Latent_R_{i}" for i in range(model.latent_r)]).to_csv(f"{D}/RNA_Factor_Loadings.csv")
    
    W_p_np = model.dec_prot.weight.detach().cpu().numpy()
    pd.DataFrame(W_p_np, index=p_names, columns=[f"Latent_P_{i}" for i in range(model.latent_p)]).to_csv(f"{D}/Prot_Factor_Loadings.csv")

    if not getattr(cfg, 'save_latents', True):
        print(f"All matrix artifacts saved to {D} (skipping per-cell latent dump; save_latents=False)")
        return

    print("Extracting Latents for all cells...")
    full_loader = DataLoader(TensorDataset(g_tensor, p_tensor, batch_tensor), batch_size=cfg.batch_size, shuffle=False)
    
    eps_r_list, eps_p_list = [], []
    z_r_list, z_p_list = [], []
    
    with torch.no_grad():
        for b_rna, b_prot, b_batch in full_loader:
            b_rna, b_prot, b_batch = b_rna.to(device), b_prot.to(device), b_batch.to(device)
            mu_r, _ = model.enc_rna(b_rna)
            mu_p, _ = model.enc_prot(b_prot)
            
            eps_r_list.append(mu_r.cpu().numpy())
            eps_p_list.append(mu_p.cpu().numpy())
            
            eps_joint = torch.cat([mu_r, mu_p], dim=1)
            # Use deterministic tau for final extraction
            z_joint, _, _ = model.scm(eps_joint, tau=0.5, alpha=1.0)
            z_r, z_p = torch.split(z_joint, [model.latent_r, model.latent_p], dim=1)
            
            z_r_list.append(z_r.cpu().numpy())
            z_p_list.append(z_p.cpu().numpy())
            
    full_eps_r = np.vstack(eps_r_list)
    full_eps_p = np.vstack(eps_p_list)
    full_z_r = np.vstack(z_r_list)
    full_z_p = np.vstack(z_p_list)
    
    pd.DataFrame(full_eps_r, index=obs_ids, columns=[f"Eps_R_{i}" for i in range(model.latent_r)]).to_csv(f"{D}/latents_exogenous_rna.csv")
    pd.DataFrame(full_eps_p, index=obs_ids, columns=[f"Eps_P_{i}" for i in range(model.latent_p)]).to_csv(f"{D}/latents_exogenous_prot.csv")
    pd.DataFrame(full_z_r, index=obs_ids, columns=[f"Z_R_{i}" for i in range(model.latent_r)]).to_csv(f"{D}/latents_causal_rna.csv")
    pd.DataFrame(full_z_p, index=obs_ids, columns=[f"Z_P_{i}" for i in range(model.latent_p)]).to_csv(f"{D}/latents_causal_prot.csv")
    
    print(f"All artifacts saved to {D}")


# ==========================================
# 4. MAIN
# ==========================================
def run_experiment(cfg: ExperimentConfig):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running experiment '{cfg.name}' on {DEVICE}")
    print(f"Results dir: {cfg.results_dir}")

    os.makedirs(cfg.results_dir, exist_ok=True)
    cfg.save()

    # Reproducibility (seed governs both torch init and the train/val/test split)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # W&B logging
    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project, entity=cfg.wandb_entity, group=cfg.wandb_group,
                name=cfg.name, config=asdict(cfg), dir=cfg.results_dir, reinit=True,
            )
        except Exception as e:
            print(f"[wandb] init failed ({e}); continuing without W&B.")
            wandb_run = None

    (g_tensor, p_tensor, batch_tensor, obs_ids, g_names, p_names,
     mask_g, mask_p, num_batches, counts_r, counts_p, cov_g, context_of_batch,
     signed_prior, u_tensor, u_n_classes) = load_data_and_priors(cfg, DEVICE)

    idx_train, idx_temp = train_test_split(np.arange(len(obs_ids)), test_size=cfg.test_size, random_state=cfg.seed)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=cfg.seed)

    def _ds(idx):
        return TensorDataset(g_tensor[idx], p_tensor[idx], batch_tensor[idx], counts_r[idx], counts_p[idx],
                             u_tensor[idx])
    if getattr(cfg, 'balanced_sampler', False):
        from torch.utils.data import WeightedRandomSampler
        ctx_tr = context_of_batch[batch_tensor[idx_train]]            # per-cell context (0/1)
        cnt = torch.bincount(ctx_tr, minlength=int(context_of_batch.max().item()) + 1).float()
        w = (1.0 / cnt.clamp_min(1.0))[ctx_tr].double()
        sampler = WeightedRandomSampler(w, len(idx_train), replacement=True)
        train_loader = DataLoader(_ds(idx_train), batch_size=cfg.batch_size, sampler=sampler)
        print(f"  balanced_sampler ON: per-context counts {cnt.tolist()} -> inverse-freq weighting")
    else:
        train_loader = DataLoader(_ds(idx_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(_ds(idx_val), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(_ds(idx_test), batch_size=cfg.batch_size, shuffle=False)

    priors = (mask_g, mask_p, cov_g, signed_prior)

    _dec_mask = None
    if getattr(cfg, 'dec_mask_rna_path', None):
        _mk = torch.load(cfg.dec_mask_rna_path, map_location='cpu')
        _dec_mask = _mk['mask'] if isinstance(_mk, dict) else _mk
        print(f"  PATHWAY DECODER: mask {tuple(_dec_mask.shape)} lambda={cfg.dec_mask_lambda} "
              f"freeze={cfg.dec_mask_freeze} (latent_dim_rna must == {_dec_mask.shape[1]})")
    model = CrossModalCausalLDVAE(
        dim_rna=g_tensor.shape[1],
        dim_prot=p_tensor.shape[1],
        num_batches=num_batches,
        dec_mask_rna=_dec_mask,
        dec_mask_lambda=getattr(cfg, 'dec_mask_lambda', 0.1),
        dec_mask_freeze=getattr(cfg, 'dec_mask_freeze', False),
        latent_dim_rna=cfg.latent_dim_rna,
        latent_dim_prot=cfg.latent_dim_prot,
        recon_likelihood=cfg.recon_likelihood,
        gate_mode=cfg.gate_mode,
        cross_modal=cfg.cross_modal,
        multiscale_A=getattr(cfg, 'multiscale_A', False),
        global_rank=getattr(cfg, 'global_rank', 8),
        global_scale=getattr(cfg, 'global_scale', 0.5),
        coupled_noise=getattr(cfg, 'coupled_noise', False),
        coupled_rank=getattr(cfg, 'coupled_rank', 8),
        context_adapt=getattr(cfg, 'context_adapt', None),
        context_cross_modal=getattr(cfg, 'context_cross_modal', False),
        n_context=int(context_of_batch.max().item()) + 1
                  if (getattr(cfg, 'context_adapt', None) or getattr(cfg, 'context_cross_modal', False)) else 1,
        rna_feature_dropout=getattr(cfg, 'rna_feature_dropout', 0.1),
        prot_feature_dropout=getattr(cfg, 'prot_feature_dropout', 0.0),
        modality_dropout_p=getattr(cfg, 'modality_dropout_p', 0.0),
        modality_dropout_mode=getattr(cfg, 'modality_dropout_mode', 'both'),
        modality_dropout_null=getattr(cfg, 'modality_dropout_null', 'zero'),
        scm_nonlinear=getattr(cfg, 'scm_nonlinear', False),
        phi=getattr(cfg, 'phi', 'tanh'),
        n_fp_iters=getattr(cfg, 'n_fp_iters', 20),
        scm_init_scale=getattr(cfg, 'scm_init_scale', 1.0),
        nonlin_gain=getattr(cfg, 'nonlin_gain', 1.0),
        fp_damping=getattr(cfg, 'fp_damping', 0.0),
        scm_mechanism=getattr(cfg, 'scm_mechanism', 'elementwise'),
        mlp_hidden=getattr(cfg, 'mlp_hidden', 8),
        eps_bottleneck=getattr(cfg, 'eps_bottleneck', False),
        eps_gate_init=getattr(cfg, 'eps_gate_init', 2.0),
        eps_bottleneck_target=getattr(cfg, 'eps_bottleneck_target', 'both'),
        eps_prior=getattr(cfg, 'eps_prior', 'gaussian'),
        eps_df=getattr(cfg, 'eps_df', 3.0),
        cond_prior_on=getattr(cfg, 'cond_prior_on', None),
        cond_prior_n_classes=u_n_classes,
        cond_prior_target=getattr(cfg, 'cond_prior_target', 'both'),
    ).to(DEVICE)
    if getattr(cfg, 'cond_prior_on', None) is not None:
        print(f"  CONDITIONAL eps prior p(eps|u): u='{cfg.cond_prior_on}' ({u_n_classes} classes) | "
              f"closed-form Gaussian-Gaussian KL(q(eps|x) || p(eps|u))")
    if getattr(cfg, 'eps_prior', 'gaussian') != 'gaussian':
        print(f"  NON-GAUSSIAN eps prior: {cfg.eps_prior}"
              + (f" (df={cfg.eps_df})" if cfg.eps_prior == 'studentt' else "")
              + " | closed-form Gaussian KL replaced by single-sample MC KL(q||p)")
    if getattr(cfg, 'eps_bottleneck', False):
        print(f"  ε-BOTTLENECK: global L0 gate on eps | target={getattr(cfg,'eps_bottleneck_target','both')} "
              f"eps_l0_lambda={cfg.eps_l0_lambda} init={cfg.eps_gate_init}")
    if getattr(cfg, 'scm_nonlinear', False):
        print(f"  SCM NONLINEAR: z=phi(A^T z)+eps | phi={getattr(cfg,'phi','tanh')} | fp_iters={getattr(cfg,'n_fp_iters',20)}")
    if getattr(cfg, 'context_adapt', None) or getattr(cfg, 'context_cross_modal', False):
        model.context_of_batch = context_of_batch.to(DEVICE)
        print(f"  context_adapt={cfg.context_adapt} | context_cross_modal={getattr(cfg,'context_cross_modal',False)} | "
              f"n_context={model.n_context} | iccite-ctrl batches={(context_of_batch==1).sum().item()}")

    # --- optional: load a pretrained checkpoint, then (optionally) freeze for finetuning ---
    if getattr(cfg, 'pretrained_checkpoint', None):
        sd = torch.load(cfg.pretrained_checkpoint, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  loaded pretrained: {cfg.pretrained_checkpoint} (missing={len(missing)} unexpected={len(unexpected)})")
    if getattr(cfg, 'finetune_mode', None) == 'cross_modal':
        for p in model.parameters():
            p.requires_grad = False
        for p in (model.scm.logits_rp, model.scm.weights_rp, model.scm.logits_pr, model.scm.weights_pr):
            p.requires_grad = True
        model.dec_prot.weight.requires_grad = True
        model.bias_prot.requires_grad = True
        ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ntot = sum(p.numel() for p in model.parameters())
        print(f"  FINETUNE cross_modal: trainable {ntr}/{ntot} params (SCM cross-modal + protein decoder)")

    if getattr(cfg, 'fix_A_rr_zero', False):
        model.scm.zero_A_rr = True
        with torch.no_grad():
            model.scm.weights_rr.zero_()   # checkpoint carries A_rr == 0 even for loaders unaware of the flag
        model.scm.weights_rr.requires_grad = False
        model.scm.logits_rr.requires_grad = False
        print("  FIX_A_RR_ZERO: latent A_rr fixed to 0 (logits_rr/weights_rr frozen, weights_rr zeroed)")

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
    )
    
    best_metric = float('inf')
    counter = 0
    history = {'train': [], 'val': [], 'test': {}, 'config': asdict(cfg)}
    
    print(f"Starting training for {cfg.epochs} epochs...")
    print(f"{'Ep':<4} | {'Rec':<8} | {'DAG':<8} | {'Prior':<8} | {'Rho':<6} | {'Intra-Sparse':<6} | {'Inter-Sparse':<6} | {'Orthogonal':<6} | {'Decoder-Sparse':<6} | {'KL-RNA':<6} | {'KL-Protein':<6} | {'Tot':<8}")
    print("-" * 150)
    
    for epoch in range(cfg.epochs):
        params, alpha, tau = cfg.get_params(epoch)
        # residual-independence TC estimator needs the dataset size (importance weighting); inject it
        # only when the penalty is active so off-runs' params dict / loss stay byte-identical.
        if getattr(cfg, 'resid_indep', None) is not None:
            params['dataset_size'] = len(idx_train)

        avg_train = train_epoch(model, train_loader, optimizer, priors, params, alpha, tau, DEVICE, cfg.grad_clip)
        avg_val = eval_epoch(model, val_loader, priors, params, alpha, tau, DEVICE)

        history['train'].append(avg_train)
        history['val'].append(avg_val)

        val_rec_loss = avg_val['rec']
        scheduler.step(val_rec_loss)

        if wandb_run is not None:
            import wandb
            log = {'epoch': epoch, 'alpha': alpha, 'tau': tau,
                   'lr': optimizer.param_groups[0]['lr']}
            log.update({f'train/{k}': v for k, v in avg_train.items()})
            log.update({f'val/{k}': v for k, v in avg_val.items()})
            wandb.log(log, step=epoch)
        
        if epoch % 5 == 0:
            print(f"{epoch:<4} | {avg_val['rec']:<8.4f} | "
                  f"{avg_val['dag']:<8.4f} | {avg_val['prior']:<8.4f} | "
                  f"{avg_val['rho']:<6.4f} | {avg_val['intra_sparse']:<6.4f} | {avg_val['inter_sparse']:<6.4f} | {avg_val['ortho']:<6.4f} | {avg_val['dec_sparse']:<6.4f} | {avg_val['kl_r']:<6.4f} | {avg_val['kl_p']:<6.4f} | gram={avg_val.get('gram', 0.0):<8.5f} | {avg_val['total']:<8.4f}")
            # reversal-discrepancy diagnostic (only when the term is ON; off-runs' logs stay identical)
            if getattr(cfg, 'lambda_reversal', 0.0) > 0:
                print(f"     [reversal] alpha={alpha:.2f} lambda={cfg.lambda_reversal:g} "
                      f"train: reversal={avg_train.get('reversal', 0.0):.5f} rev_gap(rec_rev-rec_fwd)={avg_train.get('rev_gap', 0.0):.5f} | "
                      f"val: reversal={avg_val.get('reversal', 0.0):.5f} rev_gap={avg_val.get('rev_gap', 0.0):.5f}")
            # residual-independence diagnostic (only when ON; off-runs' logs stay identical)
            if getattr(cfg, 'resid_indep', None) is not None and getattr(cfg, 'lambda_resid_indep', 0.0) > 0:
                print(f"     [resid_indep={cfg.resid_indep}] lambda(ramped)={params.get('lambda_resid_indep', 0.0):g} "
                      f"train={avg_train.get('resid_indep', 0.0):.5f} val={avg_val.get('resid_indep', 0.0):.5f}")

        # --- EARLY STOPPING (Monitor REC, not TOTAL) ---
        if epoch >= cfg.warmup_epochs:
            if val_rec_loss < best_metric:  
                best_metric = val_rec_loss
                counter = 0 
                torch.save(model.state_dict(), f"{cfg.results_dir}/best_model.pt")
                with open(f"{cfg.results_dir}/best_stats.json", "w") as f:
                    json.dump(avg_val, f)
            else:
                counter += 1
                if counter >= cfg.patience:
                    print(f"Early stopping at epoch {epoch} (Rec loss stagnated).")
                    break
        else:
            if val_rec_loss < best_metric:
                best_metric = val_rec_loss
                torch.save(model.state_dict(), f"{cfg.results_dir}/best_model.pt")

    # Save the FINAL (last-epoch) model — this is the fully-regularized model used for evaluation.
    # (best_model.pt = lowest val-rec, which lands at ~warmup end BEFORE beta ramps = under-regularized.)
    torch.save(model.state_dict(), f"{cfg.results_dir}/final_model.pt")
    with open(f"{cfg.results_dir}/final_stats.json", "w") as f:
        json.dump(avg_val, f)
    print(f"Saved final_model.pt (last epoch) alongside best_model.pt")

    # ==========================================
    # FINAL TEST EVALUATION
    # ==========================================
    print("\nUsing FINAL (last-epoch) model for evaluation + artifacts...")
    model.load_state_dict(torch.load(f"{cfg.results_dir}/final_model.pt"))
    model.eval()
    
    eval_params, _, _ = cfg.get_params(cfg.epochs - 1)
    avg_test = eval_epoch(model, test_loader, priors, eval_params, alpha=1.0, tau=0.5, device=DEVICE)
    history['test'] = avg_test
    print(f"TEST RESULTS: Total={avg_test['total']:.4f} | Rec={avg_test['rec']:.4f}")

    # ==========================================
    # ARTIFACT GENERATION
    # ==========================================
    with open(f"{cfg.results_dir}/metrics.json", "w") as f:
        json.dump(history, f, indent=4)

    if wandb_run is not None:
        import wandb
        wandb.log({f'test/{k}': v for k, v in avg_test.items()})
        wandb.run.summary['best_val_rec'] = best_metric

    save_artifacts(model, g_tensor, p_tensor, batch_tensor, obs_ids, g_names, p_names, cfg, DEVICE)

    if wandb_run is not None:
        import wandb
        wandb.finish()

    return history


# ==========================================
# CLI
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train CrossModal-CausalLDVAE")
    parser.add_argument("--preset", type=str, default=None, choices=list(PRESETS.keys()),
                        help="Use a preset config")
    parser.add_argument("--config", type=str, default=None,
                        help="Load config from JSON file")
    parser.add_argument("--dec_sparse_mode", type=str, default=None,
                        choices=["unnorm_rna", "feature_norm", "unnorm_both"])
    parser.add_argument("--latent_dim_rna", type=int, default=None)
    parser.add_argument("--latent_dim_prot", type=int, default=None)
    parser.add_argument("--rec_weight", type=float, default=None)
    parser.add_argument("--lambda_ortho", type=float, default=None)
    parser.add_argument("--lambda_dec_sparse", type=float, default=None)
    parser.add_argument("--lambda_dec_sparse_rna", type=float, default=None)
    parser.add_argument("--lambda_dec_sparse_prot", type=float, default=None)
    parser.add_argument("--lambda_sparse_intra", type=float, default=None)
    parser.add_argument("--lambda_proj_sparse", type=float, default=None)
    parser.add_argument("--lambda_sparse_inter", type=float, default=None)
    parser.add_argument("--dec_sparse_w_init", type=float, default=None)
    parser.add_argument("--dec_sparse_w_final", type=float, default=None)
    parser.add_argument("--beta_rna_final", type=float, default=None)
    parser.add_argument("--beta_protein_final", type=float, default=None)
    parser.add_argument("--dag_w_final", type=float, default=None)
    parser.add_argument("--prior_w_final", type=float, default=None)
    parser.add_argument("--phi", type=str, default=None)
    parser.add_argument("--nonlin_gain", type=float, default=None)
    parser.add_argument("--fp_damping", type=float, default=None)
    parser.add_argument("--scm_mechanism", type=str, default=None)
    parser.add_argument("--mlp_hidden", type=int, default=None)
    parser.add_argument("--lambda_stable", type=float, default=None)
    parser.add_argument("--lambda_reversal", type=float, default=None)
    parser.add_argument("--reversal_margin", type=float, default=None)
    parser.add_argument("--resid_indep", type=str, default=None, choices=["tc", "hsic"],
                        help="Residual-independence penalty on eps: 'tc' (beta-TCVAE Total Correlation) "
                             "or 'hsic' (pairwise HSIC). None = OFF (byte-identical no-op).")
    parser.add_argument("--lambda_resid_indep", type=float, default=None)
    parser.add_argument("--resid_indep_hsic_pairs", type=int, default=None)
    parser.add_argument("--stable_rho_bound", type=float, default=None)
    parser.add_argument("--scm_init_scale", type=float, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--mini", action="store_true", help="Use 20% of data for fast iteration")
    parser.add_argument("--baseline", action="store_true", help="Baseline mode: alpha=0 (no SCM), no structural losses")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scm_sparse_target", type=str, default=None, choices=["weight", "gate", "both"])
    parser.add_argument("--recon_likelihood", type=str, default=None, choices=["mse", "nb"])
    parser.add_argument("--eps_prior", type=str, default=None, choices=["gaussian", "laplace", "studentt"])
    parser.add_argument("--eps_df", type=float, default=None)
    parser.add_argument("--cond_prior_on", type=str, default=None, choices=["celltype", "donor", "disease"])
    parser.add_argument("--rna_path", type=str, default=None)
    parser.add_argument("--prot_path", type=str, default=None)
    parser.add_argument("--prior_rna_path", type=str, default=None)
    parser.add_argument("--prior_prot_path", type=str, default=None)
    parser.add_argument("--batch_col", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no_save_latents", action="store_true", help="Skip per-cell latent CSV dump")
    return parser.parse_args()


def main():
    args = parse_args()

    # Start from preset, loaded config, or defaults
    if args.config:
        cfg = ExperimentConfig.load(args.config)
    elif args.preset:
        cfg = PRESETS[args.preset]
    else:
        cfg = ExperimentConfig()

    # Override with any CLI args
    for field_name in vars(args):
        val = getattr(args, field_name)
        if val is not None and field_name not in ('preset', 'config') and hasattr(cfg, field_name):
            setattr(cfg, field_name, val)

    # store_true negation flags (handled explicitly so they don't always override defaults)
    if args.no_wandb:
        cfg.use_wandb = False
    if args.no_save_latents:
        cfg.save_latents = False

    # Regenerate results_dir if name changed but results_dir wasn't explicitly set
    if args.name and not args.results_dir:
        cfg.results_dir = f"results_{cfg.name}"

    run_experiment(cfg)


if __name__ == "__main__":
    main()