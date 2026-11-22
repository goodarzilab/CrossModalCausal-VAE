"""PRIOR-WEIGHT sweep to find the learning/memorization window for the GRN prior on v4.
Known endpoints: pw500 -> prior_grn stuck at margin (NOT learned); pw2000 (pubstrong) -> CollecTRI 0.99
(MEMORIZED, hurts perturbation). Question: is there a moderate pw that LEARNS the directed prior
(CollecTRI ~0.7-0.85, prior_grn drops below margin) WITHOUT memorizing / wrecking cross-modal RP?
Base = the pubstrong 'stabilized strong-prior' config (the one that DOES learn at pw2000), sweeping pw
DOWN. Track: prior_grn (learned?), CollecTRI AUROC (memorized if ~0.99?), RP perturbation (hurt?),
gproj RR direction (helped?). 2c, v4. Grid: pw {750,1000,1500} = 3 runs."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train import ExperimentConfig, run_experiment
DATA = f"{_CMC_DATA}"
PRIOR = f"{_CMC_ROOT}/priors"; ROOT = f"{_CMC_RESULTS}"
DDIR = f"{DATA}/combined_v4"; TAG = "vocab4_2c"
GRID = [dict(pw=pw) for pw in (750, 1000, 1500)]


def build(e):
    pw = e["pw"]
    name = f"priorw_2c_pw{pw}" + os.environ.get("CMC_SMOKE_TAG", "")
    return ExperimentConfig(
        name=name, results_dir=f"{ROOT}/{name}", batch_col="donor",
        epochs=int(os.environ.get("CMC_EPOCHS", "120")), seed=42, patience=999,
        rna_path=f"{DDIR}/rna_combined.h5ad", prot_path=f"{DDIR}/citeseq_combined.h5ad",
        prior_rna_path=f"{PRIOR}/rna_hvg_priors_collectri_sc_{TAG}.pt",
        prior_prot_path=f"{PRIOR}/protein_priors_sc_{TAG}.pt",
        beta_rna_init=1e-4, beta_rna_final=4.0, beta_protein_final=4.0,
        prior_w_final=pw,                                       # <-- swept
        free_bits_r=1.0, free_bits_p=0.5,
        dag_w_final=5.0, lambda_ortho=5.0,
        lambda_sparse_intra=0.01, lambda_sparse_inter=0.01,
        lambda_stable=20.0, stable_rho_bound=0.9,
        dec_sparse_mode="feature_norm", lambda_dec_sparse_rna=1000, lambda_dec_sparse_prot=18,
        dec_sparse_w_init=0.05, dec_sparse_w_final=0.05,
        latent_dim_rna=128, latent_dim_prot=64, recon_likelihood="mse", scm_nonlinear=False,
        save_latents=False, use_wandb=True, wandb_project="CrossModal-CausalVAE", wandb_group="priorw_sweep")


if __name__ == "__main__":
    off = int(os.environ.get("EXPERIMENT_OFFSET", 0)); cnt = int(os.environ.get("EXPERIMENT_COUNT", 1))
    for e in GRID[off:off + cnt]:
        cfg = build(e)
        print(f"=== {cfg.name} | PRIOR-W SWEEP pw={cfg.prior_w_final} (find learn/memorize window) v4 "
              f"epochs={cfg.epochs} {time.strftime('%F %T')} ===", flush=True)
        run_experiment(cfg)
