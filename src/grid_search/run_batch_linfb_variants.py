"""Lin+fb base (the BEST model) + ONE clean architectural change each, to fill gaps in the ablation.
Base = linfb_2c_fb10_b20 (prior OFF, linear SCM, free-bits 1.0/0.5, beta 2, no decoder mask, no bottleneck).
Missing clean combos vs what we already have (nlsw=+nonlinear, pathdec=+learnable-GO-decoder):
  - elfbbn : + PURE eps-bottleneck (L0 gate on eps), decoder UNCHANGED, latent stays 128 (isolates the
             bottleneck effect without the confound of also masking/freezing the decoder).
2c cohort, v4 vocab. EXPERIMENT_OFFSET/COUNT."""
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
DDIR = {"2c": f"{DATA}/combined_v4"}; PTAG = {"2c": "vocab4_2c"}
GRID = [dict(c="2c", variant="bn")]


def build(e):
    c = e["c"]; d = DDIR[c]; tag = PTAG[c]
    name = f"elfb{e['variant']}_{c}" + os.environ.get("CMC_SMOKE_TAG", "")
    base = dict(
        name=name, results_dir=f"{ROOT}/{name}", batch_col="donor",
        epochs=int(os.environ.get("CMC_EPOCHS", "120")), seed=42, warmup_epochs=20, patience=999,
        rna_path=f"{d}/rna_combined.h5ad", prot_path=f"{d}/citeseq_combined.h5ad",
        prior_rna_path=f"{PRIOR}/rna_hvg_priors_collectri_sc_{tag}.pt",
        prior_prot_path=f"{PRIOR}/protein_priors_sc_{tag}.pt",
        beta_rna_init=1e-4, beta_rna_final=2.0, beta_protein_final=2.0,
        free_bits_r=1.0, free_bits_p=0.5, prior_w_final=0,
        dag_w_final=5.0, lambda_ortho=5.0, lambda_stable=20.0, stable_rho_bound=0.9,
        lambda_sparse_intra=0.01, lambda_sparse_inter=0.01, scm_sparse_target=None,
        dec_sparse_mode="feature_norm", lambda_dec_sparse_rna=1000, lambda_dec_sparse_prot=18,
        dec_sparse_w_init=0.05, dec_sparse_w_final=0.05,
        latent_dim_rna=128, latent_dim_prot=64, recon_likelihood="mse", scm_nonlinear=False,
        save_latents=False, use_wandb=True, wandb_project="CrossModal-CausalVAE", wandb_group="linfb_variants")
    if e["variant"] == "bn":   # PURE eps-bottleneck: decoder untouched
        base.update(eps_bottleneck=True, eps_l0_lambda=1.0, eps_gate_init=2.0, eps_bottleneck_target="both")
    return ExperimentConfig(**base)


if __name__ == "__main__":
    off = int(os.environ.get("EXPERIMENT_OFFSET", 0)); cnt = int(os.environ.get("EXPERIMENT_COUNT", 1))
    for e in GRID[off:off + cnt]:
        cfg = build(e)
        print(f"=== {cfg.name} | Lin+fb + {e['variant']} v4 {time.strftime('%F %T')} ===", flush=True)
        run_experiment(cfg)
