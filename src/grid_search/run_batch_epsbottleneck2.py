"""ε-BOTTLENECK sweep #2: HIGHER L0 lambda (push n_roots lower) x gate TARGET {both, rna-only}, v4 vocab,
2c + 2ccombat. Follows up the first sweep (l0 in {0.5,2.0} -> n_roots 116/85, RR load-bearing Δ +0.006..+0.013,
dose-dependent). Questions: (1) does a STRONGER bottleneck make A_rr more load-bearing? (2) is gating ONLY
eps_r (targeting A_rr, leaving the already-load-bearing cross-modal protein path free) better than gating both?
A-sparsity kept ON (fix #2). Grid: cohort{2c,2ccombat} x target{both,rna} x l0{5.0,10.0} = 8 runs.
EXPERIMENT_OFFSET/COUNT split; CMC_VOCAB=v4, CMC_SMOKE_TAG=_v4."""
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
_VV = os.environ.get("CMC_VOCAB", "v4"); _vd = "v4" if _VV == "v4" else "v3"; _vt = "4" if _VV == "v4" else "3"
DDIR = {"2c": f"{DATA}/combined_{_vd}", "2ccombat": f"{DATA}/combined_combat_{_vd}"}
PTAG = {"2c": f"vocab{_vt}_2c", "2ccombat": f"vocab{_vt}_2ccombat"}
GRID = [dict(c=c, tgt=tgt, l0=l0) for c in ("2c", "2ccombat") for tgt in ("both", "rna") for l0 in (5.0, 10.0)]


def build(e):
    c = e["c"]; d = DDIR[c]; tag = PTAG[c]
    l0name = str(e["l0"]).replace(".0", "")
    name = f"epsbn2_{c}_l0{l0name}_{e['tgt']}" + os.environ.get("CMC_SMOKE_TAG", "")
    return ExperimentConfig(
        name=name, results_dir=f"{ROOT}/{name}", batch_col="donor",
        epochs=int(os.environ.get("CMC_EPOCHS", "100")), seed=42, patience=999,
        rna_path=f"{d}/rna_combined.h5ad", prot_path=f"{d}/citeseq_combined.h5ad",
        prior_rna_path=f"{PRIOR}/rna_hvg_priors_collectri_sc_{tag}.pt",
        prior_prot_path=f"{PRIOR}/protein_priors_sc_{tag}.pt",
        beta_rna_final=2.0, beta_protein_final=2.0, prior_w_final=0,
        free_bits_r=1.0, free_bits_p=0.5,
        eps_bottleneck=True, eps_l0_lambda=e["l0"], eps_gate_init=2.0, eps_bottleneck_target=e["tgt"],  # FIX #1 (stronger + targeted)
        scm_sparse_target="weight", lambda_sparse_intra=0.01, lambda_sparse_inter=0.01,               # FIX #2 (on)
        dag_w_final=5.0, lambda_ortho=5.0, latent_dim_rna=128, latent_dim_prot=64, recon_likelihood="mse",
        dec_sparse_mode="feature_norm", lambda_dec_sparse_rna=1000, lambda_dec_sparse_prot=18,
        dec_sparse_w_init=0.05, dec_sparse_w_final=0.05,
        lambda_stable=20.0, stable_rho_bound=0.9, scm_nonlinear=False,
        save_latents=False, use_wandb=True, wandb_project="CrossModal-CausalVAE", wandb_group="eps_bottleneck2")


if __name__ == "__main__":
    off = int(os.environ.get("EXPERIMENT_OFFSET", 0)); cnt = int(os.environ.get("EXPERIMENT_COUNT", 1))
    for e in GRID[off:off + cnt]:
        cfg = build(e)
        print(f"=== {cfg.name} | eps_bottleneck l0={cfg.eps_l0_lambda} target={cfg.eps_bottleneck_target} "
              f"sparse={cfg.lambda_sparse_intra} v4 epochs={cfg.epochs} {time.strftime('%F %T')} ===", flush=True)
        run_experiment(cfg)
