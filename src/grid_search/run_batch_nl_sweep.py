"""Nonlinear-SCM convergence sweep (prior-OFF, 2c + 2ccombat).
Diagnosis of prior nl runs: KL posterior collapse (kl_r 27->0.5) as beta ramps to 4 -> val rec bottoms
at ep20 then RISES; nonlinear A collapses (no latent signal to structure). Fix axes:
  * free_bits_r  {0.5,1.0}  -- floor per-dim KL so beta can't collapse the latent (KEY fix)
  * beta_rna_final {1,2}    -- gentler than 4 so reconstruction + structure coexist
  * phi=relu                -- avoids tanh saturation (|A^T z|>2) that killed ANM identifiability
  * scm_init_scale=0.5      -- more A signal while staying unsaturated
All prior-off, rho_bound 0.5, n_fp_iters 25, lambda_stable 20 (stability-fixed spectral penalty).
EXPERIMENT_OFFSET/COUNT split. CMC_EPOCHS/CMC_SMOKE_TAG env overrides."""
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
DATA=f"{_CMC_DATA}"
PRIOR=f"{_CMC_ROOT}/priors"; ROOT=f"{_CMC_RESULTS}"
_VV=os.environ.get("CMC_VOCAB","v3"); _vd="v4" if _VV=="v4" else "v3"; _vt="4" if _VV=="v4" else "3"; _msfx="_v4" if _VV=="v4" else ""
DDIR={"2c":f"{DATA}/combined_{_vd}","2ccombat":f"{DATA}/combined_combat_{_vd}"}
PTAG={"2c":f"vocab{_vt}_2c","2ccombat":f"vocab{_vt}_2ccombat"}
CFG=[  # (fb, beta, phi, init)
    dict(fb=0.5, beta=2.0, phi="tanh", init=0.3),
    dict(fb=1.0, beta=2.0, phi="tanh", init=0.3),
    dict(fb=0.5, beta=1.0, phi="tanh", init=0.3),
    dict(fb=1.0, beta=1.0, phi="tanh", init=0.3),
    dict(fb=1.0, beta=2.0, phi="relu", init=0.3),
    dict(fb=1.0, beta=2.0, phi="tanh", init=0.5),
]
EXP=[dict(c=c, **k) for c in ("2c","2ccombat") for k in CFG]

def build(e):
    c=e["c"]; d=DDIR[c]; tag=PTAG[c]
    name=f"nlsw_{c}_fb{e['fb']}_b{e['beta']}_{e['phi']}_i{e['init']}".replace(".","")+os.environ.get("CMC_SMOKE_TAG","")
    return ExperimentConfig(
        name=name, results_dir=f"{ROOT}/{name}", batch_col="donor",
        epochs=int(os.environ.get("CMC_EPOCHS","120")), seed=42,
        rna_path=f"{d}/rna_combined.h5ad", prot_path=f"{d}/citeseq_combined.h5ad",
        prior_rna_path=f"{PRIOR}/rna_hvg_priors_collectri_sc_{tag}.pt",
        prior_prot_path=f"{PRIOR}/protein_priors_sc_{tag}.pt",
        beta_rna_final=e["beta"], beta_protein_final=e["beta"], prior_w_final=0,  # prior OFF
        free_bits_r=e["fb"], free_bits_p=e["fb"]*0.5,
        dag_w_final=5.0, scm_sparse_target=None, lambda_sparse_intra=0.01, lambda_sparse_inter=0.01,
        lambda_ortho=5.0, latent_dim_rna=128, latent_dim_prot=64, recon_likelihood="mse",
        dec_sparse_mode="feature_norm", lambda_dec_sparse_rna=1000, lambda_dec_sparse_prot=18,
        dec_sparse_w_init=0.05, dec_sparse_w_final=0.05,
        lambda_stable=20.0, stable_rho_bound=0.5,
        scm_nonlinear=True, phi=e["phi"], n_fp_iters=25, scm_init_scale=e["init"],
        save_latents=False, use_wandb=True, wandb_project="CrossModal-CausalVAE", wandb_group="nl_sweep")

if __name__=="__main__":
    off=int(os.environ.get("EXPERIMENT_OFFSET",0)); cnt=int(os.environ.get("EXPERIMENT_COUNT",1))
    for e in EXP[off:off+cnt]:
        cfg=build(e)
        print(f"=== {cfg.name} | nl phi={cfg.phi} fb_r={cfg.free_bits_r} beta={cfg.beta_rna_final} "
              f"init={cfg.scm_init_scale} epochs={cfg.epochs} {time.strftime('%F %T')} ===", flush=True)
        run_experiment(cfg)
