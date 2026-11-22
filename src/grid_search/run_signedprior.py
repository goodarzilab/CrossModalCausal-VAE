"""Signed-prior retrain: base = EXACT Lin+fb config (linfb_2c_fb10_b20_v4) with prior OFF
(prior_w_final=0, unchanged) + the SIGNED DIRECTED prior loss ON (KnockTF + signed CollecTRI/DoRothEA).
Reconstructs the config from the existing run's config.json (like run_batch_linfb_seeds.py) so the
only changes are the signed-prior flags + name/results_dir.

Env:
  CMC_LAMBDA_SP   signed-prior loss weight (default 5.0)
  CMC_SP_MARGIN   hinge margin in std units (default 0.5)
  CMC_SMOKE=1     smoke test: mini data, few epochs, short warmup, no wandb, no latents
  CMC_DRY=1       reconstruct config + verify, do not train
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import os, sys, json, dataclasses, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train import ExperimentConfig, run_experiment

ROOT = f"{_CMC_RESULTS}"
BASE = f"{ROOT}/linfb_2c_fb10_b20_v4/config.json"
SIGNED_PRIOR = f"{_CMC_ROOT}/priors/signed_prior_2c_v4.pt"

LAMBDA_SP = float(os.environ.get("CMC_LAMBDA_SP", "5.0"))
SP_MARGIN = float(os.environ.get("CMC_SP_MARGIN", "0.5"))
SMOKE = os.environ.get("CMC_SMOKE") == "1"

base = json.load(open(BASE))
fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
dropped = sorted(set(base) - fields)
kw = {k: v for k, v in base.items() if k in fields}

# --- signed-prior flags (the ONLY additions vs the Lin+fb baseline; prior_w_final stays 0) ---
kw["signed_prior_path"] = SIGNED_PRIOR
kw["lambda_signed_prior"] = LAMBDA_SP
kw["signed_prior_margin"] = SP_MARGIN
kw["signed_prior_use_train_tfs_only"] = True

if SMOKE:
    kw["name"] = "linfb_2c_signedprior_smoke"
    kw["results_dir"] = f"{ROOT}/linfb_2c_signedprior_smoke"
    kw["mini"] = True
    kw["epochs"] = 10
    kw["warmup_epochs"] = 3
    kw["alpha_ramp_epochs"] = 3
    kw["tau_switch_epoch"] = 3
    kw["use_wandb"] = False
    kw["save_latents"] = False
else:
    kw["name"] = "linfb_2c_signedprior_v4"
    kw["results_dir"] = f"{ROOT}/linfb_2c_signedprior_v4"
    kw["use_wandb"] = False
    kw["save_latents"] = False

cfg = ExperimentConfig(**kw)

if __name__ == "__main__":
    print(f"=== SIGNED-PRIOR retrain | name={cfg.name} lambda_signed_prior={cfg.lambda_signed_prior} "
          f"margin={cfg.signed_prior_margin} prior_w_final={cfg.prior_w_final} epochs={cfg.epochs} "
          f"mini={cfg.mini} {time.strftime('%F %T')} ===", flush=True)
    print(f"  signed_prior_path={cfg.signed_prior_path}", flush=True)
    if dropped:
        print(f"  [config.json keys not in ExperimentConfig, ignored]: {dropped}", flush=True)
    if os.environ.get("CMC_DRY") == "1":
        print("  DRY RUN — config reconstructed OK, not training."); sys.exit(0)
    run_experiment(cfg)
