"""G2 (multi-seed CIs): retrain the BEST model (Lin+fb, linfb_2c_fb10_b20_v4) with additional seeds so
we can report mean±SD on every headline metric. Reconstructs the EXACT config from the existing run's
config.json (only `seed`, `name`, `results_dir` change) — guarantees the CI models are identical to the
headline model except the seed. Seed via CMC_SEED. Dry-run check via CMC_DRY=1."""
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

SEED = int(os.environ.get("CMC_SEED", "1"))
base = json.load(open(BASE))
fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
dropped = sorted(set(base) - fields)
kw = {k: v for k, v in base.items() if k in fields}
kw["seed"] = SEED
kw["name"] = f"linfb_2c_fb10_b20_v4_s{SEED}"
kw["results_dir"] = f"{ROOT}/linfb_2c_fb10_b20_v4_s{SEED}"
cfg = ExperimentConfig(**kw)

if __name__ == "__main__":
    print(f"=== G2 seed retrain | name={cfg.name} seed={cfg.seed} scm_nonlinear={cfg.scm_nonlinear} "
          f"free_bits_r={cfg.free_bits_r} beta_rna_final={cfg.beta_rna_final} prior_w_final={cfg.prior_w_final} "
          f"{time.strftime('%F %T')} ===", flush=True)
    if dropped:
        print(f"  [config.json keys not in ExperimentConfig, ignored]: {dropped}", flush=True)
    if os.environ.get("CMC_DRY") == "1":
        print("  DRY RUN — config reconstructed OK, not training."); sys.exit(0)
    run_experiment(cfg)
