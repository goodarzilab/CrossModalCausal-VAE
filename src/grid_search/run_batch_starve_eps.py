"""STARVE-EPS sweep: crank KL / reduce free-bits on the BEST model (linfb_2c_fb10_b20_v4) to test
whether starving the exogenous eps pushes representational structure OUT of eps and INTO the latent
DAG A (makes A_rr load-bearing) without collapsing reconstruction.

Reconstructs the EXACT base config from results/linfb_2c_fb10_b20_v4/config.json (like
run_batch_linfb_seeds.py) and ONLY changes the swept field(s) + name + results_dir.

GRID (EXPERIMENT_OFFSET / EXPERIMENT_COUNT select which to run):
  0  linfb_2c_fbr03_v4        free_bits_r = 0.3   (from 1.0) — mild eps starvation
  1  linfb_2c_fbr00_v4        free_bits_r = 0.0                — no free-bits floor (full KL pressure)
  2  linfb_2c_beta4_v4        beta_rna_final = 4.0 (fb_r=1.0) — higher beta
  3  linfb_2c_beta4_fbr03_v4  beta_rna_final = 4.0 AND free_bits_r = 0.3 (optional combined)
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

# name -> dict of swept overrides
GRID = [
    ("linfb_2c_fbr03_v4",       dict(free_bits_r=0.3)),
    ("linfb_2c_fbr00_v4",       dict(free_bits_r=0.0)),
    ("linfb_2c_beta4_v4",       dict(beta_rna_final=4.0)),
    ("linfb_2c_beta4_fbr03_v4", dict(beta_rna_final=4.0, free_bits_r=0.3)),
]


def build(name, overrides):
    base = json.load(open(BASE))
    fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
    dropped = sorted(set(base) - fields)
    kw = {k: v for k, v in base.items() if k in fields}
    kw["name"] = name
    kw["results_dir"] = f"{ROOT}/{name}"
    kw["wandb_group"] = "starve_eps_sweep"
    kw.update(overrides)
    return ExperimentConfig(**kw), dropped


if __name__ == "__main__":
    off = int(os.environ.get("EXPERIMENT_OFFSET", 0))
    cnt = int(os.environ.get("EXPERIMENT_COUNT", 1))
    for name, ov in GRID[off:off + cnt]:
        cfg, dropped = build(name, ov)
        print(f"=== STARVE-EPS | name={cfg.name} | overrides={ov} | free_bits_r={cfg.free_bits_r} "
              f"free_bits_p={cfg.free_bits_p} beta_rna_final={cfg.beta_rna_final} "
              f"beta_protein_final={cfg.beta_protein_final} scm_nonlinear={cfg.scm_nonlinear} "
              f"{time.strftime('%F %T')} ===", flush=True)
        if dropped:
            print(f"  [config.json keys not in ExperimentConfig, ignored]: {dropped}", flush=True)
        if os.environ.get("CMC_DRY") == "1":
            print("  DRY RUN — config reconstructed OK, not training.")
            continue
        run_experiment(cfg)
