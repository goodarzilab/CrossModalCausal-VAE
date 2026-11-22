"""Symmetric-latent Lin+fb: identical to the best model (linfb_2c_fb10_b20_v4) but protein latent = 128
(= RNA latent), so W_rp (128x128) and W_pr (128x128) are dimensionally symmetric. Purpose: test whether
the D3 finding (|W_pr| feedback >> |W_rp| translation in magnitude) is a real effect or an artifact of the
128(RNA)/64(protein) latent-dim asymmetry + decoder scaling, and give a FAIR A_pp(PPI)/W_pr comparison.
Reconstructs config from the existing run; only latent_dim_prot and name change."""
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
base = json.load(open(f"{ROOT}/linfb_2c_fb10_b20_v4/config.json"))
fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
kw = {k: v for k, v in base.items() if k in fields}
kw["latent_dim_prot"] = 128
kw["name"] = "linfb_2c_fb10_b20_v4_p128"
kw["results_dir"] = f"{ROOT}/linfb_2c_fb10_b20_v4_p128"
cfg = ExperimentConfig(**kw)

if __name__ == "__main__":
    print(f"=== symmetric-latent Lin+fb | name={cfg.name} latent_rna={cfg.latent_dim_rna} "
          f"latent_prot={cfg.latent_dim_prot} seed={cfg.seed} {time.strftime('%F %T')} ===", flush=True)
    if os.environ.get("CMC_DRY") == "1":
        print("  DRY OK"); sys.exit(0)
    run_experiment(cfg)
