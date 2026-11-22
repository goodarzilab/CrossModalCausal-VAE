"""Generate the sparsity sweep configs (809k, .sum() L1). Templated from the baseline nodag_v4 config,
overriding sparsity knobs. Grid: latent .sum() x projected .sum() x decoder group-lasso.
Calibrated: baseline sum|A_rr|~216, sum|DAD^T|~216 -> lambda ~5e-3/2e-2/5e-2/2e-1 give penalties ~1/4/11/43.
Writes results/sweep_sparse/configs/<name>.json + prints the launch list."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import os, sys, json, copy
from dataclasses import fields
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train import ExperimentConfig
R = f"{_CMC_ROOT}"
VALID = {f.name for f in fields(ExperimentConfig)}
base = {k: v for k, v in json.load(open(f"{_CMC_RESULTS}/linfb_2c_nodag_v4/config.json")).items() if k in VALID}
OUT = f"{_CMC_RESULTS}/sweep_sparse"; os.makedirs(f"{OUT}/configs", exist_ok=True)

def cfg(name, **ov):
    c = copy.deepcopy(base); c["name"] = name; c["results_dir"] = f"{OUT}/{name}"
    c.update(ov); return c

grid = []
# --- latent .sum() (proj off, decoder = baseline feature_norm) ---
for lam in [5e-3, 2e-2, 5e-2, 2e-1]:
    grid.append(cfg(f"sp_lat_{lam:g}", scm_sparse_reduction="sum",
                    lambda_sparse_intra=lam, lambda_sparse_inter=lam, lambda_proj_sparse=0.0))
# --- projected .sum() (latent = baseline mean/weak, decoder baseline) ---
for lam in [5e-3, 2e-2, 5e-2]:
    grid.append(cfg(f"sp_proj_{lam:g}", scm_sparse_reduction="sum",
                    lambda_sparse_intra=0.01, lambda_sparse_inter=0.01, lambda_proj_sparse=lam))
# --- decoder group-lasso (latent/proj = baseline) ---
for ldec in [1000, 5000]:
    grid.append(cfg(f"sp_dec_gl{ldec}", dec_sparse_mode="group_lasso",
                    lambda_dec_sparse_rna=ldec, lambda_dec_sparse_prot=18))
# --- combined latent+projected+decoder ---
grid.append(cfg("sp_comb_med", scm_sparse_reduction="sum", lambda_sparse_intra=2e-2, lambda_sparse_inter=2e-2,
                lambda_proj_sparse=2e-2, dec_sparse_mode="group_lasso", lambda_dec_sparse_rna=1000))
grid.append(cfg("sp_comb_strong", scm_sparse_reduction="sum", lambda_sparse_intra=5e-2, lambda_sparse_inter=5e-2,
                lambda_proj_sparse=5e-2, dec_sparse_mode="group_lasso", lambda_dec_sparse_rna=5000))
grid.append(cfg("sp_lat_dec", scm_sparse_reduction="sum", lambda_sparse_intra=5e-2, lambda_sparse_inter=5e-2,
                lambda_proj_sparse=0.0, dec_sparse_mode="group_lasso", lambda_dec_sparse_rna=5000))

for c in grid:
    json.dump(c, open(f"{OUT}/configs/{c['name']}.json", "w"), indent=2)
print(f"wrote {len(grid)} configs to {OUT}/configs/")
for c in grid:
    print(f"  {c['name']:16s} red={c.get('scm_sparse_reduction')} lat={c['lambda_sparse_intra']:g} "
          f"proj={c['lambda_proj_sparse']:g} dec_mode={c['dec_sparse_mode']} ldec_rna={c['lambda_dec_sparse_rna']}")
