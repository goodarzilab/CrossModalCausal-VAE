"""RNA-only CMC-VAE (pairing ablation) on the SAME corpus/cells/split as sp_dec_gl1000.

Uses src/train.py verbatim (same config: 128 RNA latents, group-lasso decoder, loss weights, 120 epochs,
same seed-dependent 80/10/10 split over the identical cell list), with two changes:
  * protein data replaced by zeros (the model never sees protein; protein reconstruction is trivial),
  * cross_modal=False (W_rp, W_pr forced to zero), so the RNA block is a self-contained RNA-only VAE + SCM.
After training, cross-modal weights and the protein decoder are zeroed IN THE SAVED CHECKPOINT so any
harness that rebuilds the model with cross_modal=True still sees an RNA-only model (protein outputs = 0).
Usage: python train_rnaonly.py <seed> <outdir>
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import sys, os, json, torch
sys.path.insert(0, f'{_CMC_ROOT}/src')
import train as T

seed, outdir = int(sys.argv[1]), sys.argv[2]
cfg = T.ExperimentConfig.load(f'{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000/config.json')
cfg.seed = seed; cfg.name = f'rnaonly_s{seed}'; cfg.results_dir = outdir
cfg.cross_modal = False; cfg.use_wandb = False; cfg.save_latents = False

_orig = T.load_data_and_priors
def _patched(cfg_, device):
    out = list(_orig(cfg_, device))
    p = out[1]
    print(f"[rnaonly] zeroing protein tensor {tuple(p.shape)} (model never sees protein)", flush=True)
    out[1] = torch.zeros_like(p)
    if out[10] is p: out[10] = out[1]
    return tuple(out)
T.load_data_and_priors = _patched
T.run_experiment(cfg)

for fn in ('final_model.pt', 'best_model.pt'):
    pth = os.path.join(outdir, fn)
    sd = torch.load(pth, map_location='cpu')
    for k in list(sd):
        if k.split('.')[-1] in ('weights_rp', 'weights_pr') or k in ('dec_prot.weight', 'bias_prot'):
            sd[k] = torch.zeros_like(sd[k])
    torch.save(sd, pth)
json.dump({'rnaonly': True, 'note': 'protein zeroed; cross_modal False; W_rp/W_pr/dec_prot zeroed post hoc'},
          open(os.path.join(outdir, 'RNAONLY.json'), 'w'))
print('RNAONLY_DONE', outdir)
