"""Review-2 M7a scratch control. Driver modelled on analysis/fig4_cross_modal/adapt/adapt.py; imports adapt_lib UNCHANGED.
usage: python -I scratch_adapt.py SEEDKEY DOMAINS ARMS      e.g. s42 frangieh,iccite,arce repro_i,scratch,scratch_x3,adapt_x3,scratch_gen
Arms (PLAN.md):
  repro_i     pretrained init, 4,000 steps   (= variant i; reproduction check)
  scratch     random init,     4,000 steps
  scratch_x3  random init,    12,000 steps
  adapt_x3    pretrained init,12,000 steps
  scratch_gen random init,    12,000 steps, lr 1e-3, BN running stats updated during training steps
Models -> adapt_scratch/models/<arm>_<dom>_<sk>/ ; diag -> adapt_scratch/diag/<arm>_<dom>_<sk>.json"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, copy
sys.path.insert(0, f"{_CMC_ROOT}/analysis/fig4_cross_modal/adapt")
import torch, numpy as np
import adapt_lib as L
from adapt_lib import log
HERE = f"{_CMC_RESULTS}/paper/analyses/review2/adapt_scratch"
ARMS = {"repro_i": dict(init="pre", steps=4000), "scratch": dict(init="rand", steps=4000),
        "scratch_x3": dict(init="rand", steps=12000), "adapt_x3": dict(init="pre", steps=12000),
        "scratch_gen": dict(init="rand", steps=12000, lr=1e-3, bn_update=True)}
# scratch_gen's lr: the Adapter's optimiser takes VARIANTS[variant]['lr'] -> register a variant identical to 'i' with lr 1e-3
L.VARIANTS["i_lr1e3"] = dict(L.VARIANTS["i"], lr=1e-3)

def fresh_model(rd, seed):
    """same architecture as the checkpoint, parameters from the constructor's default initialisers (as src/train.py)."""
    cfg = json.load(open(f"{rd}/config.json"))
    sd = torch.load(L.P.resolve_ckpt(rd), map_location="cpu"); ng, np_ = sd["dec_rna.weight"].shape[0], sd["dec_prot.weight"].shape[0]
    torch.manual_seed(seed); np.random.seed(seed)
    m = L.CrossModalCausalLDVAE(ng, np_, num_batches=1, latent_dim_rna=cfg["latent_dim_rna"], latent_dim_prot=cfg["latent_dim_prot"],
                                recon_likelihood=cfg.get("recon_likelihood", "mse"), gate_mode=cfg.get("gate_mode", "gumbel")).to(L.DEV)
    fs = m.state_dict(); assert set(fs) == set(sd) and all(fs[k].shape == sd[k].shape for k in sd)
    # non-learned buffers that are NOT parameters of the fit (e.g. SCM structural masks) must equal the checkpoint's
    for k, v in fs.items():
        if k.startswith("scm.mask"): assert torch.equal(v.cpu(), sd[k].cpu()), k
    m.eval(); assert m.dec_r_mask is None and not m.scm.scm_nonlinear and m.context_adapt is None and not m.eps_bottleneck
    return m, cfg

_orig_modes = L.Adapter.modes
def modes_bn_update(self, train):
    """scratch_gen: as Adapter.modes but BN follows train/eval (running statistics updated on training steps)."""
    self.m.enc_rna.train(train); self.m.enc_prot.train(train); self.m.scm.eval()

if __name__ == "__main__":
    SK, DOMS, AR = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
    rd0 = L.BASES["sp"][SK]; seed = L.SEEDNUM[SK]
    for dom in DOMS:
        D = L.domain_data(dom); rg = json.load(open(f"{L.OUT}/diag/ridge_{dom}.json"))   # read-only cached ridge (same split)
        for arm in AR:
            a = ARMS[arm]; rd = f"{HERE}/models/{arm}_{dom}_{SK}"
            if os.path.exists(f"{rd}/final_model.pt"): log("exists", rd); continue
            m, cfg = (L.load_model(rd0) if a["init"] == "pre" else fresh_model(rd0, seed))
            d_init = L.diag_screen(m, D)
            L.HP["max_steps"] = a["steps"]
            L.Adapter.modes = modes_bn_update if a.get("bn_update") else _orig_modes
            v = "i_lr1e3" if a.get("lr") else "i"
            info = L.run_adapt(m, D, v, seed)
            L.HP["max_steps"] = 4000; L.Adapter.modes = _orig_modes
            d = L.diag_screen(m, D)
            rec = dict(arm=arm, arm_def=a, base_rd=rd0, init=a["init"], domain=dom, seed=SK, hp=dict(L.HP, max_steps=a["steps"]),
                       variant_def=L.VARIANTS[v], best_step=info["best_step"], crit0=info["crit0"], crit_best=info["crit_best"],
                       n_steps_run=info["n_steps_run"])
            log(f"{arm} {dom} {SK}: best_step {info['best_step']}/{info['n_steps_run']} crit {info['crit0']:.2f}->{info['crit_best']:.2f} | "
                f"rna_only {d_init['rna_only']:.3f}->{d['rna_only']:.3f} with_prot {d['with_protein']:.3f} share {d['var_share_rna_path']:.3f} "
                f"rna_rec {d['rna_recon_r_median']:.3f} (ridge {rg['median']:.3f})")
            L.save_model(m, cfg, rd, rec)
            json.dump(dict(rec, hist=info["hist"], diag=d, diag_init=d_init, ridge_median=rg["median"]),
                      open(f"{HERE}/diag/{arm}_{dom}_{SK}.json", "w"), default=float)
    log("DONE")
