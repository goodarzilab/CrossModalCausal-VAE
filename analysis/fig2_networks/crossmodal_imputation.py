"""Cross-modal imputation eval (the on-thesis translation-network metric, no perturbations needed).
On held-out 2-cohort cells: predict protein from RNA only (force-drop protein) and RNA from protein
only (force-drop RNA), via the model's learned-null latent + cross-modal SCM block. Report per-feature
Pearson(predicted, true) vs the full (both-modalities-present) reconstruction ceiling.

Usage: python analysis/fig2_networks/crossmodal_imputation.py results/<run_dir>
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json
import numpy as np, torch, scanpy as sc
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.join(_CMC_ROOT, "src"))
from model import CrossModalCausalLDVAE

rd = os.path.abspath(sys.argv[1]); cfg = json.load(open(os.path.join(rd, "config.json")))
rna = sc.read_h5ad(cfg["rna_path"]); prot = sc.read_h5ad(cfg["prot_path"])
obs = sorted(set(rna.obs_names) & set(prot.obs_names)); rna = rna[obs]; prot = prot[obs]
Xr = rna.X.toarray() if hasattr(rna.X, 'toarray') else np.asarray(rna.X)
Xp = prot.X.toarray() if hasattr(prot.X, 'toarray') else np.asarray(prot.X)
Xr = ((Xr - Xr.mean(0)) / (Xr.std(0) + 1e-6)).astype(np.float32)     # match train.py global z-score
Xp = ((Xp - Xp.mean(0)) / (Xp.std(0) + 1e-6)).astype(np.float32)
seed = cfg.get("seed", 42)
idx = np.arange(len(obs)); _, tmp = train_test_split(idx, test_size=0.2, random_state=seed)
_, te = train_test_split(tmp, test_size=0.5, random_state=seed)        # held-out test split (matches train)

sd = torch.load(os.path.join(rd, os.environ.get("CMC_CKPT_FILE", "final_model.pt")), map_location="cpu"); sd.pop("context_of_batch", None)
m = CrossModalCausalLDVAE(Xr.shape[1], Xp.shape[1], 1,
                          latent_dim_rna=cfg["latent_dim_rna"], latent_dim_prot=cfg["latent_dim_prot"],
                          recon_likelihood=(cfg.get("recon_likelihood") or "mse"),
                          gate_mode=cfg.get("gate_mode", "gumbel"))
m.load_state_dict(sd, strict=False); m.eval()

xr = torch.tensor(Xr[te]); xp = torch.tensor(Xp[te]); bb = torch.zeros(len(te), dtype=torch.long)
with torch.no_grad():
    rec_full_r, rec_full_p = m(xr, xp, bb)['rec']
    rec_p_imp = m(xr, xp, bb, force_drop="protein")['rec'][1].numpy()   # protein <- RNA only
    rec_r_imp = m(xr, xp, bb, force_drop="rna")['rec'][0].numpy()       # RNA <- protein only

def percol_pearson(P, Tr):
    out = []
    for j in range(Tr.shape[1]):
        a, b = P[:, j], Tr[:, j]
        if a.std() > 1e-6 and b.std() > 1e-6:
            out.append(np.corrcoef(a, b)[0, 1])
    return np.array(out)

pP_imp = percol_pearson(rec_p_imp, Xp[te]); pP_full = percol_pearson(rec_full_p.numpy(), Xp[te])
pR_imp = percol_pearson(rec_r_imp, Xr[te]); pR_full = percol_pearson(rec_full_r.numpy(), Xr[te])
res = {"protein_from_rna_pearson": float(np.nanmean(pP_imp)), "protein_full_pearson": float(np.nanmean(pP_full)),
       "rna_from_protein_pearson": float(np.nanmean(pR_imp)), "rna_full_pearson": float(np.nanmean(pR_full)),
       "n_test_cells": int(len(te)), "n_proteins": int(Xp.shape[1]), "n_genes": int(Xr.shape[1])}
json.dump(res, open(os.path.join(rd, "crossmodal_imputation.json"), "w"), indent=2)
print(f"\n=== cross-modal imputation: {os.path.basename(rd)} (n_test={len(te)}) ===")
print(f"  PROTEIN from RNA-only : Pearson {res['protein_from_rna_pearson']:.3f}  (full-recon ceiling {res['protein_full_pearson']:.3f})")
print(f"  RNA from PROTEIN-only : Pearson {res['rna_from_protein_pearson']:.3f}  (full-recon ceiling {res['rna_full_pearson']:.3f})")
print("DONE")
