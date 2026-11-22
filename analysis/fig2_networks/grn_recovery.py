"""Projected-GRN recovery: how well the inferred gene-space GRN G = D_r A_rr D_r^T recovers known
networks. For each reference (CollecTRI, DoRothEA, coexpr, STRING) report AUROC + AUPRC-over-random
of |G| separating edges from non-edges (off-diagonal). Directed refs use directed G; symmetric refs
use the symmetrized |G|+|G|^T. Flags refs that were the training prior (circular).
Usage: python analysis/fig2_networks/grn_recovery.py results/<run>
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
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(_CMC_ROOT,"src"))
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE
rd=os.path.abspath(sys.argv[1]); cfg=json.load(open(f"{rd}/config.json"))
cohort="2ccombat" if "2ccombat" in os.path.basename(rd) else "2c"
ref=np.load(f"{_CMC_RESULTS}/ref_nets_v3_{cohort}.npz",allow_pickle=True)
rgenes=list(map(str,ref["genes"])); refmasks={k:ref[k] for k in ref.files if k!="genes"}
meta=json.load(open(f"{_CMC_RESULTS}/ref_nets_v3_{cohort}.json"))

# build projected GRN from the trained model
import scanpy as sc

import os as _os
def _ckpt(rd):
    """FINAL checkpoint by default (fully-regularized); fallback best. Override CMC_CKPT=best."""
    pref=_os.environ.get("CMC_CKPT","final")
    for n in [f"{pref}_model.pt","final_model.pt","best_model.pt"]:
        p=_os.path.join(rd,n)
        if _os.path.exists(p): return p
    return _os.path.join(rd,"best_model.pt")

ng=sc.read_h5ad(cfg["rna_path"],backed="r").shape[1]; npr=sc.read_h5ad(cfg["prot_path"],backed="r").shape[1]
model=CrossModalCausalLDVAE(ng,npr,1,latent_dim_rna=cfg["latent_dim_rna"],latent_dim_prot=cfg["latent_dim_prot"],recon_likelihood="mse")
model.load_state_dict(torch.load(_ckpt(rd),map_location="cpu"),strict=False); model.eval()
with torch.no_grad():
    A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
    Dr=model.dec_rna.weight.detach().float()
    G=(Dr@A_rr.float()@Dr.T).cpu().numpy()
model_genes=list(sc.read_h5ad(cfg["rna_path"],backed="r").var_names.astype(str))
assert model_genes==rgenes, "vocab mismatch between model and ref_nets"
np.fill_diagonal(G,0.0)
off=~np.eye(len(G),dtype=bool)
out={"run":os.path.basename(rd),"cohort":cohort,"prior":os.path.basename(cfg.get("prior_rna_path",""))}
for name,m in refmasks.items():
    directed=meta[name]["directed"]
    if directed:
        score=np.abs(G)[off]; lab=m.astype(bool)[off]
    else:
        Gs=np.abs(G)+np.abs(G).T; score=Gs[off]; lab=(m.astype(bool)|m.astype(bool).T)[off]
    if lab.sum()<3: continue
    base=lab.mean()
    out[name]={"auroc":float(roc_auc_score(lab,score)),"auprc":float(average_precision_score(lab,score)),
               "auprc_ratio":float(average_precision_score(lab,score)/base),"n_edges":int(lab.sum()),
               "prevalence":float(base),"circular":("training prior" in meta[name].get("note",""))}
json.dump(out,open(f"{rd}/grn_recovery.json","w"),indent=2)
print(f"=== GRN recovery: {out['run']} (prior={out['prior']}) ===")
for name in refmasks:
    if name in out: r=out[name]; print(f"  {name:11s} AUROC={r['auroc']:.3f}  AUPRC={r['auprc']:.3f} ({r['auprc_ratio']:.2f}x)  n={r['n_edges']}"+("  [CIRCULAR: training prior]" if r['circular'] else ""))
print("DONE")
