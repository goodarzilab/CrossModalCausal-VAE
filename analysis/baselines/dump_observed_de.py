"""Dump observed per-KO RNA log-fold-changes + DE mask for one perturbation cohort, so the DeepSEM
interventional fine-tuner can pool all cohorts in a single process. Env: CMC_DS_TAG + PERTURB_*."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os,sys,json,numpy as np
sys.path.insert(0,os.path.join(_CMC_ROOT, "analysis", "common"))
import perturb_validation_analysis as P
TAG=os.environ["CMC_DS_TAG"]; CMC_DIR=os.environ.get("CMC_DIR","results/linfb_2c_fb10_b20_100k")
cfg=json.load(open(f"{CMC_DIR}/config.json"))
pr,pp,align=P.load_shared_data(cfg["rna_path"],cfg["prot_path"])
de=P.compute_observed_de(pr,pp,align); common=list(align["common_genes"])
genes=[g for g in de if 0<np.asarray(de[g]["sig_rna"]).sum()<len(common)]
lfc=np.stack([np.asarray(de[g]["lfc_rna"]) for g in genes]); sig=np.stack([np.asarray(de[g]["sig_rna"]).astype(int) for g in genes])
np.savez_compressed(f"results/baselines/de_dump_{TAG}.npz",genes=np.array(genes,dtype=object),
                    common=np.array(common,dtype=object),lfc=lfc,sig=sig)
print(f"[{TAG}] dumped {len(genes)} KO genes x {len(common)} common genes -> de_dump_{TAG}.npz")
