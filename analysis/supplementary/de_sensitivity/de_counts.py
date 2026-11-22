"""DEG counts per dataset under the locked definition (raw MW p<0.05 AND |LFC|>=0.10 RNA / >=0.05
protein). Reports per-pert median/mean AND aggregate: total DEG-calls (summed over perts) and unique
genes/proteins ever called DE."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, numpy as np
sys.path.insert(0, "analysis/common"); sys.path.insert(0, "src")
_DATA = f"{_CMC_DATA}"
_DS = {"frangieh": (f"{_DATA}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad", f"{_DATA}/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad"),
       "iccite":   (f"{_DATA}/chen_perturb_icciteseq/export/iccite_rna.h5ad", f"{_DATA}/chen_perturb_icciteseq/export/iccite_prot.h5ad"),
       "arce":     (f"{_DATA}/arce_perturb_citeseq/arce_rna.h5ad", f"{_DATA}/arce_perturb_citeseq/arce_prot.h5ad")}
_tag = os.environ.get("CMC_DS_TAG", "frangieh")
os.environ["PERTURB_RNA_PATH"], os.environ["PERTURB_PROT_PATH"] = _DS[_tag]
os.environ["CMC_DE_LFC_RNA"] = "0.10"; os.environ["CMC_DE_LFC_PROT"] = "0.05"
import perturb_validation_analysis as P

cfg = json.load(open("results/linfb_2c_fb10_b20_v4/config.json"))
pr, pp, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
de = P.compute_observed_de(pr, pp, align)

def agg(mod):
    per = []; union = None
    for d in de.values():
        s = d.get(f"sig_{mod}", np.array([]))
        if len(s) == 0: continue
        s = np.asarray(s, bool); per.append(int(s.sum()))
        union = s.copy() if union is None else (union | s)
    per = np.array(per)
    return dict(n_pert=len(per), per_med=int(np.median(per)) if len(per) else 0,
                per_mean=round(float(per.mean()), 1) if len(per) else 0,
                per_min=int(per.min()) if len(per) else 0, per_max=int(per.max()) if len(per) else 0,
                total_calls=int(per.sum()), n_features=int(len(union)) if union is not None else 0,
                unique_ever_DE=int(union.sum()) if union is not None else 0)
print("DE_COUNTS_JSON " + json.dumps({"tag": _tag, "rna": agg("rna"), "prot": agg("prot")}))
