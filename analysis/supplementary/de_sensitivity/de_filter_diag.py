"""Sweep |LFC| thresholds per modality to pick a data-QC floor. For each dataset, reports #DE/pert
(median) and #perts retaining >=5 DE genes, at |LFC| in {0, 0.05, 0.1, 0.25} with the CURRENT raw
p<0.05 filter. RNA DE sets are large (0.25 fine); protein sets are tiny + CLR-compressed (need a
lower floor). Guides the modality-specific DE_LFC_THRESHOLD for the eval re-run."""
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
# CMC_DS_TAG only names the output — the actual perturbation data is selected by PERTURB_*_PATH env,
# which P reads AT IMPORT. So set the correct paths per dataset BEFORE importing P.
_DATA = f"{_CMC_DATA}"
_DS = {"frangieh": (f"{_DATA}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad", f"{_DATA}/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad"),
       "iccite":   (f"{_DATA}/chen_perturb_icciteseq/export/iccite_rna.h5ad", f"{_DATA}/chen_perturb_icciteseq/export/iccite_prot.h5ad"),
       "arce":     (f"{_DATA}/arce_perturb_citeseq/arce_rna.h5ad", f"{_DATA}/arce_perturb_citeseq/arce_prot.h5ad")}
_tag = os.environ.get("CMC_DS_TAG", "frangieh")
os.environ["PERTURB_RNA_PATH"], os.environ["PERTURB_PROT_PATH"] = _DS[_tag]
import perturb_validation_analysis as P

THRS = [0.0, 0.05, 0.1, 0.25]
cfg = json.load(open("results/linfb_2c_fb10_b20_v4/config.json"))
pr, pp, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
de = P.compute_observed_de(pr, pp, align)
tag = os.environ.get("CMC_DS_TAG", "frangieh")

def sweep(mod):
    lfck, pk = f"lfc_{mod}", f"pvals_{mod}"
    perpert = {t: [] for t in THRS}
    for g, d in de.items():
        if pk not in d or len(d[pk]) == 0: continue
        pv = np.asarray(d[pk]); lf = np.abs(np.asarray(d[lfck]))
        for t in THRS:
            perpert[t].append(int(((pv < 0.05) & (lf >= t)).sum()))
    n = len(perpert[THRS[0]])
    if n == 0:
        print(f"\n[{tag} / {mod}]  no evaluable perturbations"); return
    print(f"\n[{tag} / {mod}]  n_perturbations={n}")
    for t in THRS:
        v = np.array(perpert[t])
        print(f"   |LFC|>={t:<4}: median DE={np.median(v):5.0f}  mean={v.mean():6.1f}  perts w/>=5 DE: {int((v>=5).sum())}/{n}  perts w/>=3: {int((v>=3).sum())}/{n}")

sweep("rna")
if len(align.get("prot_eval_perturb_indices", [])) > 0:
    sweep("prot")
