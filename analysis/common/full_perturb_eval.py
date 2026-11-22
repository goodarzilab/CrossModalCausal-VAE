"""Unified CORRECTED perturbation eval -> one JSON per (model, dataset).

METRIC RATIONALE (see EVAL_PATHWAY.md): panel-wide DE-ranking AUROC is INFLATED by easy true-negatives
(rare, never-DE genes) and is not comparable across vocabs of different size/composition
(Fawcett 2006; Saito & Rehmsmeier 2015; BEELINE/Pratapa 2020). The corrected headline metrics are
therefore panel-robust:
  * rr/rp_auprc_ratio  = AUPRC / random-baseline on DE labels  (BEELINE-style; robust to panel padding)  [PRIMARY]
  * rr/rp_pear_de, _spear_de = signed Δ vs signed LFC on DE genes only  (panel-INDEPENDENT)              [PRIMARY]
  * rr/rp_top20, rr_top50   = Pearson-Δ on the strongest-effect genes
  * rr_auroc, rp_auroc      = kept for continuity but PANEL-SENSITIVE — do NOT compare across vocabs     [SECONDARY]
  * rr_tert_advantage       = mediated (low-|corr|) tertile: model AUROC - correlation AUROC
For fair cross-model comparison hold the ranked gene panel fixed; within one cohort's vocab it already is.
Reuses perturb_validation_analysis do-op + observed-DE + correlation baselines.
Usage: PERTURB_RNA_PATH=.. PERTURB_PROT_PATH=.. CMC_DS_TAG=<frangieh|iccite> [CMC_EVAL_CONTEXT=k] \
       python analysis/common/full_perturb_eval.py results/<run_dir>
  -> writes results/<run_dir>/full_perturb_eval_<tag>.json
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
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(_CMC_ROOT, "src"))
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE

rd = os.path.abspath(sys.argv[1]); cfg = json.load(open(os.path.join(rd, "config.json")))
pr, pp, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
de = P.compute_observed_de(pr, pp, align)
# Perturbation-quality gate: restrict to GOOD perturbations (>=cells, on-target KD, sig E-dist phenotype)
# computed once from ground truth (perturbation_qc.py). Same set for every model. Default ON when file exists.
_tag = os.environ.get("CMC_DS_TAG", "")
_qcf = os.path.join(_CMC_RESULTS, f"perturbation_qc_{_tag}.json")
if os.environ.get("CMC_QC_FILTER", "1") != "0" and _tag and os.path.exists(_qcf):
    good = set(json.load(open(_qcf))["good"])
    de = {g: d for g, d in de.items() if g in good}
    print(f"  [QC gate] restricting to {len(de)} GOOD perturbations (from {os.path.basename(_qcf)})")
model = P.load_ctx_model(CrossModalCausalLDVAE, rd, len(align["train_gene_names"]), len(align["train_prot_names"]), cfg)
eps_r, eps_p, *_ = P.encode_control_cells(model, pr, pp, align)
common = align["common_genes"]; g2i = align["train_gene_to_idx_common"]; tgi = align["train_gene_idx"]
reti = align["rna_eval_train_indices"]; preti = align["prot_eval_train_indices"]; nP = len(align["prot_eval_names"])
corr_rr = align.get("corr_train"); corr_rp = align.get("gene_prot_corr")

def auroc(lab, sc):
    lab = lab.astype(int)
    return roc_auc_score(lab, sc) if 3 <= lab.sum() < len(lab) else np.nan
def auprc(lab, sc):  # AUPRC-over-random: 1.0 = perfect, ~1.0 baseline = no signal (robust to easy-negative padding)
    lab = lab.astype(int)
    if not (3 <= lab.sum() < len(lab)): return np.nan
    base = lab.mean()
    return average_precision_score(lab, sc) / base if base > 0 else np.nan
def pear(a, b):
    return pearsonr(a, b)[0] if len(a) >= 3 and np.std(a) > 1e-9 and np.std(b) > 1e-9 else np.nan
def topk_pear(pred, obs, k):
    if len(obs) < 3: return np.nan
    idx = np.argsort(-np.abs(obs))[:min(k, len(obs))]
    return pear(pred[idx], obs[idx])

R = {k: [] for k in ["rr_auroc","rp_auroc","rr_auprc_ratio","rp_auprc_ratio","rr_pear_de","rr_spear_de",
                     "rp_pear_de","rp_spear_de","rr_top20","rr_top50","rp_top20","rr_tert_model_low","rr_tert_corr_low"]}
for g, d in de.items():
    if g not in tgi or g not in g2i: continue
    dr_c, dr_p, pp_c, pp_p = P.do_intervention_projected(model, eps_r, eps_p, tgi[g])
    dR = (dr_p - dr_c).mean(0)[reti]; dP = (pp_p - pp_c).mean(0)[preti]
    gi = g2i[g]; keep = np.array([c != g for c in common])
    sigR = d["sig_rna"][keep]; lfcR = d["lfc_rna"][keep]; effR = d["effect_rna"][keep]; dRk = dR[keep]
    # RR
    R["rr_auroc"].append(auroc(sigR, np.abs(dRk)))
    R["rr_auprc_ratio"].append(auprc(sigR, np.abs(dRk)))
    if sigR.sum() >= 3:
        R["rr_pear_de"].append(pear(dRk[sigR], lfcR[sigR])); R["rr_spear_de"].append(spearmanr(dRk[sigR], lfcR[sigR])[0])
    R["rr_top20"].append(topk_pear(dRk, lfcR, 20)); R["rr_top50"].append(topk_pear(dRk, lfcR, 50))
    # RR low-corr tertile (mediated)
    if corr_rr is not None:
        cabs = np.abs(corr_rr[gi, :])[keep]
        if len(cabs) >= 9 and sigR.sum() >= 3:
            lo = cabs <= np.quantile(cabs, 1/3)
            if sigR[lo].sum() >= 3 and sigR[lo].sum() < lo.sum():
                R["rr_tert_model_low"].append(auroc(sigR[lo], np.abs(dRk)[lo]))
                R["rr_tert_corr_low"].append(auroc(sigR[lo], cabs[lo]))
    # RP
    sp = d.get("sig_prot", np.array([])); lp = d.get("lfc_prot", np.array([]))
    if len(sp) == nP:
        R["rp_auroc"].append(auroc(sp, np.abs(dP)))
        R["rp_auprc_ratio"].append(auprc(sp, np.abs(dP)))
        if sp.sum() >= 3:
            R["rp_pear_de"].append(pear(dP[sp], lp[sp])); R["rp_spear_de"].append(spearmanr(dP[sp], lp[sp])[0])
        R["rp_top20"].append(topk_pear(dP, lp, 20))

agg = {k: float(np.nanmean(v)) if len(v) and np.any(np.isfinite(v)) else None for k, v in R.items()}
agg["rr_tert_advantage"] = (agg["rr_tert_model_low"] - agg["rr_tert_corr_low"]
                            if agg.get("rr_tert_model_low") and agg.get("rr_tert_corr_low") else None)
agg["n_perturbations"] = len([1 for g in de if g in tgi and g in g2i]); agg["n_proteins"] = nP
tag = os.environ.get("CMC_DS_TAG", "")                       # e.g. "frangieh" / "iccite" -> per-dataset json
suffix = f"_{tag}" if tag else ""
json.dump(agg, open(os.path.join(rd, f"full_perturb_eval{suffix}.json"), "w"), indent=2)
print(f"=== full_perturb_eval[{tag or 'default'}]: {os.path.basename(rd)} (n={agg['n_perturbations']}) ===")
for k, v in agg.items():
    print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
print("DONE")
