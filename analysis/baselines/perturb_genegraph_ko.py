"""Matched in-silico-KO scorer for GENE-SPACE graph baselines (GENIE3, DeepSEM), on the held-out
perturbation cohorts, using the IDENTICAL DE ground truth + eval gene set as the CMC-VAE mutilation
read-out (via perturb_validation_analysis). RNA->RNA only (these models have no protein channel).

Matched KO read-out for KO gene g:
  GENIE3 (directed importance M, unsigned):  pred[j] = M[g, j]  (g's out-edges)         -> AUROC/AUPRC
  DeepSEM (signed SEM W, x=(I-W^T)^-1 e):     pred[j] = (I-W^T)^-1[j, g]  (total g->j)   -> AUROC/AUPRC/pearDE
Scored: AUROC, AUPRC-over-random (|pred| ranks DE genes), pearDE (signed, DeepSEM only).

Usage: CMC_DS_TAG=frangieh PERTURB_RNA_PATH=... PERTURB_PROT_PATH=... \
       python perturb_genegraph_ko.py --matrix <npz> --model {genie3|deepsem} --cmc_dir <results/model>
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
import os, sys, json, argparse, numpy as np
import ctypes as _ct   # disable HW FP-exception trapping (cluster SIGFPE guard); must precede numpy math
for _l in ("libm.so.6", None):
    try: _ct.CDLL(_l).fedisableexcept(0x3f); break
    except Exception: pass
sys.path.insert(0, os.path.join(_CMC_ROOT, "analysis", "common"))
import perturb_validation_analysis as P
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--matrix", required=True); ap.add_argument("--model", choices=["genie3", "deepsem"], required=True)
ap.add_argument("--cmc_dir", default="results/linfb_2c_fb10_b20_v4")  # only for train-vocab alignment (vocab identical across 100k/full)
ap.add_argument("--out", default=None)
args = ap.parse_args()
TAG = os.environ.get("CMC_DS_TAG", "frangieh")

def auprc_ratio(lab, score):
    lab = np.asarray(lab).astype(int)
    if lab.sum() == 0 or lab.sum() == len(lab): return np.nan
    return average_precision_score(lab, score) / (lab.mean())
def auroc(lab, score):
    lab = np.asarray(lab).astype(int)
    if lab.sum() == 0 or lab.sum() == len(lab): return np.nan
    return roc_auc_score(lab, score)
def pear(pred, obs, de):
    p, o = pred[de], obs[de]
    if len(p) < 5 or np.nanstd(p) == 0 or np.nanstd(o) == 0: return np.nan
    return float(np.corrcoef(p, o)[0, 1])

# ---- DE ground truth + alignment (identical to the CMC mutilation eval) ----
cfg = json.load(open(f"{args.cmc_dir}/config.json"))
perturb_rna, perturb_prot, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
de = P.compute_observed_de(perturb_rna, perturb_prot, align)
common = list(align["common_genes"])

# ---- matrix + gene index ----
d = np.load(args.matrix, allow_pickle=True)
M = np.asarray(d["W"] if args.model == "deepsem" else d["matrix"], dtype=np.float64)
mgenes = list(np.asarray(d["genes"]).astype(str))
mrow = {g: i for i, g in enumerate(mgenes)}
signed = (args.model == "deepsem")
if signed:  # SEM influence matrix Minv = (I - W^T)^-1 ; effect g->j = Minv[j, g]
    Minv = np.linalg.inv(np.eye(M.shape[0]) - M.T)

rows = []
for g, dd in de.items():
    if g not in mrow: continue
    keep = np.array([cg != g for cg in common])
    lab = np.asarray(dd["sig_rna"])[keep].astype(int)
    if lab.sum() < 3: continue
    cols = np.array([mrow.get(cg, -1) for cg in common])
    ok = keep & (cols >= 0)
    lab = np.asarray(dd["sig_rna"])[ok].astype(int); obs = np.asarray(dd["lfc_rna"])[ok]
    ci = cols[ok]; gi = mrow[g]
    pred = Minv[ci, gi] if signed else M[gi, ci]     # signed effect g->eval genes
    a = auroc(lab, np.abs(pred)); ap_ = auprc_ratio(lab, np.abs(pred))
    pe = pear(pred, obs, lab.astype(bool)) if signed else np.nan
    rows.append(dict(g=g, n=int(len(lab)), n_de=int(lab.sum()), rr_auroc=a, rr_auprcx=ap_, rr_pearDE=pe))

import pandas as pd
df = pd.DataFrame(rows)
summ = {"model": args.model, "cohort": TAG, "n_perts": len(df), "matrix": os.path.basename(args.matrix)}
for c in ["rr_auroc", "rr_auprcx", "rr_pearDE"]:
    summ[f"{c}_mean"] = float(df[c].dropna().mean()); summ[f"{c}_sem"] = float(df[c].dropna().std(ddof=1) / max(np.sqrt(df[c].notna().sum()), 1))
out = args.out or f"results/baselines/perturb_{args.model}_{TAG}.json"
json.dump({"summary": summ, "per_pert": rows}, open(out, "w"), indent=2)
print(f"[{args.model} / {TAG}] n={len(df)}  RR AUROC {summ['rr_auroc_mean']:.3f}±{summ['rr_auroc_sem']:.3f}  "
      f"AUPRCx {summ['rr_auprcx_mean']:.3f}  pearDE {summ['rr_pearDE_mean']:.3f}  -> {out}", flush=True)
