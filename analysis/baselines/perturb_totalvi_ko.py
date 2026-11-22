"""Matched in-silico-KO read-out for totalVI, scored identically to CMC-VAE against the locked DE
ground truth. totalVI has no SCM, so its 'network' is its denoised generative correlation
(G_rr_tv = corr of denoised RNA-RNA; G_rp_tv = corr of denoised RNA-protein). Matched KO read-out for
KO gene g: RR pred[j]=G_rr_tv[g,j] (g's denoised row), RP pred[p]=G_rp_tv[g,p]. Protein-aware -> RR + RP.
Runs in the BASE env (consumes the precomputed matrices in totalvi_raw.npz).
Env: PERTURB_RNA_PATH/PERTURB_PROT_PATH + CMC_DS_TAG + CMC_DE_LFC_RNA/PROT (as in rerun_de_evals.sh).
Usage: python perturb_totalvi_ko.py --npz results/baselines/totalvi_raw.npz --cmc_dir results/linfb_2c_fb10_b20_100k"""
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
sys.path.insert(0, os.path.join(_CMC_ROOT, "analysis", "common"))
import perturb_validation_analysis as P
from sklearn.metrics import roc_auc_score, average_precision_score
ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="results/baselines/totalvi_raw.npz")
ap.add_argument("--cmc_dir", default="results/linfb_2c_fb10_b20_100k")
args = ap.parse_args(); TAG = os.environ.get("CMC_DS_TAG", "frangieh")

def auroc(lab, s):
    lab = np.asarray(lab).astype(int)
    return roc_auc_score(lab, s) if 0 < lab.sum() < len(lab) else np.nan
def auprcx(lab, s):
    lab = np.asarray(lab).astype(int)
    return average_precision_score(lab, s) / max(lab.mean(), 1e-9) if 0 < lab.sum() < len(lab) else np.nan
def pear(pred, obs, de):
    p, o = pred[de], obs[de]
    return float(np.corrcoef(p, o)[0, 1]) if len(p) >= 5 and np.std(p) > 0 and np.std(o) > 0 else np.nan

d = np.load(args.npz, allow_pickle=True)
Grr, Grp = np.asarray(d["G_rr_tv"], float), np.asarray(d["G_rp_tv"], float)
tgenes = list(np.asarray(d["genes"]).astype(str)); gidx = {g: i for i, g in enumerate(tgenes)}
np.fill_diagonal(Grr, 0.0)
cfg = json.load(open(f"{args.cmc_dir}/config.json"))
perturb_rna, perturb_prot, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
de = P.compute_observed_de(perturb_rna, perturb_prot, align)
common = list(align["common_genes"]); rr_col = np.array([gidx.get(cg, -1) for cg in common])
p_eval = np.asarray(align["prot_eval_train_indices"])   # totalVI prot columns == train prot order

rows = []
for g, dd in de.items():
    if g not in gidx: continue
    keep = np.array([cg != g for cg in common]); lab = dd["sig_rna"][keep].astype(int)
    r = {"g": g}
    if 0 < lab.sum() < len(lab):
        cols = rr_col[keep]; predr = np.array([Grr[gidx[g], c] if c >= 0 else 0.0 for c in cols])
        r["rr_auroc"] = auroc(lab, np.abs(predr)); r["rr_auprcx"] = auprcx(lab, np.abs(predr))
        de_r = lab.astype(bool); lfc = np.asarray(dd["lfc_rna"])[keep]
        r["rr_pearDE"] = pear(predr, lfc, de_r)
    labp = np.asarray(dd.get("sig_prot", []))
    if labp.size and 0 < labp.sum() < len(labp):
        predp = Grp[gidx[g], p_eval]
        r["rp_auroc"] = auroc(labp, np.abs(predp)); r["rp_auprcx"] = auprcx(labp, np.abs(predp))
        de_p = labp.astype(bool); lfcp = np.asarray(dd.get("lfc_prot", []))
        r["rp_pearDE"] = pear(predp, lfcp, de_p)
    rows.append(r)
import pandas as pd
df = pd.DataFrame(rows)
mets = ["rr_auroc", "rr_auprcx", "rr_pearDE", "rp_auroc", "rp_auprcx", "rp_pearDE"]
summ = {c: float(df[c].mean()) for c in mets if c in df}
summ |= {c + "_sem": float(df[c].std(ddof=1) / np.sqrt(df[c].notna().sum())) for c in mets if c in df}
summ["n"] = len(df); summ["model"] = "totalvi"
out = f"results/baselines/ko_totalvi_{TAG}.json"
json.dump({"summary": summ, "per_ko": rows}, open(out, "w"), indent=2)
print(f"[totalvi/{TAG}] n={len(df)} RR AUROC {summ.get('rr_auroc',float('nan')):.3f} pearDE {summ.get('rr_pearDE',float('nan')):.3f} | "
      f"RP AUROC {summ.get('rp_auroc',float('nan')):.3f} pearDE {summ.get('rp_pearDE',float('nan')):.3f} -> {out}", flush=True)
