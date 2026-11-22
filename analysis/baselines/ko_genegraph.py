"""Matched in-silico-KO read-out for gene x gene graph baselines (GENIE3 importance, DeepSEM SEM W),
scored identically to CMC-VAE against the locked DE ground truth on a held-out perturbation cohort.
GENIE3: predicted effect of KO gene g on gene j = importance[g, j] (unsigned -> AUROC/AUPRC only).
DeepSEM: propagate through (I - W^T)^-1 (signed -> AUROC/AUPRC + pearDE), the same SEM inverse as ours.
Env: PERTURB_RNA_PATH/PERTURB_PROT_PATH + CMC_DS_TAG + CMC_DE_LFC_RNA/PROT (as in rerun_de_evals.sh).
Usage: python ko_genegraph.py <matrix.npz> <genie3|deepsem> <cmc_results_dir_for_vocab>"""
import sys, os, json, numpy as np
sys.path.insert(0, "analysis/common"); import perturb_validation_analysis as P
from sklearn.metrics import roc_auc_score, average_precision_score

mat_path, kind, rd = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(os.path.join(rd, "config.json")))
d = np.load(mat_path, allow_pickle=True)
M = d["matrix"].astype(np.float64) if "matrix" in d else d["W"].astype(np.float64)
Mgenes = list(d["genes"]); gidx = {g: i for i, g in enumerate(Mgenes)}
if kind == "deepsem":
    np.fill_diagonal(M, 0.0)
    Minv = np.linalg.inv(np.eye(M.shape[0]) - M.T)   # (I - W^T)^-1 : column g = total signed effect of g
    np.fill_diagonal(Minv, 0.0)

perturb_rna, perturb_prot, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
de = P.compute_observed_de(perturb_rna, perturb_prot, align)
common = list(align["common_genes"])
colidx = np.array([gidx.get(cg, -1) for cg in common])   # matrix column for each eval gene
tag = os.environ.get("CMC_DS_TAG", "frangieh")

rows = []
for g, dd in de.items():
    if g not in gidx: continue
    keep = np.array([cg != g for cg in common]); lab = dd["sig_rna"][keep].astype(int)
    if lab.sum() < 3 or lab.sum() == len(lab): continue
    lfc = np.asarray(dd["lfc_rna"])[keep]
    cols = colidx[keep]
    if kind == "genie3":
        predf = M[gidx[g], :]                 # row g = g -> targets (importance, unsigned)
    else:
        predf = Minv[:, gidx[g]]              # column g = total signed effect of g -> others
    pred = np.array([predf[c] if c >= 0 else 0.0 for c in cols])
    r = {"g": g, "n_de": int(lab.sum())}
    if not np.allclose(np.abs(pred), 0):
        r["auroc"] = float(roc_auc_score(lab, np.abs(pred)))
        r["auprcx"] = float(average_precision_score(lab, np.abs(pred)) / max(lab.mean(), 1e-9))
    de_m = lab.astype(bool)
    if kind == "deepsem" and de_m.sum() >= 5 and np.std(pred[de_m]) > 0 and np.std(lfc[de_m]) > 0:
        r["pearDE"] = float(np.corrcoef(pred[de_m], lfc[de_m])[0, 1])
    rows.append(r)

import pandas as pd
df = pd.DataFrame(rows); summ = {c: float(df[c].mean()) for c in ["auroc", "auprcx", "pearDE"] if c in df}
summ |= {c + "_sem": float(df[c].std(ddof=1) / np.sqrt(df[c].notna().sum())) for c in ["auroc", "auprcx", "pearDE"] if c in df}
summ["n"] = len(df); summ["kind"] = kind; summ["matrix"] = os.path.basename(mat_path)
out = f"results/baselines/ko_{kind}_{tag}.json"
json.dump({"summary": summ, "per_ko": rows}, open(out, "w"), indent=2)
print(f"[{kind}/{tag}] n={len(df)} RR AUROC {summ.get('auroc',float('nan')):.3f} AUPRCx {summ.get('auprcx',float('nan')):.2f} pearDE {summ.get('pearDE',float('nan')):.3f} -> {out}", flush=True)
