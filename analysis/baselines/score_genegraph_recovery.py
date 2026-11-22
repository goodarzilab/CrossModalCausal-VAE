"""Generic gene x gene network recovery vs reference DBs (coexpr/STRING/DoRothEA/CollecTRI),
intersection-aligned. Directed refs -> forward |M|; undirected -> symmetrized. Usage:
  python score_genegraph_recovery.py <matrix.npz> <label>   # npz has 'W' or 'matrix' + 'genes'"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import sys, os, json, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
R = _CMC_ROOT
npz, label = sys.argv[1], sys.argv[2]
ref = np.load(f"{_CMC_RESULTS}/ref_nets_v3_2c.npz", allow_pickle=True)
rgenes = list(map(str, ref["genes"])); refmasks = {k: ref[k] for k in ref.files if k != "genes"}
meta = json.load(open(f"{_CMC_RESULTS}/ref_nets_v3_2c.json")); ridx = {g: i for i, g in enumerate(rgenes)}
d = np.load(npz, allow_pickle=True); M = np.asarray(d["W"] if "W" in d.files else d["matrix"], float)
mg = list(np.asarray(d["genes"]).astype(str)); gi = {g: i for i, g in enumerate(mg)}
common = [g for g in rgenes if g in gi]; ci = [gi[g] for g in common]; ri = [ridx[g] for g in common]
Ms = M[np.ix_(ci, ci)]; np.fill_diagonal(Ms, 0.0); off = ~np.eye(len(common), dtype=bool)
out = {"label": label, "n_common": len(common)}
print(f"=== {label} recovery (n_common={len(common)}) ===")
for rn in ["coexpr", "STRING", "DoRothEA", "CollecTRI"]:
    m = refmasks[rn][np.ix_(ri, ri)]; directed = meta[rn]["directed"]
    if directed: score = np.abs(Ms)[off]; lab = m.astype(bool)[off]
    else: Msy = np.abs(Ms) + np.abs(Ms).T; score = Msy[off]; lab = (m.astype(bool) | m.astype(bool).T)[off]
    if lab.sum() < 3: continue
    au = float(roc_auc_score(lab, score)); ap = float(average_precision_score(lab, score) / lab.mean())
    out[rn] = {"auroc": au, "auprcx": ap, "n_edges": int(lab.sum())}
    print(f"  {rn:11s} AUROC={au:.3f}  AUPRCx={ap:.2f}  n={int(lab.sum())}")
json.dump(out, open(f"{_CMC_RESULTS}/baselines/recovery_{label}.json", "w"), indent=2)
print(f"WROTE results/baselines/recovery_{label}.json")
