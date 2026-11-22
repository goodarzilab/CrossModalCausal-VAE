"""Per-sweep-model recovery + sparsity/Gini scorer. Reads a trained model dir (projected GRN_inferred.csv
+ final_model.pt), scores GRN recovery vs ref_nets_v3_2c, and computes latent+projected Gini/sparsity.
Perturbation (mutilation) is scored separately (block_isolated_ablation writes block_isolated_mut_*.json).
Merges everything -> <model_dir>/sweep_metrics.json.  Usage: python sweep_eval_model.py <model_dir>"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, numpy as np, pandas as pd, torch
import ctypes as _ct
for _l in ("libm.so.6", None):
    try: _ct.CDLL(_l).fedisableexcept(0x3f); break
    except Exception: pass
from sklearn.metrics import roc_auc_score
R = f"{_CMC_ROOT}"; MD = sys.argv[1].rstrip("/")
ref = np.load(f"{_CMC_RESULTS}/ref_nets_v3_2c.npz", allow_pickle=True)
rgenes = list(map(str, ref["genes"])); ridx = {g: i for i, g in enumerate(rgenes)}
meta = json.load(open(f"{_CMC_RESULTS}/ref_nets_v3_2c.json")); refm = {k: ref[k] for k in ref.files if k != "genes"}

def gini(x):
    x = np.sort(np.abs(np.asarray(x, float)).ravel()); n = x.size
    return float((2 * (np.arange(1, n + 1) * x).sum()) / (n * x.sum()) - (n + 1.0) / n) if n and x.sum() > 0 else float("nan")

def recov(M, genes):
    gi = {str(g): i for i, g in enumerate(genes)}
    common = [g for g in rgenes if g in gi]; ci = [gi[g] for g in common]; ri = [ridx[g] for g in common]
    Ms = np.abs(np.asarray(M, float)[np.ix_(ci, ci)]); np.fill_diagonal(Ms, 0); off = ~np.eye(len(common), dtype=bool)
    out = {}
    for rn in ["coexpr", "STRING", "DoRothEA", "CollecTRI"]:
        mk = refm[rn][np.ix_(ri, ri)]
        if meta[rn]["directed"]: s = Ms[off]; lab = mk.astype(bool)[off]
        else: My = Ms + Ms.T; s = My[off]; lab = (mk.astype(bool) | mk.astype(bool).T)[off]
        out[rn] = round(float(roc_auc_score(lab, s)), 4)
    return out

res = {"model_dir": MD, "name": os.path.basename(MD)}

# --- recovery (projected gene GRN) ---
Grr = pd.read_csv(f"{MD}/GRN_inferred.csv", index_col=0)
res["recovery"] = recov(Grr.values, list(Grr.index))
a = np.abs(Grr.values.copy()); np.fill_diagonal(a, 0); off = ~np.eye(a.shape[0], dtype=bool)
res["proj_gini"] = round(gini(a[off]), 4)
res["proj_sparsity"] = round(float((a[off] < 1e-6 * a[off].max()).mean()), 4)

# --- latent A_rr sparsity/gini + effective density (final model) ---
fm = torch.load(f"{MD}/final_model.pt", map_location="cpu", weights_only=False)
lg = fm["scm.logits_rr"].float().numpy(); wt = fm["scm.weights_rr"].float().numpy(); n = lg.shape[0]
gate = 1 / (1 + np.exp(-lg)); A = np.abs(gate * wt * (1 - np.eye(n))); loff = ~np.eye(n, dtype=bool)
res["latent_gini"] = round(gini(A[loff]), 4)
res["latent_mean_gate"] = round(float(gate[loff].mean()), 4)
res["latent_frac_gate>0.5"] = round(float((gate[loff] > 0.5).mean()), 4)

# --- perturbation (mutilation) from block_isolated_mut_<c>_ablation.json ---
COH = ["frangieh", "iccite", "arce"]
pert = {"rr_auroc": {}, "rp_auroc": {}, "rr_pearDE": {}, "rp_pearDE": {}}
for c in COH:
    p = f"{MD}/block_isolated_mut_{c}_ablation.json"
    if not os.path.exists(p): continue
    s = json.load(open(p)); s = s.get("summary", s)
    pert["rr_auroc"][c] = s.get("mean_rr_true"); pert["rp_auroc"][c] = s.get("mean_rp_true")
    pert["rr_pearDE"][c] = s.get("mean_rr_pearDE_true"); pert["rp_pearDE"][c] = s.get("mean_rp_pearDE_true")
def _mean(d): v = [x for x in d.values() if x is not None]; return round(float(np.mean(v)), 4) if v else None
res["perturb"] = pert
res["perturb_mean"] = {k: _mean(pert[k]) for k in pert}

json.dump(res, open(f"{MD}/sweep_metrics.json", "w"), indent=2)
print(json.dumps({k: res[k] for k in ["name", "recovery", "proj_gini", "proj_sparsity", "latent_gini",
                                       "latent_mean_gate", "perturb_mean"]}, indent=2))
