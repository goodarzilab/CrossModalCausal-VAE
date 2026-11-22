"""Control-cell mean (log-normalized / CLR) expression of every model-vocabulary gene per CITE-seq screen.
Uses the align dict cached by analysis/common/reviewer_controls.py (same control mask, same feature mapping)."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import pickle, numpy as np, scanpy as sc, sys
REPO = f"{_CMC_ROOT}"; OUT = f"{_CMC_RESULTS}/paper/analyses/operator"
DATA = f"{_CMC_DATA}"
DS = {"frangieh": f"{DATA}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad",
      "iccite": f"{DATA}/chen_perturb_icciteseq/export/iccite_rna.h5ad",
      "arce": f"{DATA}/arce_perturb_citeseq/arce_rna.h5ad"}
for tag, path in DS.items():
    b = pickle.load(open(f"{_CMC_RESULTS}/_ftcache/rc_{tag}_sp_dec_gl1000_3a5d257c9323_0.1_0.05.pkl", "rb")); al = b["align"]
    ad = sc.read_h5ad(path, backed="r"); cm = np.asarray(al["control_mask"].values, bool)
    idx = np.where(cm)[0]; pi = np.asarray(al["rna_eval_perturb_indices"]); ti = np.asarray(al["rna_eval_train_indices"])
    X = ad.X[idx] if not hasattr(ad, "isbacked") else ad[idx].to_memory().X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    Xc = X[:, pi].astype(np.float64)
    mean_log = Xc.mean(0); frac = (Xc > 0).mean(0)
    mz = (mean_log - np.asarray(al["train_rna_mean"])[ti]) / np.clip(np.asarray(al["train_rna_std"])[ti], 1e-8, None)
    np.savez(f"{OUT}/ctrl_means_{tag}.npz", common=np.array(al["common_genes"]), train_idx=ti, mean_log=mean_log, frac_expr=frac,
             mean_z=mz, n_ctrl=len(idx))
    print(tag, len(idx), "ctrl cells; median mean_log", np.median(mean_log), flush=True)
