"""Diagnose the heavy-tailedness of the RNA input: is the extreme per-gene excess-kurtosis a real
data property, a few outlier genes/cells, or a normalization artifact? Check BOTH cohorts
(combined_v4 = 2c, combined_combat_v4 = 2c+COMBAT). Reports, on log1p(.X) and raw counts (if present):
per-gene excess-kurtosis + skew distributions (mean/median/p90/p99/max), top offender genes, per-cell
library-size distribution (tiny libraries inflate CP10k), zero-fraction, and cross-cohort comparison."""
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
import scanpy as sc
from scipy.stats import kurtosis, skew
DATA = f"{_CMC_DATA}"
COH = {"2c": f"{DATA}/combined_v4", "2c+COMBAT": f"{DATA}/combined_combat_v4"}
rng = np.random.RandomState(42); N = 60000
out = {}
for tag, d in COH.items():
    a = sc.read_h5ad(f"{d}/rna_combined.h5ad")
    idx = np.sort(rng.choice(a.n_obs, min(N, a.n_obs), replace=False))
    genes = list(a.var_names.astype(str))
    X = a.X[idx]; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    r = {"n_cells_total": int(a.n_obs), "n_genes": len(genes), "n_sampled": int(len(idx)),
         "X.min": float(X.min()), "X.max": float(X.max()), "X.mean": float(X.mean()),
         "X_looks_log1p": bool(X.max() < 20 and not np.allclose(X, np.round(X))),
         "X_is_integer": bool(np.allclose(X, np.round(X))), "layers": list(a.layers.keys())}
    # per-gene excess kurtosis + skew on the (log1p) .X
    gk = kurtosis(X, axis=0, fisher=True, nan_policy="omit"); gs = skew(X, axis=0, nan_policy="omit")
    zero_frac = (X == 0).mean(0)
    def dist(v): v=np.asarray(v); return dict(mean=float(np.nanmean(v)), median=float(np.nanmedian(v)),
        p90=float(np.nanpercentile(v,90)), p99=float(np.nanpercentile(v,99)), max=float(np.nanmax(v)))
    r["logX_gene_excess_kurtosis"] = dist(gk)
    r["logX_gene_abs_skew"] = dist(np.abs(gs))
    r["gene_zero_frac"] = dist(zero_frac)
    top = np.argsort(gk)[::-1][:8]
    r["top_kurtosis_genes"] = [(genes[i], round(float(gk[i]),1), round(float(zero_frac[i]),3), round(float(X[:,i].max()),2)) for i in top]
    # per-cell library size (on raw counts if available, else expm1 of log1p)
    if "counts" in a.layers:
        C = a.layers["counts"][idx]; C = C.toarray() if hasattr(C,"toarray") else np.asarray(C)
        lib = C.sum(1); r["raw_lib"] = dict(min=float(lib.min()), p1=float(np.percentile(lib,1)),
            median=float(np.median(lib)), max=float(lib.max()))
        ck = kurtosis(np.log1p(C), axis=0, fisher=True, nan_policy="omit")
        r["rawcounts_log1p_gene_excess_kurtosis"] = dist(ck)
        rawk = kurtosis(C, axis=0, fisher=True, nan_policy="omit")
        r["rawcounts_gene_excess_kurtosis"] = dist(rawk)
    else:
        lib = np.expm1(X).sum(1); r["pseudo_lib_from_expm1"] = dict(min=float(lib.min()),
            p1=float(np.percentile(lib,1)), median=float(np.median(lib)), max=float(lib.max()))
    out[tag] = r
    print(f"\n===== {tag} ({d}) =====")
    print(f"  X: min {r['X.min']:.2f} max {r['X.max']:.2f} mean {r['X.mean']:.3f} | log1p-looking={r['X_looks_log1p']} integer={r['X_is_integer']} layers={r['layers']}")
    k=r["logX_gene_excess_kurtosis"]; print(f"  log1p gene excess-kurtosis: mean {k['mean']:.1f} MEDIAN {k['median']:.2f} p90 {k['p90']:.1f} p99 {k['p99']:.1f} max {k['max']:.0f}")
    s=r["logX_gene_abs_skew"]; print(f"  log1p gene |skew|:           mean {s['mean']:.2f} median {s['median']:.2f} p99 {s['p99']:.2f}")
    z=r["gene_zero_frac"]; print(f"  gene zero-fraction:          median {z['median']:.3f} p90 {z['p90']:.3f}")
    print(f"  top-kurtosis genes (gene, exk, zerofrac, maxval): {r['top_kurtosis_genes'][:5]}")
    if "raw_lib" in r:
        L=r["raw_lib"]; print(f"  raw library size: min {L['min']:.0f} p1 {L['p1']:.0f} median {L['median']:.0f} max {L['max']:.0f}")
        print(f"  rawcounts(log1p) gene exk: median {r['rawcounts_log1p_gene_excess_kurtosis']['median']:.2f} mean {r['rawcounts_log1p_gene_excess_kurtosis']['mean']:.1f}")
    del a, X
json.dump(out, open("results/data_skew_diag.json","w"), indent=2)
print("\nWROTE results/data_skew_diag.json")
