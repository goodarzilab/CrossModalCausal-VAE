#!/usr/bin/env python
"""Collect the eps-prior comparison table: NON-GAUSSIAN (laplace) vs GAUSSIAN baseline.
Reads recovery / symmetry+kurtosis / edge-reversal / perturbation JSONs for both runs and prints a
side-by-side table. Writes results/eps_prior/comparison.json. Usage: python analysis/supplementary/objectives/eps_prior_collect.py"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, os, numpy as np
R = f"{_CMC_ROOT}"
RUNS = {"gaussian_baseline": "gaussian_ref", "laplace": "laplace_2c"}
COHORTS = ["frangieh", "iccite", "arce"]

def load(p):
    try: return json.load(open(p))
    except Exception: return None

def mean_over_cohorts(dirname, fname_tmpl, keys):
    """Average given summary keys of <dir>/<tmpl(tag)>.json across cohorts (nan-safe)."""
    out = {k: [] for k in keys}
    per = {}
    for ds in COHORTS:
        d = load(f"{_CMC_RESULTS}/eps_prior/{dirname}/{fname_tmpl.format(ds=ds)}")
        if d is None: continue
        s = d.get("summary", d)
        per[ds] = {k: s.get(k) for k in keys}
        for k in keys:
            v = s.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)): out[k].append(v)
    return {k: (float(np.mean(v)) if v else None) for k, v in out.items()}, per

res = {}
for label, dirn in RUNS.items():
    entry = {}
    # recovery
    rec = load(f"{_CMC_RESULTS}/bio_recovery_{dirn}.json")
    if rec:
        entry["recovery"] = {k: (rec[k]["auroc"] if isinstance(rec.get(k), dict) else rec.get(k))
                             for k in ["coexpr", "STRING", "DoRothEA", "CollecTRI"]}
        entry["recovery"]["symG_projected"] = rec.get("symG")
    # symmetry + kurtosis
    sk = load(f"{_CMC_RESULTS}/eps_prior/{dirn}/eps_prior_symkurt.json")
    if sk:
        entry["A_rr_corr_A_AT"] = sk["A_rr_latent"]["corr_A_AT"]
        entry["A_rr_antisym_energy_frac"] = sk["A_rr_latent"]["antisym_energy_frac"]
        entry["G_rr_corr_G_GT"] = sk["G_rr_projected"]["corr_G_GT"]
        entry["G_rr_antisym_energy_frac"] = sk["G_rr_projected"]["antisym_energy_frac"]
        entry["eps_r_mu_mean_excess_kurtosis"] = sk["eps_r_mu_kurtosis"]["mean_excess_kurtosis"]
        entry["eps_r_mu_median_excess_kurtosis"] = sk["eps_r_mu_kurtosis"]["median_excess_kurtosis"]
        entry["eps_r_sampled_mean_excess_kurtosis"] = sk["eps_r_sampled_kurtosis"]["mean_excess_kurtosis"]
    # edge-reversal (avg over cohorts): RR/RP true vs reversed AUROC + pearDE
    er_keys = ["mean_rr_true", "mean_rr_rev", "mean_rr_pearDE_true", "mean_rr_pearDE_rev",
               "mean_rp_true", "mean_rp_rev", "mean_rp_pearDE_true", "mean_rp_pearDE_rev"]
    er, er_per = mean_over_cohorts(dirn, "edge_reversal_{ds}.json", er_keys)
    entry["edge_reversal_mean"] = er
    entry["edge_reversal_gap"] = {
        "rr_auroc_true_minus_rev": (er["mean_rr_true"] - er["mean_rr_rev"]) if er["mean_rr_true"] and er["mean_rr_rev"] else None,
        "rr_pearDE_true_minus_rev": (er["mean_rr_pearDE_true"] - er["mean_rr_pearDE_rev"]) if er["mean_rr_pearDE_true"] is not None and er["mean_rr_pearDE_rev"] is not None else None,
        "rp_auroc_true_minus_rev": (er["mean_rp_true"] - er["mean_rp_rev"]) if er["mean_rp_true"] and er["mean_rp_rev"] else None,
        "rp_pearDE_true_minus_rev": (er["mean_rp_pearDE_true"] - er["mean_rp_pearDE_rev"]) if er["mean_rp_pearDE_true"] is not None and er["mean_rp_pearDE_rev"] is not None else None,
    }
    entry["edge_reversal_per_cohort"] = er_per
    # perturbation (avg over cohorts): RR/RP AUROC + pearDE
    p_keys = ["rr_auroc", "rp_auroc", "rr_pear_de", "rp_pear_de", "rr_auprc_ratio", "rp_auprc_ratio"]
    pm, pm_per = mean_over_cohorts(dirn, "full_perturb_eval_{ds}.json", p_keys)
    entry["perturbation_mean"] = pm
    entry["perturbation_per_cohort"] = pm_per
    res[label] = entry

json.dump(res, open(f"{_CMC_RESULTS}/eps_prior/comparison.json", "w"), indent=2)

def g(label, *path):
    d = res.get(label, {})
    for p in path:
        d = d.get(p) if isinstance(d, dict) else None
        if d is None: return float("nan")
    return d if isinstance(d, (int, float)) else float("nan")

B, L = "gaussian_baseline", "laplace"
def row(name, b, l): print(f"  {name:<42s} {b:>12} {l:>12}")
print("\n==================== eps-prior comparison: GAUSSIAN baseline vs LAPLACE ====================")
print(f"  {'metric':<42s} {'GAUSSIAN':>12} {'LAPLACE':>12}")
print("  " + "-" * 68)
def f(x): return f"{x:.4f}" if isinstance(x,(int,float)) and x==x else "NA"
print("  -- eps non-Gaussianity (excess kurtosis; Gaussian=0, Laplace=3) --")
row("eps_r mu mean excess kurtosis", f(g(B,"eps_r_mu_mean_excess_kurtosis")), f(g(L,"eps_r_mu_mean_excess_kurtosis")))
row("eps_r mu median excess kurtosis", f(g(B,"eps_r_mu_median_excess_kurtosis")), f(g(L,"eps_r_mu_median_excess_kurtosis")))
row("eps_r SAMPLED mean excess kurtosis", f(g(B,"eps_r_sampled_mean_excess_kurtosis")), f(g(L,"eps_r_sampled_mean_excess_kurtosis")))
print("  -- graph symmetry / orientation --")
row("A_rr corr(A,A^T)  [1=symmetric]", f(g(B,"A_rr_corr_A_AT")), f(g(L,"A_rr_corr_A_AT")))
row("A_rr antisym energy frac  [0=sym]", f(g(B,"A_rr_antisym_energy_frac")), f(g(L,"A_rr_antisym_energy_frac")))
row("G_rr corr(G,G^T) projected", f(g(B,"G_rr_corr_G_GT")), f(g(L,"G_rr_corr_G_GT")))
row("G_rr antisym energy frac projected", f(g(B,"G_rr_antisym_energy_frac")), f(g(L,"G_rr_antisym_energy_frac")))
print("  -- recovery (AUROC vs reference nets) --")
for k in ["coexpr","STRING","DoRothEA","CollecTRI"]: row(f"recovery {k}", f(g(B,"recovery",k)), f(g(L,"recovery",k)))
print("  -- perturbation (mean over cohorts) --")
row("RR AUROC", f(g(B,"perturbation_mean","rr_auroc")), f(g(L,"perturbation_mean","rr_auroc")))
row("RP AUROC", f(g(B,"perturbation_mean","rp_auroc")), f(g(L,"perturbation_mean","rp_auroc")))
row("RR pearDE", f(g(B,"perturbation_mean","rr_pear_de")), f(g(L,"perturbation_mean","rr_pear_de")))
row("RP pearDE", f(g(B,"perturbation_mean","rp_pear_de")), f(g(L,"perturbation_mean","rp_pear_de")))
print("  -- edge-reversal inertness (gap = true - reversed; ~0 => NOT directional) --")
row("RR AUROC gap (true-rev)", f(g(B,"edge_reversal_gap","rr_auroc_true_minus_rev")), f(g(L,"edge_reversal_gap","rr_auroc_true_minus_rev")))
row("RR pearDE gap (true-rev)", f(g(B,"edge_reversal_gap","rr_pearDE_true_minus_rev")), f(g(L,"edge_reversal_gap","rr_pearDE_true_minus_rev")))
row("RP AUROC gap (true-rev)", f(g(B,"edge_reversal_gap","rp_auroc_true_minus_rev")), f(g(L,"edge_reversal_gap","rp_auroc_true_minus_rev")))
row("RP pearDE gap (true-rev)", f(g(B,"edge_reversal_gap","rp_pearDE_true_minus_rev")), f(g(L,"edge_reversal_gap","rp_pearDE_true_minus_rev")))
print("  " + "-" * 68)
print(f"  wrote {_CMC_RESULTS}/eps_prior/comparison.json")
