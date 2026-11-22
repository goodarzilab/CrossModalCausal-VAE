"""Collate the residual-independence ablation vs the sp_dec_gl1000 headline baseline.
For baseline + results/resid_indep/*, reads:
  - config.json                -> resid_indep, lambda_resid_indep
  - resid_indep_metrics.json   -> A_rr/G_rr symmetry + achieved TC / mean pairwise HSIC / mean|corr| of eps
                                  (falls back to eps_prior_symkurt.json for symmetry if absent)
  - sweep_metrics.json         -> recovery{coexpr,STRING,DoRothEA,CollecTRI}, perturb_mean{rr/rp auroc+pearDE}
  - edge_reversal_<ds>.json    -> edge-reversal inertness  gap = mean(rr_true - rr_rev) over cohorts
Prints a table and writes results/resid_indep/comparison.json.
Usage: python analysis/supplementary/objectives/collect_resid_indep.py
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, json, glob
import numpy as np
R = f"{_CMC_ROOT}"
COH = ["frangieh", "iccite", "arce"]
def J(p): return json.load(open(p)) if os.path.exists(p) else None


def orient_asym(md):
    """mean over cohorts of (rr_true - rr_rev) for AUROC and signed-pearDE (edge-reversal inertness)."""
    au, pe = [], []
    for c in COH:
        s = J(f"{md}/edge_reversal_{c}.json")
        if not s: continue
        s = s.get("summary", s)
        t, r = s.get("mean_rr_true"), s.get("mean_rr_rev")
        if t is not None and r is not None and np.isfinite(t) and np.isfinite(r): au.append(t - r)
        tp, rp = s.get("mean_rr_pearDE_true"), s.get("mean_rr_pearDE_rev")
        if tp is not None and rp is not None and np.isfinite(tp) and np.isfinite(rp): pe.append(tp - rp)
    return (round(float(np.mean(au)), 4) if au else None,
            round(float(np.mean(pe)), 4) if pe else None)


def row(md):
    cfg = J(f"{md}/config.json") or {}
    sm = J(f"{md}/sweep_metrics.json") or {}
    ri = J(f"{md}/resid_indep_metrics.json")
    sk = ri or J(f"{md}/eps_prior_symkurt.json") or {}   # symmetry fallback
    rec = sm.get("recovery", {}); pm = sm.get("perturb_mean", {})
    au_asym, pe_asym = orient_asym(md)
    r = {
        "name": os.path.basename(md.rstrip("/")),
        "resid_indep": cfg.get("resid_indep"),
        "lambda": cfg.get("lambda_resid_indep", 0.0),
        # (A) orientation
        "A_rr_corr": (sk.get("A_rr_latent") or {}).get("corr_A_AT"),
        "A_rr_antisym": (sk.get("A_rr_latent") or {}).get("antisym_energy_frac"),
        "G_rr_corr": (sk.get("G_rr_projected") or {}).get("corr_G_GT"),
        "G_rr_antisym": (sk.get("G_rr_projected") or {}).get("antisym_energy_frac"),
        # (B) achieved residual independence (only present in resid_indep_metrics.json)
        "achTC": None, "achHSIC": None, "achCorr": None,
        # recovery
        "coexpr": rec.get("coexpr"), "STRING": rec.get("STRING"),
        "DoRothEA": rec.get("DoRothEA"), "CollecTRI": rec.get("CollecTRI"),
        # perturbation
        "RR_AUROC": pm.get("rr_auroc"), "RP_AUROC": pm.get("rp_auroc"),
        "RR_pearDE": pm.get("rr_pearDE"), "RP_pearDE": pm.get("rp_pearDE"),
        # edge-reversal inertness
        "orient_gap_RR_AUROC": au_asym, "orient_gap_RR_pearDE": pe_asym,
    }
    if ri:
        r["achTC"] = (ri.get("achieved_TC_mss", {}).get("eps_sample") or {}).get("mean")
        r["achHSIC"] = (ri.get("achieved_mean_pairwise_HSIC", {}).get("eps_sample") or {}).get("mean")
        r["achCorr"] = (ri.get("achieved_mean_abs_offdiag_corr") or {}).get("eps_sample")
    return r


base = f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000"
rows = [row(base)]
for d in sorted(glob.glob(f"{_CMC_RESULTS}/resid_indep/*/")):
    if os.path.exists(f"{d}/config.json"):
        rows.append(row(d))


def f(x, w=8, p=4):
    return (f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'NA':>{w}}")


print("\n============ RESIDUAL-INDEPENDENCE ablation vs sp_dec_gl1000 baseline ============")
h1 = f"{'model':14s} {'mode':5s} {'lam':>7s} | {'A_rr':>7s} {'Aasym':>7s} {'G_rr':>7s} | {'achTC':>8s} {'achHSIC':>8s} {'ach|r|':>7s}"
print(h1); print("-" * len(h1))
for r in rows:
    tag = "  <==BASE" if r["name"] == "sp_dec_gl1000" else ""
    print(f"{r['name'][:14]:14s} {str(r['resid_indep']):5s} {r['lambda']:7g} | "
          f"{f(r['A_rr_corr'],7)} {f(r['A_rr_antisym'],7)} {f(r['G_rr_corr'],7)} | "
          f"{f(r['achTC'],8,3)} {f(r['achHSIC'],8,5)} {f(r['achCorr'],7)}{tag}")
print()
h2 = f"{'model':14s} | {'coexpr':>7s} {'DoRoth':>7s} {'CollTRI':>7s} {'STRING':>7s} | {'RR-AU':>7s} {'RP-AU':>7s} {'RRpear':>7s} {'RPpear':>7s} | {'revGapRR':>8s}"
print(h2); print("-" * len(h2))
for r in rows:
    tag = "  <==BASE" if r["name"] == "sp_dec_gl1000" else ""
    print(f"{r['name'][:14]:14s} | {f(r['coexpr'],7)} {f(r['DoRothEA'],7)} {f(r['CollecTRI'],7)} {f(r['STRING'],7)} | "
          f"{f(r['RR_AUROC'],7)} {f(r['RP_AUROC'],7)} {f(r['RR_pearDE'],7)} {f(r['RP_pearDE'],7)} | "
          f"{f(r['orient_gap_RR_AUROC'],8)}{tag}")

os.makedirs(f"{_CMC_RESULTS}/resid_indep", exist_ok=True)
json.dump({"baseline": "sp_dec_gl1000", "rows": rows},
          open(f"{_CMC_RESULTS}/resid_indep/comparison.json", "w"), indent=2)
print(f"\nwrote {_CMC_RESULTS}/resid_indep/comparison.json")
print("A_rr/G_rr corr: 1=symmetric (unoriented); DID IT ORIENT => corr drops <<0.99, antisym rises.")
print("achTC/achHSIC/ach|r|: achieved residual DEPENDENCE (lower=more independent) vs baseline.")
print("revGapRR = mean_cohort(rr_true - rr_rev); ~0 => edge reversal inert (A_rr not directional).")
