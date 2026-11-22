"""Collate the reversal-discrepancy-loss trade-off vs the sp_dec_gl1000 headline baseline.
For each model (baseline + results/rev_loss/*), reads:
  - config.json         -> lambda_reversal
  - sweep_metrics.json  -> recovery{coexpr,DoRothEA,CollecTRI,STRING}, perturb_mean.rr_auroc (RR-AUROC), rp_auroc
  - edge_reversal_<ds>.json (frangieh,iccite,arce) -> orientation-asymmetry:
        RR AUROC asym  = mean(mean_rr_true - mean_rr_rev)   over cohorts  (>0 = directional)
        RR pearDE asym = mean(mean_rr_pearDE_true - mean_rr_pearDE_rev)   over cohorts
Prints a table and writes results/rev_loss/tradeoff.json.
Usage: python analysis/supplementary/objectives/collect_rev_loss.py
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
    """mean over cohorts of (true - reversed) for RR-AUROC and RR signed-pearDE."""
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
    rec = sm.get("recovery", {}); pm = sm.get("perturb_mean", {})
    au_asym, pe_asym = orient_asym(md)
    return {
        "name": os.path.basename(md.rstrip("/")),
        "lambda_reversal": cfg.get("lambda_reversal", 0.0),
        "coexpr": rec.get("coexpr"), "DoRothEA": rec.get("DoRothEA"), "CollecTRI": rec.get("CollecTRI"),
        "STRING": rec.get("STRING"),
        "RR_AUROC": pm.get("rr_auroc"), "RP_AUROC": pm.get("rp_auroc"),
        "orient_asym_RR_AUROC": au_asym, "orient_asym_RR_pearDE": pe_asym,
    }

base = f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000"
rows = [row(base)] + [row(os.path.dirname(p)) for p in sorted(glob.glob(f"{_CMC_RESULTS}/rev_loss/*/sweep_metrics.json"))]
# include rev_loss dirs that have a config but no sweep_metrics yet (so gaps are visible)
seen = {r["name"] for r in rows}
for d in sorted(glob.glob(f"{_CMC_RESULTS}/rev_loss/*/")):
    if os.path.basename(d.rstrip("/")) not in seen and os.path.exists(f"{d}/config.json"):
        rows.append(row(d))

def f(x, w=8, p=4):
    return (f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'NA':>{w}}")
hdr = f"{'model':16s} {'lam_rev':>7s} | {'coexpr':>7s} {'DoRoth':>7s} {'CollTRI':>7s} | {'RR-AU':>7s} {'RP-AU':>7s} | {'RRasymAU':>8s} {'RRasymPe':>8s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    lam = r["lambda_reversal"]
    tag = "  <== BASELINE" if r["name"] == "sp_dec_gl1000" else ""
    print(f"{r['name'][:16]:16s} {lam:7g} | {f(r['coexpr'],7)} {f(r['DoRothEA'],7)} {f(r['CollecTRI'],7)} | "
          f"{f(r['RR_AUROC'],7)} {f(r['RP_AUROC'],7)} | {f(r['orient_asym_RR_AUROC'],8)} {f(r['orient_asym_RR_pearDE'],8)}{tag}")
os.makedirs(f"{_CMC_RESULTS}/rev_loss", exist_ok=True)
json.dump({"baseline": "sp_dec_gl1000", "rows": rows}, open(f"{_CMC_RESULTS}/rev_loss/tradeoff.json", "w"), indent=2)
print(f"\nwrote {_CMC_RESULTS}/rev_loss/tradeoff.json")
print("orient_asym_RR_* = mean over cohorts of (true - reversed); >0 => A_rr is directional (reversal hurts).")
