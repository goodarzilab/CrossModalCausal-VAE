"""Collect the sparsity sweep into a Pareto table vs the baseline. Reads every
results/sweep_sparse/*/sweep_metrics.json + the baseline results/linfb_2c_nodag_v4/sweep_metrics.json.
Flags models that MATCH/BEAT baseline perturbation while being SPARSER. Writes results/sweep_sparse/pareto.json."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, glob
R = f"{_CMC_ROOT}"
def J(p): return json.load(open(p)) if os.path.exists(p) else None
base = J(f"{_CMC_RESULTS}/linfb_2c_nodag_v4/sweep_metrics.json")
rows = [base] + sorted([J(p) for p in glob.glob(f"{_CMC_RESULTS}/sweep_sparse/*/sweep_metrics.json")],
                       key=lambda d: d["name"] if d else "")
rows = [r for r in rows if r]
b = base["perturb_mean"]
print(f"{'model':16s} {'projGini':>8s} {'projSpar':>8s} {'latGini':>7s} {'gate':>5s} | "
      f"{'RR-AU':>6s} {'RP-AU':>6s} {'RRpear':>7s} {'RPpear':>7s} | {'STRING':>6s} {'DoRo':>5s} {'flag':>16s}")
out = []
for r in rows:
    pm = r["perturb_mean"]; rec = r["recovery"]
    sparser = (r["proj_gini"] > base["proj_gini"] + 0.01) or (r["proj_sparsity"] > base["proj_sparsity"] + 0.02) \
              or (r.get("latent_mean_gate", 1) < base.get("latent_mean_gate", 1) - 0.02)
    # perturbation match: within noise on RR-AUROC AND RP-pearDE not worse by >0.03
    matches = (pm.get("rr_auroc") or 0) >= b["rr_auroc"] - 0.02 and (pm.get("rp_pearDE") or -9) >= b["rp_pearDE"] - 0.03
    beats = (pm.get("rr_auroc") or 0) > b["rr_auroc"] + 0.01 or (pm.get("rp_pearDE") or -9) > b["rp_pearDE"] + 0.02
    flag = ""
    if r["name"] != base["name"]:
        if sparser and beats: flag = "SPARSER+BEATS**"
        elif sparser and matches: flag = "SPARSER+MATCHES*"
        elif matches: flag = "matches"
        elif sparser: flag = "sparser(worse pert)"
        else: flag = "-"
    print(f"{r['name'][:16]:16s} {r['proj_gini']:8.3f} {r['proj_sparsity']:8.3f} {r['latent_gini']:7.3f} "
          f"{r.get('latent_mean_gate',float('nan')):5.2f} | {pm.get('rr_auroc',float('nan')) or float('nan'):6.3f} "
          f"{pm.get('rp_auroc',float('nan')) or float('nan'):6.3f} {pm.get('rr_pearDE',float('nan')) or float('nan'):7.3f} "
          f"{pm.get('rp_pearDE',float('nan')) or float('nan'):7.3f} | {rec['STRING']:6.3f} {rec['DoRothEA']:5.3f} {flag:>16s}")
    out.append({**{k: r[k] for k in ['name','proj_gini','proj_sparsity','latent_gini','latent_mean_gate','recovery']},
                'perturb_mean': pm, 'flag': flag})
json.dump({"baseline": base["name"], "rows": out}, open(f"{_CMC_RESULTS}/sweep_sparse/pareto.json", "w"), indent=2)
print(f"\nbaseline: RR-AU {b['rr_auroc']:.3f} RP-AU {b['rp_auroc']:.3f} RR-pear {b['rr_pearDE']:.3f} RP-pear {b['rp_pearDE']:.3f} "
      f"| projGini {base['proj_gini']:.3f} latGini {base['latent_gini']:.3f}")
print(f"models collected: {len(rows)-1}/12 sweep + baseline")
