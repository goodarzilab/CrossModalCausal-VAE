"""Collate the group_lasso cohort sweep: mean held-out RNA->RNA AUROC across the 3 CRISPR cohorts
for each training-cohort combination, to confirm the 2-cohort (Hao+Stephenson) choice under the
headline sparsity (sp_dec_gl1000). Mirrors tab:cohortsweep (which used feature_norm).
Writes results/combo_gl/cohort_sweep_gl.json and prints a ranked table."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, os, glob, numpy as np
R = f"{_CMC_ROOT}"
COH = ["frangieh", "iccite", "arce"]
rows = []
for cfg in sorted(glob.glob(f"{R}/configs/combo_gl/*_config.json")):
    name = os.path.basename(cfg).replace("_config.json", "")           # combo_hao-ste_nodag_gl
    short = name.replace("combo_", "").replace("_nodag_gl", "").replace("combat_only_nodag_gl", "combat")
    rd = f"{_CMC_RESULTS}/combo_gl/{name}"
    per = {}
    for c in COH:
        p = f"{rd}/full_perturb_eval_{c}.json"
        if os.path.exists(p):
            try: per[c] = float(json.load(open(p))["rr_auroc"])
            except Exception: pass
    k = short.count("-") + 1 if short != "combat" else 1
    vals = [per[c] for c in COH if c in per]
    rows.append({"combo": short, "k": k, "per_cohort": per,
                 "mean_rr_auroc": (round(float(np.mean(vals)), 4) if len(vals) == 3 else None),
                 "n_cohorts_done": len(vals)})

done = [r for r in rows if r["mean_rr_auroc"] is not None]
done.sort(key=lambda r: -r["mean_rr_auroc"])
print(f"{'combo':16s} k  {'Frangieh':>9s} {'Chen':>7s} {'Arce':>7s} {'mean':>7s}")
print("-" * 56)
for r in done:
    p = r["per_cohort"]
    star = "  <-- 2-cohort headline" if r["combo"] == "hao-ste" else ""
    print(f"{r['combo']:16s} {r['k']}  {p.get('frangieh',float('nan')):9.3f} {p.get('iccite',float('nan')):7.3f} "
          f"{p.get('arce',float('nan')):7.3f} {r['mean_rr_auroc']:7.3f}{star}")
missing = [r["combo"] for r in rows if r["mean_rr_auroc"] is None]
if missing: print(f"\nnot yet complete ({len(missing)}): {missing}")
else:
    best = done[0]["combo"]; hs = next((r for r in done if r["combo"] == "hao-ste"), None)
    print(f"\nBEST = {best} ({done[0]['mean_rr_auroc']}); hao-ste rank = "
          f"{[r['combo'] for r in done].index('hao-ste')+1}/{len(done)} ({hs['mean_rr_auroc'] if hs else 'NA'})")
json.dump({"rows": rows, "metric": "mean held-out RNA->RNA AUROC over 3 CRISPR cohorts",
           "sparsity": "group_lasso (sp_dec_gl1000)"},
          open(f"{_CMC_RESULTS}/combo_gl/cohort_sweep_gl.json", "w"), indent=2)
print("\nWROTE results/combo_gl/cohort_sweep_gl.json")
