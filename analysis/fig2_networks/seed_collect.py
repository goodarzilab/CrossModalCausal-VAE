"""Collect mean+-SD across seeds 42/1/2 of the top model (sp_dec_gl1000) for the headline claims:
recovery (coexpr/STRING/DoRothEA/CollecTRI), perturbation (RR/RP AUROC + pearDE), cognate fold.
Reads each seed's sweep_metrics.json + block_isolated_mut + G_rp. Writes results/sweep_sparse/seed_summary.json."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, os, numpy as np, pandas as pd
R = f"{_CMC_ROOT}"
SEEDS = {"42": "results/sweep_sparse/sp_dec_gl1000", "1": "results/sweep_sparse/sp_dec_gl1000_s1",
         "2": "results/sweep_sparse/sp_dec_gl1000_s2"}
COH = ["frangieh", "iccite", "arce"]

def cognate_fold(md):
    p = f"{R}/{md}/Transcription_RNA_to_Prot.csv"
    if not os.path.exists(p): return None
    Tr = pd.read_csv(p, index_col=0)
    def norm(s): return str(s).upper().replace("-", "").replace("_", "").replace(".", "")
    genes = [norm(g) for g in Tr.index]; prots = [norm(p_) for p_ in Tr.columns]; gup = {g: i for i, g in enumerate(genes)}
    V = Tr.values; cog = []
    for j, pr in enumerate(prots):
        for cand in [pr, pr.replace("CD", ""), pr.split("/")[0]]:
            if cand in gup: cog.append((gup[cand], j)); break
    cbp = {}; [cbp.setdefault(p_, set()).add(g) for g, p_ in cog]
    matched = np.abs(np.array([V[g, p_] for g, p_ in cog]))
    bg = np.abs(np.concatenate([np.delete(V[:, p_], list(cbp[p_])) for g, p_ in cog]))
    return float(np.median(matched) / (np.median(bg) + 1e-12))

rows = {}
for s, md in SEEDS.items():
    sm = f"{R}/{md}/sweep_metrics.json"
    if not os.path.exists(sm): rows[s] = None; continue
    d = json.load(open(sm)); pm = d["perturb_mean"]; rec = d["recovery"]
    rows[s] = {"recovery": rec, "rr_auroc": pm["rr_auroc"], "rp_auroc": pm["rp_auroc"],
               "rr_pearDE": pm["rr_pearDE"], "rp_pearDE": pm["rp_pearDE"], "cognate_fold": cognate_fold(md)}

present = {s: r for s, r in rows.items() if r}
print(f"seeds present: {list(present.keys())}")
def ms(key, sub=None):
    vals = [(present[s]["recovery"][sub] if sub else present[s][key]) for s in present if (present[s].get("recovery" if sub else key) is not None)]
    vals = [v for v in vals if v is not None]
    return (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4), len(vals))

summary = {"n_seeds": len(present), "seeds": list(present.keys())}
print(f"\n{'metric':16s} mean +- SD  (n seeds)")
for lab, key, sub in [("coexpr", None, "coexpr"), ("STRING", None, "STRING"), ("DoRothEA", None, "DoRothEA"),
                      ("CollecTRI", None, "CollecTRI"), ("RR-AUROC", "rr_auroc", None), ("RP-AUROC", "rp_auroc", None),
                      ("RR-pearDE", "rr_pearDE", None), ("RP-pearDE", "rp_pearDE", None), ("cognate_fold", "cognate_fold", None)]:
    m, sd, n = ms(key, sub); summary[lab] = {"mean": m, "sd": sd, "n": n}
    print(f"  {lab:14s} {m:.3f} +- {sd:.3f}  (n={n})")
json.dump(summary, open(f"{_CMC_RESULTS}/sweep_sparse/seed_summary.json", "w"), indent=2)
print("\nWROTE results/sweep_sparse/seed_summary.json")
