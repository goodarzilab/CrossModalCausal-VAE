#!/usr/bin/env python
"""Task-3 stats from joint bootstrap (analyze.py conventions) + merge with point results -> ../results.json"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, f"{_CMC_ROOT}/analysis/fig3_zero_shot/stats")
from pblimma import bh
OD = f"{_CMC_RESULTS}/paper/analyses/review2/fig3"
TAGS = ["frangieh", "iccite", "arce"]; SC = TAGS + ["overall"]; SEEDS = ["s42", "s1", "s2"]
R = []
for f in sorted(glob.glob(f"{OD}/boot/boot3_*.json")): R += json.load(open(f))
REP0 = [r for r in R if r["_b"] == 0][0]; BOOT = [r for r in R if r["_b"] != 0]
PK = pd.read_csv(f"{OD}/per_ko_rr_auroc_3seeds.csv")
W = PK.pivot_table(index=["screen", "ko"], columns="set", values="rr_auroc")
for kind, pref in [("cmc", "cmc_"), ("Arr_shuffled", "Arr_shuffled_"), ("Arr_zeroed", "Arr_zeroed_")]:
    W[f"{pref}mean"] = W[[f"{pref}{s}" for s in SEEDS]].mean(1)
def bci(key):
    x = np.array([r[key] for r in BOOT if r.get(key) is not None and np.isfinite(r[key])])
    sh = REP0[key] - np.median(x); xc = x + sh
    return [float(np.percentile(xc, 2.5)), float(np.percentile(xc, 97.5))], xc, len(x)
def pboot(x): return float(max(2 * min((x <= 0).mean(), (x >= 0).mean()), 1.0 / (len(x) + 1)))
out = dict(n_boot=len(BOOT), means={}, contrasts={})
for nm in W.columns:
    for sc in SC:
        v = W[nm] if sc == "overall" else W.loc[sc][nm]; v = v.dropna()
        ci, _, _ = bci(f"mean|{nm}|{sc}")
        out["means"][f"{nm}|{sc}"] = dict(mean=float(v.mean()), sem=float(v.std(ddof=1) / np.sqrt(len(v))), n=len(v), ci95_boot=ci)
# seed summary
seedsum = {}
for kind in ["cmc", "Arr_shuffled", "Arr_zeroed"]:
    for sc in SC:
        vals = [out["means"][f"{kind}_{s}|{sc}"]["mean"] for s in SEEDS]
        seedsum[f"{kind}|{sc}"] = dict(per_seed=dict(zip(SEEDS, vals)), mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)))
out["seed_summary"] = seedsum
fam = {}
for s in SEEDS + ["mean"]:
    a = f"cmc_{s}"
    for b in ["corr", "totalvi_ko", "deepsem_ko", f"Arr_shuffled_{s}", f"Arr_zeroed_{s}"]:
        panel = "3d" if b.startswith("Arr") else "3b"
        for sc in SC:
            x = W[a] if sc == "overall" else W.loc[sc][a]; y = W[b] if sc == "overall" else W.loc[sc][b]
            f = x.notna() & y.notna(); d = (x - y)[f]
            ci, xb, nb = bci(f"diff|{a}|{b}|{sc}")
            r = dict(seed=s, panel=panel, a=a, b=b, scope=sc, n=int(f.sum()), mean_a=float(x[f].mean()), mean_b=float(y[f].mean()),
                     diff=float(d.mean()), ci95_boot=ci, p_boot=pboot(xb), p_t=float(stats.ttest_rel(x[f], y[f]).pvalue),
                     p_t_unpaired=float(stats.ttest_ind(x[f], y[f]).pvalue), win_frac=float((d > 0).mean()))
            out["contrasts"][f"{a}|{b}|{sc}"] = r
            if sc != "overall": fam.setdefault((s, panel), []).append(f"{a}|{b}|{sc}")
for (s, panel), keys in fam.items():
    if panel == "3b":
        q = bh([out["contrasts"][k]["p_t"] for k in keys])
        for k, qq in zip(keys, q):
            c = out["contrasts"][k]; c["p_t_bh"] = float(qq); c["p_reported"] = float(max(qq, c["p_boot"]))
    else:
        q = bh([out["contrasts"][k]["p_boot"] for k in keys])
        for k, qq in zip(keys, q):
            c = out["contrasts"][k]; c["p_boot_bh"] = float(qq); c["p_reported"] = float(qq)
# consistency vs seed 42
cons = []
for k, c in out["contrasts"].items():
    if c["seed"] != "s42" or c["scope"] == "overall": continue
    rowsig = c["p_reported"] < 0.05; entry = dict(contrast=f"{c['b']}|{c['scope']}", panel=c["panel"], s42_diff=c["diff"], s42_p=c["p_reported"], s42_sig=rowsig)
    ok = True
    for s in ["s1", "s2", "mean"]:
        bb = c["b"].replace("_s42", f"_{s}"); cs = out["contrasts"][f"cmc_{s}|{bb}|{c['scope']}"]
        entry[f"{s}_diff"] = cs["diff"]; entry[f"{s}_p"] = cs["p_reported"]
        if rowsig and not (np.sign(cs["diff"]) == np.sign(c["diff"]) and cs["p_reported"] < 0.05): ok = False
    entry["consistent"] = ok if rowsig else None
    cons.append(entry)
out["consistency"] = cons
out["fig3e_winfrac_vs_corr_319"] = {s: out["contrasts"][f"cmc_{s}|corr|overall"]["win_frac"] for s in SEEDS + ["mean"]}
P = json.load(open(f"{OD}/point_results.json"))
res = dict(plan_sha256=open(f"{OD}/PLAN.sha256").read().split()[0],
           eps_check={t: json.load(open(f"{OD}/eps_check_{t}.json")) for t in ["frangieh", "iccite", "arce", "papalexi"] if os.path.exists(f"{OD}/eps_check_{t}.json")},
           reproduction_gate="seed-42 per-KO rr_auroc for cmc_s42/corr/totalvi_ko/deepsem_ko/Arr_shuffled/Arr_zeroed matches source_data_fig3_4_5_pb.csv, max|diff| 1.1e-16, n=174/122/23",
           task1_cross_ko=P["cross_ko"], task2_residualisation=P["residualisation"], task3_three_seed=out)
json.dump(res, open(f"{OD}/results.json", "w"), indent=1, default=float)
print(json.dumps(seedsum, indent=0)); print(pd.DataFrame(cons).to_string())
print(out["fig3e_winfrac_vs_corr_319"])
