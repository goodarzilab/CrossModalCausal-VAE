#!/usr/bin/env python
"""Aggregate r2_boot.py: percentile CIs median-centred on the point estimate (as code/reviewfix/rf_analyze_boot.py),
P_boot = two-sided sign proportion with floor 1/(B+1), BH within the stated families. Hierarchical (seed-resampled)
versions are used for every contrast that involves a 3-seed family.  usage: python -I r2_analyze.py BOOTDIR OUT_JSON"""
import sys, glob, json, numpy as np
from statsmodels.stats.multitest import multipletests
D, OUT = sys.argv[1], sys.argv[2]
keys = None; P0 = None; rows = []
for f in sorted(glob.glob(f"{D}/boot_*.npz")):
    z = np.load(f, allow_pickle=True); k = list(z["keys"]); b = z["b"]; V = z["V"]
    if keys is None: keys = k; KI = {x: i for i, x in enumerate(keys)}
    idx = np.array([k.index(x) if x in k else -1 for x in keys]) if k != keys else np.arange(len(keys))
    for j, bb in enumerate(b):
        row = np.where(idx >= 0, V[j][np.clip(idx, 0, None)], np.nan)
        if bb == 0: P0 = row
        else: rows.append((bb, row))
seen = {}; [seen.setdefault(bb, r) for bb, r in rows]
BT = np.array([seen[bb] for bb in sorted(seen)]); NB = len(BT)
print("bootstrap replicates:", NB)
def get(key):
    i = KI.get(key)
    if i is None or not np.isfinite(P0[i]): return None
    x = BT[:, i]; x = x[np.isfinite(x)]
    if len(x) < 20: return dict(point=float(P0[i]), ci=None, p=None, nb=len(x))
    xc = x + (P0[i] - np.median(x))
    p = float(max(2 * min((xc <= 0).mean(), (xc >= 0).mean()), 1.0 / (len(xc) + 1)))
    return dict(point=float(P0[i]), ci=[float(np.percentile(xc, 2.5)), float(np.percentile(xc, 97.5))], p=p, nb=len(x))
def mean(prefix, st, m, t): return get(f"mean|{prefix}|{st}|{m}|{t}")
def diff(prefix, a, b, m, t):
    h = a.endswith("3"); a2 = a + "h" if h else a; b2 = b + "h" if b.endswith("3") else b
    e = get(f"diff|{prefix}|{a2}|{b2}|{m}|{t}") if h else get(f"diff|{prefix}|{a}|{b}|{m}|{t}")
    if e is None: return None
    pt = get(f"diff|{prefix}|{a}|{b}|{m}|{t}")       # point estimate always from the plain 3-seed mean
    if pt: e["point"] = pt["point"]
    e["n"] = int(P0[KI[f"n|{prefix}|{a}|{b}|{m}|{t}"]]) if f"n|{prefix}|{a}|{b}|{m}|{t}" in KI else None
    e.update(a=a, b=b, metric=m, screen=t, prefix=prefix); return e
def family(name, items):
    items = [e for e in items if e and e.get("p") is not None]
    if items:
        q = multipletests([e["p"] for e in items], method="fdr_bh")[1]
        for e, qq in zip(items, q): e["q"] = float(qq); e["family"] = name
    return items
TAGS = ["frangieh", "iccite", "arce"]; SC = TAGS + ["all"]
RNA4 = ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]
O = dict(n_boot=NB, means={}, families={})
for k in keys:
    if k.startswith("mean|") and not k.split("|")[2].endswith("3h"):
        e = get(k)
        if e: O["means"][k[5:]] = e
F = O["families"]
F["T1_fig3_paired_vs_rnaonly"] = family("T1_fig3", [diff("F3", "clamp3", "rnaonly3", m, t) for m in RNA4 + ["rr_au_strat_ms", "rr_au_resid"] for t in TAGS])
F["T1_fig3_rnaonly_vs_baselines"] = family("T1_fig3b", [diff("F3", "rnaonly3", b, m, t) for b in ["corr", "totalvi_ko", "deepsem_ko"] for m in RNA4 for t in TAGS])
F["T1_scvi_vs_totalvi"] = family("T1_scvi", [diff("F3", a, "scvi_ko", m, t) for a in ["totalvi_ko", "totalvi_fresh"] for m in RNA4 for t in TAGS])
for fa, fb in [("paired", "rnaonly"), ("adapt", "unadpt"), ("adapt", "paired")]:
    F[f"X5_{fa}_vs_{fb}"] = family(f"X5_{fa}_{fb}", [diff(f"X5_{fa}_{fb}", f"{fa}:{a}3", f"{fb}:{a}3", m, t)
                                                     for a in ["baseline", "real", "shuf", "rs", "rb"] for m in ["rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc"] for t in SC])
F["T3_ko_invariant"] = family("T3_koinv", [diff("F3", f + "3" if f in ("clamp", "rnaonly", "adapt") else f, b, "rr_auroc", t)
                                           for f in ["clamp", "corr", "totalvi_ko", "deepsem_ko", "rnaonly", "adapt"] for b in ["ctrl_mean", "ctrl_sd"] for t in TAGS])
F["T3_stratified"] = family("T3_strat", [diff("F3", "clamp3", b, m, t) for b in ["corr", "totalvi_ko", "deepsem_ko", "edgecut"]
                                         for m in ["rr_au_strat_ms", "rr_au_strat_ms10", "rr_au_strat_sd10", "rr_au_strat_m10", "rr_au_resid"] for t in TAGS])
F["T5_fig3_adapt"] = family("T5_fig3", [diff("F3", "adapt3", b, m, t) for b in ["clamp3", "corr", "totalvi_ko", "deepsem_ko"] for m in RNA4 for t in TAGS])
LG = []
for fr, fp in [(0.05, 0.05), (0.10, 0.05), (0.15, 0.05), (0.25, 0.05), (0.10, 0.025), (0.10, 0.10)]:
    pre = f"LG_r{fr}_p{fp}"
    LG += [diff(pre, "clamp3", "corr", m, t) for m in ["rr_auroc", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe", "rp_pe_cent"] for t in TAGS]
F["T6_lfc_grid"] = family("T6", LG)
for fam in ["paired", "rnaonly", "adapt", "unadpt"]:
    F[f"T8_fig5_{fam}"] = family(f"T8_{fam}", [diff(f"F5_{fam}", f"{fam}:{a}3", f"{fam}:{b}3", m, t)
        for a, b in [("real", "baseline"), ("real", "shuf"), ("real", "mean"), ("shuf", "baseline"), ("mean", "baseline")]
        for m in ["rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval"] for t in SC])
st = [diff("F5_st4", f"st4:{a}3", f"st4:{b}3", m, "all") for a, b in [("real", "baseline"), ("real", "shuf")] for m in ["rr_pe_cent", "rr_retrieval", "rr_pe"]]
F["T8_st4_primary"] = family("T8_st4", st)
F["T8_st4_secondary"] = family("T8_st4b", [diff("F5_st4", f"st4:{a}3", f"st4:{b}3", m, t) for a, b in [("real", "mean"), ("shuf", "baseline")] for m in ["rr_pe_cent", "rr_retrieval", "rr_pe"] for t in SC] +
                                [diff("F5_st4", f"st4:{a}3", f"st4:{b}3", m, t) for a, b in [("real", "baseline"), ("real", "shuf")] for m in ["rr_pe_cent", "rr_retrieval", "rr_pe"] for t in TAGS])
json.dump(O, open(OUT, "w"), indent=1)
f3 = lambda x: "NA" if x is None else f"{x:+.3f}"
for nm, L in F.items():
    print(f"\n## {nm} ({len(L)} tests)")
    for e in L: print(f"{e['a']} - {e['b']} | {e['metric']} | {e['screen']} | n={e['n']} | {f3(e['point'])} [{f3(e['ci'][0])}, {f3(e['ci'][1])}] p={e['p']:.2g} q={e['q']:.2g}")
