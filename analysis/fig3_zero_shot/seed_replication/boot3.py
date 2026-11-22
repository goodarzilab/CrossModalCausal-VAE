#!/usr/bin/env python
"""Task 3 joint bootstrap (same procedure and RNG stream as analysis/fig3_zero_shot/stats/boot.py). usage: boot3.py TASK NREP"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from lib import TAGS, SETS
TASK, NREP = int(sys.argv[1]), int(sys.argv[2])
OD = f"{lib.OUTD}/boot"; os.makedirs(OD, exist_ok=True)
SEEDS = ["s42", "s1", "s2"]
def comps():
    L = []
    for s in SEEDS + ["mean"]:
        for b in ["corr", "totalvi_ko", "deepsem_ko"]: L.append((f"cmc_{s}", b))
        for a in ["Arr_shuffled", "Arr_zeroed"]: L.append((f"cmc_{s}", f"{a}_{s}"))
    return L
COMPS = comps()
rows = []; T0 = time.time()
for rep in range(NREP):
    b = TASK * NREP + rep
    rng = None if b == 0 else np.random.default_rng(100003 + b)
    C = lib.realise(rng)
    S = {"_b": b}; cache = {}
    for t in TAGS:
        ev = C[t].evaluable
        A = {nm: lib.rr_auroc_set(C[t], d[t], [g for g in ev if g in d[t][0]]) for nm, d in SETS.items()}
        for kind in ["cmc", "Arr_shuffled", "Arr_zeroed"]:
            pref = "cmc_" if kind == "cmc" else f"{kind}_"
            A[f"{pref}mean"] = {g: float(np.nanmean([A[f"{pref}{s}"].get(g, np.nan) for s in SEEDS])) for g in ev}
        units = sorted(ev)
        samp = rng.choice(len(units), len(units), replace=True) if rng is not None else np.arange(len(units))
        for nm, a in A.items():
            v = np.array([a.get(units[i], np.nan) for i in samp]); cache[(nm, t)] = v
            S[f"mean|{nm}|{t}"] = float(np.nanmean(v))
        S[f"n|{t}"] = len(units)
    for nm in set(k[0] for k in cache):
        v = np.concatenate([cache[(nm, t)] for t in TAGS]); S[f"mean|{nm}|overall"] = float(np.nanmean(v))
    for a, bb in COMPS:
        allv = []
        for t in TAGS:
            d = cache[(a, t)] - cache[(bb, t)]; allv.append(d); S[f"diff|{a}|{bb}|{t}"] = float(np.nanmean(d))
            if b == 0: S[f"winfrac|{a}|{bb}|{t}"] = float(np.mean(d[np.isfinite(d)] > 0))
        d = np.concatenate(allv); S[f"diff|{a}|{bb}|overall"] = float(np.nanmean(d))
        if b == 0: S[f"winfrac|{a}|{bb}|overall"] = float(np.mean(d[np.isfinite(d)] > 0)); S[f"n_pairs|{a}|{bb}|overall"] = int(np.isfinite(d).sum())
    rows.append(S); print(f"[{time.time()-T0:.0f}s] rep {b}", {t: S[f'n|{t}'] for t in TAGS}, flush=True)
json.dump(rows, open(f"{OD}/boot3_{TASK:03d}.json", "w"))
