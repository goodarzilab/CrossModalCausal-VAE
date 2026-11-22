"""Aggregate task9 gene-block bootstrap + within-gene permutation.  usage: python -I task9_aggregate.py BOOTDIR PERMDIR OUT.json
Contrasts: X3 = AUROC averaged over the 3 seeds within each replicate (same gene draw); per-seed contrasts also reported.
CI: percentile of bootstrap differences (B replicates, b>=1). P_boot = 2*min(Pr(d<=0), Pr(d>=0)), floored at 1/(B+1).
P_perm = (1 + #{|d_perm| >= |d_obs|})/(B_perm + 1).  BH within panel (2a refs / 2b programs) and within contrast family."""
import sys, json, glob, numpy as np
BD, PD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q
BR = sorted([r for f in glob.glob(f"{BD}/boot_*.json") for r in json.load(open(f))], key=lambda r: r["b"])
PR = sorted([r for f in glob.glob(f"{PD}/perm_*.json") for r in json.load(open(f))], key=lambda r: r["b"]) if PD != "none" else []
assert BR[0]["b"] == 0
keys = list(BR[0]["node"]); maps = sorted({k.split("|")[0] for k in keys}); items = list(dict.fromkeys(k.split("|", 1)[1] for k in keys))
REFS = ["coexpr", "STRING", "DoRothEA", "CollecTRI"]; PROGS = [i for i in items if i not in REFS]
CMC = [m for m in maps if m.startswith("CMC_")]; RNA = [m for m in maps if m.startswith("RNA_")]
BL = ["corr", "partial_corr", "corr_within_ct", "partial_within_ct"]
def arr(var, m, it):
    return np.array([r[var].get(f"{m}|{it}", np.nan) for r in BR], float)
def grp(var, ms, it): return np.nanmean(np.stack([arr(var, m, it) for m in ms]), 0)
def getv(var, name, it):
    if name == "CMC3": return grp(var, CMC, it)
    if name == "RNA3": return grp(var, RNA, it)
    return arr(var, name, it)
NB = len(BR) - 1
def stat(d):
    x = d[1:]; x = x[np.isfinite(x)]
    if len(x) < 20: return dict(diff=float(d[0]))
    p = max(2 * min((x <= 0).mean(), (x >= 0).mean()), 1.0 / (len(x) + 1))
    return dict(diff=float(d[0]), ci95=[float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))], p_boot=float(p), B=int(len(x)))
PERM = {}
if PR:
    pk = list(PR[0]["d"])
    for k in pk: PERM[k] = np.array([r["d"][k] for r in PR], float)
def pperm(a, b, it):
    if not PERM: return None, None
    seeds = []
    if a in ("CMC3", "RNA3"):
        A = CMC if a == "CMC3" else RNA
        for m in A:
            bb = b if b in BL else (b if b not in ("CMC3", "RNA3") else ("CMC_" if b == "CMC3" else "RNA_") + m.split("_")[1])
            k = f"{m}|{bb}|{it}" if f"{m}|{bb}|{it}" in PERM else None
            if k is None:  # stored the other direction (CMC|RNA) -> negate
                k2 = f"{bb}|{m}|{it}"
                if k2 not in PERM: return None, None
                seeds.append(-PERM[k2])
            else: seeds.append(PERM[k])
        d = np.mean(seeds, 0)
    else:
        if f"{a}|{b}|{it}" in PERM: d = PERM[f"{a}|{b}|{it}"]
        elif f"{b}|{a}|{it}" in PERM: d = -PERM[f"{b}|{a}|{it}"]
        else: return None, None
    obs, x = d[0], d[1:]
    return float((1 + (np.abs(x) >= abs(obs) - 1e-12).sum()) / (len(x) + 1)), int(len(x))
O = dict(n_boot=NB, n_perm=max(0, len(PR) - 1), maps=maps, auroc={}, contrasts=[])
for m in maps + (["CMC3"] if CMC else []) + (["RNA3"] if RNA else []):
    for it in items:
        v = getv("node", m, it); x = v[1:]
        O["auroc"][f"{m}|{it}"] = dict(point=float(v[0]), ci95_node=[float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))])
FAMS = []
def add(fam, a, b, its):
    for it in its:
        e = dict(family=fam, a=a, b=b, item=it, panel="2a" if it in REFS else "2b")
        e.update(stat(getv("node", a, it) - getv("node", b, it)))
        if it in REFS:
            s = stat(getv("source", a, it) - getv("source", b, it)); e["source_block"] = {k: s.get(k) for k in ("ci95", "p_boot")}
        e["p_perm"], e["B_perm"] = pperm(a, b, it)
        O["contrasts"].append(e)
for b in BL:
    if CMC: add("CMC3_vs_baseline", "CMC3", b, items)
    for m in CMC: add("CMCseed_vs_baseline", m, b, items)
    if RNA: add("RNA3_vs_baseline", "RNA3", b, items)
if RNA and CMC:
    add("RNA3_vs_CMC3", "RNA3", "CMC3", items)
    for m in CMC:
        r = "RNA_" + m.split("_")[1]
        if r in RNA: add("RNAseed_vs_CMCseed", r, m, items)
# BH within (family, panel[, seed])
groups = {}
for i, e in enumerate(O["contrasts"]):
    seed = e["a"] if "seed" in e["family"] else ""
    groups.setdefault((e["family"], e["panel"], seed), []).append(i)
for g, idx in groups.items():
    for key in ("p_boot", "p_perm"):
        ps = [O["contrasts"][i].get(key) for i in idx]
        if all(p is not None for p in ps):
            for i, q in zip(idx, bh(ps)): O["contrasts"][i]["q_" + key.split("_")[1]] = float(q)
json.dump(O, open(OUT, "w"), indent=1)
f = lambda x: "NA" if x is None else f"{x:.3f}"
for e in O["contrasts"]:
    if e["family"] in ("CMC3_vs_baseline", "RNA3_vs_CMC3", "RNA3_vs_baseline"):
        print(e["family"], e["b"], e["item"], f"{e['diff']:+.4f}", [round(c, 4) for c in e.get("ci95", [])], "pB", e.get("p_boot"), "qB", f(e.get("q_boot")), "pP", e.get("p_perm"), "qP", f(e.get("q_perm")))
