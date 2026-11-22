# [xmodal] COPY of code/multimodal/aggregate.py -- PATH-ONLY substitution (O from env XM_EVAL); logic unchanged.
"""Aggregate multimodal runs -> analyses/multimodal/summary.json + printable tables.
Per-KO metrics are averaged over the 3 seeds (sp_dec_gl1000 s42/s1/s2) before testing; two-sided Wilcoxon signed-rank
and paired t; 95% bootstrap CI over knockouts (2000 reps) of the mean paired difference; BH within panel
(panel = operator x direction x scope); per-seed mean differences reported to show seed consistency."""
import os, json, sys
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
O = os.environ["XM_EVAL"]
SEEDS = ["_s42", "_s1", "_s2"]; OPS = ["clamp_zero", "cutclamp_zero"]; SCOPES = ["overall", "iccite", "arce", "frangieh"]

def bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0: return p
    o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1); r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q
def boot(d, B=2000, seed=0):
    r = np.random.RandomState(seed); n = len(d); return [float(np.percentile([d[r.randint(0, n, n)].mean() for _ in range(B)], q)) for q in (2.5, 97.5)]
def kmap(rows, m): return {(r["cohort"], r["gene"]): r.get(m) for r in rows if r.get(m) is not None and np.isfinite(r.get(m))}
def seedmean(list_of_rows, m):
    maps = [kmap(rows, m) for rows in list_of_rows]; keys = set.intersection(*[set(x) for x in maps]) if maps else set()
    return {k: float(np.mean([x[k] for x in maps])) for k in keys}, maps
def compare(A, B, scope, Aseeds=None, Bseeds=None):
    """A, B: {key: value} seed-mean; returns stats of A - B."""
    ks = sorted(k for k in set(A) & set(B) if scope == "overall" or k[0] == scope)
    if len(ks) < 3: return dict(n=len(ks))
    a = np.array([A[k] for k in ks]); b = np.array([B[k] for k in ks]); d = a - b
    out = dict(n=len(ks), mean_a=float(a.mean()), mean_b=float(b.mean()), diff=float(d.mean()), ci95=boot(d),
               p_wilcoxon=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, p_t=float(ttest_rel(a, b).pvalue) if np.any(d != 0) else 1.0,
               frac_a_gt_b=float(np.mean(d > 0)))
    if Aseeds and Bseeds:
        out["per_seed_diff"] = [float(np.mean([x[k] - y[k] for k in ks if k in x and k in y])) for x, y in zip(Aseeds, Bseeds)]
    return out
def mean_ci(A, scope):
    v = np.array([x for k, x in A.items() if scope == "overall" or k[0] == scope])
    return dict(n=len(v), mean=float(v.mean()) if len(v) else None, ci95=boot(v) if len(v) >= 3 else None)

S = {}
# ============================================================== experiment 1 (main)
PM = ["rp_auroc", "rp_pe", "rp_pe_cent", "rp_retrieval"]; RM = ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]
FWD = [("ftR", "obs"), ("ftR", "ftR_shuf"), ("ftR", "ftR+corr"), ("ftR", "ftR+cridge"), ("ftR", "ftR+tv"), ("ftR", "ftR+cog"),
       ("ftR", "ridgeR+corr"), ("ftR", "ridgeR+cridge"), ("ftR", "ridgeR+cog"), ("ftR", "measRNA+corr"), ("ftR", "measRNA+cridge"), ("ftR", "measRNA+tv"), ("ftR", "measRNA+cog"),
       ("ftP", "ftR"), ("ridgeP_cx", "ftR"), ("ridgeP_fp", "ftR"), ("ftP", "ftP_shuf"), ("ftRP", "ftR"), ("ftR_shuf", "obs"), ("obs", "ftR+corr")]
REV = [("ftP", "obs"), ("ftP", "ftP_shuf"), ("ftR", "ftP"), ("ridgeP_cx", "ftP"), ("ftP_shuf", "obs")]   # ridgeP_cx rr = ridge on RNA labels (cx)
S["exp1"] = {}
for op in OPS:
    R = [json.load(open(f"{O}/runs/main_{op}{s}.json")) for s in SEEDS if os.path.exists(f"{O}/runs/main_{op}{s}.json")]
    R = [r for r in R if "main" in r]
    if not R: continue
    arms = list(R[0]["main"]); E = dict(n_seeds=len(R), arms={}, fwd={}, rev={})
    SM = {}
    for a in arms:
        SM[a] = {}
        for m in PM + RM + ["rp_pe_meanresp", "rp_auroc_meanresp", "rr_pe_meanresp", "rr_auroc_meanresp"]:
            SM[a][m] = seedmean([r["main"][a]["per_ko"] for r in R], m)
        E["arms"][a] = {sc: {m: mean_ci(SM[a][m][0], sc) for m in PM + RM + ["rp_pe_meanresp", "rp_auroc_meanresp", "rr_pe_meanresp"]} for sc in SCOPES}
    for dname, pairs, mets in [("fwd", FWD, PM), ("rev", REV, RM)]:
        for sc in SCOPES:
            tab = []
            for a, b in pairs:
                if a not in SM or b not in SM: continue
                for m in mets:
                    c = compare(SM[a][m][0], SM[b][m][0], sc, SM[a][m][1], SM[b][m][1]); c.update(a=a, b=b, metric=m); tab.append(c)
            ps = [c.get("p_wilcoxon", np.nan) for c in tab]; ok = [i for i, p in enumerate(ps) if np.isfinite(p)]
            q = bh([ps[i] for i in ok])
            for j, i in enumerate(ok): tab[i]["q_bh"] = float(q[j])
            E[dname][sc] = tab
    # cognate subsets
    cs = {}
    for name in R[0].get("cog_subsets", {}):
        for lab in ["cog", "nocog"]:
            vals = [{(x["cohort"], x["gene"]): x["pe"] for x in r["cog_subsets"][name][lab] if x["pe"] is not None and np.isfinite(x["pe"])} for r in R]
            ks = set.intersection(*[set(v) for v in vals]); cs.setdefault(name, {})[lab] = {k: float(np.mean([v[k] for v in vals])) for k in ks}
    E["cog_subsets"] = {name: {lab: {sc: mean_ci(v, sc) for sc in SCOPES} for lab, v in d.items()} for name, d in cs.items()}
    E["cog_subsets_tests"] = {f"ftR-vs-{b}|{lab}": {sc: compare(cs["ftR"][lab], cs[b][lab], sc) for sc in SCOPES}
                              for b in ["obs", "ftR_shuf", "ftR+cog", "measRNA+cog", "ftR+corr"] for lab in ["cog", "nocog"] if b in cs}
    E["ft_steps"] = {r["run"]: r.get("ft_info") for r in R}
    S["exp1"][op] = E

# ============================================================== few-shot
S["fewshot"] = {}
for op in OPS:
    R = [json.load(open(f"{O}/runs/fs_{op}{s}.json")) for s in SEEDS if os.path.exists(f"{O}/runs/fs_{op}{s}.json")]
    R = [r for r in R if "fewshot" in r]
    M = [json.load(open(f"{O}/runs/main_{op}{s}.json")) for s in SEEDS if os.path.exists(f"{O}/runs/main_{op}{s}.json")]
    M = [m for m in M if "main" in m]
    if not R: continue
    names = [k for k in R[0]["fewshot"] if k != "nlab"]
    ns = sorted(set(int(k.split("|")[1]) for k in names)); arms = sorted(set(k.split("|")[0] for k in names))
    F = dict(n_grid=ns, nlab=R[0]["fewshot"]["nlab"], curves={}, tests={})
    def arm_n(a, n, m):
        ds = sorted(set(k.split("|")[2] for k in names if k.startswith(f"{a}|{n}|")))
        per_seed = []
        for r in R:
            maps = [kmap(r["fewshot"][f"{a}|{n}|{d}"]["per_ko"], m) for d in ds]; ks = set.intersection(*[set(x) for x in maps])
            per_seed.append({k: float(np.mean([x[k] for x in maps])) for k in ks})
        ks = set.intersection(*[set(x) for x in per_seed]); return {k: float(np.mean([x[k] for x in per_seed])) for k in ks}
    for m in ["rp_pe_cent", "rp_retrieval", "rp_auroc", "rp_pe", "rr_pe_cent", "rr_retrieval", "rr_pe"]:
        ref = {"ftR": seedmean([x["main"]["ftR"]["per_ko"] for x in M], m)[0], "obs": seedmean([x["main"]["obs"]["per_ko"] for x in M], m)[0]} if M else {}
        for sc in SCOPES:
            cur = {a: {n: mean_ci(arm_n(a, n, m), sc) for n in ns} for a in arms}
            cur["n0_ftR"] = mean_ci(ref["ftR"], sc) if ref else None; cur["n0_obs"] = mean_ci(ref["obs"], sc) if ref else None
            F["curves"].setdefault(m, {})[sc] = cur
            tab = []
            for n in ns:
                for a, b in [("ftRP", "ridgeP_cx"), ("ftRP", "ridgeP_fp"), ("ftP", "ridgeP_cx"), ("ftRP", "ftP")]:
                    c = compare(arm_n(a, n, m), arm_n(b, n, m), sc); c.update(a=a, b=b, n=n); tab.append(c)
                if ref:
                    c = compare(arm_n("ftRP", n, m), ref["ftR"], sc); c.update(a="ftRP", b="ftR(n=0)", n=n); tab.append(c)
            ps = [c.get("p_wilcoxon", np.nan) for c in tab]; ok = [i for i, p in enumerate(ps) if np.isfinite(p)]; q = bh([ps[i] for i in ok])
            for j, i in enumerate(ok): tab[i]["q_bh"] = float(q[j])
            F["tests"].setdefault(m, {})[sc] = tab
    S["fewshot"][op] = F

# ============================================================== experiment 2/3 (translation from measured RNA)
f = f"{O}/runs/translate.json"
if os.path.exists(f):
    T = json.load(open(f)); X = dict(summary=T["summary"], summary_perprot=T["summary_perprot"], check=T["check"], ctrl_fit=T["ctrl_fit"],
                                      n_units_p=T["n_units_p"], discordance={m: T["discordance"][m]["summary"] for m in T["discordance"]})
    km = lambda m, met: {(r["cohort"], r["gene"]): r[met] for r in T["per_ko"][m] if r[met] is not None and np.isfinite(r[met])}
    kp = lambda m, met: {(r["cohort"], r["protein"]): r[met] for r in T["per_protein"][m] if r[met] is not None and np.isfinite(r[met])}
    BASE = ["corr", "cridge", "totalVI", "cognate", "ctrlridge", "koridge", "CMC-Wrp", "CMC-Wrp-shuffled", "CMC-Wrp<-WprT"]
    X["tests"] = {}
    for sc in SCOPES:
        tab = []
        for b in BASE:
            for met in ["pe", "auroc", "pe_cent", "retrieval", "sign_acc"]:
                c = compare(km("CMC-enc", met), km(b, met), sc); c.update(a="CMC-enc", b=b, metric=met, unit="KO"); tab.append(c)
            c = compare(kp("CMC-enc", "r"), kp(b, "r"), sc); c.update(a="CMC-enc", b=b, metric="perprot_r", unit="protein"); tab.append(c)
        for b in ["CMC-Wrp-shuffled", "CMC-Wrp<-WprT", "corr", "cognate", "ctrlridge"]:
            for met in ["pe", "pe_cent", "retrieval"]:
                c = compare(km("CMC-Wrp", met), km(b, met), sc); c.update(a="CMC-Wrp", b=b, metric=met, unit="KO"); tab.append(c)
            c = compare(kp("CMC-Wrp", "r"), kp(b, "r"), sc); c.update(a="CMC-Wrp", b=b, metric="perprot_r", unit="protein"); tab.append(c)
        ps = [c.get("p_wilcoxon", np.nan) for c in tab]; ok = [i for i, p in enumerate(ps) if np.isfinite(p)]; q = bh([ps[i] for i in ok])
        for j, i in enumerate(ok): tab[i]["q_bh"] = float(q[j])
        X["tests"][sc] = tab
    # per protein, stratified by cognate availability
    X["perprot_by_cog"] = {}
    for m in T["per_protein"]:
        for hc in [True, False]:
            v = [r["r"] for r in T["per_protein"][m] if r["has_cog"] == hc and r["r"] is not None and np.isfinite(r["r"])]
            X["perprot_by_cog"].setdefault(m, {})["cog" if hc else "nocog"] = dict(n=len(v), mean=float(np.mean(v)) if v else None)
    X["perprot_by_cog_tests"] = {}
    for b in ["corr", "cridge", "totalVI", "ctrlridge", "koridge", "cognate"]:
        for hc in [True, False]:
            A = {(r["cohort"], r["protein"]): r["r"] for r in T["per_protein"]["CMC-enc"] if r["has_cog"] == hc and r["r"] is not None and np.isfinite(r["r"])}
            B = {(r["cohort"], r["protein"]): r["r"] for r in T["per_protein"][b] if r["has_cog"] == hc and r["r"] is not None and np.isfinite(r["r"])}
            X["perprot_by_cog_tests"][f"CMC-enc-vs-{b}|{'cog' if hc else 'nocog'}"] = compare(A, B, "overall")
    # discordance: sign-accuracy on 'opposite' / 'transcript_unchanged' pairs, McNemar-style paired over (KO, protein) pairs
    from scipy.stats import binomtest
    D = {}
    rec = {m: T["discordance"][m]["records"] for m in T["discordance"] if T["discordance"][m]["records"]}
    for b in ["cognate", "ctrlridge", "corr", "koridge"]:
        for cat in ["opposite", "transcript_unchanged", "concordant"]:
            A = [r for r in rec["CMC-enc"] if r["cat"] == cat]; Bm = {(r["cohort"], r["gene"], r["protein"]): r for r in rec[b] if r["cat"] == cat}
            a_ok = np.array([np.sign(r["pred"]) == np.sign(r["obs"]) for r in A]); b_ok = np.array([np.sign(Bm[(r["cohort"], r["gene"], r["protein"])]["pred"]) == np.sign(r["obs"]) for r in A])
            n10 = int((a_ok & ~b_ok).sum()); n01 = int((~a_ok & b_ok).sum())
            D[f"CMC-enc-vs-{b}|{cat}"] = dict(n=len(A), acc_a=float(a_ok.mean()) if len(A) else None, acc_b=float(b_ok.mean()) if len(A) else None,
                                              n_a_only=n10, n_b_only=n01, p_mcnemar=float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0,
                                              n_ko=len(set((r["cohort"], r["gene"]) for r in A)),
                                              p_vs_chance_a=float(binomtest(int(a_ok.sum()), len(A), 0.5).pvalue) if len(A) else None)
    X["discordance_tests"] = D
    S["exp2"] = X

json.dump(S, open(f"{O}/summary.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and o != o) else str(o))
print("WROTE", f"{O}/summary.json")

# ------------------------------------------------------------- printable digest
def fmt(c, keys=("n", "mean_a", "mean_b", "diff")):
    if "diff" not in c: return f"n={c.get('n')}"
    return f"n={c['n']} {c['mean_a']:+.3f} vs {c['mean_b']:+.3f} d={c['diff']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] p={c['p_wilcoxon']:.2g} q={c.get('q_bh', float('nan')):.2g}" + (f" seeds {['%+.3f' % x for x in c['per_seed_diff']]}" if "per_seed_diff" in c else "")
if "--print" in sys.argv:
    for op, E in S.get("exp1", {}).items():
        print(f"\n===== EXP1 {op} (seeds={E['n_seeds']})")
        for sc in SCOPES:
            print(f"-- {sc} arms:");
            for a, v in E["arms"].items():
                print(f"   {a:16s} " + " ".join(f"{m}={v[sc][m]['mean']:+.3f}(n{v[sc][m]['n']})" for m in PM + RM if v[sc][m]["mean"] is not None))
            print(f"   mean-response ref: rp_pe {E['arms']['obs'][sc]['rp_pe_meanresp']['mean']}  rr_pe {E['arms']['obs'][sc]['rr_pe_meanresp']['mean']}")
            for d in ["fwd", "rev"]:
                for c in E[d][sc]: print(f"   {d} {c['a']:>10s} - {c['b']:<14s} {c['metric']:13s} {fmt(c)}")
    for op, F in S.get("fewshot", {}).items():
        print(f"\n===== FEWSHOT {op}")
        for m in ["rp_pe_cent", "rp_retrieval", "rp_pe", "rp_auroc"]:
            for sc in ["overall", "iccite"]:
                cur = F["curves"][m][sc]
                print(f"  {m} {sc}: n0 ftR={cur['n0_ftR']['mean'] if cur['n0_ftR'] else None} obs={cur['n0_obs']['mean'] if cur['n0_obs'] else None}")
                for a in [k for k in cur if not k.startswith("n0")]:
                    print(f"     {a:10s} " + " ".join(f"{n}:{cur[a][n]['mean']:+.3f}" for n in F["n_grid"] if cur[a][n]["mean"] is not None))
                for c in F["tests"][m][sc]:
                    if c["a"] == "ftRP" and c["b"] in ("ridgeP_cx", "ftR(n=0)"): print(f"       n={c['n']:>7} {c['a']}-{c['b']}: {fmt(c)}")
    if "exp2" in S:
        X = S["exp2"]; print("\n===== EXP2")
        for m, s in X["summary"].items():
            o = s["overall"]; pp = X["summary_perprot"][m]["overall"]
            print(f"  {m:24s} " + " ".join(f"{k}={o[k]:+.3f}" for k in ["pe", "pe_all", "auroc", "pe_cent", "retrieval", "sign_acc"] if o[k] is not None) + f" | perprot r={pp['r']:+.3f}(n{pp['n_r']})"
                  + " | " + " ".join(f"{sc}:pe={s[sc]['pe']:+.3f},cent={s[sc]['pe_cent']:+.3f},retr={s[sc]['retrieval']:.3f}" for sc in ["iccite", "arce", "frangieh"] if s[sc]["pe"] is not None))
        for sc in SCOPES:
            for c in X["tests"][sc]: print(f"   {sc} {c['a']}-{c['b']} {c['metric']}: {fmt(c)}")
        for k, v in X["perprot_by_cog_tests"].items(): print("  ", k, fmt(v))
        for k, v in X["discordance_tests"].items(): print("  ", k, v)
        print("  check:", {k: round(v["pb_vs_cellmean_r"], 3) for k, v in X["check"].items()})
