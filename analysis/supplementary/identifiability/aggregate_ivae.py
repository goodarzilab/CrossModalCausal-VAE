"""Aggregate conditional-prior (iVAE) seed replicates scored with reviewer_controls.py (RC_FIG4=1 RC_FIG3_ONLY=1).
Inputs : analyses/identifiability/ivae_eval/rc_<variant>_s<seed>.json   (variants incl. base = sp_dec_gl1000)
Outputs: analyses/identifiability/ivae_summary.json, ivae_table.md
Forward-minus-reverse = learned A_rr minus A_rr transposed (Fig. 4 control), per-KO paired, TWO-SIDED t-test,
BH within (metric x cohort) across all variant-seed rows.
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
import json, glob, os, re
import numpy as np
from scipy import stats

ANA = f"{_CMC_RESULTS}/paper/analyses/identifiability"
COH = ["overall", "frangieh", "iccite", "arce"]
MET = ["rr_pe", "rr_pe_meanresp", "rr_pe_cent", "rr_pe_shuf", "rr_retrieval", "rr_auroc",
       "rp_pe", "rp_pe_meanresp", "rp_pe_cent", "rp_pe_shuf", "rp_retrieval", "rp_auroc"]
FR = ["rr_pe", "rr_pe_cent", "rr_retrieval", "rp_pe", "rp_pe_cent", "rp_retrieval"]


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q


rows = {}
for f in sorted(glob.glob(f"{ANA}/ivae_eval/rc_*.json")):
    m = re.match(r"rc_(.+)_s(\d+)\.json", os.path.basename(f))
    if not m: continue
    R = json.load(open(f))
    if "fig4_controls" not in R: continue
    v, s = m.group(1), int(m.group(2))
    S = R["fig3_obs"]["summary"]; F4 = R["fig4_controls"]["Arr_transposed"]["paired_learned_minus_variant"]
    per = R["fig3_obs"]["per_ko"]
    rows[(v, s)] = dict(variant=v, seed=s, summary={c: {k: S[c].get(k) for k in MET} for c in COH},
                        n={c: {k: S[c].get("n_" + k) for k in MET} for c in COH},
                        fr={k: {c: F4[k].get(c, {}) for c in COH} for k in FR},
                        per_ko={(r["cohort"], r["gene"]): r for r in per},
                        structure=R.get("structure_obs", {}))

# BH over the forward-minus-reverse family (metric x cohort, all rows)
keys, ps = [], []
for rk, r in rows.items():
    for k in FR:
        for c in COH:
            p = r["fr"][k][c].get("p_t")
            if p is not None: keys.append((rk, k, c)); ps.append(p)
qs = bh(ps) if ps else []
for (rk, k, c), q in zip(keys, qs): rows[rk]["fr"][k][c]["q_bh"] = float(q)

# paired iVAE-minus-base (same seed) per KO, two-sided
def paired_vs(a, b, mtr, coh):
    ks = [k for k in a["per_ko"] if k in b["per_ko"] and (coh == "overall" or k[0] == coh)]
    x = np.array([[a["per_ko"][k].get(mtr, np.nan), b["per_ko"][k].get(mtr, np.nan)] for k in ks], float)
    x = x[np.isfinite(x).all(1)]
    if len(x) < 3: return dict(n=len(x))
    d = x[:, 0] - x[:, 1]
    return dict(n=len(d), mean_diff=float(d.mean()), ci95=[float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)))], p_t=float(stats.ttest_rel(x[:, 0], x[:, 1]).pvalue))

variants = sorted({v for v, _ in rows})
agg = {}
for v in variants:
    seeds = sorted(s for vv, s in rows if vv == v)
    A = dict(seeds=seeds, metrics={}, fwd_minus_rev={}, vs_base={})
    for c in COH:
        for k in MET:
            vals = [rows[(v, s)]["summary"][c][k] for s in seeds]
            vals = [x for x in vals if x is not None and np.isfinite(x)]
            A["metrics"][f"{c}.{k}"] = dict(mean=float(np.mean(vals)) if vals else None,
                                            sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None, per_seed=vals)
        for k in FR:
            d = [rows[(v, s)]["fr"][k][c].get("mean_diff") for s in seeds]
            d = [x for x in d if x is not None]
            q = [rows[(v, s)]["fr"][k][c].get("q_bh") for s in seeds]
            A["fwd_minus_rev"][f"{c}.{k}"] = dict(
                mean=float(np.mean(d)) if d else None, sd=float(np.std(d, ddof=1)) if len(d) > 1 else None, per_seed=d,
                per_seed_q=q, n_seeds_q_lt_05=int(sum(1 for x in q if x is not None and x < 0.05)),
                sign_consistent=bool(len(d) > 1 and (all(x > 0 for x in d) or all(x < 0 for x in d))),
                one_sample_t_p_across_seeds=float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 2 else None)
    if v != "base":
        for c in COH:
            for k in ["rr_pe", "rr_pe_cent", "rr_retrieval", "rp_pe", "rp_pe_cent", "rp_retrieval"]:
                A["vs_base"][f"{c}.{k}"] = {s: paired_vs(rows[(v, s)], rows[("base", s)], k, c) for s in seeds if ("base", s) in rows}
    agg[v] = A

out = dict(description=__doc__, rows={f"{v}_s{s}": dict(summary=r["summary"], n=r["n"], fwd_minus_rev=r["fr"], structure=r["structure"])
                                      for (v, s), r in rows.items()}, aggregate=agg)
json.dump(out, open(f"{ANA}/ivae_summary.json", "w"), indent=1, default=lambda o: None)

# markdown table
f = lambda x: "NA" if x is None else f"{x:+.3f}"
L = ["| Variant (seeds) | Cohort | RNA pearDE | mean-resp ref | centred | target-shuffled | retrieval | F−R pearDE (per seed) | F−R centred | F−R retrieval | protein pearDE | prot mean-resp | prot centred | prot retrieval |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for v in variants:
    A = agg[v]
    for c in COH:
        M = lambda k: A["metrics"][f"{c}.{k}"]
        ms = lambda k: (f"{M(k)['mean']:+.3f} ± {M(k)['sd']:.3f}" if M(k)["sd"] is not None else f(M(k)["mean"])) if M(k)["mean"] is not None else "NA"
        fr = lambda k: ", ".join(f"{x:+.3f}{'*' if (q is not None and q < 0.05) else ''}" for x, q in zip(A["fwd_minus_rev"][f"{c}.{k}"]["per_seed"], A["fwd_minus_rev"][f"{c}.{k}"]["per_seed_q"]))
        L.append(f"| {v} ({','.join(map(str, A['seeds']))}) | {c} | {ms('rr_pe')} | {ms('rr_pe_meanresp')} | {ms('rr_pe_cent')} | {ms('rr_pe_shuf')} | {ms('rr_retrieval')} | {fr('rr_pe')} | {fr('rr_pe_cent')} | {fr('rr_retrieval')} | {ms('rp_pe')} | {ms('rp_pe_meanresp')} | {ms('rp_pe_cent')} | {ms('rp_retrieval')} |")
open(f"{ANA}/ivae_table.md", "w").write("\n".join(L) + "\n\n* BH q < 0.05 (two-sided paired t over KOs; BH across all variant x seed x metric x cohort F−R tests).\n")
print("\n".join(L))
