"""Aggregate results/ft_ideas/loco/loco_{run}_{hold}_{mode}.json into tables (prints markdown, writes loco_summary.json)."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, glob, os, sys, numpy as np
D = f"{_CMC_RESULTS}/ft_ideas/loco"
RUNS = ["sp_dec_gl1000", "sp_dec_gl1000_s1", "sp_dec_gl1000_s2"]; HOLDS = ["frangieh", "iccite", "arce"]; MODES = ["orig", "relaxed"]
VAR = ["obs", "ft_standard", "ft_labelshuffle", "ft_meantarget", "ft_centered", "ft_infonce", "ft_standard_noshared"]
K = ["rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc", "rp_pe", "rp_pe_cent", "rr_pe_meanresp", "own_rr_pe_meanresp"]
def get(j, v, scope="all"):
    if v == "ft_labelshuffle":
        vs = [j["variants"][k]["summary"][scope] for k in j["variants"] if k.startswith("ft_labelshuffle")]
        return {k: float(np.nanmean([s.get(k, np.nan) for s in vs])) for k in K + ["samegene_obs_rr_pe", "othergene_obs_rr_pe", "samegene_obs_rr_pe_cent", "othergene_obs_rr_pe_cent"]} if vs else None
    return j["variants"].get(v, {}).get("summary", {}).get(scope)
J = {}
for r in RUNS:
    for h in HOLDS:
        for m in MODES:
            f = f"{D}/loco_{r}_{h}_{m}.json"
            j = json.load(open(f)) if os.path.exists(f) else None
            # merge per-variant split jobs (loco_..._part_<variant>.json) for variants missing from the main file
            for pf in sorted(glob.glob(f"{D}/loco_{r}_{h}_{m}_part_*.json")):
                pj = json.load(open(pf))
                if "elapsed_s" not in pj or j is None: continue
                for k, v in pj["variants"].items():
                    if k != "obs" and k not in j["variants"]: j["variants"][k] = v; j.setdefault("merged_from_parts", []).append(k)
                for k in ("infonce_grid", "infonce_selected"):
                    if k in pj and k not in j: j[k] = pj[k]
            need = {"obs", "ft_standard", "ft_meantarget", "ft_centered", "ft_standard_noshared", "ft_infonce", "ft_labelshuffle_0"}
            if j is not None and ("elapsed_s" in j or need <= set(j["variants"])): J[(r, h, m)] = j
OUT = {}
for m in MODES:
    for scope in ["all", "shared"]:
        print(f"\n### mode={m} scope={scope}  (mean over available seeds; n seeds in brackets)")
        print("| hold | variant | " + " | ".join(K) + " |"); print("|" + "---|" * (len(K) + 2))
        for h in HOLDS:
            for v in VAR:
                vals = [get(J[(r, h, m)], v, scope) for r in RUNS if (r, h, m) in J]; vals = [x for x in vals if x]
                if not vals: continue
                row = {k: (float(np.nanmean([x.get(k, np.nan) for x in vals])), float(np.nanstd([x.get(k, np.nan) for x in vals]))) for k in K}
                OUT[f"{m}|{scope}|{h}|{v}"] = dict(row, n_seeds=len(vals))
                print(f"| {h} | {v} [{len(vals)}] | " + " | ".join(f"{a:+.3f}±{b:.3f}" if np.isfinite(a) else "nan" for a, b in row.values()) + " |")
# info / stop reasons
print("\n### optimiser info (seed 42)")
for (r, h, m), j in sorted(J.items()):
    if r != "sp_dec_gl1000": continue
    for v, d in j["variants"].items():
        if d.get("info"): print(r, h, m, v, {k: d["info"][k] for k in d["info"] if k in ("best_step", "last_step", "stop", "rho", "n_projections", "val_at_init", "best_val")}, "relA", round(d["structure"]["rel_change_A"], 4))
print("\n### paired vs obs (seed 42), rr_pe / rr_pe_cent / rr_retrieval mean_diff (p_wilcoxon)")
for (r, h, m), j in sorted(J.items()):
    if r != "sp_dec_gl1000": continue
    for v, d in j["variants"].items():
        if "paired_vs_obs" in d:
            s = []
            for k in ["rr_pe", "rr_pe_cent", "rr_retrieval", "rp_pe"]:
                p = d["paired_vs_obs"][k]["all"]; s.append(f"{k} {p.get('mean_diff', float('nan')):+.3f} (p={p.get('p_wilcoxon', float('nan')):.2g}, n={p['n']})")
            print(h, m, v, " | ".join(s))
print("\n### calibration-subset centring (seed 42; 20% of held-out KOs as reference, scored on rest)")
for (r, h, m), j in sorted(J.items()):
    if r != "sp_dec_gl1000": continue
    for v in ["obs", "ft_standard", "ft_meantarget", "ft_centered"]:
        c = j["variants"][v]["calibration"]
        print(h, m, v, "cal: pe_cent %+.3f retr %.3f meanresp %+.3f | train-ref same subset: pe_cent %+.3f retr %.3f meanresp %+.3f" % tuple(
            [c["cal"].get(k, np.nan) for k in ["rr_pe_cent", "rr_retrieval", "rr_pe_meanresp"]] + [c["train_ref_same_subset"].get(k, np.nan) for k in ["rr_pe_cent", "rr_retrieval", "rr_pe_meanresp"]]))
print("\n### mean-response agreement across cohorts (seed-independent, data only)")
j0 = next(iter(J.values())) if J else None
if j0:
    for k, v in j0["mean_response_agreement"].items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
print("\n### shared genes: gene-specific transfer (seed 42 per-KO)")
for (r, h, m), j in sorted(J.items()):
    if r != "sp_dec_gl1000": continue
    V = j["variants"]; print(f"-- {h}/{m} shared={j['shared_genes']}")
    s = V["obs"]["summary"]["shared"]
    print("   data: same-gene obs pe %+.3f vs other-gene %+.3f | centred same %+.3f vs other %+.3f" % tuple(s.get(k, np.nan) for k in ["samegene_obs_rr_pe", "othergene_obs_rr_pe", "samegene_obs_rr_pe_cent", "othergene_obs_rr_pe_cent"]))
    for v in ["ft_standard", "ft_standard_noshared", "ft_labelshuffle_0", "ft_meantarget"]:
        ps = V[v]["paired_vs_obs"]
        print(f"   {v}: shared d_pe {ps['rr_pe']['shared'].get('mean_diff', np.nan):+.3f} d_cent {ps['rr_pe_cent']['shared'].get('mean_diff', np.nan):+.3f} d_retr {ps['rr_retrieval']['shared'].get('mean_diff', np.nan):+.3f}"
              f" | nonshared d_pe {ps['rr_pe']['nonshared'].get('mean_diff', np.nan):+.3f} d_cent {ps['rr_pe_cent']['nonshared'].get('mean_diff', np.nan):+.3f}")
    # direct: full FT minus FT without the shared genes, on shared KOs
    a = {x["gene"]: x for x in V["ft_standard_noshared"]["per_ko"]}; b = {x["gene"]: x for x in V["ft_standard"]["per_ko"]}
    for k in ["rr_pe", "rr_pe_cent", "rr_retrieval", "rp_pe"]:
        d = np.array([b[g].get(k, np.nan) - a[g].get(k, np.nan) for g in j["shared_genes"]], float); d = d[np.isfinite(d)]
        dn = np.array([b[g].get(k, np.nan) - a[g].get(k, np.nan) for g in b if g not in j["shared_genes"]], float); dn = dn[np.isfinite(dn)]
        print(f"   FT(with g) - FT(without shared): {k} shared {d.mean() if len(d) else np.nan:+.4f} (n={len(d)}, +{int((d>0).sum())}/-{int((d<0).sum())}) vs nonshared {dn.mean() if len(dn) else np.nan:+.4f}")
json.dump(OUT, open(f"{D}/loco_summary.json", "w"), indent=1)

print("\n### shared genes pooled over seeds: FT(with shared genes) - FT(without), and FT - labelshuffle FT, per held-out KO")
from scipy.stats import wilcoxon
POOL = {}
for m in MODES:
    for h in HOLDS:
        acc = {k: dict(sh=[], ns=[]) for k in ["with_minus_without_rr_pe", "with_minus_without_rr_pe_cent", "ft_minus_shuf_rr_pe_cent_shared", "ft_minus_obs_rr_pe_shared"]}
        for r in RUNS:
            j = J.get((r, h, m))
            if not j or "ft_standard_noshared" not in j["variants"]: continue
            V = j["variants"]; SHg = set(j["shared_genes"])
            a = {x["gene"]: x for x in V["ft_standard_noshared"]["per_ko"]}; b = {x["gene"]: x for x in V["ft_standard"]["per_ko"]}
            o = {x["gene"]: x for x in V["obs"]["per_ko"]}
            sh = [{x["gene"]: x for x in V[k]["per_ko"]} for k in V if k.startswith("ft_labelshuffle")]
            for g in b:
                key = "sh" if g in SHg else "ns"
                acc["with_minus_without_rr_pe"][key].append(b[g].get("rr_pe", np.nan) - a[g].get("rr_pe", np.nan))
                acc["with_minus_without_rr_pe_cent"][key].append(b[g].get("rr_pe_cent", np.nan) - a[g].get("rr_pe_cent", np.nan))
                acc["ft_minus_shuf_rr_pe_cent_shared"][key].append(b[g].get("rr_pe_cent", np.nan) - np.nanmean([s[g].get("rr_pe_cent", np.nan) for s in sh]))
                acc["ft_minus_obs_rr_pe_shared"][key].append(b[g].get("rr_pe", np.nan) - o[g].get("rr_pe", np.nan))
        for k, d in acc.items():
            s = np.array(d["sh"], float); s = s[np.isfinite(s)]; n = np.array(d["ns"], float); n = n[np.isfinite(n)]
            if not len(s): continue
            p = float(wilcoxon(s).pvalue) if len(s) > 3 and np.any(s != 0) else float("nan")
            POOL[f"{m}|{h}|{k}"] = dict(shared_mean=float(s.mean()), shared_n=len(s), p_wilcoxon=p, nonshared_mean=float(n.mean()) if len(n) else None)
            print(f"{m} {h} {k}: shared {s.mean():+.4f} (n={len(s)} KO-seeds, p={p:.2g}) | nonshared {n.mean() if len(n) else float('nan'):+.4f}")
# data-level same-gene transfer (model-free)
for h in HOLDS:
    j = J.get(("sp_dec_gl1000", h, "orig"))
    if j:
        s = j["variants"]["obs"]["summary"]["shared"]
        print(f"data {h}: same-gene obs pe {s.get('samegene_obs_rr_pe', np.nan):+.3f} vs other-gene {s.get('othergene_obs_rr_pe', np.nan):+.3f}; centred {s.get('samegene_obs_rr_pe_cent', np.nan):+.3f} vs {s.get('othergene_obs_rr_pe_cent', np.nan):+.3f}; RP same {s.get('samegene_obs_rp_pe', np.nan):+.3f} vs other {s.get('othergene_obs_rp_pe', np.nan):+.3f} (n={s.get('n_samegene_obs_rr_pe')})")
json.dump(dict(table=OUT, shared_pooled=POOL), open(f"{D}/loco_summary.json", "w"), indent=1)
