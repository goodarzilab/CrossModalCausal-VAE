"""Collect results/ft_ideas/dir/*.json (dir_orient.py outputs) -> summary table + paired tests."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon
D = f"{_CMC_RESULTS}/ft_ideas/dir"
V = {}
for f in sorted(glob.glob(f"{D}/*.json")):
    if "smoke" in f or "summary" in f: continue
    r = json.load(open(f))
    for k, v in r.items():
        if isinstance(v, dict) and "summary" in v and (k not in V): V[k] = v
def ba(df, col, sel=None):
    d = df if sel is None else df[sel]; ps = np.sign(d[col].values); ex = d.exp.values
    return 0.5 * (np.mean(ps[ex < 0] == -1) + np.mean(ps[ex > 0] == 1))
E = {k: pd.DataFrame(v["sign_edges_heldout"]) for k, v in V.items()}
rows = []
for k, v in V.items():
    s = v["summary"]["overall"]; st = v["structure"]; oc = v["orient_collectri"]; ip = v["intervention_probe"]; sg = v["sign_test_heldout"]; info = v.get("info") or {}
    rows.append(dict(variant=k, step=info.get("best_step"), stop=info.get("stop"), retr=s["rr_retrieval"], pe_cent=s["rr_pe_cent"],
                     corrA=st["corr_Arr_ArrT"], corrG=st["corr_Grr_GrrT"], recip=st["recip"], h=st["dagma_h"], dA=st["rel_change_Arr"],
                     ctri_fwd_prior=st.get("collectri_frac_fwd_gt_rev"),
                     ctri_testTF_frac=oc["test_tf"]["G_frac_fwd_gt_rev"], ctri_testTF_lr=oc["test_tf"]["G_med_logratio"],
                     ctri_trainTF_frac=oc["train_tf"]["G_frac_fwd_gt_rev"], ctri_trainTF_lr=oc["train_tf"]["G_med_logratio"],
                     ctri_all_signagree=oc["all"]["G_sign_agree"],
                     signBA_all=sg["all"]["pred"]["balacc"], signBA_all_p=sg["all"]["pred"]["p"], signBA_de=sg["de"]["pred"]["balacc"], signBA_de_p=sg["de"]["pred"]["p"],
                     signBA_cent_all=sg["all"]["pred_cent"]["balacc"], signBA_obs_ceiling=sg["all"]["obs"]["balacc"], signBA_obs_de=sg["de"]["obs"]["balacc"],
                     probe_tr_frac=ip["inner_train"]["G_frac"], probe_tr_mlr=ip["inner_train"]["G_mlr"], probe_te_frac=ip["heldout"]["G_frac"], probe_te_mlr=ip["heldout"]["G_mlr"],
                     probeF_tr_mlr=ip["inner_train"]["F_mlr"], probeF_te_mlr=ip["heldout"]["F_mlr"]))
T = pd.DataFrame(rows).set_index("variant"); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
T.to_csv(f"{D}/summary_table.csv")
print(T[["step", "stop", "retr", "pe_cent", "corrA", "recip", "h", "dA"]].round(4).to_string())
print(T[["ctri_fwd_prior", "ctri_trainTF_frac", "ctri_testTF_frac", "ctri_testTF_lr", "ctri_all_signagree"]].round(4).to_string())
print(T[["signBA_all", "signBA_all_p", "signBA_de", "signBA_de_p", "signBA_cent_all", "signBA_obs_ceiling", "signBA_obs_de"]].round(3).to_string())
print(T[["probe_tr_frac", "probe_tr_mlr", "probe_te_frac", "probe_te_mlr", "probeF_tr_mlr", "probeF_te_mlr"]].round(4).to_string())
# paired over held-out KOs
def per(k, m): return {(r["cohort"], r["gene"]): r.get(m, np.nan) for r in V[k]["per_ko"]}
def pt(a, b, m):
    A, B = per(a, m), per(b, m); ks = [x for x in B if x in A and np.isfinite(A[x]) and np.isfinite(B[x])]
    x = np.array([B[q] - A[q] for q in ks]); return dict(n=len(ks), diff=float(x.mean()), p_t=float(ttest_rel([B[q] for q in ks], [A[q] for q in ks]).pvalue))
def sign_boot(a, b, sel="all", nb=2000):
    ea, eb = E[a], E[b]; assert (ea.target.values == eb.target.values).all()
    m = np.ones(len(ea), bool) if sel == "all" else ea.de.values
    kos = ea[["cohort", "ko"]].astype(str).agg("|".join, axis=1).values; uk = np.unique(kos[m]); rng = np.random.RandomState(0)
    d0 = ba(eb[m], "pred") - ba(ea[m], "pred"); ds = []
    idx = {u: np.where((kos == u) & m)[0] for u in uk}
    for _ in range(nb):
        ix = np.concatenate([idx[u] for u in rng.choice(uk, len(uk))]); ds.append(ba(eb.iloc[ix], "pred") - ba(ea.iloc[ix], "pred"))
    ds = np.array(ds); return dict(diff=float(d0), ci=[float(np.nanpercentile(ds, 2.5)), float(np.nanpercentile(ds, 97.5))])
def probe_pair(a, b, split="heldout", m="G_mlr"):
    A = {(r["cohort"], r["ko"]): r[m] for r in V[a]["intervention_probe"][split]["per_ko"]}; B = {(r["cohort"], r["ko"]): r[m] for r in V[b]["intervention_probe"][split]["per_ko"]}
    ks = [k for k in B if k in A]; return dict(diff=float(np.mean([B[k] - A[k] for k in ks])), p_t=float(ttest_rel([B[k] for k in ks], [A[k] for k in ks]).pvalue))
OUT = {}
for k in V:
    if not k.endswith("_real"): continue
    base = k[:-5]; comp = {"obs": "observational", "shuffle": base + "_shuffle", "mean": base + "_mean"}
    OUT[k] = {}
    for cn, c in comp.items():
        if c not in V: continue
        OUT[k][cn] = dict(retr=pt(c, k, "rr_retrieval"), pe_cent=pt(c, k, "rr_pe_cent"), sign_all=sign_boot(c, k), sign_de=sign_boot(c, k, "de"),
                          probe_te_G_mlr=probe_pair(c, k), probe_tr_G_mlr=probe_pair(c, k, "inner_train"))
        o = OUT[k][cn]
        print(f"{k} vs {cn}: dRetr {o['retr']['diff']:+.4f} (p {o['retr']['p_t']:.3f}) dPEc {o['pe_cent']['diff']:+.4f} (p {o['pe_cent']['p_t']:.3f}) "
              f"dSignBA {o['sign_all']['diff']:+.3f} {np.round(o['sign_all']['ci'],3)} dSignBA_de {o['sign_de']['diff']:+.3f} {np.round(o['sign_de']['ci'],3)} "
              f"probe_te {o['probe_te_G_mlr']['diff']:+.4f} (p {o['probe_te_G_mlr']['p_t']:.3f}) probe_tr {o['probe_tr_G_mlr']['diff']:+.4f} (p {o['probe_tr_G_mlr']['p_t']:.3f})")
json.dump(OUT, open(f"{D}/summary_paired.json", "w"), indent=1)
