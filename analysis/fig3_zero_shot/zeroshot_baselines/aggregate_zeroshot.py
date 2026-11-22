"""Aggregate the 3-seed zero-shot Fig-3 comparison: per-cohort means, paired two-sided tests (CMC-VAE vs each comparator),
bootstrap 95% CIs over knockouts, BH within panel (panel = modality x metric). Writes zeroshot_summary.json/.csv."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon, ttest_1samp
B = f"{_CMC_RESULTS}/paper/analyses/baselines"
SEEDS = {"42": "sp_dec_gl1000", "1": "sp_dec_gl1000_s1", "2": "sp_dec_gl1000_s2"}
D = {s: pd.read_csv(f"{B}/zeroshot_{r}_per_ko.csv") for s, r in SEEDS.items()}
COH = ["frangieh", "iccite", "arce", "overall"]
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q
rng = np.random.RandomState(0)
def ci(d):
    bs = [np.mean(d[rng.randint(0, len(d), len(d))]) for _ in range(2000)]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def sub(df, c): return df if c == "overall" else df[df.cohort == c]

AG = ["ctrl_mean", "ctrl_var", "train_mean", "train_sd", "decoder_row_norm", "cmc_generic"]
SP = ["corr", "partial_corr", "corr_within_ct", "screen_ctrl_corr"]
PANELS = {}
for blk in ["rr", "rp"]:
    PANELS[f"{blk}_auroc"] = (f"{blk}_auroc__CMC-VAE", {k: f"{blk}_auroc__{k}" for k in AG + SP})
    PANELS[f"{blk}_retrieval"] = (f"{blk}_retrieval__CMC-VAE", {k: f"{blk}_retrieval__{k}" for k in SP})
    PANELS[f"{blk}_pe_cent"] = (f"{blk}_pe_cent__CMC-VAE", {k: f"{blk}_pe_cent__{k}" for k in SP})
    PANELS[f"{blk}_pe"] = (f"{blk}_pe__CMC-VAE", {k: f"{blk}_pe__{k}" for k in SP})
PANELS["rr_auroc_strat"] = ("rr_auroc_strat__CMC-VAE", {k: f"rr_auroc_strat__{k}" for k in SP + ["ctrl_var", "decoder_row_norm", "cmc_generic", "train_sd"]})
PANELS["rr_auroc_logit_add"] = ("rr_auroc_logit__expr+rownorm+CMC-VAE", {k: f"rr_auroc_logit__{k}" for k in
                                ["expr", "expr+rownorm", "expr+rownorm+CMC-VAE_shuffled", "expr+rownorm+cmc_generic", "expr+rownorm+corr", "expr+rownorm+partial_corr", "expr+rownorm+screen_ctrl_corr"]})
rows = []
for s, df in D.items():
    for pn, (a, comps) in PANELS.items():
        for c in COH:
            x = sub(df, c)
            for k, b in comps.items():
                if a not in x or b not in x: continue
                m = np.isfinite(x[a]) & np.isfinite(x[b]); va, vb = x[a][m].values, x[b][m].values
                if len(va) < 3: continue
                d = va - vb
                rows.append(dict(seed=s, panel=pn, cohort=c, comparator=k, n=int(len(d)), cmc=float(va.mean()), comp=float(vb.mean()), diff=float(d.mean()),
                                 ci95=ci(d), p_t=float(ttest_rel(va, vb).pvalue) if np.any(d != 0) else 1.0,
                                 p_w=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0))
T = pd.DataFrame(rows)
T["q_bh"] = np.nan
for (s, pn), g in T.groupby(["seed", "panel"]):
    T.loc[g.index, "q_bh"] = bh(g.p_t.values)
# KO specificity of rankings: own - other AUROC, one-sample two-sided
spec = []
for s, df in D.items():
    for blk in ["rr", "rp"]:
        for k in ["CMC-VAE"] + SP:
            for c in COH:
                x = sub(df, c); a, b = f"{blk}_auroc_own__{k}", f"{blk}_auroc_other__{k}"
                if a not in x: continue
                m = np.isfinite(x[a]) & np.isfinite(x[b]); d = (x[a] - x[b])[m].values
                if len(d) < 3: continue
                spec.append(dict(seed=s, block=blk, predictor=k, cohort=c, n=int(len(d)), own=float(x[a][m].mean()), other=float(x[b][m].mean()), diff=float(d.mean()),
                                 ci95=ci(d), p=float(ttest_1samp(d, 0).pvalue)))
S = pd.DataFrame(spec); S["q_bh"] = np.nan
for (s, blk), g in S.groupby(["seed", "block"]): S.loc[g.index, "q_bh"] = bh(g.p.values)
# seed-mean table of per-cohort means for every ranker
cols = [c for c in D["42"].columns if "__" in c]
M = {c: {coh: [float(sub(D[s], coh)[c].mean()) for s in D] for coh in COH} for c in cols}
MT = pd.DataFrame([dict(metric=c, cohort=coh, mean_over_seeds=float(np.nanmean(v)), sd_over_seeds=float(np.nanstd(v, ddof=1)), s42=v[0], s1=v[1], s2=v[2],
                        n=int(np.isfinite(sub(D["42"], coh)[c]).sum())) for c, d in M.items() for coh, v in d.items()])
T.to_csv(f"{B}/zeroshot_paired_tests.csv", index=False); S.to_csv(f"{B}/zeroshot_ko_specificity.csv", index=False); MT.to_csv(f"{B}/zeroshot_means.csv", index=False)
json.dump(dict(paired=T.to_dict("records"), ko_specificity=S.to_dict("records")), open(f"{B}/zeroshot_summary.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for pn in PANELS:
    x = T[(T.panel == pn)]
    print("\n==", pn); print(x.pivot_table(index=["cohort", "comparator"], columns="seed", values=["diff", "q_bh"]).round(3))
print(S.pivot_table(index=["block", "predictor", "cohort"], columns="seed", values=["diff", "q_bh"]).round(3))
