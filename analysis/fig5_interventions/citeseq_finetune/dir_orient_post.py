"""Post-hoc orientation of projected G_rr = D A_rr D^T on independent directed references (signed CollecTRI, DoRothEA A-C)
with a NODE-LEVEL null: same TF -> random non-target genes, both restricted to non-negligible pairs
(max(|G_ij|,|G_ji|) > median |G| over all off-diagonal pairs). Reads results/ft_ideas/dir/A_*.npz + Dr.npz."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import glob, json, os, numpy as np, pandas as pd
D = f"{_CMC_RESULTS}/ft_ideas/dir"; REPO = f"{_CMC_ROOT}"
z = np.load(f"{D}/Dr.npz"); Dr = z["Dr"].astype(np.float64); genes = list(z["genes"]); gi = {g: i for i, g in enumerate(genes)}
sp = json.load(open(f"{_CMC_RESULTS}/finetune_split_genes.json")); TR, TE = set(sp["train_genes"]), set(sp["test_genes"])
REFS = {"collectri": pd.read_csv(f"{D}/collectri_signed.csv"), "dorothea_ABC": pd.read_csv(f"{D}/dorothea_ABC.csv")}
def prep(df):
    df = df[(df.source != df.target) & df.source.isin(gi) & df.target.isin(gi)].copy()
    pairs = set(zip(df.source, df.target)); df = df[[(b, a) not in pairs for a, b in zip(df.source, df.target)]]
    df["si"] = df.source.map(gi); df["ti"] = df.target.map(gi)
    df["grp"] = np.where(df.source.isin(TR), "train_tf", np.where(df.source.isin(TE), "test_tf", "other_tf"))
    return df, pairs
REFS = {k: prep(v) for k, v in REFS.items()}
def orient(G, df, pairs, rng, k_null=5, nboot=1000):
    off = ~np.eye(G.shape[0], dtype=bool); thr = np.median(np.abs(G[off])); out = {}
    allt = {}
    for s, t in pairs: allt.setdefault(s, set()).add(t)
    for grp in ["train_tf", "test_tf", "other_tf", "all"]:
        d = df if grp == "all" else df[df.grp == grp]
        tf_true, tf_null = {}, {}
        for tf, sub in d.groupby("si"):
            ti = sub.ti.values; f, r = np.abs(G[tf, ti]), np.abs(G[ti, tf]); m = np.maximum(f, r) > thr
            cand = np.setdiff1d(np.arange(G.shape[0]), [tf] + [gi[x] for x in allt.get(genes[tf], ())])
            nj = rng.choice(cand, min(len(cand), k_null * len(ti)), replace=False); fn, rn = np.abs(G[tf, nj]), np.abs(G[nj, tf]); mn = np.maximum(fn, rn) > thr
            if m.sum() and mn.sum(): tf_true[tf] = (f[m] > r[m]); tf_null[tf] = (fn[mn] > rn[mn])
        if len(tf_true) < 3: continue
        tfs = list(tf_true); tv = np.concatenate([tf_true[t] for t in tfs]); nv = np.concatenate([tf_null[t] for t in tfs])
        gain = tv.mean() - nv.mean(); bs = []
        for _ in range(nboot):
            pick = rng.choice(len(tfs), len(tfs)); bs.append(np.concatenate([tf_true[tfs[i]] for i in pick]).mean() - np.concatenate([tf_null[tfs[i]] for i in pick]).mean())
        out[grp] = dict(n_tf=len(tfs), n_edges=int(len(tv)), frac_fwd_true=float(tv.mean()), frac_fwd_null=float(nv.mean()), gain=float(gain),
                        ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    return out
RES = {}
for f in sorted(glob.glob(f"{D}/A_*.npz")):
    name = os.path.basename(f)[2:-4]; A = np.load(f)["A_rr"].astype(np.float64); G = Dr @ A @ Dr.T; np.fill_diagonal(G, 0)
    RES[name] = {ref: orient(G, df, pairs, np.random.RandomState(0)) for ref, (df, pairs) in REFS.items()}
json.dump(RES, open(f"{D}/orient_post.json", "w"), indent=1)
rows = []
for n, r in RES.items():
    row = dict(variant=n)
    for ref in REFS:
        for grp in ["train_tf", "test_tf", "all"]:
            o = r[ref].get(grp)
            if o: row[f"{ref[:4]}_{grp}_gain"] = round(o["gain"], 3); row[f"{ref[:4]}_{grp}_ci"] = f"[{o['ci95'][0]:+.3f},{o['ci95'][1]:+.3f}]"; row[f"{ref[:4]}_{grp}_true"] = round(o["frac_fwd_true"], 3)
    rows.append(row)
T = pd.DataFrame(rows).set_index("variant"); T.to_csv(f"{D}/orient_post_table.csv"); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
print(T.to_string())
