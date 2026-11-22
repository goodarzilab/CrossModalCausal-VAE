"""Build a SIGNED DIRECTED prior over the 2692-gene combined_v4 vocab, merging:
  (1) KnockTF (interventional TF knockdown/KO atlas) — signed, directed, external.
  (2) signed CollecTRI + DoRothEA (decoupler) — curated mode-of-regulation (+act / -repr).

CONVENTION (matches src/train.py:482 proj_grn = D_r A_rr D_r^T  and
eval/mutilation_directed_eval.py gproj = -G_rr[TF,:]):
  prior_sign[i,j] = MODE OF REGULATION of TF i -> target j  (+1 activation, -1 repression).
  - CollecTRI/DoRothEA `weight` sign IS the mode of regulation.
  - KnockTF Log2FC is the KO/knockdown response of the target: activation => KO lowers target
    (Log2FC<0) => mode=+1.  So  mode = -sign(mean Log2FC across significant datasets).
  The eval reads -G_rr to predict a KO's LFC, so pushing G_rr toward this mode sign aligns the
  predicted perturbation-effect sign with the observed KO LFC.

KnockTF aggregation (per TF->target across datasets):
  keep rows with Corrected_P < 0.05; require >= MIN_AGREE datasets agreeing on the effect direction;
  sign = majority; confidence = #agreeing datasets.

HELD-OUT-TF split: prior TFs (rows with >=1 merged edge) -> 70% prior-train / 30% held-out (seed 42).
Only prior-train TF edges are used by the training loss (filtered in src/train.py). Held-out TFs are
scored at eval to separate GENERALIZATION from MEMORIZATION.

Outputs:
  priors/signed_prior_2c_v4.pt  {gene_names, sign[G,G], weight[G,G], mask[G,G],
                                 train_tf_idx, heldout_tf_idx, *_names, split_seed, stats}
  results/signed_prior_build.json  (edge stats + source agreement report)

Usage: python data_prep/reference_networks/build_signed_prior.py
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, time, pickle
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _CMC_ROOT
DATA = _CMC_DATA
VOCAB_CSV = f"{DATA}/combined_v4/vocab_genes.csv"
KNOCKTF_TXT = f"{DATA}/knockTF/differential_expression_genes_all_datasets(Human).txt"
KNOCKTF_CACHE = f"{HERE}/_knocktf_agg_cache.pkl"
OUT_PT = f"{REPO}/priors/signed_prior_2c_v4.pt"
OUT_JSON = f"{_CMC_RESULTS}/signed_prior_build.json"
PERTS_JSON = "/tmp/perts_by_ds.json"   # written by the perturbed-gene probe (optional, for reporting)

CORRP = 0.05          # KnockTF significance threshold (Corrected_P)
MIN_AGREE = 2         # require >=2 agreeing datasets per KnockTF edge
SPLIT_SEED = 42
HOLDOUT_FRAC = 0.30

# ---------------- vocab ----------------
vocab = pd.read_csv(VOCAB_CSV)["gene"].astype(str).tolist()
G = len(vocab)
gi = {g: i for i, g in enumerate(vocab)}
vset = set(vocab)
print(f"vocab: {G} genes")


# ---------------- (1) KnockTF ----------------
def parse_knocktf():
    """Stream-parse KnockTF; return dict (tf_idx,gene_idx) -> (mode_sign, n_agree, mean_log2fc)."""
    if os.path.exists(KNOCKTF_CACHE):
        print(f"KnockTF: loading cache {KNOCKTF_CACHE}")
        with open(KNOCKTF_CACHE, "rb") as f:
            return pickle.load(f)
    print(f"KnockTF: streaming {KNOCKTF_TXT} (chunked) ...")
    # accumulators keyed by (tf_idx, gene_idx)
    npos, nneg, sum_l2fc = {}, {}, {}
    t0 = time.time(); nrows = 0; nkept = 0
    usecols = ["TF", "Gene", "Log2FC", "Corrected_P"]
    reader = pd.read_csv(KNOCKTF_TXT, sep="\t", usecols=usecols, chunksize=1_000_000,
                         dtype={"TF": str, "Gene": str}, na_values=["", "NA", "nan"], low_memory=False)
    for ci, chunk in enumerate(reader):
        nrows += len(chunk)
        # Corrected_P / Log2FC have mixed-type rows in the raw dump -> coerce, drop non-numeric
        chunk["Corrected_P"] = pd.to_numeric(chunk["Corrected_P"], errors="coerce")
        chunk["Log2FC"] = pd.to_numeric(chunk["Log2FC"], errors="coerce")
        chunk = chunk.dropna(subset=["TF", "Gene", "Log2FC", "Corrected_P"])
        chunk = chunk[(chunk["Corrected_P"] < CORRP)]
        chunk = chunk[chunk["TF"].isin(vset) & chunk["Gene"].isin(vset)]
        chunk = chunk[chunk["Log2FC"] != 0]
        nkept += len(chunk)
        for tf, gene, l2 in zip(chunk["TF"].values, chunk["Gene"].values,
                                chunk["Log2FC"].astype(float).values):
            k = (gi[tf], gi[gene])
            if l2 > 0:
                npos[k] = npos.get(k, 0) + 1
            else:
                nneg[k] = nneg.get(k, 0) + 1
            sum_l2fc[k] = sum_l2fc.get(k, 0.0) + l2
        if (ci + 1) % 5 == 0:
            print(f"  chunk {ci+1}: {nrows:,} rows read, {nkept:,} sig-in-vocab, "
                  f"{len(sum_l2fc):,} pairs, {time.time()-t0:.0f}s")
    print(f"  DONE stream: {nrows:,} rows, {nkept:,} sig-in-vocab rows, {len(sum_l2fc):,} unique pairs")

    edges = {}
    for k in sum_l2fc:
        p, n = npos.get(k, 0), nneg.get(k, 0)
        n_agree = max(p, n)
        if n_agree < MIN_AGREE:
            continue
        eff_up = p >= n                      # majority KO-effect direction: target went UP under KO?
        mode_sign = -1 if eff_up else +1     # UP under KO -> TF represses -> mode -1
        edges[k] = (mode_sign, n_agree, sum_l2fc[k] / (p + n))
    print(f"  KnockTF edges (>= {MIN_AGREE} agreeing): {len(edges):,}")
    with open(KNOCKTF_CACHE, "wb") as f:
        pickle.dump(edges, f)
    return edges


knocktf = parse_knocktf()

# ---------------- (2) CollecTRI + DoRothEA (signed) ----------------
import decoupler as dc

def net_edges(df, conf_col=None):
    """df with source,target,weight -> dict (tf_idx,gene_idx)->sign, restricted to vocab."""
    out = {}
    for row in df.itertuples(index=False):
        s, t, w = str(row.source), str(row.target), float(row.weight)
        if s in vset and t in vset and w != 0:
            out[(gi[s], gi[t])] = int(np.sign(w))
    return out

print("CollecTRI (decoupler) ...")
ct_df = dc.op.collectri(organism="human")
collectri = net_edges(ct_df)
print(f"  CollecTRI edges in vocab: {len(collectri):,}")

print("DoRothEA (decoupler, levels A,B,C) ...")
dor_df = dc.op.dorothea(organism="human", levels=["A", "B", "C"])
dorothea = net_edges(dor_df)
print(f"  DoRothEA edges in vocab: {len(dorothea):,}")

# ---------------- source-agreement report (overlap) ----------------
def agree(a, b):
    ov = set(a) & set(b)
    if not ov:
        return dict(n_overlap=0, n_agree=0, frac_agree=None)
    na = sum(1 for k in ov if np.sign(a[k] if isinstance(a[k], int) else a[k][0]) ==
             np.sign(b[k] if isinstance(b[k], int) else b[k][0]))
    return dict(n_overlap=len(ov), n_agree=na, frac_agree=round(na / len(ov), 4))

kt_sign = {k: v[0] for k, v in knocktf.items()}
report = {
    "vocab_genes": G,
    "knocktf": dict(n_edges=len(knocktf),
                    n_act=int(sum(1 for v in knocktf.values() if v[0] > 0)),
                    n_repr=int(sum(1 for v in knocktf.values() if v[0] < 0)),
                    n_tfs=len({k[0] for k in knocktf}),
                    min_agree=MIN_AGREE, corrp=CORRP),
    "collectri": dict(n_edges=len(collectri),
                      n_act=int(sum(1 for v in collectri.values() if v > 0)),
                      n_repr=int(sum(1 for v in collectri.values() if v < 0)),
                      n_tfs=len({k[0] for k in collectri})),
    "dorothea": dict(n_edges=len(dorothea),
                     n_act=int(sum(1 for v in dorothea.values() if v > 0)),
                     n_repr=int(sum(1 for v in dorothea.values() if v < 0)),
                     n_tfs=len({k[0] for k in dorothea})),
    "agreement": {
        "knocktf_vs_collectri": agree(kt_sign, collectri),
        "knocktf_vs_dorothea": agree(kt_sign, dorothea),
        "collectri_vs_dorothea": agree(collectri, dorothea),
    },
}
print("SOURCE AGREEMENT:", json.dumps(report["agreement"], indent=2))

# ---------------- merge into a signed matrix ----------------
# weighted vote per edge; KnockTF (interventional) trusted more, scaled by #agreeing datasets (capped).
W_KT, W_CT, W_DOR = 2.0, 1.0, 1.0
KT_CAP = 3.0
votes = {}   # (i,j) -> net signed vote
supp = {}    # (i,j) -> number of independent sources supporting the final sign (filled after)
src_signs = {}  # (i,j) -> list of (source_name, sign)

for k, (ms, na, _) in knocktf.items():
    w = W_KT * min(na, KT_CAP) / KT_CAP        # 2/3..2.0
    votes[k] = votes.get(k, 0.0) + ms * w
    src_signs.setdefault(k, []).append(("knocktf", ms))
for k, s in collectri.items():
    votes[k] = votes.get(k, 0.0) + s * W_CT
    src_signs.setdefault(k, []).append(("collectri", s))
for k, s in dorothea.items():
    votes[k] = votes.get(k, 0.0) + s * W_DOR
    src_signs.setdefault(k, []).append(("dorothea", s))

sign_mat = np.zeros((G, G), dtype=np.float32)
weight_mat = np.zeros((G, G), dtype=np.float32)
n_conflict = 0
for k, v in votes.items():
    if v == 0:            # perfect tie between opposing sources -> drop (ambiguous)
        n_conflict += 1
        continue
    fs = int(np.sign(v))
    # confidence weight: #sources agreeing with final sign + magnitude of net vote (bounded)
    n_agree_src = sum(1 for _, s in src_signs[k] if np.sign(s) == fs)
    n_disagree_src = len(src_signs[k]) - n_agree_src
    if n_disagree_src > 0:
        n_conflict += 1
    w = float(min(abs(v), 4.0))      # ~[0.67 .. 4.0]; used to scale the hinge push
    sign_mat[k[0], k[1]] = fs
    weight_mat[k[0], k[1]] = w

mask = sign_mat != 0
n_edges = int(mask.sum())
tf_rows = np.where(mask.sum(axis=1) > 0)[0]
print(f"MERGED: {n_edges:,} signed edges | {int((sign_mat>0).sum()):,} act / "
      f"{int((sign_mat<0).sum()):,} repr | {len(tf_rows)} TFs | "
      f"{int((mask.sum(0)>0).sum())} targets | conflicts(dropped/flagged)={n_conflict}")

report["merged"] = dict(n_edges=n_edges, n_act=int((sign_mat > 0).sum()),
                        n_repr=int((sign_mat < 0).sum()), n_tfs=int(len(tf_rows)),
                        n_targets=int((mask.sum(0) > 0).sum()), n_conflicts=int(n_conflict),
                        source_weights=dict(knocktf=W_KT, collectri=W_CT, dorothea=W_DOR))

# ---------------- held-out-TF split (70/30, seed 42) ----------------
rng = np.random.RandomState(SPLIT_SEED)
tf_perm = tf_rows.copy()
rng.shuffle(tf_perm)
n_hold = int(round(len(tf_perm) * HOLDOUT_FRAC))
heldout_tf_idx = np.sort(tf_perm[:n_hold])
train_tf_idx = np.sort(tf_perm[n_hold:])
heldout_names = [vocab[i] for i in heldout_tf_idx]
train_names = [vocab[i] for i in train_tf_idx]
print(f"SPLIT: {len(train_tf_idx)} prior-train TFs / {len(heldout_tf_idx)} held-out TFs (seed {SPLIT_SEED})")

# overlap of split with perturbed genes per dataset (informational for the eval)
pert_overlap = {}
if os.path.exists(PERTS_JSON):
    perts = json.load(open(PERTS_JSON))
    for tag, plist in perts.items():
        ps = set(plist)
        pert_overlap[tag] = dict(
            heldout_TFs_perturbed=sorted(ps & set(heldout_names)),
            train_TFs_perturbed=sorted(ps & set(train_names)))
    report["perturbed_TF_overlap"] = {
        tag: dict(n_heldout=len(v["heldout_TFs_perturbed"]),
                  n_train=len(v["train_TFs_perturbed"]),
                  heldout=v["heldout_TFs_perturbed"])
        for tag, v in pert_overlap.items()}
    print("PERTURBED-TF OVERLAP (held-out):",
          json.dumps({t: report["perturbed_TF_overlap"][t]["heldout"] for t in pert_overlap}, indent=2))

# ---------------- save ----------------
os.makedirs(os.path.dirname(OUT_PT), exist_ok=True)
torch.save({
    "gene_names": vocab,
    "sign": torch.from_numpy(sign_mat),          # [G,G] float32 in {-1,0,+1}, sign[TF,target]
    "weight": torch.from_numpy(weight_mat),      # [G,G] float32 confidence (0 where no edge)
    "mask": torch.from_numpy(mask.astype(np.float32)),
    "train_tf_idx": torch.from_numpy(train_tf_idx.astype(np.int64)),
    "heldout_tf_idx": torch.from_numpy(heldout_tf_idx.astype(np.int64)),
    "train_tf_names": train_names,
    "heldout_tf_names": heldout_names,
    "split_seed": SPLIT_SEED,
    "holdout_frac": HOLDOUT_FRAC,
    "source": "KnockTF + signed CollecTRI + DoRothEA (mode-of-regulation)",
    "stats": report,
}, OUT_PT)
print(f"SAVED {OUT_PT}")

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(report, open(OUT_JSON, "w"), indent=2)
print(f"SAVED {OUT_JSON}")
print("DONE")
