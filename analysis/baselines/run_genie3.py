"""GENIE3 / GRNBoost2 observational-GRN baseline for CMC-VAE (PAPER_FINALIZATION_PLAN.md §3-G1).

Runs arboreto GRNBoost2 (default) or GENIE3 on the v4 vocab RNA matrix and produces a
gene x gene DIRECTED importance matrix M where M[src, tgt] = regulator->target importance,
saved on the exact 2692-gene v4 vocab order so it can be scored by score_genie3.py through
the SAME harness as causal_eval_suite.py.

Two regulator modes (both are legitimate; we run both):
  --mode tf   : candidate regulators = the CollecTRI TFs (standard GENIE3/GRNBoost2 protocol;
                only TF rows of M are populated). Used for GRN-recovery AUROC/AUPRC.
  --mode all  : candidate regulators = ALL genes (classic GENIE3 all-vs-all; dense both
                directions). Needed for an un-confounded orientation-asymmetry test that is
                apples-to-apples with CMC-VAE (both TF->tgt and tgt->TF genuinely scored).

Expression = X (log-normalized, the same matrix the VAE ingests). Cells subsampled
(stratified by dataset) for tractability; recorded in the output.

Env: a separate conda env with arboreto installed (e.g. "arboreto-env"; CPU/dask).
Usage: python run_genie3.py --mode tf|all --n-cells 50000 --seed 0 --n-workers 32 --method grnboost2|genie3
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
import os, sys, json, time, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _arboreto_patch  # noqa: F401  (MUST precede arboreto import: dask-expr empty-list fix)
from arboreto.algo import grnboost2, genie3

REPO = _CMC_ROOT
RNA_H5AD = os.environ.get("GENIE3_RNA_H5AD", f"{_CMC_DATA}/combined_v4/rna_combined.h5ad")
COLLECTRI_PRIOR = os.path.join(REPO, "priors", "rna_hvg_priors_collectri_sc_vocab4_2c.pt")
OUTDIR = os.path.join(REPO, "results", "baselines")


TF_CACHE = os.path.join(HERE, "collectri_tfs_v4_2c.json")


def load_tf_names(genes):
    """CollecTRI TFs (source genes = rows with out-edges in the training prior mask).
    Read from a JSON cache built with the base env (arboreto-env has no torch)."""
    d = json.load(open(TF_CACHE))
    assert d["gene_names"] == list(genes), "cached CollecTRI gene order != vocab order"
    tfs = [t for t in d["tf_names"] if t in set(genes)]
    return tfs


def load_expression(n_cells, seed):
    import scanpy as sc, scipy.sparse as sp
    ad = sc.read_h5ad(RNA_H5AD, backed="r")
    genes = list(ad.var_names.astype(str))
    n_total = ad.n_obs
    if n_cells and n_cells < n_total:
        ds = ad.obs["dataset"].astype(str).values
        rng = np.random.default_rng(seed)
        # stratified-by-dataset proportional subsample
        idx = []
        for d in np.unique(ds):
            pool = np.where(ds == d)[0]
            k = int(round(len(pool) / n_total * n_cells))
            k = min(k, len(pool))
            idx.append(rng.choice(pool, size=k, replace=False))
        idx = np.sort(np.concatenate(idx))
    else:
        idx = np.arange(n_total)
    print(f"[data] loading {len(idx)} / {n_total} cells (stratified by dataset)", flush=True)
    Xsub = ad.X[idx]
    if sp.issparse(Xsub):
        Xsub = Xsub.toarray()
    Xsub = np.asarray(Xsub, dtype=np.float32)
    ds_counts = pd.Series(ad.obs["dataset"].astype(str).values[idx]).value_counts().to_dict()
    return Xsub, genes, len(idx), n_total, ds_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tf", "all"], default="tf")
    ap.add_argument("--method", choices=["grnboost2", "genie3"], default="grnboost2")
    ap.add_argument("--n-cells", type=int, default=50000)
    ap.add_argument("--tag-extra", type=str, default="")   # appended to output tag (sweep: distinct files)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=32)
    ap.add_argument("--threads-per-worker", type=int, default=1)
    ap.add_argument("--processes", action="store_true", default=True)
    ap.add_argument("--no-processes", dest="processes", action="store_false")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()

    X, genes, n_used, n_total, ds_counts = load_expression(args.n_cells, args.seed)
    ng = len(genes)
    gidx = {g: i for i, g in enumerate(genes)}
    tfs = load_tf_names(genes)
    tf_names = tfs if args.mode == "tf" else list(genes)
    print(f"[cfg] method={args.method} mode={args.mode} regulators={len(tf_names)} genes={ng} cells={n_used}", flush=True)

    expr = pd.DataFrame(X, columns=genes)

    from distributed import Client, LocalCluster
    cl = LocalCluster(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker,
                      processes=args.processes, dashboard_address=None)
    client = Client(cl)
    print(f"[dask] {args.n_workers} workers processes={args.processes}", flush=True)

    algo = grnboost2 if args.method == "grnboost2" else genie3
    net = algo(expression_data=expr, tf_names=tf_names, gene_names=genes,
               client_or_address=client, seed=args.seed, verbose=True)
    client.close(); cl.close()
    print(f"[grn] {len(net)} links inferred in {time.time()-t0:.1f}s", flush=True)

    # dense directed importance matrix M[src, tgt]
    M = np.zeros((ng, ng), dtype=np.float32)
    src = net["TF"].map(gidx).values
    tgt = net["target"].map(gidx).values
    M[src, tgt] = net["importance"].values.astype(np.float32)

    tag = f"{args.method}_{args.mode}{args.tag_extra}"
    outp = os.path.join(OUTDIR, f"genie3_matrix_{tag}.npz")
    np.savez_compressed(outp, matrix=M, genes=np.array(genes),
                        tf_names=np.array(tfs), mode=args.mode, method=args.method,
                        n_cells=n_used, n_total=n_total, seed=args.seed,
                        n_links=len(net))
    meta = dict(method=args.method, mode=args.mode, n_regulators=len(tf_names),
                n_tfs=len(tfs), n_genes=ng, n_cells=n_used, n_total=n_total, seed=args.seed,
                n_links=int(len(net)), dataset_counts=ds_counts,
                nonzero_rows=int((np.abs(M).sum(1) > 0).sum()),
                elapsed_s=round(time.time() - t0, 1),
                matrix_path=outp)
    json.dump(meta, open(os.path.join(OUTDIR, f"genie3_run_{tag}.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
