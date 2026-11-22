"""Compatibility shim for arboreto 0.1.6 running on modern dask (dask-expr query planner).

arboreto.core.create_graph calls `from_delayed(delayed_meta_dfs, meta=_META_SCHEMA)`
unconditionally. With the default grnboost2/genie3 (include_meta=False) that list is
EMPTY, and modern dask-expr raises "Must supply at least one delayed object" (old dask
returned an empty frame). `all_meta_df` is never used unless include_meta=True, so we
simply make from_delayed tolerate an empty list by returning the (empty) meta frame.

This does NOT change the GRN algorithm: the actual per-target GBM/RF regressions and the
links DataFrame (from the non-empty delayed_link_dfs) are untouched.

Import this module BEFORE calling arboreto.algo.grnboost2/genie3.
"""
import arboreto.core as _ac

_orig_from_delayed = _ac.from_delayed


def _safe_from_delayed(dfs, *args, **kwargs):
    try:
        empty = (dfs is None) or (len(dfs) == 0)
    except TypeError:
        dfs = list(dfs)
        empty = (len(dfs) == 0)
    if empty:
        meta = kwargs.get("meta", None)
        if meta is None and len(args) >= 1:
            meta = args[0]
        return meta  # make_meta output is already an empty, correctly-typed DataFrame
    return _orig_from_delayed(dfs, *args, **kwargs)


_ac.from_delayed = _safe_from_delayed
print("[_arboreto_patch] patched arboreto.core.from_delayed for empty-list tolerance")


# ---------------------------------------------------------------------------
# Scatter-once fix for large-N scaling.
#
# Stock arboreto.create_graph embeds a full expression COLUMN into the Dask task
# graph for EACH of the ~2700 target genes (`delayed(expression_matrix[:, i])`),
# so the submitted graph balloons to ~expr_matrix-size (5 GiB @ 500k cells, 8.7 GiB
# @ 809k) and the scheduler stalls serializing it (0 CPU) — with BOTH the process
# and threaded schedulers. Here we scatter the full matrix ONCE as a broadcast
# future and slice each target column on the worker via np.take, so only a future
# reference + an int index enter the graph. Identical GRN math; just no graph bloat.
# ---------------------------------------------------------------------------
import numpy as _np
from dask import delayed as _delayed


def _create_graph_scatter_once(expression_matrix, gene_names, tf_names, regressor_type,
                               regressor_kwargs, client, target_genes='all', limit=None,
                               include_meta=False,
                               early_stop_window_length=_ac.EARLY_STOP_WINDOW_LENGTH,
                               repartition_multiplier=1, seed=_ac.DEMON_SEED):
    assert expression_matrix.shape[1] == len(gene_names)
    assert client, "client is required"

    tf_matrix, tf_matrix_gene_names = _ac.to_tf_matrix(expression_matrix, gene_names, tf_names)
    future_tf_matrix = client.scatter(tf_matrix, broadcast=True)
    [future_tf_matrix_gene_names] = client.scatter([tf_matrix_gene_names], broadcast=True)
    future_expr = client.scatter(expression_matrix, broadcast=True)   # <-- scatter full matrix ONCE

    delayed_link_dfs, delayed_meta_dfs = [], []
    for target_gene_index in _ac.target_gene_indices(gene_names, target_genes):
        target_gene_name = _delayed(gene_names[target_gene_index], pure=True)
        # slice the column on the worker from the scattered future (NOT embedded in the graph)
        target_gene_expression = _delayed(_np.take, pure=True)(future_expr, target_gene_index, 1)
        if include_meta:
            delayed_link_df, delayed_meta_df = _delayed(_ac.infer_partial_network, pure=True, nout=2)(
                regressor_type, regressor_kwargs, future_tf_matrix, future_tf_matrix_gene_names,
                target_gene_name, target_gene_expression, include_meta, early_stop_window_length, seed)
            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df); delayed_meta_dfs.append(delayed_meta_df)
        else:
            delayed_link_df = _delayed(_ac.infer_partial_network, pure=True)(
                regressor_type, regressor_kwargs, future_tf_matrix, future_tf_matrix_gene_names,
                target_gene_name, target_gene_expression, include_meta, early_stop_window_length, seed)
            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)

    all_links_df = _ac.from_delayed(delayed_link_dfs, meta=_ac._GRN_SCHEMA)
    all_meta_df = _ac.from_delayed(delayed_meta_dfs, meta=_ac._META_SCHEMA)
    maybe_limited_links_df = all_links_df.nlargest(limit, columns=['importance']) if limit else all_links_df
    n_parts = len(client.ncores()) * repartition_multiplier
    if include_meta:
        return maybe_limited_links_df.repartition(npartitions=n_parts), all_meta_df.repartition(npartitions=n_parts)
    return maybe_limited_links_df.repartition(npartitions=n_parts)


_ac.create_graph = _create_graph_scatter_once
import arboreto.algo as _aa
_aa.create_graph = _create_graph_scatter_once   # algo.py imported the name directly
print("[_arboreto_patch] patched create_graph: scatter expr matrix once (large-N graph-bloat fix)")
