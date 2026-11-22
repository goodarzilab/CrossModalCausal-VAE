"""Build the totalVI-ready AnnData for the CMC-VAE combined_v4 (2-cohort) data.

totalVI needs RAW COUNTS for both modalities. The combined_v4 protein file is CLR-normalized,
so we RECONSTRUCT raw ADT counts from the two source datasets (Hao h5seurat ADT assay,
Stephenson haniffa layers['raw']) and re-align them to the exact combined_v4 cells + 144-protein
panel using the authoritative STEPH_TO_HAO mapping in combine_datasets.py.

Raw RNA counts come from combined_v4/rna_combined.h5ad layers['counts'] (2692 vocab genes).

Output AnnData (saved to combined_v4/totalvi_input.h5ad):
  .X                          = raw RNA counts (809130 x 2692), sparse
  .layers['counts']           = same (totalVI reads a layer)
  .obsm['protein_expression'] = raw ADT counts (809130 x 144), DataFrame w/ protein names
  .obs['donor'], ['dataset'], ['celltype']

Run in the BASE env (needs scanpy/h5py; no scvi). Absolute paths only.
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
import os, sys, gc
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, h5py
import scipy.sparse as sp

DATA = f"{_CMC_DATA}"
CV4  = f"{DATA}/combined_v4"
HAO_H5SEURAT = f"{DATA}/hao_citeseq/hao_citeseq.h5seurat"
STEPH_H5AD   = f"{DATA}/stephenson_citeseq/haniffa21.processed.h5ad"
OUT = f"{CV4}/totalvi_input.h5ad"

sys.path.insert(0, DATA); sys.path.insert(0, _os.path.join(_CMC_ROOT, "data_prep", "citeseq"))  # [public release] combine_datasets.py
from combine_datasets import STEPH_TO_HAO, HAO_ISOTYPES  # authoritative panel mapping

def read_hao_adt_counts(f):
    grp = f["assays"]["ADT"]["counts"]
    data, indices, indptr = grp["data"][:], grp["indices"][:], grp["indptr"][:]
    n_feats = len(f["assays"]["ADT"]["features"][:])
    n_cells = len(indptr) - 1
    # stored features x cells (CSC) -> cells x features (CSR)
    return sp.csc_matrix((data, indices, indptr), shape=(n_feats, n_cells)).T.tocsr()

def main():
    print("[1] load combined_v4 protein (obs order, panel, dataset labels)...", flush=True)
    ap = sc.read_h5ad(f"{CV4}/citeseq_combined.h5ad")  # CLR (only for obs order + panel names)
    obs_names = np.array(ap.obs_names)
    panel = list(ap.var_names)                          # 144 canonical (Hao) names, target column order
    dataset = ap.obs["dataset"].astype(str).values
    donor = ap.obs["donor"].astype(str).values
    celltype = ap.obs["celltype"].astype(str).values
    n_cells = len(obs_names)
    print(f"    cells={n_cells}  panel={len(panel)}", flush=True)
    assert len(panel) == 144

    # reverse map panel(Hao) -> Stephenson AB_ name
    hao2steph = {h: s for s, h in STEPH_TO_HAO.items()}
    missing = [p for p in panel if p not in hao2steph]
    assert not missing, f"panel names missing from mapping: {missing}"

    prot_raw = np.zeros((n_cells, len(panel)), dtype=np.float32)
    filled = np.zeros(n_cells, dtype=bool)

    # ---------- Hao raw ADT ----------
    print("[2] reconstruct Hao raw ADT counts...", flush=True)
    f = h5py.File(HAO_H5SEURAT, "r")
    hao_cells = np.array([x.decode() for x in f["cell.names"][:]])
    adt_feats = np.array([x.decode() for x in f["assays"]["ADT"]["features"][:]])
    adt = read_hao_adt_counts(f).toarray().astype(np.float32)   # [n_hao, 228]
    f.close()
    feat_pos = {g: i for i, g in enumerate(adt_feats)}
    col_idx = [feat_pos[p] for p in panel]                      # panel columns in Hao ADT
    hao_sub = adt[:, col_idx]                                    # [n_hao, 144] Hao names order
    del adt; gc.collect()
    row_of = {bc: i for i, bc in enumerate(hao_cells)}
    hao_mask = dataset == "hao"
    tgt = np.where(hao_mask)[0]
    src = np.array([row_of[obs_names[i]] for i in tgt])
    prot_raw[tgt] = hao_sub[src]
    filled[tgt] = True
    print(f"    Hao: filled {hao_mask.sum()} cells; count max={hao_sub.max():.0f} int_frac={np.mean(hao_sub==np.round(hao_sub)):.3f}", flush=True)
    del hao_sub; gc.collect()

    # ---------- Stephenson raw ADT ----------
    print("[3] reconstruct Stephenson raw ADT counts (loading haniffa, ~7GB)...", flush=True)
    st = sc.read_h5ad(STEPH_H5AD)
    steph_ab = [hao2steph[p] for p in panel]                    # AB_ names in panel order
    st_var_pos = {g: i for i, g in enumerate(st.var_names)}
    miss_ab = [a for a in steph_ab if a not in st_var_pos]
    assert not miss_ab, f"AB_ names missing in haniffa: {miss_ab}"
    ab_idx = [st_var_pos[a] for a in steph_ab]
    raw = st.layers["raw"]
    raw = raw.tocsc() if sp.issparse(raw) else raw
    steph_sub = raw[:, ab_idx]
    steph_sub = steph_sub.toarray().astype(np.float32) if sp.issparse(steph_sub) else np.asarray(steph_sub, dtype=np.float32)
    steph_cells = np.array(st.obs_names)
    del st, raw; gc.collect()
    row_of = {bc: i for i, bc in enumerate(steph_cells)}
    steph_mask = dataset == "stephenson"
    tgt = np.where(steph_mask)[0]
    src = np.array([row_of[obs_names[i]] for i in tgt])
    prot_raw[tgt] = steph_sub[src]
    filled[tgt] = True
    print(f"    Steph: filled {steph_mask.sum()} cells; count max={steph_sub.max():.0f} int_frac={np.mean(steph_sub==np.round(steph_sub)):.3f}", flush=True)
    del steph_sub; gc.collect()

    assert filled.all(), f"{(~filled).sum()} cells unfilled!"
    print(f"[4] protein raw counts: shape={prot_raw.shape} min={prot_raw.min():.0f} max={prot_raw.max():.0f} "
          f"int_frac={np.mean(prot_raw==np.round(prot_raw)):.4f}", flush=True)

    # ---------- raw RNA counts ----------
    print("[5] load combined_v4 RNA raw counts (layers['counts'])...", flush=True)
    ar = sc.read_h5ad(f"{CV4}/rna_combined.h5ad")
    assert list(ar.obs_names) == list(obs_names), "RNA/protein obs order mismatch"
    Xr = ar.layers["counts"]
    Xr = sp.csr_matrix(Xr) if not sp.issparse(Xr) else Xr.tocsr()
    Xr = Xr.astype(np.float32)
    intfrac = float(np.mean(Xr.data == np.round(Xr.data))) if Xr.data.size else 1.0
    print(f"    RNA counts: shape={Xr.shape} max={Xr.max():.0f} int_frac={intfrac:.4f}", flush=True)

    # ---------- assemble ----------
    print("[6] assemble + write totalVI AnnData...", flush=True)
    obs = pd.DataFrame({"donor": donor, "dataset": dataset, "celltype": celltype}, index=obs_names)
    for c in obs.columns: obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=Xr, obs=obs, var=pd.DataFrame(index=list(ar.var_names)))
    adata.layers["counts"] = Xr.copy()
    adata.obsm["protein_expression"] = pd.DataFrame(prot_raw, index=obs_names, columns=panel)
    adata.write_h5ad(OUT)
    print(f"[DONE] wrote {OUT}\n    X {adata.shape}  protein_expression {adata.obsm['protein_expression'].shape}", flush=True)

if __name__ == "__main__":
    main()
