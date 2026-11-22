#!/usr/bin/env python
"""
Build Perturb-icCITE-seq (Chen/Wheeler FOXP3) into perturb-eval-ready h5ads, matching the
Frangieh perturb data shape so the existing perturb_validation_analysis pipeline works unchanged:
  - RNA: cells x genes, log-normalized (normalize_total 1e4 + log1p) to match training space.
          obs['sgRNA'] = PerturbedGene (NTC -> 'control' so parse_sgRNA_to_gene flags controls).
  - ADT: cells x proteins, CLR-normalized; var_names cleaned ('surface-A0007-PDL1' -> 'CD274' etc.)
          so build_protein_name_map matches the model's 144 panel.
Writes iccite_rna.h5ad / iccite_prot.h5ad into the export dir.
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
import re, numpy as np, scipy.io, scipy.sparse as sp, anndata as ad, pandas as pd, scanpy as sc

EXP = f"{_CMC_DATA}/chen_perturb_icciteseq/export"

# iccite-cleaned-name -> model-panel-name aliases (model uses CDxxx; iccite uses common names)
ALIAS = {"PDL1": "CD274", "PDL2": "CD273", "PD1": "CD279", "CTLA4": "CD152", "ICOS": "CD278",
         "TIM3": "CD366", "LAG3": "CD223", "OX40": "CD134", "41BB": "CD137", "CD154": "CD40L",
         "IL7RA": "CD127", "IL2RB": "CD122", "FAS": "CD95", "HLAABC": "HLA-ABC", "HLADR": "HLA-DR"}


def load_mtx(prefix):
    M = scipy.io.mmread(f"{EXP}/{prefix}_counts.mtx").tocsr()          # features x cells
    feats = [l.strip() for l in open(f"{EXP}/{prefix}_features.txt")]
    bcs = [l.strip() for l in open(f"{EXP}/{prefix}_barcodes.txt")]
    return sp.csr_matrix(M.T), feats, bcs                              # -> cells x features


def main():
    md = pd.read_csv(f"{EXP}/metadata.csv", index_col=0)
    sg = md["PerturbedGene"].astype(str).replace({"NTC": "control"}).values   # NTC -> control

    # --- RNA ---
    Xr, gfeat, bc_r = load_mtx("RNA")
    rna = ad.AnnData(X=Xr, obs=pd.DataFrame(index=bc_r), var=pd.DataFrame(index=gfeat))
    rna = rna[md.index.intersection(rna.obs_names)].copy() if not md.index.equals(rna.obs_names) else rna
    rna.obs["sgRNA"] = pd.Series(sg, index=md.index).reindex(rna.obs_names).values
    sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
    rna.write(f"{EXP}/iccite_rna.h5ad")
    print(f"RNA: {rna.shape}  controls={int((rna.obs['sgRNA']=='control').sum())}  perturbed_genes={rna.obs['sgRNA'].nunique()-1}")

    # --- ADT (surface protein) ---
    Xp, afeat, bc_p = load_mtx("ADT")
    clean = []
    for a in afeat:
        nm = re.sub(r"^surface-A\d+-", "", a)
        clean.append(ALIAS.get(nm.upper(), nm))
    prot = ad.AnnData(X=Xp.astype(np.float32), obs=pd.DataFrame(index=bc_p), var=pd.DataFrame(index=clean))
    prot.var_names_make_unique()
    prot.obs["sgRNA"] = pd.Series(sg, index=md.index).reindex(prot.obs_names).values
    # CLR per cell (standard ADT normalization)
    X = prot.X.toarray() if sp.issparse(prot.X) else np.asarray(prot.X)
    logX = np.log1p(X)
    prot.X = logX - logX.mean(axis=1, keepdims=True)                  # CLR
    prot.write(f"{EXP}/iccite_prot.h5ad")
    print(f"ADT: {prot.shape}  (cleaned names e.g. {', '.join(clean[:10])})")
    print("DONE")


if __name__ == "__main__":
    main()
