#!/usr/bin/env python
"""Rebuild iccite_prot.h5ad with muon.prot.pp.clr (matches training normalization).
The earlier build used a hand-rolled log1p-mean CLR -> scale mismatch (train CLR [0,10.3] vs
iccite [-1.46,6.95]). This rewrites it with the SAME muon CLR the training data used."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import re, numpy as np, scipy.io, scipy.sparse as sp, anndata as ad, pandas as pd, muon
EXP = f"{_CMC_DATA}/chen_perturb_icciteseq/export"
ALIAS = {"PDL1": "CD274", "PDL2": "CD273", "PD1": "CD279", "CTLA4": "CD152", "ICOS": "CD278",
         "TIM3": "CD366", "LAG3": "CD223", "OX40": "CD134", "41BB": "CD137", "CD154": "CD40L",
         "IL7RA": "CD127", "IL2RB": "CD122", "FAS": "CD95", "HLAABC": "HLA-ABC", "HLADR": "HLA-DR"}
M = scipy.io.mmread(f"{EXP}/ADT_counts.mtx").tocsr()                       # features x cells (raw counts)
afeat = [l.strip() for l in open(f"{EXP}/ADT_features.txt")]
bc = [l.strip() for l in open(f"{EXP}/ADT_barcodes.txt")]
clean = [ALIAS.get(re.sub(r"^surface-A\d+-", "", a).upper(), re.sub(r"^surface-A\d+-", "", a)) for a in afeat]
prot = ad.AnnData(X=sp.csr_matrix(M.T).astype(np.float32), obs=pd.DataFrame(index=bc), var=pd.DataFrame(index=clean))
prot.var_names_make_unique()
md = pd.read_csv(f"{EXP}/metadata.csv", index_col=0)
sg = md["PerturbedGene"].astype(str).replace({"NTC": "control"})
prot.obs["sgRNA"] = pd.Series(sg.values, index=md.index).reindex(prot.obs_names).values
muon.prot.pp.clr(prot)                                                     # <-- muon CLR (matches training)
prot.write(f"{EXP}/iccite_prot.h5ad")
print(f"rebuilt iccite_prot.h5ad with muon CLR: {prot.shape}  range [{prot.X.min():.3f},{prot.X.max():.3f}]")
