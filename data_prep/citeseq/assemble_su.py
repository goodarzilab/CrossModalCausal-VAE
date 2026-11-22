#!/usr/bin/env python
"""Assemble Su et al 2020 (E-MTAB-9357) 270 dense TSV.gz libraries -> su_rna_raw.h5ad + su_prot_raw.h5ad
with recovered RAW integer counts (inverse of per-cell CPM(1e6)+log1p) + SDRF disease-status join.
Inversion per cell (row): unit=expm1(min nonzero); counts=round(expm1(v)/unit).  Usage: assemble_su.py [LIMIT]"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os,sys,glob,gzip,numpy as np,pandas as pd,scanpy as sc,anndata as ad
from scipy.sparse import csr_matrix,vstack
HERE=f"{_CMC_DATA}/su_citeseq"
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 0   # 0 = all
SDRF=pd.read_csv(f"{HERE}/meta/E-MTAB-9357.sdrf.txt",sep="\t",dtype=str)
gexcol=[c for c in SDRF.columns if c=="Derived Array Data File"][0]  # col29 = gex filename
ind="Characteristics[individual]"; dis="Characteristics[disease]"; tp="Characteristics[sampling time point]"
fmap={os.path.basename(str(r[gexcol])):(str(r[ind]),str(r[dis]),str(r[tp])) for _,r in SDRF.iterrows()}
def status(d): return "healthy" if str(d).strip().lower()=="normal" else "disease"

def invert_counts(df):   # df: cells x features, log1p(CPM). returns csr of raw counts.
    M=csr_matrix(df.values.astype(np.float32))
    if M.data.size==0: return M
    ex=np.expm1(M.data.astype(np.float64))
    nnz=np.diff(M.indptr)
    if (nnz==0).any():   # guard empty rows for reduceat
        keep=nnz>0
    unit=np.minimum.reduceat(ex,M.indptr[:-1]) if (nnz>0).all() else None
    if unit is None:     # fallback: per-row loop for the rare empty row
        unit=np.ones(M.shape[0]); 
        for r in range(M.shape[0]):
            a,b=M.indptr[r],M.indptr[r+1]
            if b>a: unit[r]=ex[a:b].min()
    counts=np.round(ex/np.repeat(unit,nnz)).astype(np.float32)
    return csr_matrix((counts,M.indices,M.indptr),shape=M.shape)

gex_files=sorted(glob.glob(f"{HERE}/data/*_gex_library_*.txt.gz"))
if LIMIT: gex_files=gex_files[:LIMIT]
print(f"assembling {len(gex_files)} libraries (LIMIT={LIMIT})",flush=True)
ref_genes=ref_prots=None; Xr=[];Xp=[];obs=[]
for i,gf in enumerate(gex_files):
    bn=os.path.basename(gf); tok=bn.split("_gex_library_")[1].replace(".txt.gz","")
    pf=gf.replace("_gex_library_","_pro_library_")
    if bn not in fmap or not os.path.exists(pf): print(f"  SKIP {bn} (no SDRF/pro match)",flush=True); continue
    dg=pd.read_csv(gf,sep="\t",index_col=0,engine="c").astype(np.float32)
    dp=pd.read_csv(pf,sep="\t",index_col=0,engine="c").astype(np.float32)
    if ref_genes is None: ref_genes=list(dg.columns); ref_prots=list(dp.columns)
    else:
        if list(dg.columns)!=ref_genes: dg=dg.reindex(columns=ref_genes,fill_value=0.0)
        if list(dp.columns)!=ref_prots: dp=dp.reindex(columns=ref_prots,fill_value=0.0)
    dg=dg.loc[dp.index.intersection(dg.index)]; dp=dp.loc[dg.index]   # align cells (should be identical)
    Xr.append(invert_counts(dg)); Xp.append(invert_counts(dp))
    donor,d,t=fmap[bn]
    for bc in dg.index: obs.append((bc,donor,status(d),d,t,tok))
    if (i+1)%25==0 or i==len(gex_files)-1: print(f"  [{i+1}/{len(gex_files)}] {bn} cells={dg.shape[0]}",flush=True)
O=pd.DataFrame(obs,columns=["barcode","donor","status","disease_raw","timepoint","library"]).set_index("barcode")
O["dataset"]="su"
rna=ad.AnnData(X=vstack(Xr).tocsr(),obs=O.copy(),var=pd.DataFrame(index=ref_genes)); rna.var_names_make_unique()
prot=ad.AnnData(X=vstack(Xp).tocsr(),obs=O.copy(),var=pd.DataFrame(index=ref_prots)); prot.var_names_make_unique()
# integrality QC
s=rna.X.data[:100000]; err=np.abs(s-np.round(s)).max() if s.size else 0.0
print(f"RNA {rna.shape} PROT {prot.shape} | status {O['status'].value_counts().to_dict()} | count-integrality max-err={err:.4g}",flush=True)
if LIMIT==0:
    rna.write(f"{HERE}/su_rna_raw.h5ad"); prot.write(f"{HERE}/su_prot_raw.h5ad")
    print(f"WROTE {HERE}/su_rna_raw.h5ad + su_prot_raw.h5ad",flush=True)
else:
    print("DRY-RUN (no write)",flush=True)
