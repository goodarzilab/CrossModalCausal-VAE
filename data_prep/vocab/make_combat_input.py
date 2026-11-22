"""Build COMBAT-only CITE-seq training input aligned to the v4 vocabulary (2692 genes, 144 proteins),
for a generalization test: train CMC-VAE on COMBAT alone, then run perturbation validation on the
held-out CRISPR cohorts. Subsample cells; X is already log-normalized. ADT names are 'AB_...' -> mapped."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os,sys,h5py,json,numpy as np,scanpy as sc,warnings; warnings.filterwarnings("ignore")
from scipy import sparse
NC=int(os.environ.get("COMBAT_NCELL","120000")); OUT=f"{_CMC_DATA}/combined_COMBAT"
CB=f"{_CMC_DATA}/COMBAT_citeseq/COMBAT-CITESeq-DATA.h5ad"
v4=json.load(open(f"{_CMC_RESULTS}/v4vocab.json")); vg=v4["genes"]; vp=v4["prots"]
def npr(s): return str(s).upper().replace("AB_","").replace("HUMAN","").replace("-","").replace("_","").replace(".","")
f=h5py.File(CB,"r")
cats=[x.decode() for x in f["var/__categories/feature_types"][:]]; codes=f["var/feature_types"][:]
vn=[x.decode() for x in f["var/_index"][:]]; ge_c=cats.index("Gene Expression"); ab_c=cats.index("Antibody Capture")
gene_idx={vn[i]:i for i in range(len(vn)) if codes[i]==ge_c}
adt={npr(vn[i]):i for i in range(len(vn)) if codes[i]==ab_c}
# column indices into COMBAT for each v4 gene / protein (or -1 if absent)
gcol=np.array([gene_idx.get(g,-1) for g in vg]); pmap={npr(p):p for p in vp}; pcol=np.array([adt.get(npr(p),-1) for p in vp])
print(f"v4 gene overlap: {(gcol>=0).sum()}/{len(vg)}   v4 protein overlap (AB_-mapped): {(pcol>=0).sum()}/{len(vp)}",flush=True)
# subsample cells
Ncells=f["X"]["indptr"].shape[0]-1 if isinstance(f["X"],h5py.Group) else f["X"].attrs["shape"][0]
rng=np.random.default_rng(0); idx=np.sort(rng.choice(Ncells,min(NC,Ncells),replace=False))
print(f"subsampling {len(idx)} / {Ncells} cells",flush=True)
ad=sc.read_h5ad(CB,backed="r"); sub=ad[idx].to_memory(); X=sub.X.tocsc() if sparse.issparse(sub.X) else sparse.csc_matrix(sub.X)
def align(cols):
    out=sparse.lil_matrix((X.shape[0],len(cols)),dtype=np.float32)
    for j,c in enumerate(cols):
        if c>=0: out[:,j]=X[:,c]
    return out.tocsr()
Xr=align(gcol); Xp=align(pcol)
obs=[f"combat_{i}" for i in idx]
import anndata as an
rr=an.AnnData(Xr,obs={"cell":obs},var={"gene":vg}); rr.obs_names=obs; rr.var_names=vg; rr.obs["donor"]=sub.obs.get("scRNASeq_sample_ID",sub.obs.index).values if "scRNASeq_sample_ID" in sub.obs else "combat"
pp=an.AnnData(Xp,obs={"cell":obs},var={"prot":vp}); pp.obs_names=obs; pp.var_names=vp; pp.obs["donor"]=rr.obs["donor"].values
os.makedirs(OUT,exist_ok=True); rr.write(f"{OUT}/rna_combined.h5ad"); pp.write(f"{OUT}/citeseq_combined.h5ad")
print(f"WROTE {OUT}/rna_combined.h5ad {rr.shape} + citeseq_combined.h5ad {pp.shape}",flush=True)
