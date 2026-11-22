#!/usr/bin/env python
"""vocab v3 — PROPER vocab per best practices (fixes the v2 mistakes):
  * gene filter = min_cells=3 (NOT >=10% detection, which was ~100x too aggressive)
  * HVG = seurat_v3 (variance-stabilizing on RAW counts) + BATCH-AWARE (batch_key='dataset'),
    NOT the dispersion 'seurat' flavor that over-selected rare genes (old vocab median detection 0.6%)
  * V = HVG(2000, seurat_v3, batch-aware) ∪ (perturbation targets: Frangieh ∪ Chen, forced-in)
        ∪ (CollecTRI TFs ∩ present) ∪ (RBPs ∩ present)   -- all after min_cells=3 only
  * RNA X kept as log1p(normalize_total 1e4) (published pipeline); raw counts in layers['counts'].
Cohorts:  2c = Hao+Stephenson ;  2c_combat = Hao+Stephenson+COMBAT
Protein:  2c -> symlink existing combined/citeseq_combined.h5ad (144 panel unchanged)
          2c_combat -> harmonize COMBAT 192-ADT to the panel, intersect (~134), CLR, concat.
Usage: python build_vocab_v3.py {2c|2c_combat}
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
import os, sys, re, gc
import numpy as np, scanpy as sc, anndata as ad, pandas as pd, h5py
from scipy.sparse import csc_matrix, csr_matrix
HERE=f"{_CMC_DATA}"; sys.path.insert(0,HERE)
COHORT=sys.argv[1]; N_HVG=2000
HAO=f"{HERE}/hao_citeseq/hao_citeseq.h5seurat"; STEPH=f"{HERE}/stephenson_citeseq/haniffa21.processed.h5ad"
COMBAT=f"{HERE}/COMBAT_citeseq/COMBAT-CITESeq-DATA.h5ad"
ICCITE_MD=f"{HERE}/chen_perturb_icciteseq/export/metadata.csv"
FRANGIEH=f"{HERE}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad"

def hao_counts(assay):
    with h5py.File(HAO,"r") as f:
        g=f["assays"][assay]["counts"]; nf=len(f["assays"][assay]["features"][:])
        mat=csr_matrix(csc_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=(nf,len(g["indptr"][:])-1)).T)
        feats=np.array([x.decode() for x in f["assays"][assay]["features"][:]])
        cells=np.array([x.decode() for x in f["cell.names"][:]])
        donor=np.array([x.decode() for x in f["meta.data"]["donor"][:]])
        try: ct=np.array([x.decode() for x in f["meta.data"]["celltype.l1"][:]])
        except Exception: ct=np.array(["NA"]*len(cells))
    return ad.AnnData(X=mat, obs=pd.DataFrame({"donor":["hao_"+d for d in donor],"celltype":ct,"dataset":"hao"},index=cells),
                      var=pd.DataFrame(index=feats))

# ---------- RNA: load RAW counts per cohort ----------
print(f"[{COHORT}] loading RNA raw counts per cohort...", flush=True)
hao=hao_counts("SCT")
st=sc.read_h5ad(STEPH); rna_mask=st.var["feature_types"]=="Gene Expression"
steph=ad.AnnData(X=st[:,st.var_names[rna_mask]].layers["raw"].copy(),
    obs=pd.DataFrame({"donor":["steph_"+str(d) for d in st.obs["patient_id"].values],
        "celltype":(st.obs["initial_clustering"].astype(str).values if "initial_clustering" in st.obs.columns else "NA"),
        "dataset":"stephenson"},index=st.obs_names),
    var=pd.DataFrame(index=st.var_names[rna_mask])); del st; gc.collect()
parts=[hao,steph]
if COHORT=="2c_combat":
    cb=sc.read_h5ad(COMBAT); gmask=cb.var["feature_types"]=="Gene Expression"
    combat_rna=ad.AnnData(X=cb[:,cb.var_names[gmask]].layers["raw"].copy(),
        obs=pd.DataFrame({"donor":["combat_"+str(d) for d in cb.obs_names],
            "celltype":(cb.obs["Annotation_major_subset"].astype(str).values if "Annotation_major_subset" in cb.obs.columns else "NA"),
            "dataset":"combat"},index=cb.obs_names),
        var=pd.DataFrame(index=cb.var_names[gmask].astype(str)))
    parts.append(combat_rna); del cb; gc.collect()
common=sorted(set.intersection(*[set(p.var_names.astype(str)) for p in parts]))
print(f"[{COHORT}] common genes across cohorts: {len(common)}", flush=True)
adata=ad.concat([p[:,common].copy() for p in parts], merge="same"); del parts; gc.collect()
adata.X=csr_matrix(adata.X)
print(f"[{COHORT}] concatenated raw: {adata.shape}", flush=True)

# gene filter: min_cells=3 (standard, permissive)
sc.pp.filter_genes(adata, min_cells=3)
print(f"[{COHORT}] after min_cells=3: {adata.shape}", flush=True)

# HVG: seurat_v3 (raw counts), batch-aware; fallback to detection-floored seurat if skmisc missing
adata.layers["counts"]=adata.X.copy()
try:
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3", batch_key="dataset")
    hvg=set(adata.var_names[adata.var.highly_variable].astype(str)); how="seurat_v3 batch-aware"
except Exception as e:
    print(f"[{COHORT}] seurat_v3 failed ({e}); fallback seurat on detected>=1%", flush=True)
    det=np.asarray((adata.X>0).mean(0)).ravel(); keep=det>=0.01
    tmp=adata[:,keep].copy(); sc.pp.normalize_total(tmp,target_sum=1e4); sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=N_HVG, batch_key="dataset")
    hvg=set(tmp.var_names[tmp.var.highly_variable].astype(str)); how="seurat(det>=1%) fallback"; del tmp
present=set(adata.var_names.astype(str))

# perturbation targets (Frangieh ∪ Chen), forced in (only require min_cells=3 presence)
iccite=set(pd.read_csv(ICCITE_MD,index_col=0)["PerturbedGene"].astype(str))-{"NTC","control"}
fr=ad.read_h5ad(FRANGIEH,backed="r"); frset=set()
for s in pd.unique(fr.obs["sgRNA"].astype(str)):
    for g in str(s).split(","):
        g=g.strip()
        if not g or any(k in g.upper() for k in ["NON-GENE","NO_SITE","NON_TARGET","SCRAMBLE","CONTROL"]): continue
        m=re.match(r"^(.+)_(\d+)$",g); frset.add(m.group(1) if m else g)
fr.file.close()
targets=(iccite|frset)&present
import decoupler as dc
tfs=set(dc.op.collectri(organism="human")["source"].astype(str).unique())&present
rbps=set(l.strip() for l in open(f"{HERE}/rbp_go0003723.txt"))&present
V=sorted(hvg|targets|tfs|rbps)
print(f"[{COHORT}] HVG({how})={len(hvg)} | targets∩={len(targets)} (Fr {len(frset&present)}/{len(frset)}, "
      f"iccite {len(iccite&present)}/{len(iccite)}) | TF∩={len(tfs)} | RBP∩={len(rbps)} ==> |V|={len(V)}", flush=True)

# write RNA (X=log-norm on vocab, counts layer retained)
OUT=f"{HERE}/combined_v3" if COHORT=="2c" else f"{HERE}/combined_combat_v3"
os.makedirs(OUT,exist_ok=True)
adata=adata[:,V].copy()                       # subsets X AND layers['counts'] to the vocab
sc.pp.normalize_total(adata,target_sum=1e4); sc.pp.log1p(adata)   # X -> log-norm; layers['counts'] stays raw
adata.write(f"{OUT}/rna_combined.h5ad")
pd.Series(V).to_csv(f"{OUT}/vocab_genes.csv",index=False,header=["gene"])
print(f"[{COHORT}] wrote {OUT}/rna_combined.h5ad {adata.shape}", flush=True)

# ---------- protein ----------
dst=f"{OUT}/citeseq_combined.h5ad"
if COHORT=="2c":
    src=f"{HERE}/combined/citeseq_combined.h5ad"
    if not os.path.exists(dst): os.symlink(src,dst)
    print(f"[{COHORT}] protein symlink -> {src} (144 panel unchanged)", flush=True)
else:
    import muon
    def canon(x):
        x=re.sub(r'^AB_','',x); x=re.sub(r'^human','',x); x=re.sub(r'^mouse','',x)
        x=re.sub(r'-\d+$','',x); x=x.split('_')[0]; return x.upper().replace('-','')
    hs=ad.read_h5ad(f"{HERE}/combined/citeseq_combined.h5ad")          # Hao+Steph, 144, CLR
    hs_c={canon(p):p for p in hs.var_names.astype(str)}
    cb=ad.read_h5ad(COMBAT); abmask=cb.var["feature_types"]=="Antibody Capture"
    cbp=ad.AnnData(X=cb[:,cb.var_names[abmask]].layers["raw"].copy(),
        obs=pd.DataFrame({"donor":["combat_"+str(d) for d in cb.obs_names]},index=cb.obs_names),
        var=pd.DataFrame(index=cb.var_names[abmask].astype(str))); del cb; gc.collect()
    cb_c={canon(p):p for p in cbp.var_names.astype(str)}
    shared=sorted(set(hs_c)&set(cb_c))                                 # ~134 canonical
    # subset+rename COMBAT ADT to the panel's original names; CLR-normalize like the others
    cbp=cbp[:,[cb_c[k] for k in shared]].copy(); cbp.var_names=[hs_c[k] for k in shared]
    muon.prot.pp.clr(cbp)
    hs=hs[:,[hs_c[k] for k in shared]].copy()                          # subset Hao+Steph to shared 134
    prot=ad.concat([hs,cbp],join="inner",merge="same")
    prot.write(dst)
    print(f"[{COHORT}] protein 3-way (Hao+Steph+COMBAT) shared panel={len(shared)} -> {prot.shape}", flush=True)
print(f"DONE [{COHORT}]", flush=True)
