#!/usr/bin/env python
"""COMBAT-ONLY vocab, using the v4 procedure (build_vocab_v4.py) but on COMBAT alone + explicit QC.
QC (COMBAT fields): drop cells with QC_ngenes<200, QC_pct_mitochondrial>15, QC_total_UMI<500,
  QC_scrub_doublet_scores>0.25. Genes: min_cells>=3, drop MT-/RP[SL]\\d. HVG=seurat_v3(2000,batch-aware
  batch_key=scRNASeq_sample_ID). Keep CollecTRI TFs (HVG or det>=5%), RBP∩HVG, and force-include ALL
  perturbation targets (Frangieh∪iccite∪Arce) present+det>=5% (so the perturbation eval is testable).
  Protein = COMBAT ADT (192), CLR-normalized, names cleaned (AB_ stripped). RNA X=log1p(norm 1e4);counts in layer.
Usage: python build_vocab_combat.py [DET_TF] [N_SUBSAMPLE]"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os,sys,re,gc,numpy as np,scanpy as sc,anndata as ad,pandas as pd,h5py
from scipy.sparse import csr_matrix
HERE=f"{_CMC_DATA}"; sys.path.insert(0,HERE)
DET_TF=float(sys.argv[1]) if len(sys.argv)>1 else 0.05; NSUB=int(sys.argv[2]) if len(sys.argv)>2 else 150000; N_HVG=2000
COMBAT=f"{HERE}/COMBAT_citeseq/COMBAT-CITESeq-DATA.h5ad"; OUT=f"{HERE}/combat_v4vocab"; os.makedirs(OUT,exist_ok=True)
ICCITE_MD=f"{HERE}/chen_perturb_icciteseq/export/metadata.csv"; FRANGIEH=f"{HERE}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad"
ARCE_PS=f"{HERE}/arce_perturb_citeseq/GSE278572_protospacer_calls_per_cell.csv.gz"

print("loading COMBAT...",flush=True); cb=sc.read_h5ad(COMBAT)
X=cb.layers["raw"] if "raw" in cb.layers else cb.X
gm=(cb.var["feature_types"]=="Gene Expression").values; am=(cb.var["feature_types"]=="Antibody Capture").values
# ---- QC on cells ----
o=cb.obs; keep=np.ones(cb.n_obs,bool)
def col(n): return o[n].astype(float).values if n in o else None
for nm,lo,hi in [("QC_ngenes",200,None),("QC_total_UMI",500,None),("QC_pct_mitochondrial",None,15),("QC_scrub_doublet_scores",None,0.25)]:
    v=col(nm)
    if v is None: continue
    if lo is not None: keep&= (v>=lo)
    if hi is not None: keep&= (v<=hi)
print(f"QC: {keep.sum()}/{cb.n_obs} cells pass",flush=True)
idx=np.where(keep)[0]
if len(idx)>NSUB: idx=np.sort(np.random.default_rng(0).choice(idx,NSUB,replace=False))
rna=ad.AnnData(X=csr_matrix(X[idx][:,gm]),obs=pd.DataFrame({"dataset":"combat",
    "sample":o["scRNASeq_sample_ID"].astype(str).values[idx] if "scRNASeq_sample_ID" in o else "s"},index=cb.obs_names[idx]),
    var=pd.DataFrame(index=cb.var_names[gm].astype(str)))
adt=ad.AnnData(X=csr_matrix(X[idx][:,am]),obs=rna.obs.copy(),var=pd.DataFrame(index=cb.var_names[am].astype(str)))
del cb; gc.collect(); print(f"RNA {rna.shape}  ADT {adt.shape}",flush=True)
sc.pp.filter_genes(rna,min_cells=3); rna.layers["counts"]=rna.X.copy()
det=np.asarray((rna.layers["counts"]>0).mean(0)).ravel(); detmap=dict(zip(rna.var_names.astype(str),det))
def compute_hvg(a):   # robust: batch-aware seurat_v3 -> global seurat_v3 -> manual log-variance top-N (no loess)
    for kw in [dict(flavor="seurat_v3",batch_key="sample"),dict(flavor="seurat_v3")]:
        try:
            b=a.copy(); sc.pp.highly_variable_genes(b,n_top_genes=N_HVG,**kw)
            print(f"HVG via {kw}",flush=True); return set(b.var_names[b.var.highly_variable].astype(str))
        except Exception as e: print(f"HVG {kw} failed: {e}",flush=True)
    tmp=a.copy(); sc.pp.normalize_total(tmp,target_sum=1e4); sc.pp.log1p(tmp); Xl=tmp.X
    m=np.asarray(Xl.mean(0)).ravel(); m2=np.asarray(Xl.multiply(Xl).mean(0)).ravel() if hasattr(Xl,"multiply") else (np.asarray(Xl)**2).mean(0)
    var=m2-m*m; top=np.argsort(-var)[:N_HVG]; print("HVG via manual log-variance top-N",flush=True)
    return set(a.var_names[top].astype(str))
hvg=compute_hvg(rna); present=set(rna.var_names.astype(str))
# perturbation targets
iccite=set(pd.read_csv(ICCITE_MD,index_col=0)["PerturbedGene"].astype(str))-{"NTC","control"}
fr=ad.read_h5ad(FRANGIEH,backed="r"); frset=set()
for s in pd.unique(fr.obs["sgRNA"].astype(str)):
    for g in str(s).split(","):
        g=g.strip()
        if not g or any(k in g.upper() for k in ["NON-GENE","NO_SITE","NON_TARGET","SCRAMBLE","CONTROL"]): continue
        m=re.match(r"^(.+)_(\d+)$",g); frset.add(m.group(1) if m else g)
fr.file.close()
def _at(fc): g=str(fc); m=re.match(r"^(.+?)_\d+_CRISPRi$",g) or re.match(r"^(.+?)_\d+$",g); return m.group(1) if m else g
_aps=pd.read_csv(ARCE_PS); arce_set=set(_at(x) for x in _aps[_aps.num_features==1]["feature_call"])-{"Non-Targeting"}
def cov(s): sp={g for g in s if g in present}; return {g for g in sp if detmap[g]>=DET_TF}
targets=cov(frset)|cov(iccite)|cov(arce_set)
import decoupler as dc
tfs_all=set(dc.op.collectri(organism="human")["source"].astype(str).unique())&present
rbps_all=set(l.strip() for l in open(f"{HERE}/rbp_go0003723.txt"))&present
junk=lambda g: bool(g.startswith("MT-") or re.match(r"^RP[SL]\d",g))
hvg_clean={g for g in hvg if not junk(g)}; rbps_hvg={g for g in (rbps_all&hvg) if not junk(g)}
tfs={g for g in tfs_all if (g in hvg or detmap[g]>=DET_TF) and not junk(g)}
V=sorted(hvg_clean|tfs|rbps_hvg|targets)
print(f"COMBAT vocab |V|={len(V)} | HVG_clean={len(hvg_clean)} TF={len(tfs)} RBP_hvg={len(rbps_hvg)} targets={len(targets)}"
      f" | Fr={len(cov(frset))} iccite={len(cov(iccite))} Arce={len(cov(arce_set))}",flush=True)
rna=rna[:,V].copy(); sc.pp.normalize_total(rna,target_sum=1e4); sc.pp.log1p(rna)
rna.write(f"{OUT}/rna_combined.h5ad"); pd.Series(V).to_csv(f"{OUT}/vocab_genes.csv",index=False,header=["gene"])
# protein: clean ADT names, CLR normalize
adt.var_names=[re.sub(r"^AB_","",str(x)) for x in adt.var_names]
import numpy as _np
Xa=adt.X.toarray() if hasattr(adt.X,"toarray") else _np.asarray(adt.X)
clr=_np.log1p(Xa)-_np.log1p(Xa).mean(1,keepdims=True); adt.X=clr
adt.write(f"{OUT}/citeseq_combined.h5ad")
print(f"WROTE {OUT}: rna {rna.shape}, prot {adt.shape} (CLR)  DONE",flush=True)
