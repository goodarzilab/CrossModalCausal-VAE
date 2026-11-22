#!/usr/bin/env python
"""vocab v4 — leaner, audit-driven fixes over v3 (see results/vocab_audit.json):
  * DROP mitochondrial (MT-) and cytoplasmic ribosomal (RP[SL]\\d) genes  (inflate coexpr shortcut, track QC)
  * TFs: keep only CollecTRI TFs that are HVG OR detected >= DET_TF (default 5%)  -> removes the ~rare
    forced-in TFs that diluted GRN recovery (v3 forced ALL 844 detected TFs; original 2kHVG had ~59 strong TFs)
  * RBPs: keep only RBP ∩ HVG (highly-variable RBPs) -> trims the 1162 RBP-only block
  * TARGETS (Frangieh ∪ Chen): KEEP ALL present (perturbation eval needs them, regardless of expression)
  * HVG = seurat_v3 (2000, batch-aware batch_key='dataset') on raw counts  (same as v3)
  RNA X = log1p(normalize_total 1e4); raw counts in layers['counts'].
Cohorts:  2c = Hao+Stephenson ;  2c_combat = +COMBAT.  Protein identical to v3.
Usage: python build_vocab_v4.py {2c|2c_combat} [DET_TF]
Prints composition at several DET_TF thresholds for confirmation (does NOT write until chosen threshold)."""
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
COHORT=sys.argv[1]; DET_TF=float(sys.argv[2]) if len(sys.argv)>2 else 0.05; N_HVG=2000
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

print(f"[{COHORT}] loading RNA raw counts...", flush=True)
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
adata=ad.concat([p[:,common].copy() for p in parts], merge="same"); del parts; gc.collect()
adata.X=csr_matrix(adata.X)
sc.pp.filter_genes(adata, min_cells=3)
print(f"[{COHORT}] after min_cells=3: {adata.shape}", flush=True)
adata.layers["counts"]=adata.X.copy()
# detection (fraction cells>0) on raw counts
det=np.asarray((adata.layers["counts"]>0).mean(0)).ravel()
detmap=dict(zip(adata.var_names.astype(str), det))
# HVG seurat_v3 batch-aware
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3", batch_key="dataset")
hvg=set(adata.var_names[adata.var.highly_variable].astype(str))
present=set(adata.var_names.astype(str))

# targets (keep ALL present)
iccite=set(pd.read_csv(ICCITE_MD,index_col=0)["PerturbedGene"].astype(str))-{"NTC","control"}
fr=ad.read_h5ad(FRANGIEH,backed="r"); frset=set()
for s in pd.unique(fr.obs["sgRNA"].astype(str)):
    for g in str(s).split(","):
        g=g.strip()
        if not g or any(k in g.upper() for k in ["NON-GENE","NO_SITE","NON_TARGET","SCRAMBLE","CONTROL"]): continue
        m=re.match(r"^(.+)_(\d+)$",g); frset.add(m.group(1) if m else g)
fr.file.close()
# Arce/Marson (GSE278572) perturbation targets — force-include so all Arce perts are testable
ARCE_PS=f"{HERE}/arce_perturb_citeseq/GSE278572_protospacer_calls_per_cell.csv.gz"
def _arce_tgt(fc):
    g=str(fc); m=re.match(r"^(.+?)_\d+_CRISPRi$",g) or re.match(r"^(.+?)_\d+$",g); return m.group(1) if m else g
_aps=pd.read_csv(ARCE_PS); arce_set=set(_arce_tgt(x) for x in _aps[_aps.num_features==1]["feature_call"])-{"Non-Targeting"}
# TARGET QC FLOOR: force-include a perturbation target ONLY if detected >= DET_TGT in training data
# (a target the model can't represent gives a chance-level do-op AND dilutes the vocab). Report evaluable/dropped.
DET_TGT=DET_TF   # 5% (same as TFs)
def _cov(name,s):
    sp={g for g in s if g in present}
    ev={g for g in sp if detmap[g]>=DET_TGT}
    print(f"    {name}: total={len(s)} present(min_cells3)={len(sp)} EVALUABLE(det>={DET_TGT:.0%})={len(ev)} "
          f"dropped_lowsignal={len(sp)-len(ev)} (+{len(s)-len(sp)} absent)", flush=True)
    return ev
print(f"[{COHORT}] === PERTURBATION TARGET coverage (force-include only if training-detection >= {DET_TGT:.0%}) ===", flush=True)
tg_fr=_cov("Frangieh",frset); tg_ic=_cov("iccite",iccite); tg_ar=_cov("Arce",arce_set)
targets=tg_fr|tg_ic|tg_ar
print(f"[{COHORT}] evaluable target union = {len(targets)}", flush=True)
import decoupler as dc
tfs_all=set(dc.op.collectri(organism="human")["source"].astype(str).unique())&present
rbps_all=set(l.strip() for l in open(f"{HERE}/rbp_go0003723.txt"))&present

def is_junk(g): return bool(g.startswith("MT-") or re.match(r"^RP[SL]\d", g))

# ---- report composition at candidate DET_TF thresholds (for confirmation) ----
print(f"\n[{COHORT}] === composition at candidate DET_TF thresholds (RBP=HVG∩RBP, drop mito/ribo, keep all targets) ===")
hvg_clean={g for g in hvg if not is_junk(g)}
rbps_hvg={g for g in (rbps_all&hvg) if not is_junk(g)}
for thr in [0.03,0.05,0.10]:
    tfs={g for g in tfs_all if (g in hvg or detmap[g]>=thr) and not is_junk(g)}
    V=hvg_clean|tfs|rbps_hvg|targets
    print(f"  DET_TF={thr:.2f}: |V|={len(V)}  (HVG_clean={len(hvg_clean)}, TF={len(tfs)}, RBP_hvg={len(rbps_hvg)}, targets={len(targets)}, "
          f"rare<1%={sum(detmap[g]<0.01 for g in V)}, mito={sum(g.startswith('MT-') for g in V)}, ribo={sum(bool(re.match(r'^RP[SL]\\d',g)) for g in V)})", flush=True)

# ---- build with chosen DET_TF ----
tfs={g for g in tfs_all if (g in hvg or detmap[g]>=DET_TF) and not is_junk(g)}
V=sorted(hvg_clean|tfs|rbps_hvg|targets)
print(f"\n[{COHORT}] CHOSEN DET_TF={DET_TF} -> |V|={len(V)} | HVG_clean={len(hvg_clean)} TF={len(tfs)} RBP_hvg={len(rbps_hvg)} targets={len(targets)} "
      f"| rare<1%={sum(detmap[g]<0.01 for g in V)} mito={sum(g.startswith('MT-') for g in V)} ribo={sum(bool(re.match(r'^RP[SL]\\d',g)) for g in V)} "
      f"| evaluable_targets={len(targets)} (Fr {len(tg_fr)}, iccite {len(tg_ic)}, Arce {len(tg_ar)})", flush=True)

OUT=f"{HERE}/combined_v4" if COHORT=="2c" else f"{HERE}/combined_combat_v4"
os.makedirs(OUT,exist_ok=True)
adata=adata[:,V].copy()
sc.pp.normalize_total(adata,target_sum=1e4); sc.pp.log1p(adata)
adata.write(f"{OUT}/rna_combined.h5ad")
pd.Series(V).to_csv(f"{OUT}/vocab_genes.csv",index=False,header=["gene"])
print(f"[{COHORT}] wrote {OUT}/rna_combined.h5ad {adata.shape}", flush=True)

# protein (identical to v3)
dst=f"{OUT}/citeseq_combined.h5ad"
if COHORT=="2c":
    src=f"{HERE}/combined_v3/citeseq_combined.h5ad"
    if not os.path.exists(dst): os.symlink(os.path.realpath(src),dst)
    print(f"[{COHORT}] protein -> symlink v3 panel (unchanged)", flush=True)
else:
    src=f"{HERE}/combined_combat_v3/citeseq_combined.h5ad"
    if not os.path.exists(dst): os.symlink(os.path.realpath(src),dst)
    print(f"[{COHORT}] protein -> symlink v3 combat panel (unchanged)", flush=True)
print(f"DONE [{COHORT}] DET_TF={DET_TF}", flush=True)
