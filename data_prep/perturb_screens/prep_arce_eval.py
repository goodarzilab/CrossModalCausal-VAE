#!/usr/bin/env python
"""Prepare the Arce/Marson Perturb-CITE-seq (GSE278572) as an eval dataset in the SAME format the
perturbation pipeline expects (like perturb_cite_seq_*_processed.h5ad):
  arce_rna.h5ad  : log-norm RNA, obs['sgRNA'] = target gene ('NO_SITE' for Non-Targeting -> parsed to control)
  arce_prot.h5ad : CLR ADT (HTO+isotypes dropped), var = marker names, obs barcodes match RNA
Singlet cells only (1 guide). Writes to the arce dir. Prints #perturbations / #genes / #proteins."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, re, tempfile, warnings
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
warnings.filterwarnings("ignore")
SRC=f"{_CMC_DATA}/arce_perturb_citeseq"
OUTR=f"{SRC}/arce_rna.h5ad"; OUTP=f"{SRC}/arce_prot.h5ad"

print("loading 10x mtx (1B nnz, slow)...", flush=True)
tmp=tempfile.mkdtemp()
for std,pref in [("matrix.mtx.gz","GSE278572_matrix.mtx.gz"),("features.tsv.gz","GSE278572_features.tsv.gz"),
                 ("barcodes.tsv.gz","GSE278572_barcodes.tsv.gz")]:
    os.symlink(f"{SRC}/{pref}", f"{tmp}/{std}")
A=sc.read_10x_mtx(tmp, gex_only=False); A.var_names_make_unique()
ft=A.var["feature_types"].astype(str)
gid=A.var["gene_ids"].astype(str) if "gene_ids" in A.var else pd.Series(A.var_names,index=A.var_names)
is_hto=np.asarray(pd.Series(gid.values).str.startswith("HTO"))
is_iso=np.asarray(pd.Series(A.var_names).str.contains("Isotype",case=False))
is_gex=np.asarray(ft.values=="Gene Expression"); is_adt=np.asarray(ft.values=="Antibody Capture")
rna=A[:, is_gex].copy(); adt=A[:, (is_adt & ~is_hto & ~is_iso)].copy(); del A
print(f"RNA {rna.shape} | ADT {adt.shape}", flush=True)

# ---- perturbation target from protospacer (singlets) ----
ps=pd.read_csv(f"{SRC}/GSE278572_protospacer_calls_per_cell.csv.gz")
ps_s=ps[ps.num_features==1].copy()
def tgt(fc):
    g=str(fc); m=re.match(r"^(.+?)_\d+_CRISPRi$",g) or re.match(r"^(.+?)_\d+$",g)
    return m.group(1) if m else g
ps_s["target"]=ps_s.feature_call.map(tgt)
bc2t=dict(zip(ps_s.cell_barcode, ps_s.target))
tt=np.array([bc2t.get(b,"__none__") for b in rna.obs_names])
keep=tt!="__none__"                        # singlet-guided cells only
rna=rna[keep].copy(); adt=adt[keep].copy(); tt=tt[keep]
# sgRNA obs: Non-Targeting -> NO_SITE (parses to control); real target gene stays as-is
sg=np.where(tt=="Non-Targeting","NO_SITE",tt)
rna.obs["sgRNA"]=sg; adt.obs["sgRNA"]=sg
print(f"singlet cells: {rna.n_obs} | targets(incl NTC): {len(set(tt))}", flush=True)

# ---- RNA log-norm (match processed perturb files) ----
rna.layers["counts"]=rna.X.copy()
sc.pp.normalize_total(rna,target_sum=1e4); sc.pp.log1p(rna)
# ---- clean ADT names (strip 'Hu.'/'Mu.' species prefix + '_clone' suffix) to match training panel ----
def _clean_adt(n):
    n=re.sub(r'^(Hu|Mu|Hum|Mou)\.','',n); return n.split('_')[0]
_new=[_clean_adt(x) for x in adt.var_names.astype(str)]
_seen={}; _final=[]
for x in _new:
    if x in _seen: _seen[x]+=1; _final.append(f"{x}.{_seen[x]}")
    else: _seen[x]=0; _final.append(x)
adt.var_names=_final
# ---- ADT CLR ; var=marker names ----
Ax=adt.X.toarray() if hasattr(adt.X,"toarray") else np.asarray(adt.X)
adt.X=(np.log1p(Ax)-np.log1p(Ax).mean(1,keepdims=True)).astype(np.float32)

rna.write(OUTR); adt.write(OUTP)
print(f"WROTE {OUTR} {rna.shape}\nWROTE {OUTP} {adt.shape}", flush=True)
# summary numbers
ntc=(tt=="Non-Targeting").sum(); ntargets=len(set(tt))-(1 if "Non-Targeting" in set(tt) else 0)
print(f"ARCE_EVAL_READY: perturbations(gene targets)={ntargets} | NTC_cells={ntc} | "
      f"RNA_genes={rna.n_vars} | ADT_proteins={adt.n_vars} | total_singlet_cells={rna.n_obs}", flush=True)
print("PREP_ARCE_DONE", flush=True)
