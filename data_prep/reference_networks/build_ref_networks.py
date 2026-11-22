"""Build binary reference-network masks aligned to a v3 vocab, for projected-GRN recovery AUROC.
References: CollecTRI (directed TF->target), DoRothEA A/B/C (directed), coexpression (symmetric,
from the coexpr prior), STRING high-confidence (symmetric, API; skipped if unreachable).
Run on the LOGIN node (needs internet). Usage: python data_prep/reference_networks/build_ref_networks.py {2c|2ccombat}
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
import os, sys, json, io
import numpy as np, pandas as pd, torch
DATA=f"{_CMC_DATA}"
PRIOR=f"{_CMC_ROOT}/priors"; RES=f"{_CMC_RESULTS}"
c=sys.argv[1]; _VV=os.environ.get("CMC_VOCAB","v3"); _vd="v4" if _VV=="v4" else "v3"; _vt="4" if _VV=="v4" else "3"
ddir=f"{DATA}/combined_{_vd}" if c=="2c" else f"{DATA}/combined_combat_{_vd}"
tag=f"vocab{_vt}_{c}" if c=="2c" else f"vocab{_vt}_2ccombat"
genes=list(pd.read_csv(f"{ddir}/vocab_genes.csv")["gene"].astype(str)); idx={g:i for i,g in enumerate(genes)}; n=len(genes)
print(f"[{c}] vocab {n} genes", flush=True)
masks={}; meta={}

def add_directed(name, df):
    m=np.zeros((n,n),dtype=bool)
    for s,t in zip(df["source"].astype(str),df["target"].astype(str)):
        if s in idx and t in idx and s!=t: m[idx[s],idx[t]]=True
    masks[name]=m; meta[name]={"n_edges":int(m.sum()),"directed":True}; print(f"  {name}: {int(m.sum())} edges", flush=True)

def add_symmetric(name, pairs):
    m=np.zeros((n,n),dtype=bool)
    for a,b in pairs:
        if a in idx and b in idx and a!=b: m[idx[a],idx[b]]=True; m[idx[b],idx[a]]=True
    masks[name]=m; meta[name]={"n_edges":int(m.sum()//2),"directed":False}; print(f"  {name}: {int(m.sum()//2)} undirected edges", flush=True)

# CollecTRI (from training prior pt -> exact index alignment guaranteed via gene names)
d=torch.load(f"{PRIOR}/rna_hvg_priors_collectri_sc_{tag}.pt",map_location="cpu")
pm=d["prior_mask"]; pm=pm.numpy() if torch.is_tensor(pm) else np.asarray(pm); gn=list(map(str,d["gene_names"]))
# reorder prior mask to vocab gene order
gi2={g:i for i,g in enumerate(gn)}; perm=[gi2[g] for g in genes]
masks["CollecTRI"]=pm[np.ix_(perm,perm)].astype(bool); meta["CollecTRI"]={"n_edges":int(masks["CollecTRI"].sum()),"directed":True,"note":"training prior for prior-on"}
print(f"  CollecTRI: {int(masks['CollecTRI'].sum())} edges", flush=True)

# DoRothEA A/B/C (independent, decoupler)
import decoupler as dc
do=dc.op.dorothea(organism="human")
add_directed("DoRothEA", do[do["confidence"].isin(["A","B","C"])])

# coexpression (from coexpr prior pt; symmetric)
dc_=torch.load(f"{PRIOR}/rna_coexpr_prior_{tag}.pt",map_location="cpu")
cpm=dc_["prior_mask"]; cpm=cpm.numpy() if torch.is_tensor(cpm) else np.asarray(cpm); cgn=list(map(str,dc_["gene_names"]))
cgi={g:i for i,g in enumerate(cgn)}; cperm=[cgi[g] for g in genes]
masks["coexpr"]=cpm[np.ix_(cperm,cperm)].astype(bool); meta["coexpr"]={"n_edges":int(masks["coexpr"].sum()//2),"directed":False,"note":"training prior for coexpr models"}
print(f"  coexpr: {int(masks['coexpr'].sum())} edges (may be directed-stored)", flush=True)

# STRING high-confidence (API; symmetric; skip on failure)
try:
    import requests
    edges=set()
    for i in range(0,n,1900):
        chunk=genes[i:i+1900]
        r=requests.post("https://string-db.org/api/tsv/network",
                        data={"identifiers":"%0d".join(chunk),"species":9606,"required_score":700},timeout=90)
        r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text),sep="\t")
        for a,b in zip(df["preferredName_A"].astype(str),df["preferredName_B"].astype(str)): edges.add((a,b))
    add_symmetric("STRING",edges)
except Exception as e:
    print(f"  STRING: SKIPPED ({str(e)[:120]})",flush=True)

np.savez_compressed(f"{RES}/ref_nets_{_VV}_{c}.npz", genes=np.array(genes), **{k:v for k,v in masks.items()})
json.dump(meta,open(f"{RES}/ref_nets_{_VV}_{c}.json","w"),indent=2)
print(f"[{c}] saved {list(masks)} -> {RES}/ref_nets_v3_{c}.npz\nDONE",flush=True)
