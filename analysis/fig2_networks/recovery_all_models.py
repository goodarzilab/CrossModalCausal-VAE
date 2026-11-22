"""GRN recovery (AUROC) of projected G=D_r·A_rr·D_r^T vs CollecTRI + DoRothEA A/B (and A/B/C) for
ALL sweep models. Reuses ref genes + CollecTRI mask from ref_nets_v3_2c.npz; builds DoRothEA A/B fresh.
Usage: python analysis/fig2_networks/recovery_all_models.py"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json,os,sys,warnings,glob,numpy as np,torch,scanpy as sc
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from model import CrossModalCausalLDVAE
from sklearn.metrics import roc_auc_score
R=f"{_CMC_ROOT}"
ref=np.load(f"{_CMC_RESULTS}/ref_nets_v3_2c.npz",allow_pickle=True)
rgenes=list(map(str,ref["genes"])); ridx={g:i for i,g in enumerate(rgenes)}; NG=len(rgenes)
collectri=ref["CollecTRI"].astype(bool); dorothea_abc=ref["DoRothEA"].astype(bool)
# build DoRothEA A/B mask on the same ref-gene axis
import decoupler as dc
do=dc.op.dorothea(organism="human"); ab=do[do["confidence"].isin(["A","B"])]
M_ab=np.zeros((NG,NG),bool)
for s,t in zip(ab["source"].astype(str),ab["target"].astype(str)):
    if s in ridx and t in ridx: M_ab[ridx[s],ridx[t]]=True
print(f"ref genes={NG} | CollecTRI edges={collectri.sum()} DoRothEA A/B edges={M_ab.sum()} A/B/C={dorothea_abc.sum()}",flush=True)
REFS={"CollecTRI":collectri,"DoRothEA_AB":M_ab,"DoRothEA_ABC":dorothea_abc}  # all directed

def score_model(RD):
    c=json.load(open(f"{RD}/config.json"))
    ck=f"{RD}/final_model.pt"
    if not os.path.exists(ck): return None
    sd=torch.load(ck,map_location="cpu")
    ng,npr=sd["dec_rna.weight"].shape[0],sd["dec_prot.weight"].shape[0]
    m=CrossModalCausalLDVAE(ng,npr,1,latent_dim_rna=c["latent_dim_rna"],latent_dim_prot=c.get("latent_dim_prot",64),
        recon_likelihood=c.get("recon_likelihood","mse"),scm_nonlinear=c.get("scm_nonlinear",False),
        phi=c.get("phi","tanh"),nonlin_gain=c.get("nonlin_gain",1.0),fp_damping=c.get("fp_damping",0.0),
        scm_mechanism=c.get("scm_mechanism","elementwise"),mlp_hidden=c.get("mlp_hidden",8))
    m.load_state_dict(sd,strict=False); m.eval()
    with torch.no_grad():
        A_rr,_,_,_=m.scm.get_block_adjacencies(tau=0.5); Dr=m.dec_rna.weight.detach().float()
        G=(Dr@A_rr.float()@Dr.T).cpu().numpy()
    mg=list(sc.read_h5ad(c["rna_path"],backed="r").var_names.astype(str)); gi={g:i for i,g in enumerate(mg)}
    common=[g for g in rgenes if g in gi]; ci=[gi[g] for g in common]; ri=[ridx[g] for g in common]
    Gs=np.abs(G[np.ix_(ci,ci)]); np.fill_diagonal(Gs,0); off=~np.eye(len(common),dtype=bool)
    out={"n_common":len(common)}
    for name,M in REFS.items():
        lab=M[np.ix_(ri,ri)][off]; s=Gs[off]
        out[name]=float(roc_auc_score(lab,s)) if 0<lab.sum()<len(lab) else float("nan")
    return out

MODELS=[]
for d in sorted(glob.glob(f"{_CMC_RESULTS}/combo_*_nodag")): MODELS.append((os.path.basename(d)[6:-6],d))
for s in ["healthy","disease","pooled"]: MODELS.append((f"*{s}",f"{_CMC_RESULTS}/multi_{s}_nodag"))
MODELS.append(("2cohort",f"{_CMC_RESULTS}/linfb_2c_nodag_v4")); MODELS.append(("COMBAT-only",f"{_CMC_RESULTS}/combat_only_nodag"))
res={}
for tag,RD in MODELS:
    try:
        o=score_model(RD)
        if o: res[tag]=o; print(f"[{tag:14s}] CollecTRI={o['CollecTRI']:.3f} DoRothEA_AB={o['DoRothEA_AB']:.3f} DoRothEA_ABC={o['DoRothEA_ABC']:.3f} (n={o['n_common']})",flush=True)
        else: print(f"[{tag:14s}] no checkpoint",flush=True)
    except Exception as e: print(f"[{tag:14s}] ERR {e}",flush=True)
json.dump(res,open(f"{_CMC_RESULTS}/recovery_all_models.json","w"),indent=2)
print("WROTE results/recovery_all_models.json",flush=True)
