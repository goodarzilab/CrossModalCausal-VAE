"""CAUSAL / INTERPRETABILITY eval suite for all four blocks of one model.
Input: one results dir (config.json + best_model.pt). Runs A(GRN) B(PPI) C(translation) D(feedback)
and writes results/<run>/causal_eval_suite.json. Self-contained (no perturbation data needed).

Projected networks:  G_rr = D_r A_rr D_r^T | G_pp = D_p A_pp D_p^T
                     G_rp = D_r W_rp D_p^T (gene->prot) | G_pr = D_p W_pr D_r^T (prot->gene)
References: CollecTRI + OmniPath from the model's own prior masks (always, no internet); DoRothEA/
STRING/coexpr from cached results/ref_nets_v3_<cohort>.npz if the vocab matches.
Metrics borrowed from CausalGRN (orientation asymmetry, EPR) + DeepSEM (prior-recovery AUROC/AUPRC).
Usage: python analysis/common/causal_eval_suite.py results/<run>
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
import os, sys, json, re
import numpy as np, torch, scanpy as sc
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(_CMC_ROOT,"src"))
from model import CrossModalCausalLDVAE

import os as _os
def _ckpt(rd):
    """FINAL checkpoint by default (fully-regularized); fallback best. Override CMC_CKPT=best."""
    pref=_os.environ.get("CMC_CKPT","final")
    for n in [f"{pref}_model.pt","final_model.pt","best_model.pt"]:
        p=_os.path.join(rd,n)
        if _os.path.exists(p): return p
    return _os.path.join(rd,"best_model.pt")

RES=_CMC_RESULTS
rd=os.path.abspath(sys.argv[1]); cfg=json.load(open(f"{rd}/config.json")); out={"run":os.path.basename(rd)}

# ---------- load model + projected networks ----------
ar=sc.read_h5ad(cfg["rna_path"],backed="r"); ap=sc.read_h5ad(cfg["prot_path"],backed="r")
genes=list(ar.var_names.astype(str)); prots=list(ap.var_names.astype(str)); ng,npr=len(genes),len(prots)
_dm=None
if cfg.get("dec_mask_rna_path"):
    _mk=torch.load(cfg["dec_mask_rna_path"],map_location="cpu"); _dm=_mk["mask"] if isinstance(_mk,dict) else _mk
md=CrossModalCausalLDVAE(ng,npr,1,latent_dim_rna=cfg["latent_dim_rna"],latent_dim_prot=cfg["latent_dim_prot"],
    recon_likelihood="mse",scm_nonlinear=cfg.get("scm_nonlinear",False),phi=cfg.get("phi","tanh"),
    dec_mask_rna=_dm,dec_mask_lambda=cfg.get("dec_mask_lambda",0.1),dec_mask_freeze=cfg.get("dec_mask_freeze",False))
md.load_state_dict(torch.load(_ckpt(rd),map_location="cpu"),strict=False); md.eval()
with torch.no_grad():
    A_rr,A_pp,W_rp,W_pr=md.scm.get_block_adjacencies(tau=0.5)
    Dr=md._dec_r_w().detach().float(); Dp=md.dec_prot.weight.detach().float()  # EFFECTIVE decoder (W(x)mask if pathway)
    G_rr=(Dr@A_rr.float()@Dr.T).cpu().numpy()
    G_pp=(Dp@A_pp.float()@Dp.T).cpu().numpy()
    G_rp=(Dr@W_rp.float()@Dp.T).cpu().numpy()      # [ng, npr]  gene -> protein (translation)
    G_pr=(Dp@W_pr.float()@Dr.T).cpu().numpy()      # [npr, ng]  protein -> gene (feedback)
for M in (G_rr,G_pp): np.fill_diagonal(M,0.0)

def epr(score, lab):
    """early precision ratio = precision in top-K / pos_frac, K=#true edges (CausalGRN/DeepSEM)."""
    lab=lab.astype(bool); K=int(lab.sum())
    if K<3 or K>=len(lab): return None
    top=np.argsort(-score)[:K]; return float(lab[top].mean()/(K/len(lab)))
def recov(score, lab, symmetric=False):
    lab=lab.astype(bool)
    if lab.sum()<3 or lab.sum()>=len(lab): return None
    base=lab.mean()
    return dict(auroc=float(roc_auc_score(lab,score)), auprc_ratio=float(average_precision_score(lab,score)/base),
                epr=epr(score,lab), n_edges=int(lab.sum()), prevalence=float(base))

# ---------- reference masks aligned to model vocab ----------
def load_prior(path, names_key):
    d=torch.load(path,map_location="cpu"); m=d["prior_mask"]; m=m.numpy() if torch.is_tensor(m) else np.asarray(m)
    nm=list(map(str,d.get("gene_names",d.get("protein_names",[]))))
    return m, nm
refs_g={}  # gene-space directed masks
try:
    m,nm=load_prior(cfg["prior_rna_path"],"gene_names")
    if nm==genes: refs_g["CollecTRI"]=(m.astype(bool),True)
    else:
        gi={g:i for i,g in enumerate(nm)}; perm=[gi[g] for g in genes] if set(genes)<=set(nm) else None
        if perm is not None: refs_g["CollecTRI"]=(m[np.ix_(perm,perm)].astype(bool),True)
except Exception as e: out["collectri_err"]=str(e)[:100]
# cached DoRothEA/STRING/coexpr for this cohort
cohort="2ccombat" if "2ccombat" in os.path.basename(rd) else ("3cqc" if "3cqc" in os.path.basename(rd) else "2c")
_rv="v4" if "_v4" in os.path.basename(rd) else "v3"
refp=f"{RES}/ref_nets_{_rv}_{cohort}.npz"; refmeta={}
if os.path.exists(refp):
    ref=np.load(refp,allow_pickle=True)
    if list(map(str,ref["genes"]))==genes:
        refmeta=json.load(open(f"{RES}/ref_nets_{_rv}_{cohort}.json"))
        for k in ref.files:
            if k=="genes": continue
            refs_g[k]=(ref[k].astype(bool), refmeta[k]["directed"])

# ========== A. GRN ==========
A={}
for name,(mask,directed) in refs_g.items():
    if directed: A[name]=recov(np.abs(G_rr)[~np.eye(ng,dtype=bool)], mask[~np.eye(ng,dtype=bool)])
    else:
        Gs=np.abs(G_rr)+np.abs(G_rr).T; off=~np.eye(ng,dtype=bool)
        A[name]=recov(Gs[off], (mask|mask.T)[off])
    if A.get(name):
        note=refmeta.get(name,{}).get("note","")
        A[name]["circular"]=("prior" in note) or (name=="CollecTRI" and cfg.get("prior_w_final",0)>0) or (name=="coexpr" and "coexpr" in os.path.basename(rd))
# directional asymmetry on CollecTRI directed edges (CausalGRN orientation): |G[tf,tgt]| vs |G[tgt,tf]|
if "CollecTRI" in refs_g:
    m=refs_g["CollecTRI"][0]; ii,jj=np.where(m); keep=ii!=jj; ii,jj=ii[keep],jj[keep]
    fwd=np.abs(G_rr[ii,jj]); rev=np.abs(G_rr[jj,ii])
    try:
        u,p=mannwhitneyu(fwd,rev,alternative="greater")
        A["asymmetry"]=dict(fwd_median=float(np.median(fwd)),rev_median=float(np.median(rev)),
                            fwd_over_rev=float(np.median(fwd)/(np.median(rev)+1e-12)),mwu_p=float(p),n_pairs=int(len(ii)))
    except Exception: pass
out["A_GRN"]=A

# ========== B. PPI ==========
B={"symmetry_corr": float(np.corrcoef(np.abs(G_pp).ravel(), np.abs(G_pp).T.ravel())[0,1])}  # expect HIGH (undirected)
try:
    mp,nmp=load_prior(cfg["prior_prot_path"],"protein_names")
    if nmp==prots or set(prots)<=set(nmp):
        if nmp!=prots:
            pi={p:i for i,p in enumerate(nmp)}; perm=[pi[p] for p in prots]; mp=mp[np.ix_(perm,perm)]
        Gs=np.abs(G_pp)+np.abs(G_pp).T; off=~np.eye(npr,dtype=bool)
        B["OmniPath"]=recov(Gs[off],(mp.astype(bool)|mp.astype(bool).T)[off])
except Exception as e: B["omnipath_err"]=str(e)[:100]
out["B_PPI"]=B

# ========== C. Translation (cognate RNA -> protein) ==========
def canon_prot(x):
    x=re.sub(r'^AB_','',x); x=re.sub(r'^(human|mouse)','',x,flags=re.I); x=re.sub(r'-\d+$','',x); return x.split('_')[0].upper()
gup={g.upper():i for i,g in enumerate(genes)}
cognate=[]  # (gene_idx, prot_idx)
for j,p in enumerate(prots):
    c=canon_prot(p)
    for cand in (c, c.replace("CD",""), c+"E", c+"A"):   # light expansion (CD3->CD3E etc.)
        if cand in gup: cognate.append((gup[cand],j)); break
C={"n_cognate_pairs":len(cognate)}
if cognate:
    matched=np.array([abs(G_rp[g,p]) for g,p in cognate])
    # background: for each cognate protein, non-cognate genes
    cog_by_p={}; [cog_by_p.setdefault(p,set()).add(g) for g,p in cognate]
    bg=[]; ranks=[]
    for g,p in cognate:
        col=np.abs(G_rp[:,p]); bg.append(np.delete(col, list(cog_by_p[p])))
        ranks.append(float((col> col[g]).sum())/len(col))   # fractional rank of cognate (0=top)
    bg=np.concatenate(bg)
    u,pv=mannwhitneyu(matched,bg,alternative="greater")
    C.update(matched_median=float(np.median(matched)), bg_median=float(np.median(bg)),
             fold=float(np.median(matched)/(np.median(bg)+1e-12)), mwu_p=float(pv),
             median_cognate_rank_frac=float(np.median(ranks)), top10pct_cognate=float(np.mean(np.array(ranks)<0.1)))
out["C_translation"]=C

# ========== D. Feedback vs translation asymmetry ==========
out["D_feedback"]=dict(
    mean_abs_translation_Wrp=float(np.abs(W_rp.cpu().numpy()).mean()),
    mean_abs_feedback_Wpr=float(np.abs(W_pr.cpu().numpy()).mean()),
    mean_abs_G_rp=float(np.abs(G_rp).mean()), mean_abs_G_pr=float(np.abs(G_pr).mean()),
    translation_over_feedback=float(np.abs(G_rp).mean()/(np.abs(G_pr).mean()+1e-12)))

# ========== aliveness ==========
with torch.no_grad():
    Af=torch.cat([torch.cat([A_rr,W_rp],1),torch.cat([W_pr,A_pp],1)],0)
    # DAGMA acyclicity h(A)=trace(expm(A o A))-d on the intra RNA block (0 iff acyclic)
    AA=(A_rr.float()*A_rr.float())
    try: h_rr=float(torch.trace(torch.matrix_exp(AA))-AA.shape[0])
    except Exception: h_rr=None
    out["aliveness"]=dict(spectral_radius=float(torch.linalg.eigvals(Af).abs().max()),
        A_rr_density=float((A_rr.abs()>1e-6).float().mean()),
        dagma_h_rr=h_rr,
        G_rr_density=float((np.abs(G_rr)>1e-6).mean()),      # projected gene-space GRN density
        pathway_decoder=bool(cfg.get("dec_mask_rna_path")),
        nonlinear=bool(cfg.get("scm_nonlinear",False)))
json.dump(out, open(f"{rd}/causal_eval_suite.json","w"), indent=2)
print(f"=== causal_eval_suite: {out['run']} ===")
print(json.dumps({k:v for k,v in out.items() if k not in ("run",)}, indent=2)[:2500])
print("DONE")
