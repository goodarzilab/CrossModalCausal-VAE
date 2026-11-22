"""Sparsity/density of the learned latent DAG (A_rr/A_pp/W_rp/W_pr) and the projected GRN
G = D_r A_rr D_r^T, for every model. Latent graph: fraction of edges with |w|>eps (the real learned
sparsity) + gate open-prob. Projected GRN is mathematically dense, so report CONCENTRATION:
effective-edge fraction (|G_ij|>0.1·max), Gini of |G|, and % of total |G| mass in the top 1% of pairs.
Usage: python analysis/fig2_networks/dag_sparsity.py <model1> <model2> ...
"""
import sys, os, json
import numpy as np, torch, scanpy as sc

import os as _os
def _ckpt(rd):
    """FINAL checkpoint by default (fully-regularized); fallback best. Override CMC_CKPT=best."""
    pref=_os.environ.get("CMC_CKPT","final")
    for n in [f"{pref}_model.pt","final_model.pt","best_model.pt"]:
        p=_os.path.join(rd,n)
        if _os.path.exists(p): return p
    return _os.path.join(rd,"best_model.pt")

sys.path.insert(0, "src"); from model import CrossModalCausalLDVAE
def gini(x):
    x=np.sort(np.abs(x).ravel()); n=len(x);
    return float((2*np.arange(1,n+1)-n-1).dot(x)/(n*x.sum()+1e-12))
rows=[]
for m in sys.argv[1:]:
    rd=f"results/{m}"; cfg=json.load(open(f"{rd}/config.json"))
    ng=sc.read_h5ad(cfg["rna_path"],backed="r").shape[1]; npr=sc.read_h5ad(cfg["prot_path"],backed="r").shape[1]
    md=CrossModalCausalLDVAE(ng,npr,1,latent_dim_rna=cfg["latent_dim_rna"],latent_dim_prot=cfg["latent_dim_prot"],
        recon_likelihood="mse",scm_nonlinear=cfg.get("scm_nonlinear",False),phi=cfg.get("phi","tanh"))
    md.load_state_dict(torch.load(_ckpt(rd),map_location="cpu"),strict=False); md.eval()
    with torch.no_grad():
        A_rr,A_pp,W_rp,W_pr=md.scm.get_block_adjacencies(tau=0.5)
        p_rr,p_pp,p_rp,p_pr=md.scm.get_gate_probs()
        A_full=torch.cat([torch.cat([A_rr,W_rp],1),torch.cat([W_pr,A_pp],1)],0)
        rho=float(torch.linalg.eigvals(A_full).abs().max())
        Dr=md.dec_rna.weight; G=(Dr@A_rr@Dr.T).cpu().numpy(); np.fill_diagonal(G,0.0); Go=np.abs(G[~np.eye(len(G),dtype=bool)])
    def dens(B): B=B.detach().cpu().numpy(); off=~np.eye(*B.shape,dtype=bool) if B.shape[0]==B.shape[1] else np.ones_like(B,bool); v=np.abs(B[off]); return float((v>1e-6).mean())
    def gopen(P): P=P.detach().cpu().numpy(); off=~np.eye(*P.shape,dtype=bool) if P.shape[0]==P.shape[1] else np.ones_like(P,bool); return float((P[off]>0.5).mean())
    topmass=float(np.sort(Go)[::-1][:max(1,len(Go)//100)].sum()/(Go.sum()+1e-12))
    r=dict(model=m, nonlinear=bool(cfg.get("scm_nonlinear",False)), rho=rho,
           A_rr_edge_density=dens(A_rr), A_pp_edge_density=dens(A_pp), W_rp_density=dens(W_rp), W_pr_density=dens(W_pr),
           A_rr_gate_open=gopen(p_rr),
           G_eff_edge_frac=float((Go>0.1*Go.max()).mean()), G_gini=gini(Go), G_top1pct_mass=topmass)
    rows.append(r)
    print(f"{m:24s} nl={int(r['nonlinear'])} rho={rho:.2f} | A_rr dens={r['A_rr_edge_density']:.3f} gate_open={r['A_rr_gate_open']:.3f} "
          f"A_pp={r['A_pp_edge_density']:.3f} Wrp={r['W_rp_density']:.3f} Wpr={r['W_pr_density']:.3f} | "
          f"projG: eff_edge={r['G_eff_edge_frac']:.3f} gini={r['G_gini']:.2f} top1%mass={r['G_top1pct_mass']:.2f}",flush=True)
json.dump(rows,open("results/dag_sparsity.json","w"),indent=2); print("DONE")
