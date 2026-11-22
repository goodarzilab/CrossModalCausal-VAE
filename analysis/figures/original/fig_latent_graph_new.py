"""Network diagram of significant (pathway-mapped) CMC-VAE latents and their A_rr connections.
A: latent-level graph grouped by pathway family (node=latent). B: pathway-family cross-talk summary.
HEADLINE model (results/sweep_sparse/sp_dec_gl1000); reads the immune-geneset GSEA JSON.
Copy of fig_latent_graph.py -> reads headline JSON, saves to paper/figures/_new_latent_gsea.pdf."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os,sys,json,math,warnings,numpy as np,torch; warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.dirname(__file__)); from style import plt, save
sys.path.insert(0,"src"); from model import CrossModalCausalLDVAE
import networkx as nx; from collections import defaultdict
R=f"{_CMC_ROOT}"; rd=f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000"
c=json.load(open(f"{rd}/config.json")); sd=torch.load(f"{rd}/final_model.pt",map_location="cpu")
ng,npr=sd["dec_rna.weight"].shape[0],sd["dec_prot.weight"].shape[0]; lr=c["latent_dim_rna"]
m=CrossModalCausalLDVAE(ng,npr,1,latent_dim_rna=lr,latent_dim_prot=c.get("latent_dim_prot",64),recon_likelihood="mse")
m.load_state_dict(sd,strict=False); m.eval()
with torch.no_grad(): A_rr,_,_,_=m.scm.get_block_adjacencies(tau=0.5); A=A_rr.float().numpy(); np.fill_diagonal(A,0)
gj=json.load(open(f"{_CMC_RESULTS}/gsea_latents_immune_sp_dec_gl1000.json"))["latent_pathway"]
sig={int(k):v for k,v in gj.items() if v[2]<0.1}
def fam(p):
    p=p.upper()
    if "INTERFERON_GAMMA" in p: return "IFN-γ"
    if "INTERFERON_ALPHA" in p or "DDX58" in p or "IFIH" in p: return "IFN-α/β"
    if "MHC" in p or "ANTIGEN" in p: return "Antigen/MHC"
    if "TCR" in p or "T_CELL_RECEPTOR" in p: return "TCR"
    if "TNFA" in p or "NFKB" in p: return "TNFα/NFκB"
    if "JAK_STAT" in p: return "JAK-STAT"
    if "APOPTOSIS" in p: return "Apoptosis"
    return "Other"
FCOL={"IFN-γ":"#c53030","IFN-α/β":"#dd6b20","Antigen/MHC":"#2b6cb0","TCR":"#2f855a",
      "TNFα/NFκB":"#6b46c1","JAK-STAT":"#319795","Apoptosis":"#718096","Other":"#a0aec0"}
nodes=sorted(sig); nf={n:fam(sig[n][0]) for n in nodes}
fig,(axA,axB)=plt.subplots(1,2,figsize=(15,7.5)); plt.subplots_adjust(wspace=0.05,left=0.02,right=0.98)
# ---- A: latent-level, family-grouped ring ----
G=nx.Graph(); G.add_nodes_from(nodes)
w=sorted([(abs(A[i,j])+abs(A[j,i]),i,j) for i in nodes for j in nodes if i<j],reverse=True)
for val,i,j in w[:int(len(nodes)*1.6)]: G.add_edge(i,j,weight=val)
famlist=sorted(set(nf[n] for n in G.nodes)); RING=3.6
fc={f:(RING*math.cos(2*math.pi*i/len(famlist)), RING*math.sin(2*math.pi*i/len(famlist))) for i,f in enumerate(famlist)}
pos={}
for f in famlist:
    fn=sorted([n for n in G.nodes if nf[n]==f]); cx,cy=fc[f]
    for k2,n in enumerate(fn):
        if len(fn)==1: pos[n]=(cx,cy)
        else:
            rr=min(0.55+0.13*len(fn),1.0); ang=2*math.pi*k2/len(fn); pos[n]=(cx+rr*math.cos(ang), cy+rr*math.sin(ang))
ews=[G[u][v]["weight"] for u,v in G.edges]; mx=max(ews) if ews else 1
nx.draw_networkx_edges(G,pos,ax=axA,width=[0.3+2.2*e/mx for e in ews],edge_color="#d0d7e2",alpha=0.55)
nx.draw_networkx_nodes(G,pos,ax=axA,node_size=210,node_color=[FCOL[nf[n]] for n in G.nodes],edgecolors="white",linewidths=1.0)
nx.draw_networkx_labels(G,pos,labels={n:str(n) for n in G.nodes},ax=axA,font_size=5.3,font_color="white",font_weight="bold")
for f in famlist:  # family label pushed radially OUTSIDE its cluster
    cx,cy=fc[f]; nrm=math.hypot(cx,cy) or 1; lx,ly=cx/nrm*(RING+2.0), cy/nrm*(RING+2.0)
    axA.text(lx,ly,f,ha="center",va="center",fontsize=9.5,fontweight="bold",color=FCOL[f])
axA.set_xlim(-6.0,6.0); axA.set_ylim(-6.0,6.0); axA.axis("off")
axA.set_title("Latent programs & their $A_{rr}$ connections  (node = latent, grouped by pathway)",fontsize=10.5)
# ---- B: pathway-family cross-talk ----
fams=sorted(set(nf.values())); F2=nx.Graph(); F2.add_nodes_from(fams); fw=defaultdict(float)
for i in nodes:
    for j in nodes:
        if i!=j and nf[i]!=nf[j]: fw[tuple(sorted((nf[i],nf[j])))]+=abs(A[i,j])
mxf=max(fw.values()) if fw else 1
for (a,b),val in fw.items():
    if val>0.12*mxf: F2.add_edge(a,b,weight=val)
posF={f:(math.cos(2*math.pi*i/len(fams)), math.sin(2*math.pi*i/len(fams))) for i,f in enumerate(sorted(fams))}
fe=[F2[u][v]["weight"] for u,v in F2.edges]; mxe=max(fe) if fe else 1
nx.draw_networkx_edges(F2,posF,ax=axB,width=[0.6+5*e/mxe for e in fe],edge_color="#b7c0cc",alpha=0.7)
sz=[520+280*sum(1 for n in nodes if nf[n]==f) for f in F2.nodes]
nx.draw_networkx_nodes(F2,posF,ax=axB,node_size=sz,node_color=[FCOL[f] for f in F2.nodes],edgecolors="white",linewidths=1.5)
for f in F2.nodes:  # label radially OUTSIDE the node
    x,y=posF[f]; axB.text(x*1.34,y*1.34,f,ha="center",va="center",fontsize=9,fontweight="bold",color=FCOL[f])
axB.set_xlim(-1.75,1.75); axB.set_ylim(-1.75,1.75); axB.axis("off")
axB.set_title("Pathway-family cross-talk  (edge = summed $A_{rr}$ between families)",fontsize=10.5)
# NOTE: S9 is now the healthy-vs-disease comparison (fig_latent_graph_hd.py). This single-model version
# is kept as a supplementary/exploratory artifact under its own filename.
save(fig,f"{R}/paper/figures/_singlemodel_latent_gsea.pdf"); print(f"WROTE _singlemodel_latent_gsea.pdf: A={len(G)}n/{G.number_of_edges()}e, B={len(F2)}fam")
