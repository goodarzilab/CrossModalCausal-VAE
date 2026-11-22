"""GSEA on the CMC-VAE latent 'programs': each RNA latent j has a decoder loading D_r[:,j] over genes.
Run GSEA (decoupler) on those loadings vs MSigDB Hallmark -> which pathway each latent encodes.
Then use the latent GRN A_rr to find which pathway-programs regulate which (strongly-connected pathways).
Usage: python analysis/fig2_networks/gsea_latents.py results/linfb_2c_nodag_v4"""
import os,sys,json,warnings,numpy as np,pandas as pd,torch; warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); import scanpy as sc; from model import CrossModalCausalLDVAE
import decoupler as dc
rd=sys.argv[1] if len(sys.argv)>1 else "results/linfb_2c_nodag_v4"
c=json.load(open(f"{rd}/config.json")); sd=torch.load(f"{rd}/final_model.pt",map_location="cpu")
ng,npr=sd["dec_rna.weight"].shape[0],sd["dec_prot.weight"].shape[0]; lr=c["latent_dim_rna"]
genes=list(sc.read_h5ad(c["rna_path"],backed="r").var_names.astype(str))
m=CrossModalCausalLDVAE(ng,npr,1,latent_dim_rna=lr,latent_dim_prot=c.get("latent_dim_prot",64),recon_likelihood="mse")
m.load_state_dict(sd,strict=False); m.eval()
Dr=m.dec_rna.weight.detach().numpy()                          # [n_gene, lr]
with torch.no_grad(): A_rr,_,_,_=m.scm.get_block_adjacencies(tau=0.5); A=A_rr.float().numpy(); np.fill_diagonal(A,0)
# latents x genes loading matrix
mat=pd.DataFrame(Dr.T, index=[f"L{j}" for j in range(lr)], columns=genes)
if os.environ.get("GSEA_SETS","hallmark")=="immune":
    cache=json.load(open("analysis/fig2_networks/_pathway_genesets_cache.json"))
    H=pd.DataFrame([(k,g) for k,gs in cache.items() for g in gs], columns=["source","target"])
else:
    H=dc.op.hallmark(organism="human"); H.columns=["source","target"] if list(H.columns)[:2]!=["source","target"] else H.columns
H=H[H["target"].isin(genes)]
print(f"Hallmark sets: {H['source'].nunique()}  genes overlapping vocab: {H['target'].nunique()}")
# GSEA per latent
try:
    dc.mt.gsea(data=mat, net=H, tmin=5); es=mat.obsm_es if False else None
    es=dc.pp.get_obsm if False else None
except Exception:
    es=None
# robust: use gseapy prerank per latent
import gseapy as gp
gs={s:H.loc[H.source==s,"target"].tolist() for s in H.source.unique()}
lat_path={}
for j in range(lr):
    rnk=mat.iloc[j].sort_values(ascending=False)
    try:
        r=gp.prerank(rnk=rnk.reset_index(), gene_sets=gs, min_size=5, max_size=500, permutation_num=200, seed=0, threads=4, no_plot=True, verbose=False)
        res=r.res2d.sort_values("NES",key=lambda x:x.astype(float).abs(),ascending=False)
        top=res.iloc[0]; lat_path[j]=(top["Term"], float(top["NES"]), float(top["FDR q-val"]))
    except Exception as e:
        lat_path[j]=("(none)",0.0,1.0)
# report
sig=[(j,)+v for j,v in lat_path.items() if v[2]<0.1]
print(f"\n{len(sig)}/{lr} latents map to a Hallmark pathway at FDR<0.1")
from collections import Counter
pc=Counter(v[0] for v in lat_path.values() if lat_path[[k for k,vv in lat_path.items() if vv is v][0]][2]<0.1) if False else Counter(v[1][0] for v in enumerate(lat_path.values()) if v[1][2]<0.1)
print("\nTop pathways by #latents encoding them:")
for p,n in pc.most_common(12): print(f"  {n:2d} latents  {p}")
# strongly-connected pathways via A_rr (edge i->j, annotate with latent pathways)
lab={j:lat_path[j][0].replace('HALLMARK_','') for j in range(lr)}
thr=np.percentile(np.abs(A),99); ii,jj=np.where(np.abs(A)>thr)
edges=Counter((lab[i],lab[j]) for i,j in zip(ii,jj) if lat_path[i][2]<0.1 and lat_path[j][2]<0.1 and lab[i]!=lab[j])
print("\nTop pathway-program -> pathway-program regulatory edges (A_rr, top-1% weights):")
for (a,b),n in edges.most_common(12): print(f"  {n:2d}  {a}  ->  {b}")
json.dump({"latent_pathway":{str(j):lat_path[j] for j in lat_path},"pathway_edges":{f"{a}->{b}":n for (a,b),n in edges.most_common(40)}},
          open(f"results/gsea_latents_{os.environ.get('GSEA_SETS','hallmark')}_{os.path.basename(rd)}.json","w"),indent=2)
print(f"\nWROTE results/gsea_latents_{os.path.basename(rd)}.json")
