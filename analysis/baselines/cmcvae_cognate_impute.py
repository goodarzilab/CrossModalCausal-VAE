"""Per-protein CMC-VAE imputation + cognate-mRNA proxy (base env).

Reproduces the exact protocol of analysis/fig2_networks/crossmodal_imputation.py (global per-column
z-score, seed-42 80/10/10 split, force_drop="protein" learned-null imputation) but
SAVES the per-protein Pearson vector (the base script only saves the mean 0.363).
Also computes the single-cognate-mRNA proxy per protein (task 3 / A2 biological floor).

Memory-safe: column stats come from the sparse matrix (identical to np.std ddof=0 on
dense) and only the ~80k test rows are densified — the job cgroup is capped at 80 GB.

Test cells are defined identically to crossmodal_imputation.py, so they match the
sciPENN held-out-cell test set (all three h5ad share obs_names in identical order).

Outputs results/baselines/_cmcvae_cognate.json  (consumed by make_impute_figure.py).
Run with the BASE conda env python.
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
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

ROOT = _CMC_ROOT
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import CrossModalCausalLDVAE

RUN = os.path.join(ROOT, "results", "linfb_2c_fb10_b20_v4")    # CMC-VAE-v4 (combined_v4, mean 0.382)
V4_RNA = f"{_CMC_DATA}/combined_v4/rna_combined.h5ad"
OUT = os.path.join(ROOT, "results", "baselines", "_cmcvae_cognate.json")

def col_mean_std(X):
    """Per-column mean/std matching numpy default (ddof=0), from sparse or dense."""
    if sp.issparse(X):
        n = X.shape[0]
        mean = np.asarray(X.mean(0)).ravel()
        sq = np.asarray(X.multiply(X).mean(0)).ravel()
        var = np.maximum(sq - mean**2, 0.0)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)
    return X.mean(0).astype(np.float32), X.std(0).astype(np.float32)

cfg = json.load(open(os.path.join(RUN, "config.json")))
print("[cmcvae] reading RNA(vocab) + protein headers ...", flush=True)
rna = sc.read_h5ad(cfg["rna_path"]); prot = sc.read_h5ad(cfg["prot_path"])
common = sorted(set(rna.obs_names) & set(prot.obs_names))
prot_names = list(prot.var_names)

seed = cfg.get("seed", 42)
idx = np.arange(len(common))
_, tmp = train_test_split(idx, test_size=0.2, random_state=seed)
_, te = train_test_split(tmp, test_size=0.5, random_state=seed)     # 10% held-out test
test_names = [common[i] for i in te]
print(f"[cmcvae] n_common={len(common)} n_test={len(te)} n_prot={len(prot_names)}", flush=True)

# global per-column stats over the common cells (order-independent) -> matches eval z-score
rmean, rstd = col_mean_std(rna[common].X)
pmean, pstd = col_mean_std(prot[common].X)

# densify only the test rows
Xr_te = rna[test_names].X; Xr_te = Xr_te.toarray() if sp.issparse(Xr_te) else np.asarray(Xr_te)
Xp_te = prot[test_names].X; Xp_te = Xp_te.toarray() if sp.issparse(Xp_te) else np.asarray(Xp_te)
Xp_te_raw = Xp_te.astype(np.float32).copy()
Xr_te_z = ((Xr_te - rmean) / (rstd + 1e-6)).astype(np.float32)
Xp_te_z = ((Xp_te - pmean) / (pstd + 1e-6)).astype(np.float32)
n_genes_vocab = Xr_te_z.shape[1]
del rna

sd = torch.load(os.path.join(RUN, "best_model.pt"), map_location="cpu"); sd.pop("context_of_batch", None)
m = CrossModalCausalLDVAE(n_genes_vocab, len(prot_names), 1,
                          latent_dim_rna=cfg["latent_dim_rna"], latent_dim_prot=cfg["latent_dim_prot"],
                          recon_likelihood=(cfg.get("recon_likelihood") or "mse"),
                          gate_mode=cfg.get("gate_mode", "gumbel"))
m.load_state_dict(sd, strict=False); m.eval()

xr = torch.tensor(Xr_te_z); xp = torch.tensor(Xp_te_z); bb = torch.zeros(len(te), dtype=torch.long)
with torch.no_grad():
    rec_p_imp = m(xr, xp, bb, force_drop="protein")["rec"][1].numpy()   # protein <- RNA only

def percol_pearson(P, Tr):
    out = np.full(Tr.shape[1], np.nan)
    for j in range(Tr.shape[1]):
        a, b = P[:, j], Tr[:, j]
        if np.std(a) > 1e-6 and np.std(b) > 1e-6:
            out[j] = np.corrcoef(a, b)[0, 1]
    return out

cmc_perprot = percol_pearson(rec_p_imp, Xp_te_z)
print(f"[cmcvae] protein<-RNA per-protein Pearson mean={np.nanmean(cmc_perprot):.4f} "
      f"(sanity vs 0.382 v4) n_valid={int(np.sum(~np.isnan(cmc_perprot)))}", flush=True)

# ---- cognate-mRNA proxy on the v4 panel (sciPENN's RNA panel), same test cells ----
# Curated CITE-seq ADT -> cognate gene-symbol map (naive canon_prot only matched 24/144;
# standard CD-antigen aliases, e.g. CD11c->ITGAX, must be resolved for a fair biological floor).
ADT2GENE = {
 'B7-H4':'VTCN1','CD123':'IL3RA','CD141':'THBD','CD15':'FUT4','CD16':'FCGR3A','CD161':'KLRB1',
 'CD1A':'CD1A','CD1C':'CD1C','CD1D':'CD1D','CD2':'CD2','CD20':'MS4A1','CD209':'CD209','CD21':'CR2',
 'CD22':'CD22','CD226':'CD226','CD235AB':'GYPA','CD24':'CD24','CD244':'CD244','CD25':'IL2RA',
 'CD27':'CD27','CD274':'CD274','CD28':'CD28','CD303':'CLEC4C','CD304':'NRP1','CD34':'CD34',
 'CD36':'CD36','CD3':'CD3E','CD4':'CD4','CD40':'CD40','CD44':'CD44','CD45':'PTPRC','CD45RA':'PTPRC',
 'CD45RO':'PTPRC','CD47':'CD47','CD52':'CD52','CD56':'NCAM1','CD64':'FCGR1A','CD69':'CD69','CD70':'CD70',
 'CD71':'TFRC','CD79B':'CD79B','CD8':'CD8A','CD80':'CD80','CD81':'CD81','CD83':'CD83','CD86':'CD86',
 'CD96':'CD96','CD99':'CD99','CLEC12A':'CLEC12A','CX3CR1':'CX3CR1','HLA-DR':'HLA-DRA','IGD':'IGHD',
 'IGM':'IGHM','TIGIT':'TIGIT','TSLPR':'CRLF2','XCR1':'XCR1','PODOPLANIN':'PDPN','CD103':'ITGAE',
 'CD106':'VCAM1','CD107A':'LAMP1','CD112':'NECTIN2','CD117':'KIT','CD11A/CD18':'ITGAL','CD11B':'ITGAM',
 'CD11C':'ITGAX','CD122':'IL2RB','CD124':'IL4R','CD127':'IL7R','CD133':'PROM1','CD137':'TNFRSF9',
 'CD138':'SDC1','CD144':'CDH5','CD146':'MCAM','CD152':'CTLA4','CD154':'CD40LG','CD155':'PVR',
 'CD158E1':'KIR3DL1','CD169':'SIGLEC1','CD178':'FASLG','CD18':'ITGB2','CD184':'CXCR4','CD185':'CXCR5',
 'CD193':'CCR3','CD194':'CCR4','CD195':'CCR5','CD196':'CCR6','CD204':'MSR1','CD206':'MRC1','CD207':'CD207',
 'CD223':'LAG3','CD252':'TNFSF4','CD26':'DPP4','CD267':'TNFRSF13B','CD268':'TNFRSF13C','CD269':'TNFRSF17',
 'CD272':'BTLA','CD273':'PDCD1LG2','CD275':'ICOSLG','CD278':'ICOS','CD279':'PDCD1','CD29':'ITGB1',
 'CD294':'PTGDR2','CD30':'TNFRSF8','CD305':'LAIR1','CD307D':'FCRL4','CD307E':'FCRL5','CD309':'KDR',
 'CD31':'PECAM1','CD314':'KLRK1','CD319':'SLAMF7','CD324':'CDH1','CD335':'NCR1','CD337':'NCR3','CD35':'CR1',
 'CD357':'TNFRSF18','CD366':'HAVCR2','CD370':'CLEC9A','CD38':'CD38','CD39':'ENTPD1','CD41':'ITGA2B',
 'CD49B':'ITGA2','CD49D':'ITGA4','CD54':'ICAM1','CD57':'B3GAT1','CD62P':'SELP','CD66A/C/E':'CEACAM1',
 'CD66B':'CEACAM8','CD73':'NT5E','CD90':'THY1','CD95':'FAS','CD98':'SLC3A2','INTEGRIN-7':'ITGB7',
 'LOX-1':'OLR1','CD19':'CD19','CD14':'CD14','CD158':'KIR2DL1','CD158B':'KIR2DL3',
}
def cognate_symbol(p):
    up = p.upper()
    if up in ADT2GENE: return ADT2GENE[up]               # full name first (Integrin-7, LOX-1, CD66a/c/e)
    stripped = re.sub(r"-\d+$", "", up)                   # strip TotalSeq clone tag (CD3-1 -> CD3)
    if stripped in ADT2GENE: return ADT2GENE[stripped]
    return re.sub(r"^AB_", "", stripped).split("_")[0]    # naive fallback

print("[cmcvae] reading RNA(v4) for cognate proxy ...", flush=True)
rv4 = sc.read_h5ad(V4_RNA)
Xv4_te = rv4[test_names].X; Xv4_te = Xv4_te.toarray() if sp.issparse(Xv4_te) else np.asarray(Xv4_te)
v4_genes = {g.upper(): i for i, g in enumerate(rv4.var_names)}

cognate_r = np.full(len(prot_names), np.nan)
cognate_gene = [None] * len(prot_names)
for j, p in enumerate(prot_names):
    c = cognate_symbol(p); gi = v4_genes.get(c)
    if gi is not None:
        a = Xv4_te[:, gi]; b = Xp_te_raw[:, j]
        if np.std(a) > 1e-6 and np.std(b) > 1e-6:
            cognate_r[j] = np.corrcoef(a, b)[0, 1]; cognate_gene[j] = c
n_cog = int(np.sum(~np.isnan(cognate_r)))
print(f"[cognate] matched {n_cog}/{len(prot_names)} proteins to a v4 transcript; "
      f"mean cognate Pearson={np.nanmean(cognate_r):.4f}", flush=True)

json.dump({
    "run": os.path.basename(RUN),
    "n_test_cells": int(len(te)),
    "test_idx_first10": te[:10].tolist(),
    "test_name_first3": test_names[:3],
    "protein_names": prot_names,
    "cmcvae_protein_from_rna_pearson": [None if np.isnan(v) else float(v) for v in cmc_perprot],
    "cmcvae_mean_pearson": float(np.nanmean(cmc_perprot)),
    "cognate_gene": cognate_gene,
    "cognate_mrna_pearson": [None if np.isnan(v) else float(v) for v in cognate_r],
    "cognate_mean_pearson": float(np.nanmean(cognate_r)),
    "n_cognate_matched": n_cog,
}, open(OUT, "w"), indent=2)
print(f"[cmcvae] wrote {OUT}\nDONE", flush=True)
