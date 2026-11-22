"""PPI + feedback eval for the CMC-VAE protein-side blocks (A_pp latent PPI, W_pr protein->RNA feedback).

We have no protein perturbation data, so these blocks are evaluated by REFERENCE-NETWORK RECOVERY
(mirrors the GRN side / analysis/common/causal_eval_suite.py), plus baselines.

  Task 1  PPI recovery of projected  G_pp = D_p @ A_pp @ D_p^T  vs protein-protein references
          restricted to the 144-antibody panel:  STRING (>=700) | OmniPath (the B_PPI prior mask) | CORUM.
  Task 2  Baselines on the SAME references/harness:
            (i)  raw protein coexpression = |corr of the 144 CLR proteins across cells|  (trivial PPI baseline)
            (ii) totalVI protein-protein  = |corr of totalVI denoised proteins| (loaded in scvi-env)
          KEY question: does A_pp recover PPI ABOVE raw protein coexpression?
  Task 3  Feedback W_pr characterization (exploratory, no clean reference): project G_pr = D_p @ W_pr @ D_r^T,
          Gini/sparsity, "reverse cognate" enrichment (protein X -> its own transcript X) w/ permutation null,
          and |W_rp| vs |W_pr| / |G_rp| vs |G_pr| magnitude (translation vs feedback; latent-dim confounded).

Reuses the EXACT projection + recov()/epr() metrics + canon_prot from analysis/common/causal_eval_suite.py.
Model is PRIOR-OFF (prior_w_final=0): OmniPath recovery is then honest, not circular (prior-ON memorises it).

Outputs:  results/baselines/ppi_feedback_eval.json  and  results/figures/slides/fig_ppi_eval.png
Usage:    python analysis/fig2_networks/ppi_feedback_eval.py [results/<run>]      (default: results/linfb_2c_fb10_b20_v4)
Env:      base env for everything; totalVI baseline auto-shells out to the scvi-env interpreter.
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
import os, sys, json, re, io, subprocess, tempfile
import numpy as np, torch, scanpy as sc
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_CMC_ROOT, "src"))
from model import CrossModalCausalLDVAE

REPO = _CMC_ROOT
RES = os.path.join(REPO, "results")
BASE = os.path.join(RES, "baselines"); os.makedirs(BASE, exist_ok=True)
FIGDIR = os.path.join(RES, "figures", "slides"); os.makedirs(FIGDIR, exist_ok=True)
SCVI_PY = f"{_CMC_SCVI_PY}"
CV4 = f"{_CMC_DATA}/combined_v4"
TVI_INPUT = f"{CV4}/totalvi_input.h5ad"
CITESEQ = f"{CV4}/citeseq_combined.h5ad"
TVI_MODEL = f"{BASE}/totalvi_model"
N_CORR = int(os.environ.get("PPI_N_CELLS", "30000"))   # cells for coexpr / totalVI corr matrices

rd = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(RES, "linfb_2c_fb10_b20_v4")
run = os.path.basename(rd)
cfg = json.load(open(f"{rd}/config.json"))
out = {"run": run, "prior_off": cfg.get("prior_w_final", 0) == 0, "n_corr_cells": N_CORR}

# ----------------------------------------------------------------------------- metrics (from causal_eval_suite)
def epr(score, lab):
    lab = lab.astype(bool); K = int(lab.sum())
    if K < 3 or K >= len(lab): return None
    top = np.argsort(-score)[:K]; return float(lab[top].mean() / (K / len(lab)))

def recov(score, lab):
    lab = lab.astype(bool)
    if lab.sum() < 3 or lab.sum() >= len(lab): return None
    base = lab.mean()
    return dict(auroc=float(roc_auc_score(lab, score)),
                auprc_ratio=float(average_precision_score(lab, score) / base),
                epr=epr(score, lab), n_edges=int(lab.sum()), prevalence=float(base))

def canon_prot(x):  # EXACT copy from analysis/common/causal_eval_suite.py
    x = re.sub(r'^AB_', '', x); x = re.sub(r'^(human|mouse)', '', x, flags=re.I)
    x = re.sub(r'-\d+$', '', x); return x.split('_')[0].upper()

def gini(v):
    v = np.sort(np.abs(np.asarray(v, dtype=np.float64)).ravel())
    if v.sum() == 0: return 0.0
    n = len(v); cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)

def score_all(methods, mask, npr):
    """methods: dict name->[npr,npr] score matrix. mask: bool [npr,npr]. Symmetric off-diagonal recovery."""
    off = ~np.eye(npr, dtype=bool)
    lab = (mask | mask.T)[off]
    res = {}
    for nm, M in methods.items():
        if M is None: res[nm] = None; continue
        Gs = np.abs(M) + np.abs(M).T
        res[nm] = recov(Gs[off], lab)
    return res

# ----------------------------------------------------------------------------- load model + projected blocks
ar = sc.read_h5ad(cfg["rna_path"], backed="r"); ap = sc.read_h5ad(cfg["prot_path"], backed="r")
genes = list(ar.var_names.astype(str)); prots = list(ap.var_names.astype(str)); ng, npr = len(genes), len(prots)
_dm = None
if cfg.get("dec_mask_rna_path"):
    _mk = torch.load(cfg["dec_mask_rna_path"], map_location="cpu"); _dm = _mk["mask"] if isinstance(_mk, dict) else _mk
md = CrossModalCausalLDVAE(ng, npr, 1, latent_dim_rna=cfg["latent_dim_rna"], latent_dim_prot=cfg["latent_dim_prot"],
    recon_likelihood="mse", scm_nonlinear=cfg.get("scm_nonlinear", False), phi=cfg.get("phi", "tanh"),
    dec_mask_rna=_dm, dec_mask_lambda=cfg.get("dec_mask_lambda", 0.1), dec_mask_freeze=cfg.get("dec_mask_freeze", False))
_ckpt = next((f"{rd}/{n}" for n in ["final_model.pt", "best_model.pt"] if os.path.exists(f"{rd}/{n}")), f"{rd}/best_model.pt")
md.load_state_dict(torch.load(_ckpt, map_location="cpu"), strict=False); md.eval()
with torch.no_grad():
    A_rr, A_pp, W_rp, W_pr = md.scm.get_block_adjacencies(tau=0.5)
    Dr = md._dec_r_w().detach().float(); Dp = md.dec_prot.weight.detach().float()
    G_pp = (Dp @ A_pp.float() @ Dp.T).cpu().numpy()
    G_rp = (Dr @ W_rp.float() @ Dp.T).cpu().numpy()   # [ng, npr] gene -> protein (translation)
    G_pr = (Dp @ W_pr.float() @ Dr.T).cpu().numpy()   # [npr, ng] protein -> gene (feedback)
np.fill_diagonal(G_pp, 0.0)
W_rp_np = W_rp.detach().cpu().numpy(); W_pr_np = W_pr.detach().cpu().numpy()
out["latent_dims"] = {"rna": cfg["latent_dim_rna"], "prot": cfg["latent_dim_prot"],
                      "W_rp_shape": list(W_rp_np.shape), "W_pr_shape": list(W_pr_np.shape)}
out["G_pp_symmetry_corr"] = float(np.corrcoef(np.abs(G_pp).ravel(), np.abs(G_pp).T.ravel())[0, 1])

# ----------------------------------------------------------------------------- protein -> gene-symbol map
# canon_prot (reused) gives valid HGNC symbols only for antibodies named as their gene (CD14, CD19, ...).
# For CD-antigens the antibody name != gene symbol, and the training prior's mygene name_map is noisy
# (pseudogenes / antisense, e.g. CD25->ISG20). We therefore use a curated antibody->HGNC map for the
# STRING/CORUM references (canon_prot as fallback). All three methods use the SAME map, so the CMC-VAE-vs-
# baseline comparison is fair regardless of residual mapping error.
CURATED = {
 'B7-H4':'VTCN1','CD123':'IL3RA','CD141':'THBD','CD15':'FUT4','CD158':'KIR2DL1','CD158b':'KIR2DL3','CD16':'FCGR3A',
 'CD161':'KLRB1','CD20':'MS4A1','CD21':'CR2','CD235ab':'GYPA','CD25':'IL2RA','CD274':'CD274','CD303':'CLEC4C',
 'CD304':'NRP1','CD3-1':'CD3E','CD4-1':'CD4','CD44-1':'CD44','CD45-1':'PTPRC','CD45RA':'PTPRC','CD45RO':'PTPRC',
 'CD56-1':'NCAM1','CD64':'FCGR1A','CD71':'TFRC','CD79b':'CD79B','CD8':'CD8A','CD103':'ITGAE','CD106':'VCAM1',
 'CD107a':'LAMP1','CD112':'NECTIN2','CD117':'KIT','CD11a/CD18':'ITGAL','CD11b-1':'ITGAM','CD11c':'ITGAX',
 'CD122':'IL2RB','CD124':'IL4R','CD127':'IL7R','CD133-1':'PROM1','CD137':'TNFRSF9','CD138-1':'SDC1','CD144':'CDH5',
 'CD146':'MCAM','CD152':'CTLA4','CD154':'CD40LG','CD155':'PVR','CD158e1':'KIR3DL1','CD169':'SIGLEC1','CD178':'FASLG',
 'CD18':'ITGB2','CD184':'CXCR4','CD185':'CXCR5','CD193':'CCR3','CD194':'CCR4','CD195':'CCR5','CD196':'CCR6',
 'CD204':'MSR1','CD206':'MRC1','CD223':'LAG3','CD252':'TNFSF4','CD26-1':'DPP4','CD267':'TNFRSF13B','CD268':'TNFRSF13C',
 'CD269':'TNFRSF17','CD272':'BTLA','CD273':'PDCD1LG2','CD275-1':'ICOSLG','CD278':'ICOS','CD279':'PDCD1','CD29':'ITGB1',
 'CD294':'PTGDR2','CD30':'TNFRSF8','CD305':'LAIR1','CD307d':'FCRL4','CD307e':'FCRL5','CD309':'KDR','CD31':'PECAM1',
 'CD314':'KLRK1','CD319':'SLAMF7','CD324':'CDH1','CD335':'NCR1','CD337':'NCR3','CD35':'CR1','CD357':'TNFRSF18',
 'CD366':'HAVCR2','CD370':'CLEC9A','CD38-1':'CD38','CD39':'ENTPD1','CD41':'ITGA2B','CD49b':'ITGA2','CD49d':'ITGA4',
 'CD54':'ICAM1','CD57':'B3GAT1','CD62P':'SELP','CD66a/c/e':'CEACAM1','CD66b':'CEACAM8','CD73':'NT5E','CD90':'THY1',
 'CD95':'FAS','CD98':'SLC3A2','CLEC12A':'CLEC12A','CX3CR1':'CX3CR1','HLA-DR':'HLA-DRA','IgD':'IGHD','IgM':'IGHM',
 'TIGIT':'TIGIT','TSLPR':'CRLF2','XCR1':'XCR1','Podoplanin':'PDPN','Integrin-7':'ITGB7','LOX-1':'OLR1','TCR-1':'TRAC',
 'TCR-2':'TRBC1','TCR-V-7.2':'TRAV1-2','TCR-V-9':'TRGV9',
}
sym = [CURATED.get(p, canon_prot(p)) for p in prots]
gene2idx = {}
for i, s in enumerate(sym):
    if s: gene2idx.setdefault(s, []).append(i)
uniq_genes = sorted(gene2idx)
out["panel_gene_map"] = {"n_proteins": npr, "n_unique_genes": len(uniq_genes), "n_curated": int(sum(p in CURATED for p in prots))}

# ----------------------------------------------------------------------------- references (cached, else network)
REF_CACHE = f"{BASE}/ppi_ref_nets.npz"
refs = {}   # name -> bool [npr,npr] symmetric mask

def sym_mask_from_genepairs(pairs):
    m = np.zeros((npr, npr), dtype=bool)
    for a, b in pairs:
        if a in gene2idx and b in gene2idx and a != b:
            for i in gene2idx[a]:
                for j in gene2idx[b]:
                    if i != j: m[i, j] = m[j, i] = True
    return m

# OmniPath = the model's own protein prior mask restricted to the panel (EXACTLY the B_PPI reference)
try:
    d = torch.load(cfg["prior_prot_path"], map_location="cpu")
    mp = d["prior_mask"]; mp = mp.numpy() if torch.is_tensor(mp) else np.asarray(mp)
    nmp = list(map(str, d.get("protein_names", [])))
    if nmp != prots and set(prots) <= set(nmp):
        pi = {p: i for i, p in enumerate(nmp)}; perm = [pi[p] for p in prots]; mp = mp[np.ix_(perm, perm)]
    refs["OmniPath"] = mp.astype(bool)
except Exception as e:
    out["omnipath_err"] = str(e)[:120]

if os.path.exists(REF_CACHE):
    z = np.load(REF_CACHE, allow_pickle=True)
    if list(map(str, z["prots"])) == prots:
        for k in z.files:
            if k in ("prots",): continue
            refs[k] = z[k].astype(bool)
        out["refs_from_cache"] = True

if "STRING" not in refs or "CORUM" not in refs:
    # STRING high-confidence (>=700)
    try:
        import requests, pandas as pd
        r = requests.post("https://string-db.org/api/tsv/network",
                          data={"identifiers": "%0d".join(uniq_genes), "species": 9606, "required_score": 700}, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="\t")
        refs["STRING"] = sym_mask_from_genepairs(zip(df["preferredName_A"].astype(str), df["preferredName_B"].astype(str)))
    except Exception as e:
        out["string_err"] = str(e)[:120]
    # CORUM protein complexes (within-complex pairs = positives), via OmniPath Complexes
    try:
        from omnipath.requests import Complexes
        cx = Complexes.get(); cx = cx[cx["sources"].astype(str).str.contains("CORUM")]
        pairs = []
        for comp in cx["components_genesymbols"].dropna().astype(str):
            members = [g for g in comp.split("_") if g in gene2idx]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    pairs.append((members[a], members[b]))
        refs["CORUM"] = sym_mask_from_genepairs(pairs)
    except Exception as e:
        out["corum_err"] = str(e)[:120]
    # persist cache (STRING/CORUM only; OmniPath comes from the prior each run)
    save = {k: refs[k] for k in ("STRING", "CORUM") if k in refs}
    if save:
        np.savez_compressed(REF_CACHE, prots=np.array(prots), **save)

out["ref_edge_counts"] = {k: int((m | m.T).sum() // 2) for k, m in refs.items()}

# ----------------------------------------------------------------------------- baselines: raw coexpr + totalVI
GPP_CACHE = f"{BASE}/totalvi_gpp.npz"
G_pp_raw = None; G_pp_tv = None
if not os.path.exists(GPP_CACHE) and os.path.exists(TVI_MODEL) and os.path.exists(SCVI_PY):
    helper = f'''
import os, numpy as np, scanpy as sc
from sklearn.model_selection import train_test_split
import scvi
NC = {N_CORR}; SEED = 42
ad = sc.read_h5ad("{TVI_INPUT}", backed="r"); N = ad.n_obs
idx = np.arange(N); idx_tr, _ = train_test_split(idx, test_size=0.2, random_state=SEED)
sub = np.random.RandomState(SEED).choice(len(idx_tr), size=min(NC, len(idx_tr)), replace=False)
gpos = np.sort(idx_tr[sub])
adt = ad[gpos].to_memory().copy()
scvi.model.TOTALVI.setup_anndata(adt, layer="counts", protein_expression_obsm_key="protein_expression", batch_key="donor")
m = scvi.model.TOTALVI.load("{TVI_MODEL}", adata=adt)
_, pden = m.get_normalized_expression(adt, n_samples=1, return_numpy=True)
pden = np.asarray(pden, dtype=np.float32)
# matched raw CLR proteins for the SAME cells
ce = sc.read_h5ad("{CITESEQ}", backed="r")
pos = {{n: i for i, n in enumerate(map(str, ce.obs_names))}}
rows = np.array([pos[n] for n in map(str, adt.obs_names)])
order = np.argsort(rows); Xclr = np.asarray(ce.X[rows[order]]); inv = np.argsort(order); Xclr = Xclr[inv].astype(np.float32)
def zc(M): return (M - M.mean(0)) / (M.std(0) + 1e-8)
n = pden.shape[0]
Gtv = (zc(pden).T @ zc(pden)) / (n - 1); np.fill_diagonal(Gtv, 0.0)
Graw = (zc(Xclr).T @ zc(Xclr)) / (n - 1); np.fill_diagonal(Graw, 0.0)
np.savez_compressed("{GPP_CACHE}", prots=np.array(list(map(str, adt.obsm["protein_expression"].columns))),
    G_pp_tv=Gtv.astype(np.float32), G_pp_raw=Graw.astype(np.float32), n_cells=np.array([n]))
print("HELPER_OK", n)
'''
    tf = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False); tf.write(helper); tf.close()
    print(f"[totalVI] running denoised-protein inference in scvi-env ({N_CORR} cells)...", flush=True)
    try:
        p = subprocess.run([SCVI_PY, tf.name], capture_output=True, text=True, timeout=2400)
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-3:])
        print(f"[totalVI] {tail}", flush=True)
        out["totalvi_helper_ok"] = "HELPER_OK" in p.stdout
    except Exception as e:
        out["totalvi_helper_err"] = str(e)[:200]
    finally:
        os.unlink(tf.name)

if os.path.exists(GPP_CACHE):
    z = np.load(GPP_CACHE, allow_pickle=True)
    if list(map(str, z["prots"])) == prots:
        G_pp_tv = z["G_pp_tv"]; G_pp_raw = z["G_pp_raw"]; out["totalvi_n_cells"] = int(z["n_cells"][0])
    else:
        out["gpp_cache_prot_mismatch"] = True

# fallback: compute raw coexpr in base env from CLR if the cache is unavailable
if G_pp_raw is None:
    print("[coexpr] computing raw protein coexpression in base env (totalVI cache unavailable)...", flush=True)
    ce = sc.read_h5ad(CITESEQ, backed="r")
    rows = np.sort(np.random.RandomState(42).choice(ce.n_obs, size=min(N_CORR, ce.n_obs), replace=False))
    Xclr = np.asarray(ce.X[rows]).astype(np.float32)
    Z = (Xclr - Xclr.mean(0)) / (Xclr.std(0) + 1e-8)
    G_pp_raw = (Z.T @ Z) / (len(rows) - 1); np.fill_diagonal(G_pp_raw, 0.0)
    out["raw_coexpr_fallback_base_env"] = True

# ----------------------------------------------------------------------------- Task 1+2: recovery scoring
methods = {"CMC_VAE_Gpp": G_pp, "raw_coexpr": G_pp_raw, "totalVI": G_pp_tv}
ppi = {}
for refname, mask in refs.items():
    ppi[refname] = score_all(methods, mask, npr)
out["PPI_recovery"] = ppi

# added-value deltas: CMC-VAE vs the raw-protein-coexpr baseline (the crux question)
delta = {}
for refname, r in ppi.items():
    a = (r.get("CMC_VAE_Gpp") or {}).get("auroc"); b = (r.get("raw_coexpr") or {}).get("auroc")
    t = (r.get("totalVI") or {}).get("auroc")
    delta[refname] = {"auroc_CMC_minus_coexpr": (None if a is None or b is None else round(a - b, 4)),
                      "auroc_CMC_minus_totalVI": (None if a is None or t is None else round(a - t, 4))}
out["PPI_added_value_vs_coexpr"] = delta

# how much of G_pp is explained by raw coexpr (are they the same object?)
if G_pp_raw is not None:
    off = ~np.eye(npr, dtype=bool)
    gpp_s = (np.abs(G_pp) + np.abs(G_pp).T)[off]; raw_s = np.abs(G_pp_raw)[off]
    out["G_pp_vs_raw_coexpr_spearman"] = float(spearmanr(gpp_s, raw_s).correlation)

# ----------------------------------------------------------------------------- Task 3: feedback W_pr
# cognate pairs (gene_idx, prot_idx) via canon_prot -> model gene vocab (EXACT causal_eval_suite logic)
gup = {g.upper(): i for i, g in enumerate(genes)}
cognate = []
for j, p in enumerate(prots):
    c = canon_prot(p)
    for cand in (c, c.replace("CD", ""), c + "E", c + "A"):
        if cand in gup: cognate.append((gup[cand], j)); break

def cognate_enrichment(Mat, pair_axis):
    """Mat oriented so Mat[key, gene] ; pair_axis picks (gene,prot)->(row,col). Returns enrichment + perm-null p."""
    matched, ranks, bg = [], [], []
    cog_by_key = {}
    idxpairs = [(p, g) for (g, p) in cognate] if pair_axis == "prot_row" else [(g, p) for (g, p) in cognate]
    for key, tgt in idxpairs: cog_by_key.setdefault(key, set()).add(tgt)
    for key, tgt in idxpairs:
        row = np.abs(Mat[key])
        matched.append(row[tgt])
        bg.append(np.delete(row, list(cog_by_key[key])))
        ranks.append(float((row > row[tgt]).sum()) / len(row))
    matched = np.array(matched); bg = np.concatenate(bg)
    u, pv = mannwhitneyu(matched, bg, alternative="greater")
    obs = float(np.median(matched))
    # permutation null: shuffle the target index within each row's full range
    rng = np.random.RandomState(0); B = 2000; ncol = Mat.shape[1]; cnt = 0
    keys = [k for k, _ in idxpairs]
    for _ in range(B):
        perm_med = np.median([np.abs(Mat[k])[rng.randint(ncol)] for k in keys])
        if perm_med >= obs: cnt += 1
    return dict(n_pairs=len(idxpairs), matched_median=obs, bg_median=float(np.median(bg)),
                fold=float(obs / (np.median(bg) + 1e-12)), mwu_p=float(pv),
                median_cognate_rank_frac=float(np.median(ranks)), top10pct_cognate=float(np.mean(np.array(ranks) < 0.1)),
                perm_null_p=float((cnt + 1) / (B + 1)))

fb = {
    "gini_G_pr": gini(G_pr), "gini_W_pr": gini(W_pr_np),
    "gini_G_rp": gini(G_rp), "gini_W_rp": gini(W_rp_np),
    "reverse_cognate_Gpr": cognate_enrichment(G_pr, "prot_row"),   # protein X -> its own transcript X (feedback)
    "forward_cognate_Grp": cognate_enrichment(G_rp.T, "prot_row"),  # transcript X -> protein X (translation, for contrast)
}
# magnitude: translation (W_rp / G_rp) vs feedback (W_pr / G_pr)
fb["magnitude"] = {
    "mean_abs_W_rp": float(np.abs(W_rp_np).mean()), "mean_abs_W_pr": float(np.abs(W_pr_np).mean()),
    "mean_abs_G_rp": float(np.abs(G_rp).mean()), "mean_abs_G_pr": float(np.abs(G_pr).mean()),
    "Grp_over_Gpr": float(np.abs(G_rp).mean() / (np.abs(G_pr).mean() + 1e-12)),
    "Wrp_over_Wpr": float(np.abs(W_rp_np).mean() / (np.abs(W_pr_np).mean() + 1e-12)),
    "confound_note": f"latent dims asymmetric ({cfg['latent_dim_rna']} RNA / {cfg['latent_dim_prot']} prot); "
                     "W_rp is [Lr,Lp], W_pr is [Lp,Lr] -> raw magnitudes not directly comparable.",
}
# unconfounded symmetric 128/128 model if trained
P128 = os.path.join(RES, "linfb_2c_fb10_b20_v4_p128")
if os.path.exists(f"{P128}/final_model.pt"):
    try:
        c2 = json.load(open(f"{P128}/config.json"))
        m2 = CrossModalCausalLDVAE(ng, npr, 1, latent_dim_rna=c2["latent_dim_rna"], latent_dim_prot=c2["latent_dim_prot"],
            recon_likelihood="mse", scm_nonlinear=c2.get("scm_nonlinear", False), phi=c2.get("phi", "tanh"))
        m2.load_state_dict(torch.load(f"{P128}/final_model.pt", map_location="cpu"), strict=False); m2.eval()
        with torch.no_grad():
            _, _, w_rp2, w_pr2 = m2.scm.get_block_adjacencies(tau=0.5)
            Dr2 = m2._dec_r_w().detach().float(); Dp2 = m2.dec_prot.weight.detach().float()
            g_rp2 = (Dr2 @ w_rp2.float() @ Dp2.T).cpu().numpy(); g_pr2 = (Dp2 @ w_pr2.float() @ Dr2.T).cpu().numpy()
        fb["magnitude_unconfounded_p128"] = {
            "mean_abs_W_rp": float(np.abs(w_rp2.cpu().numpy()).mean()), "mean_abs_W_pr": float(np.abs(w_pr2.cpu().numpy()).mean()),
            "mean_abs_G_rp": float(np.abs(g_rp2).mean()), "mean_abs_G_pr": float(np.abs(g_pr2).mean()),
            "Grp_over_Gpr": float(np.abs(g_rp2).mean() / (np.abs(g_pr2).mean() + 1e-12))}
    except Exception as e:
        fb["p128_err"] = str(e)[:150]
else:
    fb["magnitude_unconfounded_p128"] = "NOT AVAILABLE (linfb_2c_fb10_b20_v4_p128/final_model.pt not yet written)"
out["D_feedback"] = fb

# ----------------------------------------------------------------------------- save json
outpath = f"{BASE}/ppi_feedback_eval.json"
json.dump(out, open(outpath, "w"), indent=2)
print(f"\n=== ppi_feedback_eval: {run} (prior_off={out['prior_off']}) ===")
print("REF edges:", out["ref_edge_counts"])
print("\nPPI RECOVERY (AUROC):")
print(f"  {'reference':<10} {'CMC-VAE Gpp':>12} {'raw coexpr':>12} {'totalVI':>10}")
for rn, r in ppi.items():
    def a(m): v = (r.get(m) or {}); return f"{v['auroc']:.3f}" if v.get("auroc") is not None else "  n/a"
    print(f"  {rn:<10} {a('CMC_VAE_Gpp'):>12} {a('raw_coexpr'):>12} {a('totalVI'):>10}")
print("\nAdded value (CMC-VAE AUROC - coexpr AUROC):", {k: v["auroc_CMC_minus_coexpr"] for k, v in delta.items()})
print("G_pp vs raw-coexpr spearman:", round(out.get("G_pp_vs_raw_coexpr_spearman", float('nan')), 3))
print("Reverse-cognate (W_pr, prot->own transcript): fold=%.2f mwu_p=%.1e perm_p=%.3f" % (
    fb["reverse_cognate_Gpr"]["fold"], fb["reverse_cognate_Gpr"]["mwu_p"], fb["reverse_cognate_Gpr"]["perm_null_p"]))
print(f"-> {outpath}")

# ----------------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
C = {"CMC_VAE_Gpp": "#1f4e79", "raw_coexpr": "#9aa0a6", "totalVI": "#b5651d"}
LAB = {"CMC_VAE_Gpp": "CMC-VAE  G_pp", "raw_coexpr": "raw protein coexpr", "totalVI": "totalVI"}
ref_order = [r for r in ["STRING", "OmniPath", "CORUM"] if r in ppi]
meth_order = ["CMC_VAE_Gpp", "raw_coexpr", "totalVI"]
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.4))
xs = np.arange(len(ref_order)); w = 0.26
for mi, m in enumerate(meth_order):
    vals = [((ppi[r].get(m) or {}).get("auroc") or np.nan) for r in ref_order]
    axs[0].bar(xs + (mi - 1) * w, vals, w, color=C[m], label=LAB[m])
    for xi, v in zip(xs + (mi - 1) * w, vals):
        if not np.isnan(v): axs[0].text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
axs[0].axhline(0.5, ls="--", lw=1, color="k"); axs[0].text(len(ref_order) - 0.5, 0.51, "chance", fontsize=9, ha="right")
axs[0].set_xticks(xs); axs[0].set_xticklabels([f"{r}\n({out['ref_edge_counts'].get(r,0)} edges)" for r in ref_order])
axs[0].set_ylabel("PPI recovery AUROC"); axs[0].set_ylim(0.4, max(0.9, np.nanmax([
    (ppi[r].get(m) or {}).get("auroc") or 0 for r in ref_order for m in meth_order]) + 0.06))
axs[0].set_title("Does latent A_pp recover PPI above protein coexpression?"); axs[0].legend(fontsize=10, loc="upper right")
# panel 2: AUPRC-over-random (log)
for mi, m in enumerate(meth_order):
    vals = [((ppi[r].get(m) or {}).get("auprc_ratio") or np.nan) for r in ref_order]
    axs[1].bar(xs + (mi - 1) * w, vals, w, color=C[m], label=LAB[m])
axs[1].axhline(1.0, ls="--", lw=1, color="k"); axs[1].text(len(ref_order) - 0.5, 1.03, "random", fontsize=9, ha="right")
axs[1].set_xticks(xs); axs[1].set_xticklabels(ref_order); axs[1].set_ylabel("AUPRC / random")
axs[1].set_title("Precision enrichment over random"); axs[1].set_yscale("log")
fig.suptitle(f"Protein-side PPI recovery of A_pp  (model {run}, prior-OFF) — vs raw coexpr & totalVI",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
figpath = f"{FIGDIR}/fig_ppi_eval.png"; fig.savefig(figpath); plt.close(fig)
print(f"-> {figpath}\nDONE")
