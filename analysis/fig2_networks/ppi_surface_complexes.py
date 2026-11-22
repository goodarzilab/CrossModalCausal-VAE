"""Biology-appropriate PPI validation for the CMC-VAE protein block (G_pp).

MOTIVATION
----------
analysis/fig2_networks/ppi_feedback_eval.py scores G_pp against STRING / OmniPath / CORUM. Those are
INTRACELLULAR-complex references, which are the WRONG gold standard for a 144-antibody
CELL-SURFACE marker panel (surface receptors rarely form the cytoplasmic complexes those
databases enumerate) -- so G_pp lands at/below chance there. This script builds the
biology-appropriate test: a hand-curated set of bona-fide CELL-SURFACE complexes /
receptor-ligand pairs whose BOTH members are on the panel, then asks whether G_pp ranks
those pairs above the rest of the panel -- and, crucially, whether it does so ABOVE
lineage co-expression (the honest null: most surface co-detection is just shared cell type).

ANALYSES
--------
  P1  Curated surface-complex recovery (primary): |G_pp| ranking of curated positive
      protein-protein edges vs all other panel pairs. AUROC + AUPRC/random + label-
      permutation null. Overlaid on the SAME positives: raw-CLR |corr| and totalVI.
  P2  Lineage-deconfounded added value (the crux): stronger baselines =
      within-celltype co-expression (regress out celltype means) and a graphical-lasso
      partial-correlation network. Re-score P1 positives. Does G_pp beat lineage co-expr?
  P3  Module recovery: spectral community detection on |G_pp| vs curated lineage modules
      (ARI / NMI) with a label-permutation null; compared to co-expression modules.

Positives / modules are keyed by PANEL PROTEIN NAME (identical order across G_pp, totalVI
and the CLR h5ad), so the CMC-VAE-vs-baseline comparison needs no antibody->gene mapping.

Outputs:  results/ppi_surface_complexes.json   and   paper/figures/_new_ppi_surface.pdf(.png)
Usage:    python analysis/fig2_networks/ppi_surface_complexes.py [results/<run>]
          (default run: results/sweep_sparse/sp_dec_gl1000)
Env:      PPI_N_CELLS (default 150000) cells subsampled from the CLR h5ad for co-expression.
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
import numpy as np, pandas as pd, scanpy as sc
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _CMC_ROOT
RES = os.path.join(REPO, "results")
CV4 = f"{_CMC_DATA}/combined_v4"
CITESEQ = f"{CV4}/citeseq_combined.h5ad"
TVI_NPZ = f"{RES}/baselines/totalvi_gpp.npz"
N_CELLS = int(os.environ.get("PPI_N_CELLS", "150000"))
SEED = 42

rd = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(RES, "sweep_sparse", "sp_dec_gl1000")
run = os.path.basename(rd)
out = {"run": run, "run_dir": rd, "n_cells_coexpr": N_CELLS, "seed": SEED}

# ---------------------------------------------------------------- reused harness (from analysis/fig2_networks/ppi_feedback_eval.py)
def canon_prot(x):  # EXACT copy from analysis/fig2_networks/ppi_feedback_eval.py
    x = re.sub(r'^AB_', '', x); x = re.sub(r'^(human|mouse)', '', x, flags=re.I)
    x = re.sub(r'-\d+$', '', x); return x.split('_')[0].upper()

# CURATED antibody->HGNC map (from analysis/fig2_networks/ppi_feedback_eval.py) -- used here only for annotation.
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

# ---------------------------------------------------------------- P1 curated SURFACE-complex gold standard
# Each entry: complex/interaction name -> (list of (protA, protB) panel-name edges, kind).
#   kind='cis'   : stable same-cell physical heterocomplex (co-detected on one cell -> co-expression expected).
#   kind='trans' : receptor-ligand pair engaged ACROSS two cells (immune synapse) -> within-cell co-expression
#                  is NOT expected; a mechanistic latent model may still link them. Reported separately.
# Every pair below was checked so BOTH members are present in the 144-antibody panel.
CURATED_COMPLEXES = {
  # --- CIS: same-cell surface heterocomplexes ---
  "BCR co-receptor (CD19/CD21/CD81)":      (["CD19-CD21", "CD19-CD81", "CD21-CD81"], "cis"),
  "TCR-CD3 (TRAC/TRBC/CD3E)":              (["CD3-1-TCR-1", "CD3-1-TCR-2", "TCR-1-TCR-2"], "cis"),
  "beta2 integrins (CD18 heterodimers)":   (["CD11a/CD18-CD18", "CD11b-1-CD18", "CD11c-CD18"], "cis"),
  "beta1/beta7 integrins":                 (["CD49d-CD29", "CD49b-CD29", "CD49d-Integrin-7", "CD103-Integrin-7"], "cis"),
  "IL-2 receptor (CD25/CD122)":            (["CD25-CD122"], "cis"),
  "Fc-gamma receptors (CD16/CD64)":        (["CD16-CD64"], "cis"),   # family co-receptors (myeloid), not a heterodimer
  # --- TRANS: receptor-ligand pairs across the immune synapse ---
  "CD28/CTLA4 - B7 costim":                (["CD28-CD80", "CD28-CD86", "CD152-CD80", "CD152-CD86"], "trans"),
  "PD-1 - PD-L axis":                      (["CD279-CD274", "CD279-CD273"], "trans"),
  "ICOS - ICOSL":                          (["CD278-CD275-1"], "trans"),
  "CD40 - CD40L":                          (["CD40-CD154"], "trans"),
  "CD27 - CD70":                           (["CD27-CD70"], "trans"),
  "FAS - FASL":                            (["CD95-CD178"], "trans"),
  "DNAM1/TIGIT/CD96 - PVR/NECTIN2":        (["CD226-CD155", "CD226-CD112", "TIGIT-CD155", "TIGIT-CD112", "CD96-CD155"], "trans"),
  "ICAM/VCAM - integrin adhesion":         (["CD54-CD11a/CD18", "CD54-CD11b-1", "CD106-CD49d", "CD106-Integrin-7"], "trans"),
  "MHC-II axis (LAG3/CD4 - HLA-DR)":       (["CD223-HLA-DR", "CD4-1-HLA-DR"], "trans"),
}

# ---------------------------------------------------------------- P3 curated lineage MODULES (panel names)
# CD16 is intentionally omitted (dual NK / CD16-monocyte membership -> ambiguous partition label).
LINEAGE_MODULES = {
  "T":       ["CD3-1", "CD4-1", "CD8", "CD2", "CD28"],
  "B":       ["CD19", "CD20", "CD21", "CD22", "CD79b"],
  "Myeloid": ["CD11b-1", "CD11c", "CD14", "CD64"],
  "NK":      ["CD56-1", "CD314", "CD335"],
}

# ---------------------------------------------------------------- load G_pp (projected D_p A_pp D_p^T)
df = pd.read_csv(f"{rd}/PPI_inferred.csv", index_col=0)
prots = list(map(str, df.columns))
assert list(map(str, df.index)) == prots, "PPI_inferred.csv must be square/aligned"
npr = len(prots)
pidx = {p: i for i, p in enumerate(prots)}
G_pp = df.values.astype(np.float64)
np.fill_diagonal(G_pp, 0.0)

def parse_edge(e):
    """'CD19-CD21' -> ('CD19','CD21'); handles panel names that themselves contain '-' or '/'."""
    for a in sorted(prots, key=len, reverse=True):
        if e.startswith(a + "-") and e[len(a) + 1:] in pidx:
            return a, e[len(a) + 1:]
    raise ValueError(f"cannot parse edge {e!r} into two panel proteins")

# build positive mask + cis/trans partition, validate membership
pos_mask = np.zeros((npr, npr), dtype=bool)
cis_mask = np.zeros((npr, npr), dtype=bool)
edge_list, missing = [], []
for name, (edges, kind) in CURATED_COMPLEXES.items():
    for e in edges:
        a, b = parse_edge(e)
        if a not in pidx or b not in pidx:
            missing.append(e); continue
        i, j = pidx[a], pidx[b]
        pos_mask[i, j] = pos_mask[j, i] = True
        if kind == "cis":
            cis_mask[i, j] = cis_mask[j, i] = True
        edge_list.append({"complex": name, "kind": kind, "a": a, "b": b,
                          "a_gene": CURATED.get(a, canon_prot(a)), "b_gene": CURATED.get(b, canon_prot(b))})
assert not missing, f"curated edges with off-panel members: {missing}"
out["curated"] = {"n_edges": len(edge_list), "n_cis": int(cis_mask[np.triu_indices(npr, 1)].sum()),
                  "n_trans": len(edge_list) - int(cis_mask[np.triu_indices(npr, 1)].sum()),
                  "n_complexes": len(CURATED_COMPLEXES), "edges": edge_list}

iu = np.triu_indices(npr, k=1)                    # unique off-diagonal pairs
lab = pos_mask[iu].astype(bool)                    # C(144,2)=10296 pairs, ~len(edge_list) positives
lab_cis = cis_mask[iu].astype(bool)
N_pairs = lab.size; K_pos = int(lab.sum())

def sym_score(M):
    """symmetric magnitude score on unique pairs."""
    S = np.abs(M) + np.abs(M).T
    return S[iu]

# ---------------------------------------------------------------- load totalVI + raw CLR co-expr baselines
G_tv = None; G_raw_npz = None
if os.path.exists(TVI_NPZ):
    z = np.load(TVI_NPZ, allow_pickle=True)
    zp = list(map(str, z["prots"]))
    perm = [zp.index(p) for p in prots] if zp != prots else slice(None)
    G_tv = z["G_pp_tv"][np.ix_(perm, perm)] if zp != prots else z["G_pp_tv"]
    G_raw_npz = z["G_pp_raw"][np.ix_(perm, perm)] if zp != prots else z["G_pp_raw"]
    out["totalvi_n_cells"] = int(z["n_cells"][0])
else:
    out["totalvi_missing"] = True

# ---------------------------------------------------------------- load CLR proteins + celltype (co-expression)
print(f"[load] {CITESEQ} (subsample {N_CELLS} cells, seed {SEED}) ...", flush=True)
ad = sc.read_h5ad(CITESEQ, backed="r")
assert list(map(str, ad.var_names)) == prots, "h5ad var order must match panel"
n_tot = ad.n_obs
rng = np.random.RandomState(SEED)
sel = np.sort(rng.choice(n_tot, size=min(N_CELLS, n_tot), replace=False))
X = np.asarray(ad.X[sel]).astype(np.float64)                      # [N,144] CLR
celltype = ad.obs["celltype"].values[sel].astype(str)
out["n_cells_loaded"] = int(X.shape[0]); out["n_celltypes"] = int(len(np.unique(celltype)))

def abscorr(M):
    C = np.corrcoef(M.T); C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0); return np.abs(C)

# marginal co-expression (recomputed on this subsample, apples-to-apples with within-lineage)
G_marg = abscorr(X)

# within-lineage co-expression: subtract per-celltype means (regress out the celltype factor), then corr
Xr = X.copy()
for ct in np.unique(celltype):
    m = celltype == ct
    Xr[m] -= X[m].mean(axis=0, keepdims=True)
G_within = abscorr(Xr)

# graphical-lasso partial-correlation network on standardized CLR proteins
G_partial = None
try:
    from sklearn.covariance import GraphicalLassoCV
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    gl = GraphicalLassoCV(alphas=6, cv=3, max_iter=200).fit(Xz)
    P = gl.precision_; d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d); np.fill_diagonal(pc, 0.0)
    G_partial = np.abs(pc)
    out["glasso_alpha"] = float(gl.alpha_)
except Exception as e:
    out["glasso_err"] = str(e)[:200]
    try:  # fallback: shrunk-covariance partial correlation (Ledoit-Wolf precision)
        from sklearn.covariance import LedoitWolf
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
        P = LedoitWolf().fit(Xz).precision_; d = np.sqrt(np.diag(P))
        pc = -P / np.outer(d, d); np.fill_diagonal(pc, 0.0)
        G_partial = np.abs(pc); out["partial_corr_fallback"] = "LedoitWolf"
    except Exception as e2:
        out["glasso_fallback_err"] = str(e2)[:200]

# ---------------------------------------------------------------- recovery scoring + permutation null
def perm_null_auc(score, labels, B=20000, seed=0):
    """label-permutation null for AUROC via rank-sum U (fast, exact-rank)."""
    r = rankdata(score); N = len(r); K = int(labels.sum())
    obs = (r[labels].sum() - K * (K + 1) / 2.0) / (K * (N - K))
    rs = np.random.RandomState(seed); cnt = 0
    for _ in range(B):
        s = rs.choice(N, K, replace=False)
        a = (r[s].sum() - K * (K + 1) / 2.0) / (K * (N - K))
        if a >= obs: cnt += 1
    return float(obs), float((cnt + 1) / (B + 1))

def recovery(score, labels):
    labels = labels.astype(bool); K = int(labels.sum()); base = labels.mean()
    auc, p = perm_null_auc(score, labels)
    return dict(auroc=round(auc, 4), auprc_ratio=round(float(average_precision_score(labels, score) / base), 3),
                perm_p=round(p, 5), n_pos=K, n_pairs=int(labels.size), prevalence=round(float(base), 5),
                auroc_sklearn=round(float(roc_auc_score(labels, score)), 4))

METHODS = {"CMC_VAE_Gpp": G_pp, "marginal_coexpr": G_marg, "within_lineage_coexpr": G_within,
           "partial_corr": G_partial, "totalVI": G_tv}
scores = {k: (sym_score(M) if M is not None else None) for k, M in METHODS.items()}

# P1: primary (all curated) + cis-only subset
out["P1_surface_complex_recovery"] = {"all_curated": {}, "cis_only": {}}
for k, s in scores.items():
    if s is None: continue
    out["P1_surface_complex_recovery"]["all_curated"][k] = recovery(s, lab)
    if lab_cis.sum() >= 3:
        out["P1_surface_complex_recovery"]["cis_only"][k] = recovery(s, lab_cis)

# P2: lineage-deconfounded added value (delta vs the stronger co-expression baselines)
p1a = out["P1_surface_complex_recovery"]["all_curated"]
g = p1a["CMC_VAE_Gpp"]["auroc"]
out["P2_lineage_deconfounded"] = {
    "auroc": {k: p1a[k]["auroc"] for k in p1a},
    "Gpp_minus_marginal":       round(g - p1a["marginal_coexpr"]["auroc"], 4),
    "Gpp_minus_within_lineage": round(g - p1a["within_lineage_coexpr"]["auroc"], 4) if "within_lineage_coexpr" in p1a else None,
    "Gpp_minus_partial_corr":   round(g - p1a["partial_corr"]["auroc"], 4) if "partial_corr" in p1a else None,
    "Gpp_minus_totalVI":        round(g - p1a["totalVI"]["auroc"], 4) if "totalVI" in p1a else None,
}
# how much is G_pp just re-encoding co-expression?
out["Gpp_vs_marginal_spearman"] = round(float(spearmanr(scores["CMC_VAE_Gpp"], scores["marginal_coexpr"]).correlation), 3)
out["Gpp_vs_within_spearman"] = round(float(spearmanr(scores["CMC_VAE_Gpp"], scores["within_lineage_coexpr"]).correlation), 3)

# ---------------------------------------------------------------- P3: module recovery (spectral) vs lineage modules
labeled = [(p, lin) for lin, ps in LINEAGE_MODULES.items() for p in ps if p in pidx]
lab_prots = [p for p, _ in labeled]
lab_ix = np.array([pidx[p] for p in lab_prots])
true_lin = np.array([lin for _, lin in labeled])
_, true_int = np.unique(true_lin, return_inverse=True)
K_mod = len(LINEAGE_MODULES)

def module_recovery(M, k_full, B=20000, seed=0):
    """spectral-cluster the full 144 |M| graph into k_full comms; score ARI/NMI on labeled markers."""
    A = np.abs(M) + np.abs(M).T; np.fill_diagonal(A, 0.0)
    sc_ = SpectralClustering(n_clusters=k_full, affinity="precomputed", assign_labels="discretize",
                             random_state=seed).fit(A)
    pred = sc_.labels_[lab_ix]
    ari = adjusted_rand_score(true_int, pred); nmi = normalized_mutual_info_score(true_int, pred)
    rs = np.random.RandomState(seed); n = len(pred); cnt_a = cnt_n = 0
    for _ in range(B):
        pp = rs.permutation(true_int)
        if adjusted_rand_score(pp, pred) >= ari: cnt_a += 1
        if normalized_mutual_info_score(pp, pred) >= nmi: cnt_n += 1
    return dict(ari=round(float(ari), 4), nmi=round(float(nmi), 4),
                ari_perm_p=round((cnt_a + 1) / (B + 1), 5), nmi_perm_p=round((cnt_n + 1) / (B + 1), 5),
                k_clusters=k_full)

out["P3_module_recovery"] = {"n_labeled_proteins": len(lab_prots), "n_modules": K_mod,
                             "labeled": {p: lin for p, lin in labeled}, "by_method": {}}
mod_methods = {"CMC_VAE_Gpp": G_pp, "marginal_coexpr": G_marg,
               "within_lineage_coexpr": G_within, "partial_corr": G_partial}
for k, M in mod_methods.items():
    if M is None: continue
    out["P3_module_recovery"]["by_method"][k] = {str(kf): module_recovery(M, kf) for kf in (K_mod,)}

# ---------------------------------------------------------------- save JSON
os.makedirs(RES, exist_ok=True)
outpath = f"{RES}/ppi_surface_complexes.json"
json.dump(out, open(outpath, "w"), indent=2)

# ---------------------------------------------------------------- console summary
def a(d, k): return d[k]["auroc"] if k in d else float("nan")
print(f"\n=== PPI surface-complex validation: {run} ===")
print(f"curated positives: {out['curated']['n_edges']} edges "
      f"({out['curated']['n_cis']} cis / {out['curated']['n_trans']} trans) over {N_pairs} panel pairs")
print("\nP1 surface-complex recovery AUROC (all curated):")
for k in ["CMC_VAE_Gpp", "marginal_coexpr", "within_lineage_coexpr", "partial_corr", "totalVI"]:
    if k in p1a: print(f"   {k:<24} AUROC={p1a[k]['auroc']:.3f}  AUPRC/rand={p1a[k]['auprc_ratio']:.2f}  perm_p={p1a[k]['perm_p']:.4f}")
print("\nP2 lineage-deconfounded deltas (G_pp - baseline AUROC):")
print(f"   vs marginal={out['P2_lineage_deconfounded']['Gpp_minus_marginal']:+.3f}  "
      f"vs within-lineage={out['P2_lineage_deconfounded']['Gpp_minus_within_lineage']:+.3f}  "
      f"vs partial-corr={out['P2_lineage_deconfounded']['Gpp_minus_partial_corr']:+.3f}  "
      f"vs totalVI={out['P2_lineage_deconfounded']['Gpp_minus_totalVI']:+.3f}")
print(f"   G_pp~marginal spearman={out['Gpp_vs_marginal_spearman']}  G_pp~within spearman={out['Gpp_vs_within_spearman']}")
print("\nP3 module recovery (ARI / NMI vs lineage, spectral K=%d):" % K_mod)
for k, d in out["P3_module_recovery"]["by_method"].items():
    r = d[str(K_mod)]; print(f"   {k:<24} ARI={r['ari']:.3f} (p={r['ari_perm_p']:.3f})  NMI={r['nmi']:.3f} (p={r['nmi_perm_p']:.3f})")
print(f"\n-> {outpath}")

# ---------------------------------------------------------------- figure (manuscript style)
sys.path.insert(0, os.path.join(_CMC_ROOT, "analysis", "figures", "original"))
from style import plt, C, panel_label, save
import numpy as np

FIG = os.path.join(REPO, "paper", "figures", "_new_ppi_surface.pdf")
os.makedirs(os.path.dirname(FIG), exist_ok=True)
COL = {"CMC_VAE_Gpp": C["model"], "marginal_coexpr": C["muted"], "within_lineage_coexpr": C["accent"],
       "partial_corr": C["pos"], "totalVI": C["base"]}
LBL = {"CMC_VAE_Gpp": "CMC-VAE $G_{pp}$", "marginal_coexpr": "marginal\nco-expr", "within_lineage_coexpr": "within-lineage\nco-expr",
       "partial_corr": "partial\ncorr", "totalVI": "totalVI"}

fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.8))

# Panel A: surface-complex recovery AUROC (G_pp vs marginal co-expr vs totalVI) + chance + perm-p
axA = axs[0]; mA = [m for m in ["CMC_VAE_Gpp", "marginal_coexpr", "totalVI"] if m in p1a]
vA = [p1a[m]["auroc"] for m in mA]; pA = [p1a[m]["perm_p"] for m in mA]
bars = axA.bar(range(len(mA)), vA, color=[COL[m] for m in mA], width=0.66)
for i, (v, p) in enumerate(zip(vA, pA)):
    star = "n.s." if p > 0.05 else ("p<0.001" if p < 1e-3 else f"p={p:.3f}")
    axA.text(i, v + 0.015, f"{v:.2f}\n{star}", ha="center", va="bottom", fontsize=6.5)
axA.axhline(0.5, ls="--", lw=0.9, color=C["muted"]); axA.text(len(mA) - 0.5, 0.505, "chance", fontsize=6.5, ha="right", color=C["muted"])
axA.set_xticks(range(len(mA))); axA.set_xticklabels([LBL[m] for m in mA], fontsize=7)
axA.set_ylim(0.4, max(0.9, max(vA) + 0.12)); axA.set_ylabel("surface-complex AUROC")
axA.set_title(f"A  Surface-complex recovery\n({out['curated']['n_edges']} curated edges)", loc="left")
panel_label(axA, "")

# Panel B: lineage-deconfounded comparison (same positives, stronger baselines)
axB = axs[1]; mB = [m for m in ["CMC_VAE_Gpp", "marginal_coexpr", "within_lineage_coexpr", "partial_corr"] if m in p1a]
vB = [p1a[m]["auroc"] for m in mB]
axB.bar(range(len(mB)), vB, color=[COL[m] for m in mB], width=0.7)
for i, v in enumerate(vB): axB.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8)
axB.axhline(0.5, ls="--", lw=0.9, color=C["muted"]); axB.text(len(mB) - 0.5, 0.505, "chance", fontsize=6.5, ha="right", color=C["muted"])
axB.set_xticks(range(len(mB))); axB.set_xticklabels([LBL[m] for m in mB], fontsize=6.6)
axB.set_ylim(0.4, max(0.9, max(vB) + 0.1)); axB.set_ylabel("surface-complex AUROC")
axB.set_title("B  Lineage-deconfounded\nadded value", loc="left")

# Panel C: module recovery ARI / NMI (grouped) with null (chance ~ 0)
axC = axs[2]; mC = [m for m in ["CMC_VAE_Gpp", "marginal_coexpr", "within_lineage_coexpr"] if m in out["P3_module_recovery"]["by_method"]]
km = str(K_mod); xC = np.arange(len(mC)); w = 0.36
ari = [out["P3_module_recovery"]["by_method"][m][km]["ari"] for m in mC]
nmi = [out["P3_module_recovery"]["by_method"][m][km]["nmi"] for m in mC]
axC.bar(xC - w / 2, ari, w, color=[COL[m] for m in mC], label="ARI")
axC.bar(xC + w / 2, nmi, w, color=[COL[m] for m in mC], alpha=0.5, hatch="//", label="NMI")
for i, m in enumerate(mC):
    p = out["P3_module_recovery"]["by_method"][m][km]["ari_perm_p"]
    axC.text(xC[i] - w / 2, ari[i] + 0.02, f"{ari[i]:.2f}", ha="center", va="bottom", fontsize=6.2)
    if p < 0.05: axC.text(xC[i] - w / 2, -0.06, "*", ha="center", va="top", fontsize=10, color=COL[m])
axC.axhline(0.0, lw=0.8, color=C["ink"])
axC.set_xticks(xC); axC.set_xticklabels([LBL[m] for m in mC], fontsize=6.6)
axC.set_ylabel("cluster agreement"); axC.set_ylim(-0.1, 1.0)
axC.set_title(f"C  Lineage-module recovery\n(spectral, K={K_mod}; * perm p<0.05)", loc="left")
from matplotlib.patches import Patch
axC.legend(handles=[Patch(fc=C["ink"], label="ARI"), Patch(fc=C["ink"], alpha=0.5, hatch="//", label="NMI")],
           loc="upper right", fontsize=6.5)

fig.tight_layout()
save(fig, FIG)
print("DONE")
