"""
Single-experiment analysis: generate training curves + perturbation validation plots
for one results directory.

Usage:
  python run_analysis.py <results_dir>
  python run_analysis.py results_combined_gs17

Adapted from latent_dag_vae_grid_search/run_grid_search_analysis_v2.py
to work on individual experiment directories from the main VAE_analysis folder.
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
import sys, os, json, warnings, re, argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, rankdata, norm
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(_CMC_ROOT, "src"))
from model import CrossModalCausalLDVAE
warnings.filterwarnings('ignore')

# ==========================================
# PLOTTING SETUP & CONSTANTS
# ==========================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
sns.set_context("paper")
sns.set_style("white")

TITLE_SIZE  = 22
LABEL_SIZE  = 18
TICK_SIZE   = 14
LEGEND_SIZE = 14
LINE_WIDTH  = 2.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths — train paths come from config.json inside the results directory;
# these are fallbacks. Perturb-CITE-seq (Frangieh) lives on lab storage.
DATA_ROOT = os.environ.get("CMC_DATA_ROOT",
                           f"{_CMC_DATA}")
DEFAULT_TRAIN_RNA_PATH = os.path.join(DATA_ROOT, "combined3", "rna_combined.h5ad")
DEFAULT_TRAIN_PROT_PATH = os.path.join(DATA_ROOT, "combined3", "citeseq_combined.h5ad")
PERTURB_RNA_PATH = os.environ.get(
    "PERTURB_RNA_PATH", os.path.join(DATA_ROOT, "perturb_citeseq", "perturb_cite_seq_rna_processed.h5ad"))
PERTURB_PROT_PATH = os.environ.get(
    "PERTURB_PROT_PATH", os.path.join(DATA_ROOT, "perturb_citeseq", "perturb_cite_seq_protein_processed.h5ad"))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIN_CELLS_PER_PERTURBATION = 30
DE_PVAL_THRESHOLD = 0.05
# Effect-size floor on |LFC| for the DE ground-truth (in addition to raw p<0.05). Modality-specific:
# RNA sets are large so a 0.25 floor (Seurat default) removes trivial-effect genes; protein CLR values
# are compressed so protein uses a lower floor. Override via env for the eval re-run / sensitivity.
DE_LFC_RNA = float(os.environ.get("CMC_DE_LFC_RNA", "0.25"))
DE_LFC_PROT = float(os.environ.get("CMC_DE_LFC_PROT", "0.1"))


def resolve_ckpt(rd):
    """FINAL checkpoint by default (fully-regularized, last epoch); fallback best_model.pt for old runs.
    Override with CMC_CKPT=best."""
    pref=os.environ.get("CMC_CKPT","final")
    for n in [f"{pref}_model.pt","final_model.pt","best_model.pt"]:
        p=os.path.join(rd,n)
        if os.path.exists(p): return p
    return os.path.join(rd,"best_model.pt")


def load_ctx_model(ModelCls, results_dir, n_g, n_p, cfg):
    """Build a model matching the trained architecture (context_adapt / multiscale / coupled read from
    cfg + checkpoint), load weights, set eval_context from $CMC_EVAL_CONTEXT. Backward compatible
    (plain models: all flags off, eval_context unused)."""
    import os
    sd = torch.load(resolve_ckpt(results_dir), map_location=DEVICE)
    ctx = cfg.get("context_adapt")
    ccm = cfg.get("context_cross_modal", False)
    n_ctx = 1
    if ctx == "latent" and "ctx_shift" in sd: n_ctx = sd["ctx_shift"].shape[0]
    elif ctx == "decoder" and "ctx_dec_bias_r" in sd: n_ctx = sd["ctx_dec_bias_r"].shape[0]
    if ccm and "scm.weights_rp" in sd and sd["scm.weights_rp"].dim() == 3:
        n_ctx = max(n_ctx, sd["scm.weights_rp"].shape[0])
    _dm = None
    if cfg.get("dec_mask_rna_path"):
        _mk = torch.load(cfg["dec_mask_rna_path"], map_location="cpu")
        _dm = _mk["mask"] if isinstance(_mk, dict) else _mk
    m = ModelCls(n_g, n_p, num_batches=1,
                 latent_dim_rna=cfg["latent_dim_rna"], latent_dim_prot=cfg["latent_dim_prot"],
                 recon_likelihood=cfg.get("recon_likelihood", "mse"), gate_mode=cfg.get("gate_mode", "gumbel"),
                 multiscale_A=cfg.get("multiscale_A", False), coupled_noise=cfg.get("coupled_noise", False),
                 context_adapt=ctx, n_context=n_ctx, context_cross_modal=ccm,
                 scm_nonlinear=cfg.get("scm_nonlinear", False), phi=cfg.get("phi", "tanh"),
                 n_fp_iters=cfg.get("n_fp_iters", 20),
                 dec_mask_rna=_dm, dec_mask_lambda=cfg.get("dec_mask_lambda", 0.1),
                 dec_mask_freeze=cfg.get("dec_mask_freeze", False),
                 eps_bottleneck=cfg.get("eps_bottleneck", False),
                 eps_gate_init=cfg.get("eps_gate_init", 2.0),
                 eps_bottleneck_target=cfg.get("eps_bottleneck_target", "both")).to(DEVICE)
    sd.pop("context_of_batch", None)   # training-only buffer ([n_batches]); eval uses eval_context
    m.load_state_dict(sd, strict=False); m.eval()
    m.eval_context = int(os.environ.get("CMC_EVAL_CONTEXT", "0"))
    if ctx or ccm:
        print(f"  [load_ctx_model] context_adapt={ctx} context_cross_modal={ccm} n_context={n_ctx} eval_context={m.eval_context}")
    return m
MAX_CONTROL_CELLS = 20000  # match reference run_grid_search_analysis_v2.py exactly (seed 42 subsample)
TOP_K_LIST = [10, 20, 50, 100, 200]


# ==========================================
# PLOT 1: TRAINING CURVES
# ==========================================
def plot_training_curves(results_dir, exp_name):
    metrics_path = os.path.join(results_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"  [SKIP] No metrics.json in {results_dir}")
        return False

    with open(metrics_path) as f:
        metrics = json.load(f)

    epochs = [m + 1 for m in range(len(metrics['train']))]
    train_m = {k: [m[k] for m in metrics['train']] for k in metrics['train'][0].keys() if k != 'epoch'}
    val_m = {k: [m[k] for m in metrics['val']] for k in metrics['val'][0].keys() if k != 'epoch'}

    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle(f'Training Metrics: {exp_name}', fontsize=TITLE_SIZE, fontweight='bold', y=1.02)

    plot_configs = [
        ('rec', 0, 0, 'Reconstruction Loss', 'Causal Reconstruction'),
        ('kl_r', 0, 1, 'KL Divergence', 'KL Divergence (RNA)'),
        ('kl_p', 0, 2, 'KL Divergence', 'KL Divergence (Protein)'),
        ('dag', 1, 0, 'DAGMA', 'DAGMA Acyclicity'),
        ('intra_sparse', 1, 1, 'Sparsity Loss', 'Intra-Modality Sparsity'),
        ('inter_sparse', 1, 2, 'Sparsity Loss', 'Inter-Modality Sparsity'),
        ('dec_sparse', 2, 0, 'Sparsity Loss', 'Decoder Sparsity'),
        ('ortho', 2, 1, 'Orthogonal Loss', 'Weight Orthogonality'),
        ('prior', 2, 2, 'Prior Loss', 'Biological Prior'),
    ]

    for key, r, c, ylabel, title in plot_configs:
        ax = axes[r, c]
        if key in train_m:
            ax.plot(epochs, train_m[key], label='Train', linewidth=LINE_WIDTH, color='#3498db')
            ax.plot(epochs, val_m[key], label='Val', linewidth=LINE_WIDTH, color='#e74c3c')

            ax.set_title(title, fontsize=LABEL_SIZE, fontweight='bold', pad=15)
            ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
            ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)

            ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
            ax.legend(fontsize=LEGEND_SIZE, frameon=False)
            ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")
    return True


# ==========================================
# DATA LOADING
# ==========================================
def parse_sgRNA_to_gene(sgrna_series):
    def _parse(val):
        if pd.isna(val) or str(val).lower() == 'nan':
            return 'control'
        s = str(val).strip()
        if any(kw in s.upper() for kw in ['NON-GENE', 'NO_SITE', 'NON_TARGET', 'SCRAMBLE']):
            return 'control'
        match = re.match(r'^(.+)_(\d+)$', s)
        return match.group(1) if match else s
    return sgrna_series.apply(_parse)


def build_protein_name_map(train_names, perturb_names):
    mapping = {}
    train_set = set(train_names)
    for pn in perturb_names:
        if pn in train_set:
            mapping[pn] = pn

    def normalize(name):
        n = name.strip()
        for suffix in ['_TotalSeqB', '_TotalSeqC', '_TotalSeq-B', '_TotalSeq-C',
                        '-TotalSeqB', '-TotalSeqC', '_control', '_isotype',
                        '_PROT', '_protein', '_ADT']:
            if n.endswith(suffix):
                n = n[:-len(suffix)]
        return n.replace('_', '-').replace('.', '-').lower()

    norm_to_train = {normalize(tn): tn for tn in train_names}
    for pn in perturb_names:
        if pn not in mapping:
            norm_pn = normalize(pn)
            if norm_pn in norm_to_train:
                mapping[pn] = norm_to_train[norm_pn]
    return mapping


def _chunked_mean_std(adata, chunk_size=50000):
    """Compute per-feature mean and std without loading full dense matrix at once."""
    n = adata.n_obs
    X = adata.X
    mean = np.zeros(adata.n_vars, dtype=np.float64)
    M2 = np.zeros(adata.n_vars, dtype=np.float64)
    count = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = X[start:end]
        if hasattr(chunk, 'toarray'):
            chunk = chunk.toarray()
        chunk = chunk.astype(np.float64)
        batch_n = chunk.shape[0]
        batch_mean = chunk.mean(axis=0)
        batch_var = chunk.var(axis=0)
        delta = batch_mean - mean
        new_count = count + batch_n
        mean = (count * mean + batch_n * batch_mean) / new_count
        M2 = M2 + batch_n * batch_var + delta**2 * count * batch_n / new_count
        count = new_count
    std = np.sqrt(M2 / count)
    return mean.astype(np.float32), (std + 1e-6).astype(np.float32)


def _chunked_corrcoef(adata, col_indices, chunk_size=50000):
    """Compute correlation matrix for selected columns using chunked Welford + outer products.
    Returns abs(corr) with diagonal zeroed — same as np.corrcoef but O(chunk) memory."""
    n = adata.n_obs
    X = adata.X
    p = len(col_indices)

    # Pass 1: mean (already have it but recompute for selected cols to be safe)
    mean = np.zeros(p, dtype=np.float64)
    count = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = X[start:end][:, col_indices]
        if hasattr(chunk, 'toarray'):
            chunk = chunk.toarray()
        chunk = chunk.astype(np.float64)
        batch_n = chunk.shape[0]
        delta = chunk.mean(axis=0) - mean
        new_count = count + batch_n
        mean += delta * batch_n / new_count
        count = new_count

    # Pass 2: sum of (x_i - mean)(x_j - mean) via chunked outer product
    cov = np.zeros((p, p), dtype=np.float64)
    var = np.zeros(p, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = X[start:end][:, col_indices]
        if hasattr(chunk, 'toarray'):
            chunk = chunk.toarray()
        centered = chunk.astype(np.float64) - mean
        cov += centered.T @ centered
        var += (centered ** 2).sum(axis=0)

    # Normalize to correlation
    std = np.sqrt(var / count)
    std[std < 1e-12] = 1.0  # avoid division by zero
    corr = (cov / count / np.outer(std, std)).astype(np.float32)  # SIGNED (abs applied at AUROC use-sites)
    np.fill_diagonal(corr, 0)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def _chunked_cross_corrcoef(adata_or_X1, col_indices_1, X2_dense, chunk_size=50000):
    """Compute abs cross-correlation between selected columns of adata and a dense matrix.
    Returns [len(col_indices_1) x X2_dense.shape[1]] correlation matrix."""
    if hasattr(adata_or_X1, 'X'):
        X1 = adata_or_X1.X
        n = adata_or_X1.n_obs
    else:
        X1 = adata_or_X1
        n = X1.shape[0]

    p1 = len(col_indices_1)
    p2 = X2_dense.shape[1]

    # Pass 1: means
    mean1 = np.zeros(p1, dtype=np.float64)
    mean2 = np.zeros(p2, dtype=np.float64)
    count = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        c1 = X1[start:end][:, col_indices_1]
        if hasattr(c1, 'toarray'):
            c1 = c1.toarray()
        c2 = X2_dense[start:end].astype(np.float64)
        batch_n = c1.shape[0]
        new_count = count + batch_n
        mean1 += (c1.astype(np.float64).mean(axis=0) - mean1) * batch_n / new_count
        mean2 += (c2.mean(axis=0) - mean2) * batch_n / new_count
        count = new_count

    # Pass 2: cross-covariance + variances
    cross_cov = np.zeros((p1, p2), dtype=np.float64)
    var1 = np.zeros(p1, dtype=np.float64)
    var2 = np.zeros(p2, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        c1 = X1[start:end][:, col_indices_1]
        if hasattr(c1, 'toarray'):
            c1 = c1.toarray()
        c1 = c1.astype(np.float64) - mean1
        c2 = X2_dense[start:end].astype(np.float64) - mean2
        cross_cov += c1.T @ c2
        var1 += (c1 ** 2).sum(axis=0)
        var2 += (c2 ** 2).sum(axis=0)

    std1 = np.sqrt(var1 / count)
    std2 = np.sqrt(var2 / count)
    std1[std1 < 1e-12] = 1.0
    std2[std2 < 1e-12] = 1.0
    corr = (cross_cov / count / np.outer(std1, std2)).astype(np.float32)  # SIGNED (abs at AUROC use-sites)
    return np.nan_to_num(corr, nan=0.0)


def load_shared_data(train_rna_path, train_prot_path):
    """Load all data — memory-efficient: computes training stats in chunks,
    extracts only common-gene columns for correlation (full dataset, no subsample),
    then frees training data before loading perturbation data."""
    import gc
    print("=" * 60)
    print("LOADING SHARED DATA")
    print("=" * 60)

    # --- Training data: names, mean/std, correlation columns ---
    print("Loading training data (memory-efficient)...")
    train_rna = sc.read_h5ad(train_rna_path)
    train_prot = sc.read_h5ad(train_prot_path)
    train_gene_names = train_rna.var_names.tolist()
    train_prot_names = train_prot.var_names.tolist()

    common_train_obs = sorted(set(train_rna.obs_names) & set(train_prot.obs_names))
    train_rna = train_rna[common_train_obs]
    train_prot = train_prot[common_train_obs]
    print(f"  Training cells: {len(common_train_obs)}")

    # Compute mean/std in chunks (full dataset)
    print("  Computing RNA mean/std (chunked)...")
    train_rna_mean, train_rna_std = _chunked_mean_std(train_rna)
    print("  Computing protein mean/std (chunked)...")
    train_prot_mean, train_prot_std = _chunked_mean_std(train_prot)

    # Peek at perturb var names to determine common genes (without loading data)
    perturb_rna_backed = sc.read_h5ad(PERTURB_RNA_PATH, backed='r')
    perturb_prot_backed = sc.read_h5ad(PERTURB_PROT_PATH, backed='r')
    perturb_gene_names = perturb_rna_backed.var_names.tolist()
    perturb_prot_names = perturb_prot_backed.var_names.tolist()
    perturb_rna_backed.file.close()
    perturb_prot_backed.file.close()

    train_gene_idx = {g: i for i, g in enumerate(train_gene_names)}
    train_prot_idx = {p: i for i, p in enumerate(train_prot_names)}
    common_genes = sorted(set(train_gene_names) & set(perturb_gene_names))
    prot_map = build_protein_name_map(train_prot_names, perturb_prot_names)

    prot_eval_names_pre = [tn for pn, tn in prot_map.items() if tn in train_prot_idx]

    # Compute correlation matrices using chunked method (never materializes full dense matrix)
    common_cols = np.array([train_gene_idx[g] for g in common_genes])
    train_gene_to_idx_common = {g: i for i, g in enumerate(common_genes)}

    # ---- corr cache: identical across all MODELS for a given (train data, common genes, proteins) ----
    prot_train_cols = np.array([train_prot_idx[pn] for pn in prot_eval_names_pre]) if prot_eval_names_pre else np.array([], dtype=int)
    import hashlib
    _cachedir = os.path.join(_CMC_RESULTS, "corr_cache"); os.makedirs(_cachedir, exist_ok=True)
    _key = hashlib.md5(("|".join([os.path.abspath(train_rna_path), str(len(common_train_obs)),
                                  ",".join(common_genes), ",".join(prot_eval_names_pre)])).encode()).hexdigest()[:16]
    _cf = os.path.join(_cachedir, f"corr_{_key}.npz")
    corr_train = None; gene_prot_corr = None
    if os.environ.get("CMC_CORR_CACHE","1")!="0" and os.path.exists(_cf):
        try:
            _z=np.load(_cf); corr_train=_z["corr_train"]; gene_prot_corr=(_z["gene_prot_corr"] if "gene_prot_corr" in _z.files else None)
            print(f"  [corr cache HIT] {os.path.basename(_cf)} (skipped RNA-RNA + gene-prot recompute)")
        except Exception: corr_train=None
    if corr_train is None:
        print(f"  Pre-computing training correlation baseline (RNA-RNA, {len(common_train_obs)} cells, {len(common_genes)} genes, chunked)...")
        corr_train = _chunked_corrcoef(train_rna, common_cols, chunk_size=50000)
        if len(prot_train_cols) > 0:
            print(f"  Pre-computing gene-protein correlation ({len(common_genes)} genes x {len(prot_eval_names_pre)} proteins, chunked)...")
            X_prot_common = train_prot.X[:, prot_train_cols]
            if hasattr(X_prot_common, 'toarray'): X_prot_common = X_prot_common.toarray()
            X_prot_common = X_prot_common.astype(np.float32)
            gene_prot_corr = _chunked_cross_corrcoef(train_rna, common_cols, X_prot_common, chunk_size=50000)
            del X_prot_common
        try:
            if gene_prot_corr is not None: np.savez(_cf, corr_train=corr_train, gene_prot_corr=gene_prot_corr)
            else: np.savez(_cf, corr_train=corr_train)
            print(f"  [corr cache SAVE] {os.path.basename(_cf)}")
        except Exception as e: print(f"  [corr cache save failed] {str(e)[:80]}")

    # Free full training AnnData before loading perturbation data
    del train_rna, train_prot
    gc.collect()

    # --- Now load perturbation data (training data is freed) ---
    print("Loading perturbation data...")
    perturb_rna = sc.read_h5ad(PERTURB_RNA_PATH)
    perturb_prot = sc.read_h5ad(PERTURB_PROT_PATH)

    common_obs = sorted(set(perturb_rna.obs_names) & set(perturb_prot.obs_names))
    perturb_rna = perturb_rna[common_obs].copy()
    perturb_prot = perturb_prot[common_obs].copy()
    print(f"  Aligned perturbation cells: {len(common_obs)}")

    perturb_rna.obs['target_gene'] = parse_sgRNA_to_gene(perturb_rna.obs['sgRNA'])

    target_genes = perturb_rna.obs['target_gene']
    control_mask = target_genes == 'control'

    perturb_gene_idx = {g: i for i, g in enumerate(perturb_gene_names)}
    perturb_prot_idx = {p: i for i, p in enumerate(perturb_prot_names)}

    rna_eval_perturb_indices = np.array([perturb_gene_idx[g] for g in common_genes])
    rna_eval_train_indices = np.array([train_gene_idx[g] for g in common_genes])

    prot_eval_perturb_indices, prot_eval_train_indices, prot_eval_names = [], [], []
    for pn, tn in prot_map.items():
        if pn in perturb_prot_idx and tn in train_prot_idx:
            prot_eval_perturb_indices.append(perturb_prot_idx[pn])
            prot_eval_train_indices.append(train_prot_idx[tn])
            prot_eval_names.append(tn)
    prot_eval_perturb_indices = np.array(prot_eval_perturb_indices) if prot_eval_perturb_indices else np.array([], dtype=int)
    prot_eval_train_indices = np.array(prot_eval_train_indices) if prot_eval_train_indices else np.array([], dtype=int)

    print(f"  Gene overlap: {len(common_genes)}/{len(train_gene_names)}")
    print(f"  Protein overlap: {len(prot_eval_names)}/{len(train_prot_names)} ({prot_eval_names})")

    align = {
        'common_genes': common_genes,
        'prot_map': prot_map,
        'prot_eval_names': prot_eval_names,
        'train_gene_names': train_gene_names,
        'train_prot_names': train_prot_names,
        'train_gene_idx': train_gene_idx,
        'train_prot_idx': train_prot_idx,
        'control_mask': control_mask,
        'rna_eval_perturb_indices': rna_eval_perturb_indices,
        'rna_eval_train_indices': rna_eval_train_indices,
        'prot_eval_perturb_indices': prot_eval_perturb_indices,
        'prot_eval_train_indices': prot_eval_train_indices,
        'train_rna_mean': train_rna_mean,
        'train_rna_std': train_rna_std,
        'train_prot_mean': train_prot_mean,
        'train_prot_std': train_prot_std,
        'corr_train': corr_train,
        'train_gene_to_idx_common': train_gene_to_idx_common,
        'gene_prot_corr': gene_prot_corr,
    }

    print("Shared data loaded.\n")
    return perturb_rna, perturb_prot, align


# ==========================================
# PERTURBATION VALIDATION FUNCTIONS
# ==========================================
def fast_mannwhitney_pvals(X_pert, X_ctrl):
    n1, n2 = X_pert.shape[0], X_ctrl.shape[0]
    X_all = np.vstack([X_pert, X_ctrl])
    ranks = np.apply_along_axis(rankdata, 0, X_all)
    R1 = ranks[:n1, :].sum(axis=0)
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U1 - mu) / (sigma + 1e-10)
    return 2 * norm.sf(np.abs(z))


def compute_observed_de(perturb_rna, perturb_prot, align):
    control_mask = align['control_mask']
    target_genes = perturb_rna.obs['target_gene']
    rna_eval_idx = align['rna_eval_perturb_indices']
    prot_eval_idx = align['prot_eval_perturb_indices']
    common_genes = align['common_genes']

    X_rna_full = perturb_rna.X.toarray() if hasattr(perturb_rna.X, 'toarray') else np.array(perturb_rna.X)
    X_prot_full = perturb_prot.X.toarray() if hasattr(perturb_prot.X, 'toarray') else np.array(perturb_prot.X)

    X_rna = X_rna_full[:, rna_eval_idx]
    X_prot = X_prot_full[:, prot_eval_idx] if len(prot_eval_idx) > 0 else np.zeros((X_rna_full.shape[0], 0))

    ctrl_idx = np.where(control_mask.values)[0]
    X_rna_ctrl = X_rna[ctrl_idx]
    X_prot_ctrl = X_prot[ctrl_idx]
    ctrl_mean_rna = X_rna_ctrl.mean(0)
    ctrl_mean_prot = X_prot_ctrl.mean(0)

    common_genes_set = set(common_genes)
    unique_perts = target_genes[~control_mask].unique()

    de_results = {}
    for pert_gene in unique_perts:
        if pert_gene not in common_genes_set:
            continue
        pert_idx = np.where((target_genes == pert_gene).values)[0]
        if len(pert_idx) < MIN_CELLS_PER_PERTURBATION:
            continue

        X_rna_p = X_rna[pert_idx]
        X_prot_p = X_prot[pert_idx]

        lfc_rna = X_rna_p.mean(0) - ctrl_mean_rna
        lfc_prot = X_prot_p.mean(0) - ctrl_mean_prot

        ctrl_std_rna = X_rna_ctrl.std(0) + 1e-8
        ctrl_std_prot = X_prot_ctrl.std(0) + 1e-8
        effect_rna = np.abs(lfc_rna) / np.sqrt((ctrl_std_rna**2 + (X_rna_p.std(0)+1e-8)**2) / 2)
        effect_prot = np.abs(lfc_prot) / np.sqrt((ctrl_std_prot**2 + (X_prot_p.std(0)+1e-8)**2) / 2)

        pvals_rna = fast_mannwhitney_pvals(X_rna_p, X_rna_ctrl)
        pvals_prot = fast_mannwhitney_pvals(X_prot_p, X_prot_ctrl) if len(prot_eval_idx) > 0 else np.ones(0)

        de_results[pert_gene] = {
            'lfc_rna': lfc_rna, 'lfc_prot': lfc_prot,
            'effect_rna': effect_rna, 'effect_prot': effect_prot,
            'pvals_rna': pvals_rna, 'pvals_prot': pvals_prot,
            'sig_rna': (pvals_rna < DE_PVAL_THRESHOLD) & (np.abs(lfc_rna) >= DE_LFC_RNA),
            'sig_prot': ((pvals_prot < DE_PVAL_THRESHOLD) & (np.abs(lfc_prot) >= DE_LFC_PROT)) if len(pvals_prot) > 0 else np.array([], dtype=bool),
            'n_cells': len(pert_idx),
        }

    return de_results


def encode_control_cells(model, perturb_rna, perturb_prot, align):
    control_mask = align['control_mask']
    ctrl_idx = np.where(control_mask.values)[0]
    if MAX_CONTROL_CELLS is not None and len(ctrl_idx) > MAX_CONTROL_CELLS:
        rng = np.random.RandomState(42)
        ctrl_idx = rng.choice(ctrl_idx, MAX_CONTROL_CELLS, replace=False)

    X_rna_raw = perturb_rna.X.toarray() if hasattr(perturb_rna.X, 'toarray') else np.array(perturb_rna.X)
    X_prot_raw = perturb_prot.X.toarray() if hasattr(perturb_prot.X, 'toarray') else np.array(perturb_prot.X)

    n_train_genes = len(align['train_gene_names'])
    n_train_prots = len(align['train_prot_names'])

    X_rna_full = np.zeros((len(ctrl_idx), n_train_genes), dtype=np.float32)
    X_prot_full = np.zeros((len(ctrl_idx), n_train_prots), dtype=np.float32)

    for p_idx, t_idx in zip(align['rna_eval_perturb_indices'], align['rna_eval_train_indices']):
        X_rna_full[:, t_idx] = (X_rna_raw[ctrl_idx, p_idx] - align['train_rna_mean'][t_idx]) / align['train_rna_std'][t_idx]
    for p_idx, t_idx in zip(align['prot_eval_perturb_indices'], align['prot_eval_train_indices']):
        X_prot_full[:, t_idx] = (X_prot_raw[ctrl_idx, p_idx] - align['train_prot_mean'][t_idx]) / align['train_prot_std'][t_idx]

    eps_r_list, eps_p_list = [], []
    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(ctrl_idx), batch_size):
            end = min(start + batch_size, len(ctrl_idx))
            b_rna = torch.FloatTensor(X_rna_full[start:end]).to(DEVICE)
            b_prot = torch.FloatTensor(X_prot_full[start:end]).to(DEVICE)
            mu_r, _ = model.enc_rna(b_rna)
            mu_p, _ = model.enc_prot(b_prot)
            # context-adapt: per-context latent affine (no-op unless model.context_adapt=='latent')
            ej = model.apply_ctx_eps(torch.cat([mu_r, mu_p], 1), getattr(model, 'eval_context', 0))
            mu_r, mu_p = torch.split(ej, [model.latent_r, model.latent_p], 1)
            eps_r_list.append(mu_r.cpu())
            eps_p_list.append(mu_p.cpu())

    eps_r_all = torch.cat(eps_r_list, 0); eps_p_all = torch.cat(eps_p_list, 0)
    # ε-bottleneck: apply the trained global L0 gate (deterministic in eval) so the do-op runs on the
    # SAME gated exogenous input the model reconstructs from (closed dims -> 0 exogenous, filled via A).
    if getattr(model, 'eps_bottleneck', False):
        with torch.no_grad():
            gate, _ = model._eps_gate()          # model.eval() -> deterministic open-prob gate
        gate = gate.cpu()
        eps_r_all = eps_r_all * gate[:model.latent_r].unsqueeze(0)
        eps_p_all = eps_p_all * gate[model.latent_r:].unsqueeze(0)
    # also return the z-scored control inputs (for the encoder-based intervention, which re-encodes
    # a counterfactual where the target gene is set to its zero-expression z-score)
    return eps_r_all, eps_p_all, ctrl_idx, X_rna_full, X_prot_full


def do_intervention_encoder(model, X_rna_ctrl_z, X_prot_ctrl_z, target_gene_idx, ko_zvalue):
    """Encoder-based intervention (avoids the decoder pseudo-inverse Dᵀ): set the target gene to its
    zero-expression z-score in the INPUT, re-encode through the (nonlinear) encoder to get the latent
    shift, propagate through (I-Aᵀ)⁻¹, decode. Δx̂ = D[(I-Aᵀ)⁻¹(ε_KO − ε_ctrl)]."""
    batch_size = 256
    with torch.no_grad():
        A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
        A_full = torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0).to(DEVICE)
        I = torch.eye(A_full.shape[0], device=DEVICE)
        inv = torch.linalg.inv((I - A_full.T + 1e-6 * I).cpu()).to(DEVICE)
        lr, lp = model.latent_r, model.latent_p
        deltas = []
        for s in range(0, X_rna_ctrl_z.shape[0], batch_size):
            e = min(s + batch_size, X_rna_ctrl_z.shape[0])
            xr = torch.FloatTensor(X_rna_ctrl_z[s:e]).to(DEVICE)
            xp = torch.FloatTensor(X_prot_ctrl_z[s:e]).to(DEVICE)
            xr_ko = xr.clone(); xr_ko[:, target_gene_idx] = ko_zvalue
            mu_r, _ = model.enc_rna(xr);    mu_p, _ = model.enc_prot(xp)
            mu_r_ko, _ = model.enc_rna(xr_ko)
            eps = torch.cat([mu_r, mu_p], 1); eps_ko = torch.cat([mu_r_ko, mu_p], 1)
            z = (inv @ eps.T).T; z_ko = (inv @ eps_ko.T).T
            rec = model.dec_rna(z[:, :lr]) + model.bias_rna
            rec_ko = model.dec_rna(z_ko[:, :lr]) + model.bias_rna
            deltas.append((rec_ko - rec).cpu().numpy())
    import numpy as np
    return np.vstack(deltas)


def do_intervention_projected(model, eps_r, eps_p, target_gene_idx):
    batch_size = 256
    rec_rna_ctrl_list, rec_rna_pert_list = [], []
    rec_prot_ctrl_list, rec_prot_pert_list = [], []

    with torch.no_grad():
        A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5, ctx=int(getattr(model, 'eval_context', 0)))
        top = torch.cat([A_rr, W_rp], dim=1)
        bot = torch.cat([W_pr, A_pp], dim=1)
        A_full = torch.cat([top, bot], dim=0).to(DEVICE)
        # propagate through the model's structural solver: linear (I-A^T)^-1 eps, or nonlinear
        # z=phi(A^T z)+eps fixed-point when scm_nonlinear -> the do-op is exact for both.
        solve = lambda e: model.scm._scm_solve(A_full, e)
        latent_r = model.latent_r
        latent_p = model.latent_p
        W_r = model.dec_rna.weight
        # ε-bottleneck: gate the injected KO perturbation so it enters only via exogenous (root) dims
        # and must reach endogenous dims through A (faithful to how the bottleneck model reconstructs)
        _egate = None
        if getattr(model, 'eps_bottleneck', False):
            _egate, _ = model._eps_gate(); _egate = _egate.to(DEVICE)

        for start in range(0, eps_r.shape[0], batch_size):
            end = min(start + batch_size, eps_r.shape[0])
            e_r = eps_r[start:end].to(DEVICE)
            e_p = eps_p[start:end].to(DEVICE)
            eps_joint = torch.cat([e_r, e_p], dim=1)

            z_ctrl = solve(eps_joint)
            z_r_ctrl, z_p_ctrl = torch.split(z_ctrl, [latent_r, latent_p], dim=1)
            rec_r_ctrl = model.dec_rna(z_r_ctrl) + model.bias_rna
            rec_p_ctrl = model.dec_prot(z_p_ctrl) + model.bias_prot
            rec_r_ctrl, rec_p_ctrl = model.apply_ctx_dec(rec_r_ctrl, rec_p_ctrl, getattr(model, 'eval_context', 0))

            delta_x = torch.zeros_like(rec_r_ctrl)
            delta_x[:, target_gene_idx] = -rec_r_ctrl[:, target_gene_idx]

            delta_z_r = delta_x @ W_r
            delta_z_p = torch.zeros_like(z_p_ctrl)
            delta_eps_joint = torch.cat([delta_z_r, delta_z_p], dim=1)
            if _egate is not None:
                delta_eps_joint = delta_eps_joint * _egate.unsqueeze(0)

            eps_pert = eps_joint + delta_eps_joint
            z_pert = solve(eps_pert)
            z_r_pert, z_p_pert = torch.split(z_pert, [latent_r, latent_p], dim=1)

            rec_r_pert = model.dec_rna(z_r_pert) + model.bias_rna
            rec_p_pert = model.dec_prot(z_p_pert) + model.bias_prot
            rec_r_pert, rec_p_pert = model.apply_ctx_dec(rec_r_pert, rec_p_pert, getattr(model, 'eval_context', 0))

            rec_rna_ctrl_list.append(rec_r_ctrl.cpu())
            rec_rna_pert_list.append(rec_r_pert.cpu())
            rec_prot_ctrl_list.append(rec_p_ctrl.cpu())
            rec_prot_pert_list.append(rec_p_pert.cpu())

    return (torch.cat(rec_rna_ctrl_list, 0).numpy(),
            torch.cat(rec_rna_pert_list, 0).numpy(),
            torch.cat(rec_prot_ctrl_list, 0).numpy(),
            torch.cat(rec_prot_pert_list, 0).numpy())


# ==========================================
# EVALUATION: RNA + CROSS-MODAL
# ==========================================
def _compute_prot_metrics(pred_delta_prot, de_data, n_prots):
    obs_lfc_prot = de_data['lfc_prot']
    obs_effect_prot = de_data['effect_prot']
    obs_sig_prot = de_data['sig_prot']

    r = {}
    if n_prots >= 3:
        rho_eff, _ = spearmanr(np.abs(pred_delta_prot), obs_effect_prot)
        r['spearman_rho_prot_effect'] = rho_eff

        rho_signed, _ = spearmanr(pred_delta_prot, obs_lfc_prot)
        r['spearman_rho_prot_signed'] = rho_signed

        n_sig = int(obs_sig_prot.sum())
        if n_sig >= 1 and n_sig < n_prots:
            try:
                r['auroc_prot'] = roc_auc_score(obs_sig_prot.astype(int), np.abs(pred_delta_prot))
            except:
                pass

        nonzero = obs_lfc_prot != 0
        if nonzero.sum() >= 2:
            dir_agree = np.sign(pred_delta_prot[nonzero]) == np.sign(obs_lfc_prot[nonzero])
            r['direction_accuracy_prot'] = float(dir_agree.mean())

    return r


def evaluate_in_silico(model, eps_r, eps_p, de_results, align):
    common_genes = align['common_genes']
    rna_eval_train_indices = align['rna_eval_train_indices']
    prot_eval_train_indices = align['prot_eval_train_indices']
    prot_eval_names = align['prot_eval_names']
    train_gene_idx_map = align['train_gene_idx']
    n_prots = len(prot_eval_names)

    results = []

    for i, (pert_gene, de_data) in enumerate(de_results.items()):
        if pert_gene not in train_gene_idx_map:
            continue
        target_idx = train_gene_idx_map[pert_gene]

        rec_rna_ctrl, rec_rna_pert, rec_prot_ctrl, rec_prot_pert = do_intervention_projected(
            model, eps_r, eps_p, target_idx)

        pred_delta_rna = (rec_rna_pert - rec_rna_ctrl).mean(axis=0)[rna_eval_train_indices]
        pred_delta_prot = (rec_prot_pert - rec_prot_ctrl).mean(axis=0)[prot_eval_train_indices] if n_prots > 0 else np.array([])

        obs_lfc_rna = de_data['lfc_rna']
        obs_effect_rna = de_data['effect_rna']
        obs_sig_rna = de_data['sig_rna']

        gene_idx_in_common = None
        for j, g in enumerate(common_genes):
            if g == pert_gene:
                gene_idx_in_common = j
                break

        eval_mask = np.ones(len(common_genes), dtype=bool)
        if gene_idx_in_common is not None:
            eval_mask[gene_idx_in_common] = False

        pred_delta = np.abs(pred_delta_rna[eval_mask])
        obs_effect = obs_effect_rna[eval_mask]
        obs_lfc = obs_lfc_rna[eval_mask]
        obs_sig = obs_sig_rna[eval_mask]
        pred_signed = pred_delta_rna[eval_mask]

        r = {'perturbation': pert_gene, 'n_cells': de_data['n_cells'],
             'n_genes_eval': int(eval_mask.sum()), 'n_de_genes': int(obs_sig.sum())}

        # RNA metrics
        rho, rho_p = spearmanr(pred_delta, obs_effect)
        r['spearman_rho_effect'] = rho
        rho_signed, _ = spearmanr(pred_signed, obs_lfc)
        r['spearman_rho_signed'] = rho_signed

        n_sig = int(obs_sig.sum())
        if n_sig >= 3 and n_sig < len(obs_sig):
            try:
                r['auroc'] = roc_auc_score(obs_sig.astype(int), pred_delta)
            except:
                pass

        if n_sig >= 3:
            sig_mask_local = obs_sig.astype(bool)
            direction_agree = np.sign(pred_signed[sig_mask_local]) == np.sign(obs_lfc[sig_mask_local])
            r['direction_accuracy'] = float(direction_agree.mean())

        # Protein metrics
        if n_prots > 0 and len(pred_delta_prot) > 0:
            prot_metrics = _compute_prot_metrics(pred_delta_prot, de_data, n_prots)
            r.update(prot_metrics)

        results.append(r)
        if (i + 1) % 20 == 0:
            print(f"    Evaluated {i+1}/{len(de_results)} perturbations...")

    return results


def evaluate_static_baselines(de_results, align, model_dir):
    grn_path = os.path.join(model_dir, "GRN_inferred.csv")
    if not os.path.exists(grn_path):
        return []

    grn = pd.read_csv(grn_path, index_col=0)
    common_genes = align['common_genes']
    results = []

    for pert_gene, de_data in de_results.items():
        if pert_gene not in grn.index:
            continue
        pred_row = grn.loc[pert_gene]
        obs_effect = de_data['effect_rna']
        obs_sig = de_data['sig_rna']

        scores, effects, labels = [], [], []
        for j, g in enumerate(common_genes):
            if g == pert_gene:
                continue
            scores.append(np.abs(pred_row.get(g, 0.0)))
            effects.append(obs_effect[j])
            labels.append(int(obs_sig[j]))

        scores = np.array(scores)
        effects = np.array(effects)
        labels = np.array(labels)

        r = {'perturbation': pert_gene, 'n_de_genes': int(labels.sum())}
        rho, _ = spearmanr(scores, effects)
        r['spearman_rho'] = rho
        if labels.sum() >= 3 and labels.sum() < len(labels):
            try:
                r['auroc'] = roc_auc_score(labels, scores)
            except:
                pass
        results.append(r)

    return results


def _run_correlation_eval(corr, gene_to_idx, common_genes, de_results):
    results = []
    for pert_gene, de_data in de_results.items():
        if pert_gene not in gene_to_idx:
            continue
        pidx = gene_to_idx[pert_gene]
        obs_effect = de_data['effect_rna']
        obs_sig = de_data['sig_rna']

        scores, effects, labels = [], [], []
        for j, g in enumerate(common_genes):
            if g == pert_gene:
                continue
            if g not in gene_to_idx:
                continue
            scores.append(abs(corr[pidx, gene_to_idx[g]]))  # AUROC ranks by |corr| (corr is signed)
            effects.append(obs_effect[j])
            labels.append(int(obs_sig[j]))

        scores = np.array(scores)
        effects = np.array(effects)
        labels = np.array(labels)

        valid = np.isfinite(scores)
        scores, effects, labels = scores[valid], effects[valid], labels[valid]
        if len(scores) < 10:
            continue

        r = {'perturbation': pert_gene, 'n_de_genes': int(labels.sum())}
        rho, _ = spearmanr(scores, effects)
        r['spearman_rho'] = rho
        if labels.sum() >= 3 and labels.sum() < len(labels):
            try:
                r['auroc'] = roc_auc_score(labels, scores)
            except:
                pass
        results.append(r)
    return results


def compute_correlation_baseline_perturb(de_results, perturb_rna, align):
    control_mask = align['control_mask']
    common_genes = align['common_genes']
    rna_eval_idx = align['rna_eval_perturb_indices']

    X = perturb_rna.X.toarray() if hasattr(perturb_rna.X, 'toarray') else np.array(perturb_rna.X)
    ci = np.where(control_mask.values)[0]
    X_sub = X[ci][:, rna_eval_idx]
    corr = np.abs(np.corrcoef(X_sub.T))
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0)
    gene_to_idx = {g: i for i, g in enumerate(common_genes)}
    return _run_correlation_eval(corr, gene_to_idx, common_genes, de_results)


def compute_correlation_baseline_training(de_results, align):
    return _run_correlation_eval(
        align['corr_train'], align['train_gene_to_idx_common'],
        align['common_genes'], de_results)


def compute_cross_modal_corr_baseline(de_results, align):
    gene_prot_corr = align['gene_prot_corr']
    if gene_prot_corr is None:
        return []

    common_genes = align['common_genes']
    prot_eval_names = align['prot_eval_names']
    train_gene_to_idx = align['train_gene_to_idx_common']
    n_prots = len(prot_eval_names)

    results = []
    for pert_gene, de_data in de_results.items():
        if pert_gene not in train_gene_to_idx:
            continue
        gi = train_gene_to_idx[pert_gene]
        pred_scores = np.abs(gene_prot_corr[gi, :])  # AUROC ranks by |corr| (corr is signed)

        obs_effect_prot = de_data['effect_prot']
        obs_lfc_prot = de_data['lfc_prot']
        obs_sig_prot = de_data['sig_prot']

        r = {'perturbation': pert_gene}

        if n_prots >= 3:
            rho_eff, _ = spearmanr(pred_scores, obs_effect_prot)
            r['spearman_rho_prot_effect'] = rho_eff

            rho_signed, _ = spearmanr(pred_scores, np.abs(obs_lfc_prot))
            r['spearman_rho_prot_signed'] = rho_signed

            n_sig = int(obs_sig_prot.sum())
            if n_sig >= 1 and n_sig < n_prots:
                try:
                    r['auroc_prot'] = roc_auc_score(obs_sig_prot.astype(int), pred_scores)
                except:
                    pass

        results.append(r)

    return results


def compute_direction_baselines(de_results, align):
    """Direction-accuracy baselines (a fair comparator beyond random 0.5):
      - signed training-corr: predict sign(Delta_j for KO of g) = -sign(corr(g, j)).
        (genes positively co-expressed with g go DOWN when g is knocked out)
      - all-down marginal: predict every DE feature is downregulated.
    Computed on the SAME DE set as the CMC direction metric (RR: >=3 DE genes; RP: >=1 DE protein)."""
    corr_s = align.get('corr_train')        # signed [n_common x n_common], common-gene order
    gp_s = align.get('gene_prot_corr')      # signed [n_common x n_prot_eval]
    g2i = align['train_gene_to_idx_common']
    common = align['common_genes']
    rows = []
    for g, de in de_results.items():
        if g not in g2i:
            continue
        gi = g2i[g]
        r = {'perturbation': g}
        # --- RNA -> RNA ---
        sig, lfc = de['sig_rna'], de['lfc_rna']
        de_j = [j for j in range(len(common)) if sig[j] and common[j] != g]
        if len(de_j) >= 3 and corr_s is not None:
            obs = np.sign(lfc[de_j])
            r['signed_corr_rna_direction'] = float((-np.sign(corr_s[gi, de_j]) == obs).mean())
            r['alldown_rna_direction'] = float((obs < 0).mean())
        # --- RNA -> Protein ---
        sigp = de.get('sig_prot', np.array([])); lfcp = de.get('lfc_prot', np.array([]))
        de_k = [k for k in range(len(sigp)) if sigp[k]]
        if len(de_k) >= 1 and gp_s is not None:
            obsp = np.sign(lfcp[de_k])
            r['signed_corr_rnaProt_direction'] = float((-np.sign(gp_s[gi, de_k]) == obsp).mean())
            r['alldown_rnaProt_direction'] = float((obsp < 0).mean())
        rows.append(r)
    return rows


# ==========================================
# PLOTTING
# ==========================================
def _plot_bar_panel(ax, methods, metric_key, ylabel, random_val, title):
    names, means, sems, colors = [], [], [], []
    for method_name, mdata in methods.items():
        vals = mdata.get(metric_key, [])
        if not vals:
            continue
        names.append(method_name)
        means.append(np.mean(vals))
        sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        colors.append(mdata['color'])

    names.append('Random')
    means.append(random_val)
    sems.append(0)
    colors.append('#f0f0f0')

    if len(names) <= 1:
        ax.set_visible(False)
        return

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=sems, color=colors, edgecolor='black',
                  capsize=5, width=0.65, linewidth=1.5)

    for bar, m, s in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.008,
                f'{m:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=TICK_SIZE-2)

    ax.axhline(random_val, color='red', ls='--', alpha=0.6, lw=LINE_WIDTH)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=TICK_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=LABEL_SIZE, fontweight='bold', pad=15)

    ax.tick_params(axis='y', labelsize=TICK_SIZE)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_all_methods_bar(in_silico_results, static_results, corr_results,
                         corr_train_results, cm_corr_results,
                         results_dir, exp_name):
    """
    3-panel plot: RNA→RNA AUROC, RNA→Protein AUROC, RNA→RNA Direction Accuracy.
    """
    def _extract_vals(results, key):
        return [r[key] for r in results if key in r and not np.isnan(r.get(key, np.nan))]

    # Collect RNA AUROC
    rna_methods = {}
    if in_silico_results:
        rna_methods['CMC-VAE\n(Projected)'] = {'auroc': _extract_vals(in_silico_results, 'auroc'), 'color': '#2171b5'}
    if static_results:
        rna_methods['Direct Edge\nWeights'] = {'auroc': _extract_vals(static_results, 'auroc'), 'color': '#fd8d3c'}
    if corr_train_results:
        rna_methods['Corr Baseline\n(Training Data)'] = {'auroc': _extract_vals(corr_train_results, 'auroc'), 'color': '#bdbdbd'}

    # Collect Protein AUROC
    prot_methods = {}
    if in_silico_results:
        p_aurocs = _extract_vals(in_silico_results, 'auroc_prot')
        if p_aurocs:
            prot_methods['CMC-VAE\n(RNA→Prot)'] = {'auroc': p_aurocs, 'color': '#2171b5'}
    if cm_corr_results:
        cm_aurocs = _extract_vals(cm_corr_results, 'auroc_prot')
        if cm_aurocs:
            prot_methods['Gene-Prot Corr\n(Training)'] = {'auroc': cm_aurocs, 'color': '#bdbdbd'}

    # Collect RNA Direction Accuracy (only CMC-VAE)
    rna_dir_methods = {}
    if in_silico_results:
        dir_vals = _extract_vals(in_silico_results, 'direction_accuracy')
        if dir_vals:
            rna_dir_methods['CMC-VAE\n(Projected)'] = {'direction': dir_vals, 'color': '#2171b5'}

    # Collect Protein Direction Accuracy (only CMC-VAE)
    prot_dir_methods = {}
    if in_silico_results:
        pdir_vals = _extract_vals(in_silico_results, 'direction_accuracy_prot')
        if pdir_vals:
            prot_dir_methods['CMC-VAE\n(RNA→Prot)'] = {'direction': pdir_vals, 'color': '#2171b5'}

    has_prot = len(prot_methods) > 0 and any(len(m['auroc']) > 0 for m in prot_methods.values())
    has_rna_dir = len(rna_dir_methods) > 0 and any(len(m['direction']) > 0 for m in rna_dir_methods.values())
    has_prot_dir = len(prot_dir_methods) > 0 and any(len(m['direction']) > 0 for m in prot_dir_methods.values())

    # 2x2 grid: AUROC left, Direction right; RNA top, Protein bottom
    n_rows = 1 + (1 if has_prot or has_prot_dir else 0)
    n_cols = 1 + (1 if has_rna_dir or has_prot_dir else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 7 * n_rows), squeeze=False)

    # Row 1, Col 1: RNA AUROC
    n_rna_perts = len(rna_methods.get('CMC-VAE\n(Projected)', {}).get('auroc', []))
    _plot_bar_panel(axes[0, 0], rna_methods, 'auroc', 'AUROC', 0.5,
                    f'RNA→RNA: AUROC\n({n_rna_perts} perturbations)')

    # Row 1, Col 2: RNA Direction Accuracy
    if n_cols > 1 and has_rna_dir:
        n_dir_perts = len(rna_dir_methods.get('CMC-VAE\n(Projected)', {}).get('direction', []))
        _plot_bar_panel(axes[0, 1], rna_dir_methods, 'direction', 'Direction Accuracy', 0.5,
                        f'RNA→RNA: Direction Accuracy\n({n_dir_perts} perturbations)')
    elif n_cols > 1:
        axes[0, 1].set_visible(False)

    # Row 2, Col 1: Protein AUROC
    if n_rows > 1 and has_prot:
        n_prot_perts = len(prot_methods.get('CMC-VAE\n(RNA→Prot)', {}).get('auroc', []))
        _plot_bar_panel(axes[1, 0], prot_methods, 'auroc', 'AUROC', 0.5,
                        f'RNA→Protein: AUROC\n({n_prot_perts} perturbations)')
    elif n_rows > 1:
        axes[1, 0].set_visible(False)

    # Row 2, Col 2: Protein Direction Accuracy
    if n_rows > 1 and n_cols > 1 and has_prot_dir:
        n_pdir_perts = len(prot_dir_methods.get('CMC-VAE\n(RNA→Prot)', {}).get('direction', []))
        _plot_bar_panel(axes[1, 1], prot_dir_methods, 'direction', 'Direction Accuracy', 0.5,
                        f'RNA→Protein: Direction Accuracy\n({n_pdir_perts} perturbations)')
    elif n_rows > 1 and n_cols > 1:
        axes[1, 1].set_visible(False)

    plt.suptitle(f'Zero-shot Intervention Perturbation Prediction\nUsing Perturb-Cite Seq on CMC-VAE',
                 fontsize=TITLE_SIZE, fontweight='bold')

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path = os.path.join(results_dir, 'all_methods_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Run training curves + perturbation validation on a single experiment directory.")
    parser.add_argument("results_dir", type=str,
                        help="Path to the experiment results directory (must contain best_model.pt and config.json)")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    exp_name = os.path.basename(results_dir).replace("results_", "")

    if not os.path.isdir(results_dir):
        print(f"ERROR: {results_dir} is not a directory")
        sys.exit(1)

    print(f"Device: {DEVICE}")
    print(f"Experiment: {exp_name}")
    print(f"Results dir: {results_dir}\n")

    # === Step 1: Training curves ===
    print("=" * 60)
    print("STEP 1: GENERATING TRAINING CURVES")
    print("=" * 60)
    plot_training_curves(results_dir, exp_name)

    # === Step 2: Perturbation validation ===
    model_path = resolve_ckpt(results_dir)
    if not os.path.exists(model_path):
        print(f"\n[SKIP] No best_model.pt — skipping perturbation validation.")
        return

    # Determine training data paths from experiment's config.json
    config_path = os.path.join(results_dir, "config.json")
    train_rna_path = DEFAULT_TRAIN_RNA_PATH
    train_prot_path = DEFAULT_TRAIN_PROT_PATH
    num_batches = 8
    latent_dim_rna = 128
    latent_dim_prot = 64

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        train_rna_path = cfg.get("rna_path", train_rna_path)
        train_prot_path = cfg.get("prot_path", train_prot_path)
        latent_dim_rna = cfg.get("latent_dim_rna", latent_dim_rna)
        latent_dim_prot = cfg.get("latent_dim_prot", latent_dim_prot)
        print(f"  Loaded config from {config_path}")
        print(f"    rna_path: {train_rna_path}")
        print(f"    prot_path: {train_prot_path}")

    print("\n" + "=" * 60)
    print("STEP 2: PERTURBATION VALIDATION")
    print("=" * 60)

    perturb_rna, perturb_prot, align = load_shared_data(train_rna_path, train_prot_path)

    print("Computing observed DE...")
    de_results = compute_observed_de(perturb_rna, perturb_prot, align)
    print(f"  {len(de_results)} valid perturbations\n")

    print("Computing correlation baselines...")
    corr_results = compute_correlation_baseline_perturb(de_results, perturb_rna, align)
    corr_train_results = compute_correlation_baseline_training(de_results, align)
    cm_corr_results = compute_cross_modal_corr_baseline(de_results, align)
    dir_baselines = compute_direction_baselines(de_results, align)
    print(f"  Cross-modal corr baseline: {len(cm_corr_results)} perturbations evaluated\n")

    n_genes = len(align['train_gene_names'])
    n_prots = len(align['train_prot_names'])

    # Count unique donors from training data to set num_batches
    train_rna = sc.read_h5ad(train_rna_path)
    if 'donor' in train_rna.obs.columns:
        num_batches = train_rna.obs['donor'].nunique()
    del train_rna

    print(f"Loading model (dim_rna={n_genes}, dim_prot={n_prots}, "
          f"num_batches={num_batches}, latent_rna={latent_dim_rna}, latent_prot={latent_dim_prot})...")
    _rd = os.path.dirname(model_path)
    _cfg = json.load(open(os.path.join(_rd, "config.json"))) if os.path.exists(os.path.join(_rd, "config.json")) else \
           {"latent_dim_rna": latent_dim_rna, "latent_dim_prot": latent_dim_prot}
    model = load_ctx_model(CrossModalCausalLDVAE, _rd, n_genes, n_prots, _cfg)

    eps_r, eps_p, ctrl_idx, _Xr_ctrl_z, _Xp_ctrl_z = encode_control_cells(model, perturb_rna, perturb_prot, align)
    in_silico_results = evaluate_in_silico(model, eps_r, eps_p, de_results, align)
    static_results = evaluate_static_baselines(de_results, align, results_dir)

    # Print summary
    if in_silico_results:
        aurocs = [r['auroc'] for r in in_silico_results if 'auroc' in r]
        rhos = [r['spearman_rho_effect'] for r in in_silico_results if not np.isnan(r.get('spearman_rho_effect', np.nan))]
        dirs = [r['direction_accuracy'] for r in in_silico_results if 'direction_accuracy' in r]
        print(f"\n  RNA  — AUROC: {np.mean(aurocs):.4f}" if aurocs else "\n  No RNA AUROC")
        print(f"         Spearman: {np.mean(rhos):.4f}" if rhos else "")
        print(f"         Direction: {np.mean(dirs):.4f}" if dirs else "")

        p_aurocs = [r['auroc_prot'] for r in in_silico_results if 'auroc_prot' in r]
        p_rhos = [r['spearman_rho_prot_effect'] for r in in_silico_results
                  if not np.isnan(r.get('spearman_rho_prot_effect', np.nan))]
        print(f"  Prot — AUROC: {np.mean(p_aurocs):.4f}" if p_aurocs else "  No Prot AUROC")
        print(f"         Spearman: {np.mean(p_rhos):.4f}" if p_rhos else "")

    # Plot
    plot_all_methods_bar(in_silico_results, static_results, corr_results,
                         corr_train_results, cm_corr_results,
                         results_dir, exp_name)

    # Save a compact baselines summary (means) so corr / gene-prot-corr / direct-edge
    # baselines are recorded per run — not only in the plot.
    def _mean(results, key):
        vals = [r[key] for r in (results or []) if key in r and not np.isnan(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else None
    summary = {
        'max_control_cells': MAX_CONTROL_CELLS,
        'n_perturbations_rna': len([r for r in (in_silico_results or []) if 'auroc' in r]),
        'n_perturbations_prot': len([r for r in (in_silico_results or []) if 'auroc_prot' in r]),
        'cmcvae_rna_auroc': _mean(in_silico_results, 'auroc'),
        'cmcvae_rna_direction': _mean(in_silico_results, 'direction_accuracy'),
        'cmcvae_rnaProt_auroc': _mean(in_silico_results, 'auroc_prot'),
        'cmcvae_rnaProt_direction': _mean(in_silico_results, 'direction_accuracy_prot'),
        'direct_edge_rna_auroc': _mean(static_results, 'auroc'),
        'corr_train_rna_auroc': _mean(corr_train_results, 'auroc'),
        'geneprot_corr_rnaProt_auroc': _mean(cm_corr_results, 'auroc_prot'),
        # direction-accuracy baselines (fairer than random 0.5)
        'signed_corr_rna_direction': _mean(dir_baselines, 'signed_corr_rna_direction'),
        'alldown_rna_direction': _mean(dir_baselines, 'alldown_rna_direction'),
        'signed_corr_rnaProt_direction': _mean(dir_baselines, 'signed_corr_rnaProt_direction'),
        'alldown_rnaProt_direction': _mean(dir_baselines, 'alldown_rnaProt_direction'),
    }
    with open(os.path.join(results_dir, 'baselines_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved baselines_summary.json: {summary}")

    # Per-perturbation baseline AUROCs (for common-set / paired CMC-vs-baseline comparison in the aggregator)
    try:
        base_rows = {}
        for r in (corr_train_results or []):
            if r.get('perturbation') is not None and 'auroc' in r:
                base_rows.setdefault(r['perturbation'], {})['corr_train_rna_auroc'] = r['auroc']
        for r in (cm_corr_results or []):
            if r.get('perturbation') is not None and 'auroc_prot' in r:
                base_rows.setdefault(r['perturbation'], {})['geneprot_corr_rnaProt_auroc'] = r['auroc_prot']
        for r in (dir_baselines or []):
            if r.get('perturbation') is not None:
                d = base_rows.setdefault(r['perturbation'], {})
                for k in ('signed_corr_rna_direction', 'alldown_rna_direction',
                          'signed_corr_rnaProt_direction', 'alldown_rnaProt_direction'):
                    if k in r:
                        d[k] = r[k]
        if base_rows:
            pd.DataFrame.from_dict(base_rows, orient='index').rename_axis('perturbation').to_csv(
                os.path.join(results_dir, 'baselines_perperturbation.csv'))
            print(f"  Saved baselines_perperturbation.csv ({len(base_rows)} perturbations)")
    except Exception as e:
        print(f"  [warn] per-perturbation baseline dump failed: {e}")

    # Save results
    if in_silico_results:
        pd.DataFrame(in_silico_results).to_csv(
            os.path.join(results_dir, 'perturbation_results.csv'), index=False)
        print(f"  Saved perturbation_results.csv")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
