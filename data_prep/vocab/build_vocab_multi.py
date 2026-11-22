#!/usr/bin/env python
"""build_vocab_multi.py SPLIT  (SPLIT in {pooled, healthy, disease})

Consolidated CITE-seq training data for CMC-VAE across 4 datasets in three cell-splits.
Adapts the PROVEN backbone build_vocab_v4.py (RNA vocab + target logic) and the robust
HVG cascade from build_vocab_combat.py, adds the `su` dataset + per-cell disease-status
labels, per-cell data QC, split-filtering, and a properly cell-aligned RNA + protein pair
with a fresh vocab per split.

Datasets (liu is DROPPED):
  hao         : hao_citeseq.h5seurat  (SCT raw counts; ADT = canonical 144-panel naming).  ALL cells HEALTHY.
  stephenson  : haniffa21.processed.h5ad (layers['raw']; obs 'Status').  Healthy->HEALTHY; Covid,Non_covid->DISEASE; LPS->DROP.
  combat      : COMBAT-CITESeq-DATA.h5ad (layers['raw']; obs 'Source').  HV->HEALTHY; everything else->DISEASE.
  su          : su_rna_raw.h5ad + su_prot_raw.h5ad (raw counts; obs['status'] in {healthy,disease}).

SPLIT semantics: healthy = HEALTHY cells; disease = DISEASE cells; pooled = HEALTHY u DISEASE (LPS always dropped).
  (disease split has NO hao cells -> hao contributes 0 and is skipped for disease.)

Per-cell QC (raw counts): n_genes>=200 AND total_counts>=500 AND pct_mito<15 (pct_mito from '^MT-' genes);
  additionally, where a doublet score exists (combat: QC_scrub_doublet_scores) require <0.25.

RNA vocab (per split): gene INTERSECTION over datasets present -> concat(merge='same') -> filter_genes(min_cells=3)
  -> HVG seurat_v3 n_top=2000 batch_key='dataset' (robust cascade fallback) -> drop MT-/RP[SL]\\d ('is_junk')
  -> TFs CollecTRI (HVG or det>=DET_TF) -> RBP n HVG -> force-include evaluable perturbation TARGETS
  (Frangieh u iccite u Arce, present AND det>=DET_TGT).  V = HVG_clean u TF u RBP_hvg u targets.
  X = log1p(normalize_total 1e4); raw counts in layers['counts']; obs carries 'dataset' and 'status'.

Protein (canonical hao-style panel, CELL-ALIGNED to RNA): map each dataset's ADT -> canonical panel names,
  FINAL panel = intersection over datasets present; CLR-normalize per dataset; concatenate.  Asserts that
  citeseq_combined.obs_names == rna_combined.obs_names (identical & same order).

Output per split: combined_{SPLIT}/{rna_combined.h5ad, citeseq_combined.h5ad, vocab_genes.csv}.

DRY-RUN: set env DRYRUN=1 (optionally DRY_CAP=3000) -> subsample each dataset to DRY_CAP cells (seeded),
  write to combined_{SPLIT}_DRY/ instead, and print full diagnostics.  Does NOT touch the real output dir.

Usage:  DRYRUN=1 python build_vocab_multi.py pooled
        python build_vocab_multi.py disease            # full build
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
import os, sys, re, gc
import numpy as np, scanpy as sc, anndata as ad, pandas as pd, h5py
from scipy.sparse import csc_matrix, csr_matrix

HERE = f"{_CMC_DATA}"
sys.path.insert(0, HERE); sys.path.insert(0, _os.path.join(_CMC_ROOT, "data_prep", "citeseq"))  # [public release] combine_datasets.py
from combine_datasets import STEPH_TO_HAO, HAO_ISOTYPES  # reuse the verbatim Stephenson AB_->hao mapping

# ----------------------------- config -----------------------------
if len(sys.argv) < 2 or sys.argv[1] not in ("pooled", "healthy", "disease"):
    sys.exit("Usage: [DRYRUN=1] python build_vocab_multi.py {pooled|healthy|disease} [DET_TF]")
SPLIT   = sys.argv[1]
DET_TF  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
DET_TGT = DET_TF
N_HVG   = 2000
DRYRUN  = os.environ.get("DRYRUN", "0") not in ("0", "", "false", "False")
DRY_CAP = int(os.environ.get("DRY_CAP", "3000"))
SEED    = 0

HAO      = f"{HERE}/hao_citeseq/hao_citeseq.h5seurat"
STEPH    = f"{HERE}/stephenson_citeseq/haniffa21.processed.h5ad"
COMBAT   = f"{HERE}/COMBAT_citeseq/COMBAT-CITESeq-DATA.h5ad"
SU_RNA   = f"{HERE}/su_citeseq/su_rna_raw.h5ad"
SU_PROT  = f"{HERE}/su_citeseq/su_prot_raw.h5ad"
ICCITE_MD = f"{HERE}/chen_perturb_icciteseq/export/metadata.csv"
FRANGIEH  = f"{HERE}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad"
ARCE_PS   = f"{HERE}/arce_perturb_citeseq/GSE278572_protospacer_calls_per_cell.csv.gz"

OUT = os.environ.get("OUT_DIR") or (f"{HERE}/combined_{SPLIT}" + ("_DRY" if DRYRUN else ""))

# which datasets are eligible to contribute cells to this split (hao is all-HEALTHY)
DATASETS = ["hao", "stephenson", "combat", "su"]
if SPLIT == "disease":
    DATASETS = ["stephenson", "combat", "su"]
# COHORTS env restricts to an arbitrary subset (combinatorial cohort sweep); preserves canonical order
_COH = os.environ.get("COHORTS")
if _COH:
    want = {c.strip() for c in _COH.split(",") if c.strip()}
    DATASETS = [d for d in DATASETS if d in want]
    assert DATASETS, f"COHORTS={_COH} left no eligible datasets for SPLIT={SPLIT}"

def log(*a):
    print(f"[{SPLIT}{'/DRY' if DRYRUN else ''}]", *a, flush=True)

log(f"config: DET_TF={DET_TF} N_HVG={N_HVG} DRYRUN={DRYRUN} DRY_CAP={DRY_CAP} OUT={OUT}")
log(f"datasets present: {DATASETS}")

# ----------------------------- helpers -----------------------------
def hao_assay(assay):
    """Read a hao h5seurat assay -> (cells x features) CSR AnnData with obs donor/celltype/dataset."""
    with h5py.File(HAO, "r") as f:
        g = f["assays"][assay]["counts"]; nf = len(f["assays"][assay]["features"][:])
        mat = csr_matrix(csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                                    shape=(nf, len(g["indptr"][:]) - 1)).T)
        feats = np.array([x.decode() for x in f["assays"][assay]["features"][:]])
        cells = np.array([x.decode() for x in f["cell.names"][:]])
        donor = np.array([x.decode() for x in f["meta.data"]["donor"][:]])
        try: ct = np.array([x.decode() for x in f["meta.data"]["celltype.l1"][:]])
        except Exception: ct = np.array(["NA"] * len(cells))
    obs = pd.DataFrame({"donor": ["hao_" + d for d in donor], "celltype": ct,
                        "dataset": "hao", "status": "HEALTHY"}, index=cells)
    return ad.AnnData(X=mat, obs=obs, var=pd.DataFrame(index=feats))

def canon(x):
    """Canonicalize a protein name for cross-dataset matching (hao-style namespace)."""
    x = re.sub(r'^AB_', '', str(x)); x = re.sub(r'^human', '', x); x = re.sub(r'^mouse', '', x)
    x = re.sub(r'-\d+$', '', x); x = x.split('_')[0]
    return x.upper().replace('-', '')

def is_junk(g):
    return bool(g.startswith("MT-") or re.match(r"^RP[SL]\d", g))

def qc_mask(rna, doublet=None):
    """Per-cell QC on raw counts. Returns (keep_bool, stats_dict)."""
    X = rna.X if not hasattr(rna.X, "tocsr") else rna.X.tocsr()
    total  = np.asarray(X.sum(1)).ravel()
    ngenes = np.asarray((X > 0).sum(1)).ravel()
    mt_mask = np.asarray(rna.var_names.str.match(r'^MT-'))
    n_mt = int(mt_mask.sum())
    if n_mt > 0:
        mt = np.asarray(X[:, mt_mask].sum(1)).ravel()
        pct_mt = 100.0 * mt / np.maximum(total, 1)
    else:
        pct_mt = np.zeros_like(total, dtype=float)
    keep = (ngenes >= 200) & (total >= 500) & (pct_mt < 15)
    n_after_basic = int(keep.sum())
    n_doublet_drop = 0
    if doublet is not None:
        d = np.asarray(doublet, dtype=float)
        dmask = d < 0.25
        n_doublet_drop = int((keep & ~dmask).sum())
        keep &= dmask
    return keep, dict(n_mt_genes=n_mt, n_after_basic=n_after_basic, n_doublet_drop=n_doublet_drop)

def subsample_idx(n_pool_idx, rng):
    """Given the positional indices of the split-eligible pool, return a (sorted) subsample for dry-run."""
    if not DRYRUN or len(n_pool_idx) <= DRY_CAP:
        return np.sort(n_pool_idx)
    return np.sort(rng.choice(n_pool_idx, DRY_CAP, replace=False))

# ----------------------------- per-dataset loaders -----------------------------
# Each returns (rna_ad, adt_ad, stats) on the SAME surviving cells in the SAME order, or (None,None,stats).
STEPH_STATUS = {"Healthy": "HEALTHY", "Covid": "DISEASE", "Non_covid": "DISEASE", "LPS": None}

def split_keep(status_labels):
    """Boolean mask of cells whose mapped status belongs to this split."""
    s = np.asarray(status_labels, dtype=object)
    if SPLIT == "healthy": return s == "HEALTHY"
    if SPLIT == "disease": return s == "DISEASE"
    return (s == "HEALTHY") | (s == "DISEASE")   # pooled (drops None/LPS)

def load_hao():
    rng = np.random.default_rng(SEED)
    rna = hao_assay("SCT")
    n_total = rna.n_obs
    status = rna.obs["status"].values                       # all HEALTHY
    pool = np.where(split_keep(status))[0]
    n_pool = len(pool)
    idx = subsample_idx(pool, rng)
    rna = rna[idx].copy()
    keep, qcs = qc_mask(rna, doublet=None)
    rna = rna[keep].copy()
    adt = hao_assay("ADT")[idx][keep].copy()                # same cells, same order
    adt.obs = rna.obs.copy()
    stats = dict(n_total=n_total, n_pool=n_pool, n_sub=len(idx), n_qc=int(keep.sum()),
                 n_healthy=int((rna.obs["status"] == "HEALTHY").sum()),
                 n_disease=int((rna.obs["status"] == "DISEASE").sum()), **qcs)
    return rna, adt, stats

def load_stephenson():
    rng = np.random.default_rng(SEED + 1)
    b = ad.read_h5ad(STEPH, backed="r")
    n_total = b.n_obs
    status = np.array([STEPH_STATUS.get(str(x), None) for x in b.obs["Status"].values], dtype=object)
    pool = np.where(split_keep(status))[0]
    n_pool = len(pool)
    idx = subsample_idx(pool, rng)
    mem = b[idx].to_memory(); b.file.close()
    gmask = (mem.var["feature_types"] == "Gene Expression").values
    amask = (mem.var["feature_types"] == "Antibody Capture").values
    st_status = status[idx]
    obs = pd.DataFrame({"donor": ["steph_" + str(d) for d in mem.obs["patient_id"].values],
                        "celltype": mem.obs["initial_clustering"].astype(str).values if "initial_clustering" in mem.obs else "NA",
                        "dataset": "stephenson", "status": st_status}, index=mem.obs_names)
    rna = ad.AnnData(X=csr_matrix(mem[:, gmask].layers["raw"].copy()),
                     obs=obs.copy(), var=pd.DataFrame(index=mem.var_names[gmask].astype(str)))
    adt = ad.AnnData(X=csr_matrix(mem[:, amask].layers["raw"].copy()),
                     obs=obs.copy(), var=pd.DataFrame(index=mem.var_names[amask].astype(str)))
    del mem; gc.collect()
    keep, qcs = qc_mask(rna, doublet=None)
    rna = rna[keep].copy(); adt = adt[keep].copy()
    stats = dict(n_total=n_total, n_pool=n_pool, n_sub=len(idx), n_qc=int(keep.sum()),
                 n_healthy=int((rna.obs["status"] == "HEALTHY").sum()),
                 n_disease=int((rna.obs["status"] == "DISEASE").sum()), **qcs)
    return rna, adt, stats

def load_combat():
    rng = np.random.default_rng(SEED + 2)
    b = ad.read_h5ad(COMBAT, backed="r")
    n_total = b.n_obs
    src = b.obs["Source"].astype(str).values
    status = np.where(src == "HV", "HEALTHY", "DISEASE").astype(object)
    pool = np.where(split_keep(status))[0]
    n_pool = len(pool)
    idx = subsample_idx(pool, rng)
    dbl_all = b.obs["QC_scrub_doublet_scores"].astype(float).values if "QC_scrub_doublet_scores" in b.obs else None
    mem = b[idx].to_memory(); b.file.close()
    gmask = (mem.var["feature_types"] == "Gene Expression").values
    amask = (mem.var["feature_types"] == "Antibody Capture").values
    cb_status = status[idx]
    obs = pd.DataFrame({"donor": ["combat_" + str(d) for d in mem.obs_names],
                        "celltype": mem.obs["Annotation_major_subset"].astype(str).values if "Annotation_major_subset" in mem.obs else "NA",
                        "dataset": "combat", "status": cb_status}, index=mem.obs_names)
    rna = ad.AnnData(X=csr_matrix(mem[:, gmask].layers["raw"].copy()),
                     obs=obs.copy(), var=pd.DataFrame(index=mem.var_names[gmask].astype(str)))
    adt = ad.AnnData(X=csr_matrix(mem[:, amask].layers["raw"].copy()),
                     obs=obs.copy(), var=pd.DataFrame(index=mem.var_names[amask].astype(str)))
    del mem; gc.collect()
    doublet = dbl_all[idx] if dbl_all is not None else None
    keep, qcs = qc_mask(rna, doublet=doublet)
    rna = rna[keep].copy(); adt = adt[keep].copy()
    stats = dict(n_total=n_total, n_pool=n_pool, n_sub=len(idx), n_qc=int(keep.sum()),
                 n_healthy=int((rna.obs["status"] == "HEALTHY").sum()),
                 n_disease=int((rna.obs["status"] == "DISEASE").sum()), **qcs)
    return rna, adt, stats

def load_su():
    rng = np.random.default_rng(SEED + 3)
    br = ad.read_h5ad(SU_RNA, backed="r")
    n_total = br.n_obs
    raw_status = br.obs["status"].astype(str).values
    status = np.where(raw_status == "healthy", "HEALTHY", "DISEASE").astype(object)
    pool = np.where(split_keep(status))[0]
    n_pool = len(pool)
    idx = subsample_idx(pool, rng)
    rna_mem = br[idx].to_memory(); br.file.close()
    bp = ad.read_h5ad(SU_PROT, backed="r")
    adt_mem = bp[idx].to_memory(); bp.file.close()
    assert list(rna_mem.obs_names) == list(adt_mem.obs_names), "su RNA/PROT obs order mismatch"
    su_status = status[idx]
    obs = pd.DataFrame({"donor": ["su_" + str(d) for d in rna_mem.obs["donor"].values],
                        "celltype": "NA", "dataset": "su", "status": su_status}, index=rna_mem.obs_names)
    rna = ad.AnnData(X=csr_matrix(rna_mem.X.copy()), obs=obs.copy(),
                     var=pd.DataFrame(index=rna_mem.var_names.astype(str)))
    adt = ad.AnnData(X=csr_matrix(adt_mem.X.copy()), obs=obs.copy(),
                     var=pd.DataFrame(index=adt_mem.var_names.astype(str)))
    del rna_mem, adt_mem; gc.collect()
    keep, qcs = qc_mask(rna, doublet=None)
    rna = rna[keep].copy(); adt = adt[keep].copy()
    stats = dict(n_total=n_total, n_pool=n_pool, n_sub=len(idx), n_qc=int(keep.sum()),
                 n_healthy=int((rna.obs["status"] == "HEALTHY").sum()),
                 n_disease=int((rna.obs["status"] == "DISEASE").sum()), **qcs)
    return rna, adt, stats

LOADERS = {"hao": load_hao, "stephenson": load_stephenson, "combat": load_combat, "su": load_su}

# ----------------------------- load all present datasets -----------------------------
log("loading datasets (QC + split filter + dry-subsample)...")
rna_parts, adt_parts, ALL_STATS = {}, {}, {}
for name in DATASETS:
    r, a, s = LOADERS[name]()
    ALL_STATS[name] = s
    log(f"  {name:11s} total={s['n_total']:>8d}  split_eligible={s['n_pool']:>8d}  "
        f"subsampled={s['n_sub']:>7d}  after_QC={s['n_qc']:>7d}  "
        f"(MTgenes={s['n_mt_genes']}, doublet_drop={s['n_doublet_drop']})  "
        f"H={s['n_healthy']} D={s['n_disease']}")
    if r.n_obs > 0:
        rna_parts[name] = r; adt_parts[name] = a
    else:
        log(f"  {name}: 0 cells after filtering -> excluded from this split")

present = [n for n in DATASETS if n in rna_parts]
if not present:
    sys.exit(f"ERROR: no cells survive for split={SPLIT}")
log(f"datasets contributing cells: {present}")

# ============================= RNA VOCAB =============================
log("=== RNA ===")
common = sorted(set.intersection(*[set(rna_parts[n].var_names.astype(str)) for n in present]))
log(f"gene INTERSECTION across {present}: {len(common)}")
adata = ad.concat([rna_parts[n][:, common].copy() for n in present], merge="same")
adata.X = csr_matrix(adata.X)
del rna_parts; gc.collect()
log(f"concatenated raw: {adata.shape}")

sc.pp.filter_genes(adata, min_cells=3)
log(f"after min_cells=3: {adata.shape}")
adata.layers["counts"] = adata.X.copy()
det = np.asarray((adata.layers["counts"] > 0).mean(0)).ravel()
detmap = dict(zip(adata.var_names.astype(str), det))
present_genes = set(adata.var_names.astype(str))

# ---- HVG: robust cascade (batch-aware seurat_v3 -> global seurat_v3 -> manual log-variance top-N) ----
def compute_hvg(a):
    for kw in [dict(flavor="seurat_v3", batch_key="dataset"), dict(flavor="seurat_v3")]:
        try:
            b = a.copy(); sc.pp.highly_variable_genes(b, n_top_genes=N_HVG, **kw)
            log(f"HVG via {kw}"); return set(b.var_names[b.var.highly_variable].astype(str))
        except Exception as e:
            log(f"HVG {kw} failed: {e}")
    tmp = a.copy(); sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp); Xl = tmp.X
    m = np.asarray(Xl.mean(0)).ravel()
    m2 = np.asarray(Xl.multiply(Xl).mean(0)).ravel() if hasattr(Xl, "multiply") else (np.asarray(Xl) ** 2).mean(0)
    var = m2 - m * m; top = np.argsort(-var)[:N_HVG]
    log("HVG via manual log-variance top-N (fallback)")
    return set(a.var_names[top].astype(str))
hvg = compute_hvg(adata)

# ---- perturbation TARGETS (verbatim logic from build_vocab_v4.py lines ~66-92) ----
iccite = set(pd.read_csv(ICCITE_MD, index_col=0)["PerturbedGene"].astype(str)) - {"NTC", "control"}
fr = ad.read_h5ad(FRANGIEH, backed="r"); frset = set()
for s in pd.unique(fr.obs["sgRNA"].astype(str)):
    for g in str(s).split(","):
        g = g.strip()
        if not g or any(k in g.upper() for k in ["NON-GENE", "NO_SITE", "NON_TARGET", "SCRAMBLE", "CONTROL"]): continue
        m = re.match(r"^(.+)_(\d+)$", g); frset.add(m.group(1) if m else g)
fr.file.close()
def _arce_tgt(fc):
    g = str(fc); m = re.match(r"^(.+?)_\d+_CRISPRi$", g) or re.match(r"^(.+?)_\d+$", g); return m.group(1) if m else g
_aps = pd.read_csv(ARCE_PS); arce_set = set(_arce_tgt(x) for x in _aps[_aps.num_features == 1]["feature_call"]) - {"Non-Targeting"}
def _cov(name, s):
    sp = {g for g in s if g in present_genes}
    ev = {g for g in sp if detmap[g] >= DET_TGT}
    log(f"    {name}: total={len(s)} present(min_cells3)={len(sp)} EVALUABLE(det>={DET_TGT:.0%})={len(ev)} "
        f"dropped_lowsignal={len(sp)-len(ev)} (+{len(s)-len(sp)} absent)")
    return ev
log(f"=== PERTURBATION TARGET coverage (force-include only if training-detection >= {DET_TGT:.0%}) ===")
tg_fr = _cov("Frangieh", frset); tg_ic = _cov("iccite", iccite); tg_ar = _cov("Arce", arce_set)
targets = tg_fr | tg_ic | tg_ar
log(f"evaluable target union = {len(targets)}")

import decoupler as dc
tfs_all = set(dc.op.collectri(organism="human")["source"].astype(str).unique()) & present_genes
rbps_all = set(l.strip() for l in open(f"{HERE}/rbp_go0003723.txt")) & present_genes

hvg_clean = {g for g in hvg if not is_junk(g)}
rbps_hvg  = {g for g in (rbps_all & hvg) if not is_junk(g)}
tfs       = {g for g in tfs_all if (g in hvg or detmap[g] >= DET_TF) and not is_junk(g)}
V = sorted(hvg_clean | tfs | rbps_hvg | targets)
log(f"CHOSEN DET_TF={DET_TF} -> |V|={len(V)} | HVG_clean={len(hvg_clean)} TF={len(tfs)} RBP_hvg={len(rbps_hvg)} "
    f"targets={len(targets)} | rare<1%={sum(detmap[g]<0.01 for g in V)} "
    f"mito={sum(g.startswith('MT-') for g in V)} ribo={sum(bool(re.match(r'^RP[SL]\\d',g)) for g in V)} "
    f"| evaluable_targets: Fr={len(tg_fr)} iccite={len(tg_ic)} Arce={len(tg_ar)}")

adata = adata[:, V].copy()
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
rna_finite = bool(np.isfinite(adata.X.data).all())
log(f"RNA X finite: {rna_finite}  shape={adata.shape}")
assert rna_finite, "RNA X contains non-finite values"

# ============================= PROTEIN (canonical panel, cell-aligned) =============================
log("=== PROTEIN ===")
# canonical panel = valid Stephenson<->hao pairs (hao-style naming); keys via canon()
with h5py.File(HAO, "r") as f:
    hao_prots = set(x.decode() for x in f["assays"]["ADT"]["features"][:]) - set(HAO_ISOTYPES)
_stb = ad.read_h5ad(STEPH, backed="r")
steph_prots = set(_stb.var_names[_stb.var["feature_types"] == "Antibody Capture"]); _stb.file.close()
valid_pairs = [(s, h) for s, h in STEPH_TO_HAO.items() if s in steph_prots and h in hao_prots]
CANON_PANEL = [h for s, h in valid_pairs]
HAO_TO_STEPH = {h: s for s, h in valid_pairs}
hao_key2name = {}
for h in CANON_PANEL:
    hao_key2name.setdefault(canon(h), h)   # first-wins keeps deterministic display name
panel_keys = set(hao_key2name)
log(f"canonical panel: {len(CANON_PANEL)} hao names -> {len(panel_keys)} canon keys")

# per-present-dataset canon-key -> original ADT var name.
#   hao/combat/su : canon() on the raw ADT names (hao-style / AB_-stripped / clone-stripped).
#   stephenson    : MUST use the STEPH_TO_HAO dict (canon cannot do gene-symbol->CD remaps, e.g. AB_ITGAE->CD103).
def keymap_canon(adt, drop_iso=False):
    m = {}
    for v in adt.var_names.astype(str):
        if drop_iso and ("Isotype" in v or v.startswith("Mouse_IgG") or v.startswith("Rat_IgG") or v.startswith("Rag-IgG")):
            continue
        m.setdefault(canon(v), v)          # first-wins on duplicate canon keys
    return m
dataset_keymaps = {}
for n in present:
    if n == "stephenson":
        avars = set(adt_parts[n].var_names.astype(str))
        km = {}
        for s, h in valid_pairs:           # canon(hao target) -> stephenson AB_ name
            if s in avars: km.setdefault(canon(h), s)
        dataset_keymaps[n] = km
    else:
        dataset_keymaps[n] = keymap_canon(adt_parts[n], drop_iso=(n == "su"))

# final panel = intersection of canon keys over present datasets, restricted to the canonical namespace
sets = [panel_keys] + [set(dataset_keymaps[n]) for n in present]
final_keys = set.intersection(*sets)
final_display = sorted(hao_key2name[k] for k in final_keys)   # canonical hao-style names
log(f"per-dataset panel-matched counts: " +
    ", ".join(f"{n}={len(panel_keys & set(dataset_keymaps[n]))}" for n in present))
log(f"FINAL common protein panel = {len(final_display)} markers")

# import CLR (muon preferred, manual fallback)
try:
    import muon
    def clr_inplace(a):
        muon.prot.pp.clr(a)
    _clr_how = "muon.prot.pp.clr"
except Exception as e:
    def clr_inplace(a):
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        L = np.log1p(X.astype(np.float64))
        a.X = (L - L.mean(1, keepdims=True)).astype(np.float32)
    _clr_how = f"manual log1p-rowmean (muon unavailable: {e})"
log(f"CLR method: {_clr_how}")

final_keys_sorted_by_display = [(canon(h), h) for h in final_display]
prot_parts = []
for n in present:
    km = dataset_keymaps[n]
    a = adt_parts[n]
    if n == "hao":
        orig_cols = [h for (_, h) in final_keys_sorted_by_display]          # hao names are canonical
    elif n == "stephenson":
        orig_cols = [HAO_TO_STEPH[h] for (_, h) in final_keys_sorted_by_display]
    else:  # combat / su -> via canon key
        orig_cols = [km[k] for (k, _) in final_keys_sorted_by_display]
    sub = a[:, orig_cols].copy()
    sub.var_names = final_display                                           # rename to canonical display
    Xa = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    sub.X = np.asarray(Xa, dtype=np.float32)
    clr_inplace(sub)
    prot_parts.append(sub)
    log(f"  {n}: ADT {a.shape} -> panel {sub.shape} CLR")

prot = ad.concat(prot_parts, merge="same")
del prot_parts, adt_parts; gc.collect()

# ---- alignment + sanity assertions ----
Xp = prot.X.toarray() if hasattr(prot.X, "toarray") else np.asarray(prot.X)
n_allnan = int(np.isnan(Xp).all(axis=0).sum())
n_anynan = int(np.isnan(Xp).any())
aligned = list(prot.obs_names) == list(adata.obs_names)
log(f"protein shape={prot.shape}  all-NaN columns={n_allnan}  any-NaN={n_anynan}")
log(f"RNA/PROT obs_names identical & same order: {aligned}")
assert aligned, "FATAL: rna.obs_names != prot.obs_names (order-identical) — cell alignment broken"
assert n_allnan == 0, "FATAL: protein has all-NaN column(s)"

# ============================= SUMMARY =============================
log("================= SUMMARY =================")
log(f"split={SPLIT}  n_cells={adata.n_obs}")
per_ds = adata.obs["dataset"].value_counts().to_dict()
per_st = adata.obs["status"].value_counts().to_dict()
log(f"per-dataset cells: {per_ds}")
log(f"per-status  cells: {per_st}")
log(f"|V|={len(V)}  (HVG_clean={len(hvg_clean)} TF={len(tfs)} RBP_hvg={len(rbps_hvg)} targets={len(targets)})")
log(f"protein panel size={prot.n_vars}")
log(f"protein panel names: {list(final_display)}")
log(f"RNA X finite={rna_finite}  protein all-NaN cols={n_allnan}  alignment={aligned}")

# ============================= WRITE =============================
os.makedirs(OUT, exist_ok=True)
adata.write(f"{OUT}/rna_combined.h5ad")
prot.write(f"{OUT}/citeseq_combined.h5ad")
pd.Series(V).to_csv(f"{OUT}/vocab_genes.csv", index=False, header=["gene"])
log(f"WROTE {OUT}/{{rna_combined.h5ad, citeseq_combined.h5ad, vocab_genes.csv}}"
    + ("  (DRY-RUN subsample)" if DRYRUN else ""))
log(f"DONE [{SPLIT}]{' DRY' if DRYRUN else ''}")
