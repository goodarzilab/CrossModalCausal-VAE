#!/usr/bin/env python
"""Condition-matched, donor/condition-aware pseudobulk DE for the three CITE-seq perturbation screens.

usage: python -I pb_de.py {frangieh|iccite|arce}

Units
  Frangieh : stratum = culture condition (Control / IFN-gamma / Co-culture); KO pseudobulk = guide x condition,
             cells carrying exactly one targeting guide; controls = cells carrying only non-targeting guides
             (ONE_NON-GENE_SITE / NO_SITE), pooled into 20 fixed random NT-guide groups per condition.
             (Cells with no detected guide and cells with a targeting + a non-targeting guide are excluded; the
             original pipeline counted the latter as controls.)  Protein: CLR re-derived from raw ADT counts
             (raw_CITE_expression.csv.gz, SCP1064 export); the provided log1p layer is kept as a sensitivity.
  Chen     : stratum = donor (A/B); KO pseudobulk = guide x donor; controls = 20 NTC guides x donor.
  Arce     : stratum = hashtag state (Resting/Stimulated x Treg/Teff, HTO 1-4 demultiplexed from the GEO matrix;
             donors were pooled before hashing and cannot be separated); KO pseudobulk = guide x state;
             controls = 9 non-targeting guides x state.
Pseudobulk value = mean of the stored log-normalized RNA / CLR protein over the cells of the pseudobulk (>=10 cells).
Test: pblimma.limma_ko (KO + stratum design, empirical-Bayes moderated, two-sided), BH per knockout.
Features = the model-vocabulary evaluation features in the exact order used by analysis/common/reviewer_controls.py (align cache).
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
import os, sys, re, zlib, pickle, json, time
import numpy as np, pandas as pd, h5py, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pblimma import limma_ko, bh

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
TAG = sys.argv[1]
REPO = f"{_CMC_ROOT}"
D = f"{_CMC_DATA}"
OUT = f"{_CMC_RESULTS}/paper/analyses/stats/de/pb_{TAG}.pkl"
MINC = 10; MIN_UNIT_CELLS = 30; N_NT_GROUPS = 20
FILES = {"frangieh": (f"{D}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad", f"{D}/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad"),
         "iccite": (f"{D}/chen_perturb_icciteseq/export/iccite_rna.h5ad", f"{D}/chen_perturb_icciteseq/export/iccite_prot.h5ad"),
         "arce": (f"{D}/arce_perturb_citeseq/arce_rna.h5ad", f"{D}/arce_perturb_citeseq/arce_prot.h5ad")}
cache = pickle.load(open(f"{_CMC_RESULTS}/_ftcache/rc_{TAG}_sp_dec_gl1000_3a5d257c9323_0.1_0.05.pkl", "rb"))   # read-only
al, de_orig = cache["align"], cache["de"]
common = list(al["common_genes"]); rci = np.asarray(al["rna_eval_perturb_indices"]); pci = np.asarray(al["prot_eval_perturb_indices"])
log(TAG, "eval genes", len(common), "eval proteins", len(pci))

rna_f, prot_f = FILES[TAG]
# ---------------- read eval columns of RNA (stored log-normalized X) ----------------
with h5py.File(rna_f, "r") as f:
    obs_names = np.array([x.decode() if isinstance(x, bytes) else x for x in f["obs"][f["obs"].attrs.get("_index", "_index")][:]])
    X = f["X"]
    if isinstance(X, h5py.Dataset):
        n = X.shape[0]; R = np.empty((n, len(rci)), np.float32); order = np.argsort(rci); srt = rci[order]
        for s in range(0, n, 20000):
            blk = X[s:s + 20000, :]; R[s:s + 20000, order] = blk[:, srt]
            log(" rna rows", s)
    else:
        shape = X.attrs["shape"]; enc = X.attrs.get("encoding-type", "csr_matrix")
        M = (sp.csr_matrix if "csr" in str(enc) else sp.csc_matrix)((X["data"][:], X["indices"][:], X["indptr"][:]), shape=tuple(shape))
        R = M.tocsr()[:, rci].toarray().astype(np.float32); del M
log("RNA eval matrix", R.shape)
P_ad = ad.read_h5ad(prot_f)
assert list(P_ad.obs_names) == list(obs_names), "rna/prot obs mismatch"
PX = P_ad.X.toarray() if sp.issparse(P_ad.X) else np.asarray(P_ad.X)
PRT = PX[:, pci].astype(np.float32) if len(pci) else np.zeros((len(obs_names), 0), np.float32)
prot_names_eval = [str(P_ad.var_names[i]) for i in pci]
PRT_alt = None

# ---------------- cell annotation ----------------
ntre = re.compile(r"NON-GENE|NO_SITE|NON_TARGET|NON-TARGET|SCRAMBLE", re.I)
ann = pd.DataFrame(index=obs_names)
extra = {}
if TAG == "frangieh":
    sg = pd.read_csv(f"{D}/perturb_citeseq/all_sgRNA_assignments.txt").set_index("Cell")["sgRNAs"].fillna("")
    meta = pd.read_csv(f"{D}/perturb_citeseq/RNA_metadata.csv", skiprows=[1]).set_index("NAME")
    ann["sg"] = sg.reindex(obs_names).fillna("").values; ann["stratum"] = meta["condition"].reindex(obs_names).values
    def cls(s):
        if s == "": return "none"
        gs = s.split(","); nt = [bool(ntre.search(g)) for g in gs]
        return "NT_only" if all(nt) else ("KO+NT" if any(nt) else ("KO1" if len(gs) == 1 else "KOmulti"))
    ann["cls"] = ann.sg.map(cls)
    ann["gene"] = np.where(ann.cls == "KO1", ann.sg.str.replace(r"_\d+$", "", regex=True), np.where(ann.cls == "NT_only", "control", ""))
    ann["guide"] = np.where(ann.cls == "KO1", ann.sg, np.where(ann.cls == "NT_only",
                            ["NTgrp%02d" % (zlib.crc32(s.encode()) % N_NT_GROUPS) for s in ann.sg], ""))
    # original pipeline's control pool (parse_sgRNA_to_gene): any string containing a non-targeting tag -> control
    extra["orig_control_pool"] = {k: int(v) for k, v in ann.loc[ann.sg.map(lambda s: bool(ntre.search(s))), "cls"].value_counts().items()}
    extra["cell_classes"] = {k: int(v) for k, v in ann.cls.value_counts().items()}
    # protein CLR from raw ADT counts
    raw = pd.read_csv(f"{D}/perturb_citeseq/raw_CITE_expression.csv.gz", index_col=0)
    raw.index = [str(i).replace(" protein", "") for i in raw.index]
    log("raw ADT rows:", list(raw.index))
    use = [p for p in raw.index if p in set(map(str, P_ad.var_names))]          # the 20 panel proteins of the processed layer
    rawm = raw.loc[use].T.reindex(obs_names)
    miss = rawm.isna().any(axis=1).sum(); log("cells missing raw ADT:", int(miss))
    L = np.log1p(rawm.fillna(0).values.astype(np.float64)); clr = L - L.mean(1, keepdims=True)
    clr_df = pd.DataFrame(clr, index=obs_names, columns=use)
    PRT_alt = PRT.copy()                                                         # provided log1p layer (sensitivity)
    PRT = clr_df[prot_names_eval].values.astype(np.float32)
    extra["clr_denominator_proteins"] = use; extra["cells_missing_raw_adt"] = int(miss)
    # agreement between provided log1p and log1p(raw) for sanity
    lr = np.log1p(rawm.fillna(0)[prot_names_eval].values)
    extra["corr_provided_vs_log1p_raw"] = float(np.corrcoef(lr.ravel(), PRT_alt.ravel())[0, 1])
elif TAG == "iccite":
    md = pd.read_csv(f"{D}/chen_perturb_icciteseq/export/metadata.csv", index_col=0).reindex(obs_names)
    ann["stratum"] = md.Donor.values; ann["guide"] = md.gRNA.values
    ann["gene"] = np.where(md.PerturbedGene.values == "NTC", "control", md.PerturbedGene.values)
    extra["lanes"] = int(md["orig.ident"].nunique())
elif TAG == "arce":
    bcs = pd.read_csv(f"{D}/arce_perturb_citeseq/GSE278572_barcodes.tsv.gz", header=None)[0].values
    tri = pd.read_csv(f"{_CMC_RESULTS}/paper/analyses/stats/de/arce_hto_triplets.tsv", sep="\t", header=None, names=["f", "c", "v"])
    H = np.zeros((len(bcs), 4)); H[tri.c.values - 1, tri.f.values - 36804] = tri.v.values
    Hd = pd.DataFrame(H, index=bcs, columns=["Resting_Treg", "Stimulated_Treg", "Resting_Teff", "Stimulated_Teff"]).reindex(obs_names).fillna(0)
    srt = np.sort(Hd.values, 1); top, sec = srt[:, -1], srt[:, -2]
    ok = (top >= 10) & (top >= 2 * np.maximum(sec, 1))
    ann["stratum"] = np.where(ok, Hd.columns.values[Hd.values.argmax(1)], "")
    extra["hto_assigned_frac"] = float(ok.mean()); extra["hto_counts"] = {k: int(v) for k, v in ann.stratum.value_counts().items()}
    ps = pd.read_csv(f"{D}/arce_perturb_citeseq/GSE278572_protospacer_calls_per_cell.csv.gz")
    ps = ps[ps.num_features == 1].set_index("cell_barcode")["feature_call"]
    g = ps.reindex(obs_names).fillna("").values
    ann["guide"] = g
    ann["gene"] = [("control" if s.startswith("Non-Targeting") else re.sub(r"_\d+_CRISPRi$", "", s)) if s else "" for s in g]
    ann.loc[ann.stratum == "", "gene"] = ""
log("annotation:", ann.gene.eq("control").sum(), "control cells;", (ann.gene.ne("") & ann.gene.ne("control")).sum(), "KO cells")

# ---------------- pseudobulks ----------------
ann["i"] = np.arange(len(ann))
use_cells = ann[ann.gene != ""]
grp = use_cells.groupby(["gene", "guide", "stratum"])["i"].apply(np.array)
PB = []
for (g, gd, st), idx in grp.items():
    if len(idx) < MINC: continue
    PB.append(dict(gene=g, guide=gd, stratum=st, n=len(idx), idx=idx, y_r=R[idx].mean(0), y_p=PRT[idx].mean(0),
                   y_p_alt=(PRT_alt[idx].mean(0) if PRT_alt is not None else None)))
log("pseudobulks:", len(PB))
ct = [b for b in PB if b["gene"] == "control"]
Yc_r = np.stack([b["y_r"] for b in ct]); Yc_p = np.stack([b["y_p"] for b in ct]); sc_ = np.array([b["stratum"] for b in ct])
Yc_pa = np.stack([b["y_p_alt"] for b in ct]) if PRT_alt is not None else None
ko_genes = sorted(set(b["gene"] for b in PB) - {"control"})
DE, KO_PB = {}, {}
allko_cells = use_cells[use_cells.gene != "control"].groupby("gene").size()
elig = dict(n_targets_any=int(len(allko_cells)), n_targets_ge30cells=int((allko_cells >= MIN_UNIT_CELLS).sum()))
for g in ko_genes:
    kb = [b for b in PB if b["gene"] == g]
    ncell = sum(b["n"] for b in kb)
    if ncell < MIN_UNIT_CELLS or len(kb) < 2: continue
    Yk_r = np.stack([b["y_r"] for b in kb]); Yk_p = np.stack([b["y_p"] for b in kb]); sk = np.array([b["stratum"] for b in kb])
    br, pr, info = limma_ko(Yk_r, sk, Yc_r, sc_)
    if not np.isfinite(br).any(): continue
    bp, pp_, _ = limma_ko(Yk_p, sk, Yc_p, sc_) if Yk_p.shape[1] else (np.zeros(0), np.ones(0), None)
    rec = dict(lfc_rna=br, pvals_rna=pr, q_rna=bh(pr), lfc_prot=bp, pvals_prot=pp_, q_prot=bh(pp_) if len(pp_) else pp_,
               n_cells=int(ncell), n_pb=len(kb), n_guides=len(set(b["guide"] for b in kb)), strata=sorted(set(sk.tolist())), info=info)
    if PRT_alt is not None:
        Yk_pa = np.stack([b["y_p_alt"] for b in kb]); ba, pa, _ = limma_ko(Yk_pa, sk, Yc_pa, sc_)
        rec.update(lfc_prot_log1p=ba, pvals_prot_log1p=pa, q_prot_log1p=bh(pa))
    DE[g] = rec
    KO_PB[g] = dict(Y_r=Yk_r, Y_p=Yk_p, Y_pa=(np.stack([b["y_p_alt"] for b in kb]) if PRT_alt is not None else None), strata=sk,
                    n=np.array([b["n"] for b in kb]), guides=[b["guide"] for b in kb])
elig["n_units_ge30cells_ge2pb"] = len(DE)
log("units with pseudobulk DE:", len(DE))

# control cells for the joint bootstrap (eval features only)
ct_idx = np.concatenate([b["idx"] for b in ct]); ct_pb = np.concatenate([np.full(b["n"], k) for k, b in enumerate(ct)])
out = dict(tag=TAG, common_genes=common, prot_names=prot_names_eval, de=DE, ko_pb=KO_PB,
           ct=dict(Y_r=Yc_r, Y_p=Yc_p, Y_pa=Yc_pa, strata=sc_, n=np.array([b["n"] for b in ct]), guides=[b["guide"] for b in ct],
                   cells_r=R[ct_idx].astype(np.float32), cells_p=PRT[ct_idx].astype(np.float32),
                   cells_pa=(PRT_alt[ct_idx].astype(np.float32) if PRT_alt is not None else None), cell_pb=ct_pb),
           eligibility=elig, extra=extra,
           pb_strata_counts={k: int(v) for k, v in pd.Series([b["stratum"] for b in PB]).value_counts().items()},
           n_control_cells=int(len(ct_idx)), n_control_pb=len(ct))
# comparison with the original cell-level definition
cmp = []
for g, r in DE.items():
    if g in de_orig:
        o = de_orig[g]; m = np.isfinite(r["lfc_rna"])
        cmp.append(dict(gene=g, r_lfc=float(np.corrcoef(o["lfc_rna"][m], r["lfc_rna"][m])[0, 1]),
                        n_sig_orig=int(o["sig_rna"].sum()), n_sig_pb_bh=int(((r["q_rna"] < 0.05) & (np.abs(r["lfc_rna"]) >= 0.10)).sum())))
out["compare_orig"] = cmp
pickle.dump(out, open(OUT, "wb"), protocol=4)
c = pd.DataFrame(cmp)
log("median corr(orig LFC, PB LFC) =", round(float(c.r_lfc.median()), 3), "| median #DE orig", c.n_sig_orig.median(), "PB-BH", c.n_sig_pb_bh.median())
log("extra:", json.dumps(extra, default=str)[:2000])
log("WROTE", OUT)
