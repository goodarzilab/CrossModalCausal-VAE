"""Score the totalVI baseline IDENTICALLY to analysis/common/causal_eval_suite.py, and lay the CMC-VAE numbers
alongside. Reads results/baselines/totalvi_raw.npz (from run_totalvi.py) + the SAME reference masks
(results/ref_nets_v4_2c.npz) + the SAME CMC-VAE prior for CollecTRI, then applies the metric functions
copied verbatim from causal_eval_suite.py:
  - recov()/epr()  : recovery AUROC / AUPRC-ratio / EPR
  - canon_prot()   : cognate transcript->protein matcher (C_translation)
  - orientation asymmetry on CollecTRI directed edges

Writes results/baselines/totalvi.json (totalVI metrics + CMC-VAE side-by-side) and prints a table.
Env: base (python) — needs numpy/scipy/sklearn/torch(for prior load).
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
import numpy as np, torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _CMC_ROOT
RES = f"{_CMC_RESULTS}"
OUTDIR = f"{RES}/baselines"
CMC_RUN = f"{RES}/linfb_2c_fb10_b20_v4"          # CMC-VAE best model (for side-by-side numbers)

# =================== metric fns copied verbatim from causal_eval_suite.py ===================
def epr(score, lab):
    lab = lab.astype(bool); K = int(lab.sum())
    if K < 3 or K >= len(lab): return None
    top = np.argsort(-score)[:K]; return float(lab[top].mean() / (K / len(lab)))
def recov(score, lab, symmetric=False):
    lab = lab.astype(bool)
    if lab.sum() < 3 or lab.sum() >= len(lab): return None
    base = lab.mean()
    return dict(auroc=float(roc_auc_score(lab, score)),
                auprc_ratio=float(average_precision_score(lab, score) / base),
                epr=epr(score, lab), n_edges=int(lab.sum()), prevalence=float(base))
def canon_prot(x):
    x = re.sub(r'^AB_', '', x); x = re.sub(r'^(human|mouse)', '', x, flags=re.I)
    x = re.sub(r'-\d+$', '', x); return x.split('_')[0].upper()
# ===========================================================================================

# ------- load totalVI raw artifacts -------
d = np.load(f"{OUTDIR}/totalvi_raw.npz", allow_pickle=True)
genes = list(map(str, d["genes"])); prots = list(map(str, d["prots"]))
ng, npr = len(genes), len(prots)
G_rr = np.asarray(d["G_rr_tv"], dtype=np.float64)     # symmetric denoised co-expr correlation
G_rp = np.asarray(d["G_rp_tv"], dtype=np.float64)     # gene x protein denoised correlation
np.fill_diagonal(G_rr, 0.0)
run_meta = json.load(open(f"{OUTDIR}/totalvi_run_meta.json"))

out = {"baseline": "totalVI", "scvi_version": run_meta.get("scvi_version"),
       "convergence": run_meta["convergence"], "note_no_leakage": run_meta["imputation"]}

# ------- reference masks (identical to causal_eval_suite: ref_nets_v4_2c + CMC prior CollecTRI) -------
refs_g = {}
cmc_cfg = json.load(open(f"{CMC_RUN}/config.json"))
def load_prior(path):
    dd = torch.load(path, map_location="cpu"); m = dd["prior_mask"]
    m = m.numpy() if torch.is_tensor(m) else np.asarray(m)
    nm = list(map(str, dd.get("gene_names", dd.get("protein_names", []))))
    return m, nm
try:
    m, nm = load_prior(cmc_cfg["prior_rna_path"])
    if nm == genes:
        refs_g["CollecTRI"] = (m.astype(bool), True)
    elif set(genes) <= set(nm):
        gi = {g: i for i, g in enumerate(nm)}; perm = [gi[g] for g in genes]
        refs_g["CollecTRI"] = (m[np.ix_(perm, perm)].astype(bool), True)
except Exception as e:
    out["collectri_err"] = str(e)[:120]
refp = f"{RES}/ref_nets_v4_2c.npz"
ref = np.load(refp, allow_pickle=True)
assert list(map(str, ref["genes"])) == genes, "ref_nets vocab mismatch with totalVI genes"
refmeta = json.load(open(f"{RES}/ref_nets_v4_2c.json"))
for k in ref.files:
    if k == "genes": continue
    refs_g[k] = (ref[k].astype(bool), refmeta[k]["directed"])

# ========== A. GRN recovery (identical scoring; totalVI corr is symmetric) ==========
A = {}
off = ~np.eye(ng, dtype=bool)
for name, (mask, directed) in refs_g.items():
    if directed:
        A[name] = recov(np.abs(G_rr)[off], mask[off])
    else:
        Gs = np.abs(G_rr) + np.abs(G_rr).T
        A[name] = recov(Gs[off], (mask | mask.T)[off])
# orientation asymmetry on CollecTRI directed edges (expected ~1.0 -> fails orientation, by symmetry)
if "CollecTRI" in refs_g:
    m = refs_g["CollecTRI"][0]; ii, jj = np.where(m); keep = ii != jj; ii, jj = ii[keep], jj[keep]
    fwd = np.abs(G_rr[ii, jj]); rev = np.abs(G_rr[jj, ii])
    try:
        u, p = mannwhitneyu(fwd, rev, alternative="greater")
        A["asymmetry"] = dict(fwd_median=float(np.median(fwd)), rev_median=float(np.median(rev)),
                              fwd_over_rev=float(np.median(fwd) / (np.median(rev) + 1e-12)),
                              mwu_p=float(p), n_pairs=int(len(ii)))
    except Exception:
        pass
out["A_GRN"] = A

# ========== C. Translation cognate enrichment (identical to causal_eval_suite C_translation) ==========
gup = {g.upper(): i for i, g in enumerate(genes)}
cognate = []
for j, p in enumerate(prots):
    c = canon_prot(p)
    for cand in (c, c.replace("CD", ""), c + "E", c + "A"):
        if cand in gup:
            cognate.append((gup[cand], j)); break
C = {"n_cognate_pairs": len(cognate)}
if cognate:
    matched = np.array([abs(G_rp[g, p]) for g, p in cognate])
    cog_by_p = {}; [cog_by_p.setdefault(p, set()).add(g) for g, p in cognate]
    bg = []; ranks = []
    for g, p in cognate:
        col = np.abs(G_rp[:, p]); bg.append(np.delete(col, list(cog_by_p[p])))
        ranks.append(float((col > col[g]).sum()) / len(col))
    bg = np.concatenate(bg)
    u, pv = mannwhitneyu(matched, bg, alternative="greater")
    C.update(matched_median=float(np.median(matched)), bg_median=float(np.median(bg)),
             fold=float(np.median(matched) / (np.median(bg) + 1e-12)), mwu_p=float(pv),
             median_cognate_rank_frac=float(np.median(ranks)),
             top10pct_cognate=float(np.mean(np.array(ranks) < 0.1)))
out["C_translation"] = C

# specificity/sparsity of the translation map (D1 differentiator): concentration of |G_rp|
def gini(x):
    x = np.sort(np.abs(x).ravel()); n = len(x)
    if x.sum() == 0: return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
out["translation_specificity"] = {"gini_G_rp": gini(G_rp),
                                  "top1pct_mass": float(np.sort(np.abs(G_rp).ravel())[::-1][:max(1, G_rp.size//100)].sum()/np.abs(G_rp).sum())}

# ========== imputation summary ==========
out["imputation"] = run_meta["imputation"]

# ==================== attach CMC-VAE side-by-side ====================
cmc = json.load(open(f"{CMC_RUN}/causal_eval_suite.json"))
cmc_imp = None; cmc_imp_src = None
# prefer the v4 best-model imputation (SAME combined_v4 data as totalVI); fall back to published 0.363
for cand in ["linfb_2c_fb10_b20_v4", "p2c_mdrop_base"]:
    p = f"{RES}/{cand}/crossmodal_imputation.json"
    if os.path.exists(p): cmc_imp = json.load(open(p)); cmc_imp_src = cand; break
out["CMC_VAE"] = {
    "A_GRN": {k: (cmc["A_GRN"][k] if k in cmc["A_GRN"] else None)
              for k in ["CollecTRI", "DoRothEA", "coexpr", "STRING", "asymmetry"]},
    "C_translation": cmc["C_translation"],
    "imputation": {"protein_from_rna_pearson": (cmc_imp or {}).get("protein_from_rna_pearson"),
                   "protein_full_pearson": (cmc_imp or {}).get("protein_full_pearson"),
                   "source_run": cmc_imp_src},
}
json.dump(out, open(f"{OUTDIR}/totalvi.json", "w"), indent=2)

# ==================== comparison table ====================
def g(dd, *ks):
    for k in ks:
        if dd is None: return None
        dd = dd.get(k) if isinstance(dd, dict) else None
    return dd
tv, cv = out, out["CMC_VAE"]
def fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}g}" if abs(x) < 1e-2 or abs(x) >= 1000 else f"{x:.{nd}f}")
rows = [
    ("RNA->protein imputation (Pearson)",
     fmt(tv["imputation"]["per_protein_pearson_clr_mean"]),
     fmt(g(cv, "imputation", "protein_from_rna_pearson")),
     f"totalVI Spearman {fmt(tv['imputation']['per_protein_spearman_clr_mean'])}; CMC ceiling {fmt(g(cv,'imputation','protein_full_pearson'))}; CMC src={g(cv,'imputation','source_run')}"),
    ("Cognate translation fold (x over bg)",
     fmt(g(tv, "C_translation", "fold"), 3),
     fmt(g(cv, "C_translation", "fold"), 3),
     f"n_cognate {g(tv,'C_translation','n_cognate_pairs')}"),
    ("Cognate top-10% fraction",
     fmt(g(tv, "C_translation", "top10pct_cognate")),
     fmt(g(cv, "C_translation", "top10pct_cognate")), ""),
    ("Cognate MWU p",
     fmt(g(tv, "C_translation", "mwu_p")), fmt(g(cv, "C_translation", "mwu_p")), ""),
    ("GRN coexpr AUROC", fmt(g(tv, "A_GRN", "coexpr", "auroc")), fmt(g(cv, "A_GRN", "coexpr", "auroc")), ""),
    ("GRN STRING AUROC", fmt(g(tv, "A_GRN", "STRING", "auroc")), fmt(g(cv, "A_GRN", "STRING", "auroc")), ""),
    ("GRN CollecTRI AUROC", fmt(g(tv, "A_GRN", "CollecTRI", "auroc")), fmt(g(cv, "A_GRN", "CollecTRI", "auroc")), ""),
    ("GRN DoRothEA AUROC", fmt(g(tv, "A_GRN", "DoRothEA", "auroc")), fmt(g(cv, "A_GRN", "DoRothEA", "auroc")), ""),
    ("Orientation asymmetry (fwd/rev)", fmt(g(tv, "A_GRN", "asymmetry", "fwd_over_rev")),
     fmt(g(cv, "A_GRN", "asymmetry", "fwd_over_rev")),
     f"totalVI MWU p {fmt(g(tv,'A_GRN','asymmetry','mwu_p'))}; CMC MWU p {fmt(g(cv,'A_GRN','asymmetry','mwu_p'))}"),
]
# write summary table to file (tsv)
with open(f"{OUTDIR}/totalvi_summary.tsv", "w") as fh:
    fh.write("metric\ttotalVI\tCMC_VAE\tnote\n")
    for name, a, b, note in rows:
        fh.write(f"{name}\t{a}\t{b}\t{note}\n")

w = 38
line = "| {:<38} | {:>12} | {:>12} | {}".format
print("\n" + "=" * 100)
print("totalVI baseline vs CMC-VAE  (scored via identical causal_eval_suite harness, v4 2c vocab)")
print("=" * 100)
print(line("METRIC", "totalVI", "CMC-VAE", "note"))
print("|" + "-" * 40 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 30)
for name, a, b, note in rows:
    print(line(name, a, b, note))
print("=" * 100)
print(f"totalVI convergence: {out['convergence']}")
print(f"translation specificity (Gini of |G_rp|): totalVI {out['translation_specificity']['gini_G_rp']:.3f}")
print(f"\nwrote {OUTDIR}/totalvi.json")
