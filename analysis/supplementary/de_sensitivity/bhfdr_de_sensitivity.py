"""BH-FDR sensitivity of the DE ground truth (model-independent).

For each held-out cohort, recompute the per-knockout DE sets under
  (a) the paper's definition: raw Mann-Whitney p<0.05 AND |LFC|>=floor
  (b) BH-FDR q<0.05 AND |LFC|>=floor         (correction across features, per knockout)
and also without the |LFC| floor, to show the floor's contribution.

Reuses perturb_validation_analysis.compute_observed_de, which already returns raw
pvals_rna/pvals_prot + lfc_rna/lfc_prot, so the DE calls are identical to the paper's
except for the added FDR step. No model, no GPU.

Floors (paper): RNA 0.10, protein 0.05.
Output: results/bhfdr_de_sensitivity.json
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, numpy as np

sys.path.insert(0, "analysis/common"); sys.path.insert(0, "src")

DATA = f"{_CMC_DATA}"
COHORTS = {
    "frangieh": (f"{DATA}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad",
                 f"{DATA}/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad"),
    "iccite":   (f"{DATA}/chen_perturb_icciteseq/export/iccite_rna.h5ad",
                 f"{DATA}/chen_perturb_icciteseq/export/iccite_prot.h5ad"),
    "arce":     (f"{DATA}/arce_perturb_citeseq/arce_rna.h5ad",
                 f"{DATA}/arce_perturb_citeseq/arce_prot.h5ad"),
}
FLOOR = {"rna": 0.10, "prot": 0.05}
PVAL = 0.05

# set a perturb path before importing so module globals are valid at import
os.environ.setdefault("PERTURB_RNA_PATH", COHORTS["frangieh"][0])
os.environ.setdefault("PERTURB_PROT_PATH", COHORTS["frangieh"][1])
os.environ["CMC_DE_LFC_RNA"] = "0"      # we apply floors ourselves
os.environ["CMC_DE_LFC_PROT"] = "0"
import perturb_validation_analysis as P

CFG = json.load(open("results/sweep_sparse/sp_dec_gl1000/config.json"))


def bh_fdr(p):
    """Benjamini-Hochberg q-values for a 1-D array of p-values."""
    p = np.asarray(p, float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def summarize(de, mod):
    floor = FLOOR[mod]
    pk, lk = f"pvals_{mod}", f"lfc_{mod}"
    raw_floor, fdr_floor, raw_nofloor, fdr_nofloor, n_feat = [], [], [], [], None
    perts = 0
    for g, d in de.items():
        pv = np.asarray(d.get(pk, []), float)
        lf = np.abs(np.asarray(d.get(lk, []), float))
        if pv.size == 0:
            continue
        perts += 1
        n_feat = pv.size
        q = bh_fdr(pv)
        raw_floor.append(int(((pv < PVAL) & (lf >= floor)).sum()))
        fdr_floor.append(int(((q < PVAL) & (lf >= floor)).sum()))
        raw_nofloor.append(int((pv < PVAL).sum()))
        fdr_nofloor.append(int((q < PVAL).sum()))
    if perts == 0:
        return None

    def stats(x):
        x = np.asarray(x)
        return dict(total=int(x.sum()), median=float(np.median(x)), mean=float(x.mean()),
                    perts_ge1=int((x >= 1).sum()), perts_ge5=int((x >= 5).sum()))

    rf, ff = stats(raw_floor), stats(fdr_floor)
    retention = (ff["total"] / rf["total"]) if rf["total"] else float("nan")
    return dict(
        n_perturbations=perts, n_features_tested=int(n_feat),
        expected_null_calls_per_ko=round(n_feat * PVAL, 1),
        raw_pfloor=rf, bhfdr_floor=ff,
        raw_nofloor=stats(raw_nofloor), bhfdr_nofloor=stats(fdr_nofloor),
        retention_fdr_over_raw=round(retention, 3),
    )


results = {"_meta": {"pval": PVAL, "floor_rna": FLOOR["rna"], "floor_prot": FLOOR["prot"],
                     "note": "DE = (p or q < 0.05) AND |LFC| >= floor; BH-FDR per knockout across features"}}
for tag, (rna_p, prot_p) in COHORTS.items():
    print("=" * 70, flush=True)
    print(f"COHORT {tag}", flush=True)
    P.PERTURB_RNA_PATH = rna_p
    P.PERTURB_PROT_PATH = prot_p
    pr, pp, align = P.load_shared_data(CFG["rna_path"], CFG["prot_path"])
    de = P.compute_observed_de(pr, pp, align)
    results[tag] = {"rna": summarize(de, "rna"), "prot": summarize(de, "prot")}
    print(json.dumps(results[tag], indent=1), flush=True)
    del pr, pp, align, de
    import gc; gc.collect()

with open("results/bhfdr_de_sensitivity.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nWROTE results/bhfdr_de_sensitivity.json", flush=True)

# compact table
print("\n==== SUMMARY (DE calls per knockout; floor = paper) ====", flush=True)
print(f"{'cohort':9s} {'mod':5s} {'#feat':>6s} {'exp.null':>8s} {'raw med':>8s} {'fdr med':>8s} "
      f"{'raw tot':>8s} {'fdr tot':>8s} {'retain':>7s} {'raw≥5':>6s} {'fdr≥5':>6s}", flush=True)
for tag in COHORTS:
    for mod in ("rna", "prot"):
        s = results[tag][mod]
        if not s:
            continue
        print(f"{tag:9s} {mod:5s} {s['n_features_tested']:6d} {s['expected_null_calls_per_ko']:8.1f} "
              f"{s['raw_pfloor']['median']:8.1f} {s['bhfdr_floor']['median']:8.1f} "
              f"{s['raw_pfloor']['total']:8d} {s['bhfdr_floor']['total']:8d} "
              f"{s['retention_fdr_over_raw']:7.2f} {s['raw_pfloor']['perts_ge5']:6d} {s['bhfdr_floor']['perts_ge5']:6d}",
              flush=True)
