"""Score the GENIE3/GRNBoost2 baseline matrices with the SAME harness as causal_eval_suite.py.

Reuses (verbatim) the recovery function `recov` (AUROC / AUPRC-ratio / EPR) and the CollecTRI
orientation-asymmetry (Mann-Whitney |score[tf->tgt]| vs |score[tgt->tf]|) from
analysis/common/causal_eval_suite.py, against results/ref_nets_v4_2c.npz (CollecTRI / DoRothEA / coexpr /
STRING). Directed refs scored on |M| off-diagonal; symmetric refs on |M|+|M|.T. This makes the
GENIE3 numbers apples-to-apples with CMC-VAE's projected GRN G_rr.

Orientation note: the TF-list matrix only populates TF rows, so reverse edges tgt->tf are
structural zeros unless tgt is itself a TF -> a naive asymmetry on it is CONFOUNDED (would look
like perfect orientation for free). We therefore report:
  (a) asymmetry on the ALL-genes matrix over ALL CollecTRI edges  <- the fair, apples-to-apples
      test vs CMC-VAE (both directions genuinely scored, same n_pairs),
  (b) asymmetry on the TF-list matrix restricted to TF->TF CollecTRI edges (both endpoints TFs,
      so both directions genuine),
  (c) the naive TF-list asymmetry, flagged as confounded (for transparency only).

Usage: python score_genie3.py
Writes results/baselines/genie3.json.
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
import os, sys, json
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _CMC_ROOT
RES = os.path.join(REPO, "results")
OUTDIR = os.path.join(RES, "baselines")
REFP = os.path.join(RES, "ref_nets_v4_2c.npz")
REFJ = os.path.join(RES, "ref_nets_v4_2c.json")

# ---- CMC-VAE reference numbers (linfb_2c_fb10_b20_v4/causal_eval_suite.json A_GRN) ----
CMC_VAE = {
    "run": "linfb_2c_fb10_b20_v4",
    "recovery": {
        "CollecTRI": {"auroc": 0.6090, "auprc_ratio": 1.406},
        "DoRothEA":  {"auroc": 0.6388, "auprc_ratio": 1.520},
        "coexpr":    {"auroc": 0.9348, "auprc_ratio": 22.04},
        "STRING":    {"auroc": 0.7035, "auprc_ratio": 5.173},
    },
    "asymmetry_CollecTRI": {"fwd_over_rev": 1.0207, "mwu_p": 0.2430, "n_pairs": 10275},
}

# ---- harness functions (verbatim from analysis/common/causal_eval_suite.py) ----
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


def score_matrix(Mpath):
    d = np.load(Mpath, allow_pickle=True)
    M = d["matrix"].astype(np.float64)
    genes = list(map(str, d["genes"]))
    ng = len(genes)
    np.fill_diagonal(M, 0.0)
    ref = np.load(REFP, allow_pickle=True)
    assert list(map(str, ref["genes"])) == genes, "ref vocab != matrix vocab"
    refmeta = json.load(open(REFJ))
    off = ~np.eye(ng, dtype=bool)

    A = {}
    for name in ref.files:
        if name == "genes": continue
        mask = ref[name].astype(bool)
        directed = refmeta[name]["directed"]
        if directed:
            A[name] = recov(np.abs(M)[off], mask[off])
        else:
            Gs = np.abs(M) + np.abs(M).T
            A[name] = recov(Gs[off], (mask | mask.T)[off])

    # orientation asymmetry on CollecTRI directed edges (== causal_eval_suite)
    asym = {}
    cm = ref["CollecTRI"].astype(bool)
    ii, jj = np.where(cm); keep = ii != jj; ii, jj = ii[keep], jj[keep]
    fwd = np.abs(M[ii, jj]); rev = np.abs(M[jj, ii])
    def _mwu(f, r):
        try:
            u, p = mannwhitneyu(f, r, alternative="greater")
            return dict(fwd_median=float(np.median(f)), rev_median=float(np.median(r)),
                        fwd_over_rev=float(np.median(f) / (np.median(r) + 1e-12)),
                        mwu_p=float(p), n_pairs=int(len(f)))
        except Exception as e:
            return {"err": str(e)[:120]}
    asym["all_collectri_edges"] = _mwu(fwd, rev)

    # TF->TF-restricted (both endpoints are TFs => both directions genuinely scored)
    nz_rows = set(np.where(np.abs(M).sum(1) > 0)[0].tolist())  # genes that have out-edges (== regulators)
    tftf = np.array([k for k in range(len(ii)) if ii[k] in nz_rows and jj[k] in nz_rows], dtype=int)
    if len(tftf) >= 3:
        asym["tf_to_tf_edges"] = _mwu(np.abs(M[ii[tftf], jj[tftf]]), np.abs(M[jj[tftf], ii[tftf]]))
    return dict(recovery=A, asymmetry=asym,
                meta=dict(mode=str(d["mode"]), method=str(d["method"]),
                          n_cells=int(d["n_cells"]), n_total=int(d["n_total"]),
                          seed=int(d["seed"]), n_links=int(d["n_links"]),
                          nonzero_rows=int((np.abs(M).sum(1) > 0).sum())))


def main():
    out = {"baseline": "GENIE3/GRNBoost2 (arboreto)", "cmc_vae": CMC_VAE, "runs": {}}
    found = False
    for method in ("grnboost2", "genie3"):
        for mode in ("tf", "all"):
            p = os.path.join(OUTDIR, f"genie3_matrix_{method}_{mode}.npz")
            if os.path.exists(p):
                found = True
                key = f"{method}_{mode}"
                out["runs"][key] = score_matrix(p)
                print(f"[scored] {key}", flush=True)
    if not found:
        print("no matrices found in", OUTDIR); sys.exit(1)

    # ---- pick the headline GENIE3 numbers: recovery from TF-list, asymmetry from ALL-genes ----
    def pick(method):
        tf = out["runs"].get(f"{method}_tf"); al = out["runs"].get(f"{method}_all")
        h = {}
        if tf:
            h["recovery_tf_list"] = {k: (dict(auroc=round(v["auroc"], 4), auprc_ratio=round(v["auprc_ratio"], 3)) if v else None)
                                     for k, v in tf["recovery"].items()}
            if "tf_to_tf_edges" in tf["asymmetry"]:
                h["asymmetry_tf_to_tf"] = tf["asymmetry"]["tf_to_tf_edges"]
            h["asymmetry_tf_list_naive_CONFOUNDED"] = tf["asymmetry"]["all_collectri_edges"]
        if al:
            h["recovery_all_genes"] = {k: (dict(auroc=round(v["auroc"], 4), auprc_ratio=round(v["auprc_ratio"], 3)) if v else None)
                                       for k, v in al["recovery"].items()}
            h["asymmetry_all_genes_FAIR"] = al["asymmetry"]["all_collectri_edges"]
        return h
    for method in ("grnboost2", "genie3"):
        if any(k.startswith(method) for k in out["runs"]):
            out.setdefault("headline", {})[method] = pick(method)

    json.dump(out, open(os.path.join(OUTDIR, "genie3.json"), "w"), indent=2)
    print("\n=== headline ===")
    print(json.dumps(out.get("headline", {}), indent=2))
    print("wrote", os.path.join(OUTDIR, "genie3.json"))


if __name__ == "__main__":
    main()
