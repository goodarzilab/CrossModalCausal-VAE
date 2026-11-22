"""Aggregate the STARVE-EPS sweep into results/starve_eps_sweep.json + results/figures/slides/fig_starve_eps.png.
Compares {base, fbr0.3, fbr0.0, beta4, (beta4_fbr03)} on:
  eps/z cell-type silhouette + gap (eps_vs_z_structure.json)
  A_rr load-bearing: mutil RR AUROC real vs shuffled A_rr (block_isolated_mut_<ds>_ablation.json)
  recon MSE (final_stats.json / best_stats.json)
  perturbation RR/RP AUROC + pearDE (mutil_directed_<ds>.json)
  GRN/bio: coexpr AUROC, TF->gene asymmetry, translation cognate fold (causal_eval_suite.json)
Idempotent: reads whatever outputs exist, marks missing as null. Run repeatedly as batteries finish.
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
import os, sys, json, math
import numpy as np

ROOT = f"{_CMC_RESULTS}"
DSETS = ["frangieh", "iccite", "arce"]

# label -> (results_dir, eps_vs_z_path)
RUNS = [
    ("base",        f"{ROOT}/linfb_2c_fb10_b20_v4",   f"{ROOT}/eps_vs_z_structure.json"),
    ("fbr0.3",      f"{ROOT}/linfb_2c_fbr03_v4",      f"{ROOT}/linfb_2c_fbr03_v4/eps_vs_z_structure.json"),
    ("fbr0.0",      f"{ROOT}/linfb_2c_fbr00_v4",      f"{ROOT}/linfb_2c_fbr00_v4/eps_vs_z_structure.json"),
    ("beta4",       f"{ROOT}/linfb_2c_beta4_v4",      f"{ROOT}/linfb_2c_beta4_v4/eps_vs_z_structure.json"),
    ("beta4_fbr03", f"{ROOT}/linfb_2c_beta4_fbr03_v4",f"{ROOT}/linfb_2c_beta4_fbr03_v4/eps_vs_z_structure.json"),
]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def dig(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def collect(label, rd, epsp):
    r = {"label": label, "results_dir": rd}
    # ---- config sanity ----
    cfg = load(f"{rd}/config.json")
    r["free_bits_r"] = dig(cfg, "free_bits_r")
    r["beta_rna_final"] = dig(cfg, "beta_rna_final")
    # ---- eps/z structure ----
    ez = load(epsp)
    for scope, key in [("rna", "RNA (eps_r vs z_r)"), ("joint", "JOINT (eps vs z)")]:
        blk = dig(ez, key) or {}
        se, sz = blk.get("silhouette_eps"), blk.get("silhouette_z")
        r[f"sil_eps_{scope}"] = se
        r[f"sil_z_{scope}"] = sz
        r[f"sil_gap_{scope}"] = (sz - se) if (se is not None and sz is not None) else None
        r[f"w1_eps_{scope}"] = blk.get("w1_normal_eps_mean")
        r[f"w1_z_{scope}"] = blk.get("w1_normal_z_mean")
    # ---- recon MSE ----
    fst = load(f"{rd}/final_stats.json"); bst = load(f"{rd}/best_stats.json")
    r["recon_final"] = dig(fst, "rec"); r["recon_best"] = dig(bst, "rec")
    r["kl_r_final"] = dig(fst, "kl_r"); r["kl_r_best"] = dig(bst, "kl_r")
    r["rho_final"] = dig(fst, "rho")
    # ---- A_rr load-bearing + perturbation, per dataset ----
    r["perds"] = {}
    for ds in DSETS:
        lb = load(f"{rd}/block_isolated_mut_{ds}_ablation.json")
        md = load(f"{rd}/mutil_directed_{ds}.json")
        s = dig(lb, "summary") or {}
        rr_true = s.get("mean_rr_true"); rr_shuf = s.get("mean_rr_shufArr"); rr_zero = s.get("mean_rr_zeroArr")
        r["perds"][ds] = {
            # A_rr load-bearing (from block-isolated ablation)
            "Arr_rr_true": rr_true, "Arr_rr_shuf": rr_shuf, "Arr_rr_zero": rr_zero,
            "Arr_loadbearing_gap": (rr_true - rr_shuf) if (rr_true is not None and rr_shuf is not None) else None,
            "Arr_pearDE_true": s.get("mean_rr_pearDE_true"), "Arr_pearDE_shuf": s.get("mean_rr_pearDE_shufArr"),
            "Wrp_rp_true": s.get("mean_rp_true"), "Wrp_rp_shuf": s.get("mean_rp_shufWrp"),
            # perturbation (mutil read-out)
            "rr_auroc": dig(md, "mutil_rr_auroc"), "rr_pearDE": dig(md, "mutil_rr_pearDE"),
            "rp_auroc": dig(md, "mutil_rp_auroc"), "rp_pearDE": dig(md, "mutil_rp_pearDE"),
            "n": dig(md, "n") or s.get("n"),
        }
    # ---- GRN / bio ----
    cs = load(f"{rd}/causal_eval_suite.json")
    r["coexpr_auroc"] = dig(cs, "A_GRN", "coexpr", "auroc")
    r["collectri_auroc"] = dig(cs, "A_GRN", "CollecTRI", "auroc")
    r["asym_fwd_over_rev"] = dig(cs, "A_GRN", "asymmetry", "fwd_over_rev")
    r["asym_mwu_p"] = dig(cs, "A_GRN", "asymmetry", "mwu_p")
    r["translation_fold"] = dig(cs, "C_translation", "fold")
    r["translation_mwu_p"] = dig(cs, "C_translation", "mwu_p")
    r["spectral_radius"] = dig(cs, "aliveness", "spectral_radius")
    return r


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  -  "
    return f"{x:.{nd}f}"


def main():
    rows = [collect(*a) for a in RUNS]
    # keep only runs that have at least a config (avoid empty rows for not-yet-trained)
    present = [r for r in rows if r.get("free_bits_r") is not None or r["label"] == "base"]
    out = {"runs": present}
    os.makedirs(f"{ROOT}/figures/slides", exist_ok=True)
    json.dump(out, open(f"{ROOT}/starve_eps_sweep.json", "w"), indent=2)

    # ---------- text table ----------
    def line(cells, w):
        return " | ".join(str(c).ljust(w[i]) for i, c in enumerate(cells))
    labels = [r["label"] for r in present]
    print("\n==================== STARVE-EPS SWEEP ====================")
    print(f"{'metric':<26} " + " ".join(f"{l:>13}" for l in labels))
    def prow(name, fn, nd=3):
        print(f"{name:<26} " + " ".join(f"{fmt(fn(r), nd):>13}" for r in present))
    prow("free_bits_r",        lambda r: r["free_bits_r"], 2)
    prow("beta_rna_final",     lambda r: r["beta_rna_final"], 1)
    print("-- structure --")
    prow("sil_eps (RNA)",      lambda r: r["sil_eps_rna"])
    prow("sil_z   (RNA)",      lambda r: r["sil_z_rna"])
    prow("sil_gap z-eps (RNA)",lambda r: r["sil_gap_rna"])
    prow("sil_gap z-eps(JOINT)",lambda r: r["sil_gap_joint"])
    print("-- recon --")
    prow("recon_final (MSE)",  lambda r: r["recon_final"], 2)
    prow("kl_r_final",         lambda r: r["kl_r_final"], 2)
    print("-- A_rr load-bearing (Frangieh) --")
    prow("RR AUROC real",      lambda r: r["perds"]["frangieh"]["Arr_rr_true"])
    prow("RR AUROC shufArr",   lambda r: r["perds"]["frangieh"]["Arr_rr_shuf"])
    prow("load-bearing gap",   lambda r: r["perds"]["frangieh"]["Arr_loadbearing_gap"])
    print("-- perturbation (mutil, Frangieh) --")
    prow("RR AUROC",           lambda r: r["perds"]["frangieh"]["rr_auroc"])
    prow("RR pearDE",          lambda r: r["perds"]["frangieh"]["rr_pearDE"])
    prow("RP AUROC",           lambda r: r["perds"]["frangieh"]["rp_auroc"])
    prow("RP pearDE",          lambda r: r["perds"]["frangieh"]["rp_pearDE"])
    print("-- GRN / bio --")
    prow("coexpr AUROC",       lambda r: r["coexpr_auroc"])
    prow("asym fwd/rev",       lambda r: r["asym_fwd_over_rev"])
    prow("asym mwu_p",         lambda r: r["asym_mwu_p"], 4)
    prow("translation fold",   lambda r: r["translation_fold"], 1)
    print("=========================================================\n")

    # ---------- figure ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(present)); labs = labels
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        C_EPS, C_Z, C_TRUE, C_SHUF = "#4C78A8", "#F58518", "#54A24B", "#B0B0B0"

        # (0,0) eps vs z silhouette (RNA)
        a = ax[0, 0]; w = 0.38
        a.bar(x - w/2, [r["sil_eps_rna"] or np.nan for r in present], w, label="eps", color=C_EPS)
        a.bar(x + w/2, [r["sil_z_rna"] or np.nan for r in present], w, label="z", color=C_Z)
        a.set_title("Cell-type silhouette: eps vs z (RNA)"); a.set_xticks(x); a.set_xticklabels(labs, rotation=25, ha="right")
        a.legend(); a.axhline(0, color="k", lw=.5)

        # (0,1) silhouette gap (z - eps): did structure move into A?
        a = ax[0, 1]
        a.bar(x, [r["sil_gap_rna"] or np.nan for r in present], color=C_Z)
        a.set_title("Silhouette GAP (z - eps), RNA\n(higher = more structure in A)"); a.set_xticks(x); a.set_xticklabels(labs, rotation=25, ha="right")
        a.axhline(0, color="k", lw=.5)

        # (0,2) recon MSE
        a = ax[0, 2]
        a.bar(x, [r["recon_final"] or np.nan for r in present], color="#E45756")
        base_rec = present[0]["recon_final"]
        if base_rec: a.axhline(base_rec, color="k", ls="--", lw=1, label="base")
        a.set_title("Recon MSE (final)\n(collapse = big jump up)"); a.set_xticks(x); a.set_xticklabels(labs, rotation=25, ha="right"); a.legend()

        # (1,0) A_rr load-bearing: real vs shuffled RR AUROC (Frangieh)
        a = ax[1, 0]; w = 0.38
        a.bar(x - w/2, [r["perds"]["frangieh"]["Arr_rr_true"] or np.nan for r in present], w, label="A_rr real", color=C_TRUE)
        a.bar(x + w/2, [r["perds"]["frangieh"]["Arr_rr_shuf"] or np.nan for r in present], w, label="A_rr shuffled", color=C_SHUF)
        a.set_ylim(0.45, None); a.axhline(0.5, color="k", ls=":", lw=.8)
        a.set_title("A_rr load-bearing (Frangieh)\nRR AUROC real vs shuffled"); a.set_xticks(x); a.set_xticklabels(labs, rotation=25, ha="right"); a.legend()

        # (1,1) direction: RR & RP pearDE (Frangieh)
        a = ax[1, 1]; w = 0.38
        a.bar(x - w/2, [r["perds"]["frangieh"]["rr_pearDE"] or np.nan for r in present], w, label="RR pearDE", color=C_EPS)
        a.bar(x + w/2, [r["perds"]["frangieh"]["rp_pearDE"] or np.nan for r in present], w, label="RP pearDE", color=C_Z)
        a.axhline(0, color="k", lw=.5)
        a.set_title("Perturbation DIRECTION: pearDE (Frangieh)\n(Markov wall: RR stays ~0?)"); a.set_xticks(x); a.set_xticklabels(labs, rotation=25, ha="right"); a.legend()

        # (1,2) GRN asymmetry (fwd/rev) — expect ~1 regardless
        a = ax[1, 2]
        a.bar(x, [r["asym_fwd_over_rev"] or np.nan for r in present], color="#72B7B2")
        a.axhline(1.0, color="k", ls=":", lw=.8, label="symmetric=1")
        a.set_title("TF->gene asymmetry (fwd/rev)\n(1 = misoriented / no asymmetry)"); a.set_xticks(x); a.set_xticklabels(labs, rotation=25, ha="right"); a.legend()

        fig.suptitle("STARVE-EPS sweep: does cranking KL push structure into the latent DAG A?", fontsize=14, y=1.00)
        fig.tight_layout()
        fig.savefig(f"{ROOT}/figures/slides/fig_starve_eps.png", dpi=130, bbox_inches="tight")
        print(f"WROTE {ROOT}/figures/slides/fig_starve_eps.png")
    except Exception as e:
        print(f"[figure skipped: {e}]")

    print(f"WROTE {ROOT}/starve_eps_sweep.json")


if __name__ == "__main__":
    main()
