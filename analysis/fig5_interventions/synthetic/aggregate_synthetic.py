"""Aggregate the synthetic studies -> tables (JSON + markdown) and figure panels.
Inputs : analyses/identifiability/synthetic/oracle_table.json, synthetic/v2_cmcvae_*.json
Outputs: analyses/identifiability/synthetic_summary.json, synthetic_tables.md
         figures/identifiability/SuppFig4a_synthetic_orientation.{pdf,png}
         figures/identifiability/SuppFig_synthetic_interventional_recovery.{pdf,png} (+ source-data CSV)
95% CIs: t-interval over independent replicates (n = 10 oracle, n = 5 CMC-VAE per cell); pooled tests across the
three noise families use n = 15 replicates per latent size (two-sided paired t and Wilcoxon), BH within table.
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
import json, glob, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, f"{_CMC_ROOT}/analysis/figures/original")
from style import C, save, panel_label, plt  # noqa

ROOT = f"{_CMC_RESULTS}/paper"
ANA = f"{ROOT}/analyses/identifiability"; FIG = f"{ROOT}/figures/identifiability"; os.makedirs(FIG, exist_ok=True)
FAMS = ["gaussian", "laplace", "t8"]; FL = {"gaussian": "Gaussian", "laplace": "Laplace", "t8": "Student-t(8)"}


def tci(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0: return dict(mean=None, lo=None, hi=None, n=0)
    if len(v) == 1 or np.std(v) == 0: return dict(mean=float(v.mean()), lo=float(v.mean()), hi=float(v.mean()), n=len(v))
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return dict(mean=float(v.mean()), lo=float(v.mean() - h), hi=float(v.mean() + h), n=len(v))


def fmt(d): return "NA" if d["mean"] is None else f"{d['mean']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    d = b - a
    return dict(n=int(len(d)), mean_diff=float(d.mean()), ci95=[tci(d)["lo"], tci(d)["hi"]],
                p_t=float(stats.ttest_rel(b, a).pvalue) if np.std(d) > 0 else 1.0,
                p_wilcoxon=float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)


OUT = {}; MD = []
# ------------------------------------------------------------------ oracle table
O = json.load(open(f"{ANA}/synthetic/oracle_table.json"))["records"]
MD += ["### Supplementary Table 3 (revised, part A) — oracle-access orientation accuracy",
       "Random DAG with P latent nodes (≤2 parents/node, |w|~U(0.3,0.8)), G = 10P features, x = Dz + 10% noise, "
       "n = 10,000 samples, 10 independent replicates per cell (new graph, decoder and noise); mean [95% t-CI]; chance 0.5.", "",
       "| P | Noise | #edges | Second-order (any covariance read-out) | DirectLiNGAM, true z | DirectLiNGAM, true decoder D⁺x | DirectLiNGAM, decoder learned from x (PCA): causal-order agreement |",
       "|---|---|---|---|---|---|---|"]
OUT["oracle"] = {}
for P in [8, 16, 32]:
    for f in FAMS:
        rs = [r for r in O if r["P"] == P and r["family"] == f]
        if not rs: continue
        e = {k: tci([r.get(k) for r in rs]) for k in ["second_order", "lingam_oracle_z", "lingam_oracle_D", "lingam_learned_D",
                                                    "lingam_oracle_z_order", "lingam_oracle_D_order", "lingam_learned_D_order"]}
        e["n_edges"] = float(np.mean([r["n_edges"] for r in rs])); OUT["oracle"][f"P{P}_{f}"] = e
        MD.append(f"| {P} | {FL[f]} | {e['n_edges']:.0f} | 0.50 (ties by construction) | {fmt(e['lingam_oracle_z'])} | {fmt(e['lingam_oracle_D'])} | {fmt(e['lingam_learned_D_order'])} |")

# ------------------------------------------------------------------ CMC-VAE fits
R = [json.load(open(f))[0] for f in sorted(glob.glob(f"{ANA}/synthetic/v2_cmcvae_*.json"))]
OUT["cmcvae_n_fits"] = len(R)
g = lambda r, path: (lambda x: x)(eval("r" + "".join(f"[{p!r}]" for p in path.split("."))))
COLS = [("recon_r2", "recon R²"), ("latent_match_absr_rna", "latent match |r|"),
        ("obs.A_perm_rr", "A_rr orient."), ("obs.G_marker_rr", "G_rr orient."), ("obs.G_marker_rp", "RNA→prot orient. (G)"),
        ("obs.corr_Grr_GrrT", "corr(G_rr,G_rrᵀ)"), ("obs.lingam_learned_z", "DirectLiNGAM on learned z"),
        ("oracle_lingam_true_z", "DirectLiNGAM on true z")]
MD += ["", "### Supplementary Table 3 (revised, part B) — CMC-VAE fitted to planted block-SCM data (observational training)",
       "Planted RNA DAG (l_r latents, ≤2 parents) + RNA→protein edges (no protein→RNA) + sparse protein DAG; marker-structured "
       "linear decoder (each feature one primary latent, dense N(0,0.15²) off-loadings), 10% observation noise; n = 10,000 cells; "
       "CMC-VAE (src/model.py, headline loss weights/schedules, 120 epochs) with the true latent sizes; 5 replicates per cell "
       "(new graph/decoder/noise/initialisation); mean [95% t-CI]; orientation chance 0.5.", "",
       "| l_r/l_p (genes/proteins) | Noise | " + " | ".join(c[1] for c in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
OUT["cmcvae_obs"] = {}
for dr, lab in [(8, "8/4 (200/40)"), (16, "16/8 (400/80)")]:
    for f in FAMS:
        rs = [r for r in R if r["dr"] == dr and r["family"] == f]
        if not rs: continue
        e = {c: tci([g(r, c) for r in rs]) for c, _ in COLS}; OUT["cmcvae_obs"][f"dr{dr}_{f}"] = e
        MD.append(f"| {lab} | {FL[f]} | " + " | ".join(fmt(e[c]) for c, _ in COLS) + " |")

# ------------------------------------------------------------------ interventional fine-tuning
FTC = [("before_structure.G_marker_rr", "after_structure.G_marker_rr", "G_rr orient."),
       ("before_structure.G_marker_rr_covered", "after_structure.G_marker_rr_covered", "G_rr orient., source intervened"),
       ("before_structure.G_marker_rr_uncovered", "after_structure.G_marker_rr_uncovered", "G_rr orient., source not intervened"),
       ("before_structure.A_perm_rr", "after_structure.A_perm_rr", "A_rr orient. (matched)"),
       ("before_structure.G_marker_rp", "after_structure.G_marker_rp", "RNA→prot orient. (G)"),
       ("before_test.pearDE_down", "after_test.pearDE_down", "held-out downstream pearDE"),
       ("before_test.pearDE_cent", "after_test.pearDE_cent", "held-out centred pearDE"),
       ("before_test.retrieval", "after_test.retrieval", "held-out retrieval")]
MD += ["", "### Synthetic interventional fine-tuning with known targets (latent graph only; encoders/decoder frozen)",
       "KO = perfect intervention do(z_k = −2 s.d.) on the latent marked by the target gene (2,000 cells/KO); model operator = "
       "paper eq. 4 (decoder-guided mutilation + knock-down shock of the target gene). 'partial': both marker-gene KOs of 75% of "
       "RNA latents, test on KOs of the held-out 25% latents. 'all': one marker-gene KO per RNA latent (one intervention per "
       "node), test on the other marker gene. Before → after [paired difference, 95% CI; two-sided paired t over replicates, "
       "BH within table]; n = 15 replicates (3 noise families × 5) per latent size.", "",
       "| l_r | Regime | Read-out | Before | After | Δ [95% CI] | q (BH) |", "|---|---|---|---|---|---|---|"]
OUT["ft"] = {}; rowsP = []
for dr in [8, 16]:
    for rg in ["partial", "all"]:
        rs = [r for r in R if r["dr"] == dr and rg in r.get("ft", {})]
        if not rs: continue
        for b, a, lab in FTC:
            if rg == "all" and "uncovered" in b: continue
            bv = [g(r["ft"][rg], b) for r in rs]; av = [g(r["ft"][rg], a) for r in rs]
            bv = np.array([x if x is not None else np.nan for x in bv], float); av = np.array([x if x is not None else np.nan for x in av], float)
            pr = paired(bv, av); OUT["ft"][f"dr{dr}_{rg}_{lab}"] = dict(before=tci(bv), after=tci(av), paired=pr)
            rowsP.append((dr, rg, lab, tci(bv), tci(av), pr))
        # forward vs reverse after FT on held-out
        fv = np.array([r["ft"][rg]["after_test"]["pearDE_down"] for r in rs]); rv = np.array([r["ft"][rg]["after_test_rev"]["pearDE_down"] for r in rs])
        pr = paired(rv, fv); OUT["ft"][f"dr{dr}_{rg}_fwd_minus_rev_after"] = dict(fwd=tci(fv), rev=tci(rv), paired=pr)
        rowsP.append((dr, rg, "held-out downstream pearDE: reversed A_rr → learned (after FT)", tci(rv), tci(fv), pr))
        fv0 = np.array([r["ft"][rg]["before_test"]["pearDE_down"] for r in rs]); rv0 = np.array([r["ft"][rg]["before_test_rev"]["pearDE_down"] for r in rs])
        pr = paired(rv0, fv0); OUT["ft"][f"dr{dr}_{rg}_fwd_minus_rev_before"] = dict(fwd=tci(fv0), rev=tci(rv0), paired=pr)
        rowsP.append((dr, rg, "held-out downstream pearDE: reversed A_rr → learned (before FT)", tci(rv0), tci(fv0), pr))
        mr = np.array([r["ft"][rg]["after_test"]["meanresp"] for r in rs]); OUT["ft"][f"dr{dr}_{rg}_meanresp_ref"] = tci(mr)
        tr = np.array([r["ft"][rg]["after_train"]["pearDE_down"] for r in rs]); trr = np.array([r["ft"][rg]["after_train_rev"]["pearDE_down"] for r in rs])
        OUT["ft"][f"dr{dr}_{rg}_train_fwd_rev_after"] = dict(fwd=tci(tr), rev=tci(trr))
q = bh([x[5]["p_t"] for x in rowsP])
for (dr, rg, lab, b, a, pr), qq in zip(rowsP, q):
    pr["q_bh"] = float(qq)
    MD.append(f"| {dr} | {rg} | {lab} | {fmt(b)} | {fmt(a)} | {pr['mean_diff']:+.2f} [{pr['ci95'][0]:+.2f}, {pr['ci95'][1]:+.2f}] | {qq:.2g} |")
for dr in [8, 16]:
    for rg in ["partial", "all"]:
        k = f"dr{dr}_{rg}_meanresp_ref"
        if k in OUT["ft"]:
            MD.append(f"\nl_r={dr} {rg}: model-free training-mean-response reference on held-out downstream features = {fmt(OUT['ft'][k])}; "
                      f"training KOs after FT, downstream pearDE learned {fmt(OUT['ft'][f'dr{dr}_{rg}_train_fwd_rev_after']['fwd'])} vs reversed A_rr {fmt(OUT['ft'][f'dr{dr}_{rg}_train_fwd_rev_after']['rev'])}.")
# per family FT summary (G_rr orientation)
OUT["ft_by_family"] = {}
for dr in [8, 16]:
    for f in FAMS:
        for rg in ["partial", "all"]:
            rs = [r for r in R if r["dr"] == dr and r["family"] == f and rg in r.get("ft", {})]
            if rs:
                OUT["ft_by_family"][f"dr{dr}_{f}_{rg}"] = dict(
                    before=tci([r["ft"][rg]["before_structure"]["G_marker_rr"] for r in rs]),
                    after=tci([r["ft"][rg]["after_structure"]["G_marker_rr"] for r in rs]))
json.dump(OUT, open(f"{ANA}/synthetic_summary.json", "w"), indent=1)
open(f"{ANA}/synthetic_tables.md", "w").write("\n".join(MD) + "\n")
print("\n".join(MD))

# ------------------------------------------------------------------ Figure: updated Supp Fig 4a
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axs[0]
keys = [("second_order", "2nd-order\n(G-type)", C["muted"]), ("lingam_oracle_z", "LiNGAM\ntrue z", C["base"]),
        ("lingam_oracle_D", "LiNGAM\ntrue D⁺x", "#f6ad55"), ("lingam_learned_D_order", "LiNGAM learned D\n(causal order)", C["accent"])]
w = 0.2
for i, f in enumerate(FAMS):
    rs = [r for r in O if r["P"] == 16 and r["family"] == f]
    for j, (k, lab, col) in enumerate(keys):
        v = np.array([r[k] for r in rs]); x = i + (j - 1.5) * w; e = tci(v)
        ax.bar(x, e["mean"], w * 0.9, color=col, label=lab if i == 0 else None)
        ax.errorbar(x, e["mean"], yerr=[[e["mean"] - e["lo"]], [e["hi"] - e["mean"]]], color=C["ink"], lw=0.7, capsize=1.5)
        ax.scatter(x + np.random.default_rng(j).uniform(-w * 0.25, w * 0.25, len(v)), v, s=3, color=C["ink"], alpha=0.5, zorder=3)
ax.axhline(0.5, ls=":", color=C["muted"], lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels([FL[f] for f in FAMS])
ax.set_ylim(0, 1.05); ax.set_ylabel("Orientation accuracy"); ax.set_title("Oracle access (P = 16, n = 10⁴, 10 reps)")
ax.legend(ncol=2, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.12)); panel_label(ax, "a")
ax = axs[1]
keys2 = [("obs.G_marker_rr", "CMC-VAE G_rr", C["model"]), ("obs.A_perm_rr", "CMC-VAE A_rr", "#63b3ed"),
         ("obs.G_marker_rp", "CMC-VAE RNA→prot", C["neg"]), ("obs.lingam_learned_z", "LiNGAM on CMC-VAE z", C["accent"])]
for i, f in enumerate(FAMS):
    for j, (k, lab, col) in enumerate(keys2):
        for dd, mk in [(8, "o"), (16, "s")]:
            rs = [r for r in R if r["dr"] == dd and r["family"] == f]
            if not rs: continue
            v = np.array([g(r, k) for r in rs]); e = tci(v); x = i + (j - 1.5) * w + (-0.045 if dd == 8 else 0.045)
            ax.errorbar(x, e["mean"], yerr=[[e["mean"] - e["lo"]], [e["hi"] - e["mean"]]], fmt=mk, ms=3, color=col, lw=0.8, capsize=1.5,
                        label=lab if (i == 0 and dd == 8) else None)
ax.axhline(0.5, ls=":", color=C["muted"], lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels([FL[f] for f in FAMS])
ax.set_ylim(-0.03, 1.05); ax.set_ylabel("Orientation accuracy"); ax.set_title("CMC-VAE fitted (○ l_r=8, □ l_r=16; 5 reps)")
ax.legend(fontsize=6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12)); panel_label(ax, "b")
fig.tight_layout(); save(fig, f"{FIG}/SuppFig4a_synthetic_orientation.pdf")

# ------------------------------------------------------------------ Figure: interventional recovery
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
import csv
src = open(f"{FIG}/SuppFig_synthetic_interventional_recovery_source.csv", "w", newline=""); cw = csv.writer(src)
cw.writerow(["l_r", "family", "seed", "regime", "metric", "before", "after"])
for r in R:
    for rg in ["partial", "all"]:
        if rg not in r.get("ft", {}): continue
        s = r["ft"][rg]
        for mname, b, a in [("G_rr_orient", s["before_structure"]["G_marker_rr"], s["after_structure"]["G_marker_rr"]),
                            ("G_rr_orient_covered", s["before_structure"].get("G_marker_rr_covered"), s["after_structure"].get("G_marker_rr_covered")),
                            ("G_rr_orient_uncovered", s["before_structure"].get("G_marker_rr_uncovered"), s["after_structure"].get("G_marker_rr_uncovered")),
                            ("RNA_to_prot_orient", s["before_structure"]["G_marker_rp"], s["after_structure"]["G_marker_rp"]),
                            ("heldout_down_pearDE_learned", s["before_test"]["pearDE_down"], s["after_test"]["pearDE_down"]),
                            ("heldout_down_pearDE_reversedArr", s["before_test_rev"]["pearDE_down"], s["after_test_rev"]["pearDE_down"])]:
            cw.writerow([r["dr"], r["family"], r["seed"], rg, mname, b, a])
src.close()
FC = {"gaussian": C["muted"], "laplace": C["model"], "t8": C["base"]}
for ai, (rg, met, title) in enumerate([("all", "G_marker_rr", "RNA–RNA orientation,\none KO per latent"),
                                        ("partial", "G_marker_rr_covered", "RNA–RNA orientation,\n75% latents (intervened sources)")]):
    ax = axs[ai]
    for f in FAMS:
        for dd, mk in [(8, "o"), (16, "s")]:
            rs = [r for r in R if r["dr"] == dd and r["family"] == f and rg in r.get("ft", {})]
            for r in rs:
                b = r["ft"][rg]["before_structure"].get(met); a = r["ft"][rg]["after_structure"].get(met)
                if b is None or a is None or not (np.isfinite(b) and np.isfinite(a)): continue
                ax.plot([0, 1], [b, a], color=FC[f], lw=0.5, alpha=0.5); ax.scatter([0, 1], [b, a], s=6, marker=mk, color=FC[f])
    for f in FAMS: ax.plot([], [], color=FC[f], label=FL[f])
    ax.axhline(0.5, ls=":", color=C["muted"], lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(["observational", "+ interventional\nFT"])
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.03, 1.05); ax.set_ylabel("Orientation accuracy (G = DADᵀ)"); ax.set_title(title)
    if ai == 0: ax.legend(fontsize=6, loc="lower right")
    panel_label(ax, "abc"[ai])
ax = axs[2]
for k, (rg, lab) in enumerate([("all", "one KO/latent\n(unseen genes)"), ("partial", "75% latents\n(unseen latents)")]):
    for dd, off in [(8, -0.12), (16, 0.12)]:
        rs = [r for r in R if r["dr"] == dd and rg in r.get("ft", {})]
        if not rs: continue
        fv = np.array([r["ft"][rg]["after_test"]["pearDE_down"] for r in rs]); rv = np.array([r["ft"][rg]["after_test_rev"]["pearDE_down"] for r in rs])
        x0 = k * 1.0 + off
        for a_, b_ in zip(rv, fv): ax.plot([x0 - 0.05, x0 + 0.05], [a_, b_], color=C["shuf"], lw=0.5)
        ax.scatter(np.full(len(rv), x0 - 0.05), rv, s=5, color=C["shuf"], label="A_rr reversed" if (k == 0 and dd == 8) else None)
        ax.scatter(np.full(len(fv), x0 + 0.05), fv, s=5, color=C["model"], marker="o" if dd == 8 else "s", label="learned (after FT)" if (k == 0 and dd == 8) else None)
        p = paired(rv, fv)
        ax.text(x0, 1.0, ("***" if p["p_t"] < 1e-3 else "**" if p["p_t"] < 1e-2 else "*" if p["p_t"] < 0.05 else "ns"), ha="center", fontsize=7)
ax.axhline(0, color=C["muted"], lw=0.6); ax.set_xticks([0, 1]); ax.set_xticklabels([l for _, l in [("all", "one KO/latent\n(unseen genes)"), ("partial", "75% latents\n(unseen latents)")]])
ax.set_ylim(-0.6, 1.08); ax.set_ylabel("Held-out downstream pearDE"); ax.set_title("Direction is load-bearing after FT\n(○ l_r=8, □ l_r=16)")
ax.legend(fontsize=6, loc="lower left"); panel_label(ax, "c")
fig.tight_layout(); save(fig, f"{FIG}/SuppFig_synthetic_interventional_recovery.pdf")
