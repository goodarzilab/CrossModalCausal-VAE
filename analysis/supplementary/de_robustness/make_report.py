#!/usr/bin/env python
"""Assemble results.json + REPORT.md for the M10 DE-robustness analysis from runs/perturbseq_rescore.json and
runs/papalexi_rescore.json.  Conclusion paragraph is read from code/conclusion.md.  usage: python -I make_report.py"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import json, os
DER = f"{_CMC_RESULTS}/paper/analyses/review2/de_robust"
PS = json.load(open(f"{DER}/runs/perturbseq_rescore.json")); PA = json.load(open(f"{DER}/runs/papalexi_rescore.json"))
CONC = open(f"{DER}/code/conclusion.md").read().strip()
EVN = {"orig": "original (cell MW raw P<0.05)", "bh": "(A) cell MW BH q<0.05", "pb": "(B) pseudobulk limma BH q<0.05",
       "pbmask": "(B') PB mask + cell LFC", "pbgem": "(B'') PB per gem group", "orig_pbunits": "original DE, PB units only"}
ARMS = [("obs", "Observational"), ("real", "Fine-tuned"), ("shuffle_avg", "Label-shuffled"), ("infonce", "FT+InfoNCE"), ("infonce_shuffle_avg", "InfoNCE shuffled")]
f3 = lambda x: "–" if x is None else f"{x:.3f}"
def pf(p): return "–" if p is None else (f"{p:.1g}" if p < 0.001 else f"{p:.3f}")

R = dict(description="Reviewer M10: DE ground-truth robustness (BH-corrected cell-level MW; pseudobulk limma-trend BH) for Replogle Perturb-seq (Fig. 5) and Papalexi ECCITE-seq (Fig. 3c / untouched screen). No retraining; Perturb-seq fine-tuning re-run deterministically with the original (raw-P) targets, reproducing the published per-KO values to <1e-7; only the evaluation DE changes. Seed-42 full re-runs with fine-tuning targets built from the new DE are reported separately (train{bh,pb}).",
         perturbseq=dict(de_summary=PS["de_summary"], repro=PS["repro"], arms=PS["arms"], ds_arms=PS.get("ds_arms", {}),
                         contrasts={k: {ev: {c: {x: v[x] for x in ["mean_diff", "ci95_hier", "p_wilcoxon_seedavg", "q_bh", "seed_diffs", "sign_consistent", "n_ko"] if x in v}
                                             for c, v in p.items()} for ev, p in d.items()} for k, d in PS["contrasts"].items()},
                         n_test={t: {ev: PS["runs"][t][ev]["n_test"] for ev in PS["runs"][t]} for t in PS["runs"] if t.endswith("s42_orig")}),
         papalexi=dict(de_counts=PA["de_counts"], pb_info=PA["pb_info"], published_check=PA["published_check"],
                       variants={v: {k: PA["variants"][v][k] for k in ["units", "n_units", "n_S", "de_per_unit_median", "table", "paired", "vs_chance"]} for v in PA["variants"]}),
         conclusion=CONC)
json.dump(R, open(f"{DER}/results.json", "w"), indent=1)

L = ["# M10 — DE ground-truth robustness (Papalexi ECCITE-seq; Replogle K562/Jurkat Perturb-seq)", "",
     "## Conclusion", "", CONC, "",
     "## What was done", "",
     "* **DE definitions.** *orig* = published: cell-level two-sided Mann–Whitney vs pooled NT cells, raw P<0.05 & |LFC|≥0.10. "
     "**(A) bh** = same test, Benjamini–Hochberg across features within each KO, q<0.05 & |LFC|≥0.10. "
     "**(B) pb** = pseudobulk limma-trend-like moderated t (`code/stats/pblimma.limma_ko`, the code used for the CITE-seq development screens), BH q<0.05 & |LFC_pb|≥0.10, observed response = pseudobulk LFC. "
     "Perturb-seq pseudobulks: the 55 (Jurkat) / 48 (K562) gem groups cut into 8 contiguous blocks of ~equal size (KO pseudobulk ≥5 cells, ≥3 pseudobulks; control pseudobulk = all NT cells of the block; design KO + block, i.e. block-matched LFC with KO×block variance; median 8 pseudobulks, df 7). "
     "Papalexi pseudobulks: guide × replicate (rep1–3), ≥10 cells, controls = NT guide × replicate (26 pseudobulks), design KO + replicate. "
     "Sensitivities: *pbmask* (PB significance mask but cell-level LFC as response), *pbgem* (Perturb-seq pseudobulk per individual gem group, ≥3 cells), *orig_pbunits* (Papalexi: original DE restricted to the units still evaluable under PB).",
     "* **Everything else unchanged**: evaluable KO = ≥3 DE genes excluding the target; held-out KOs = evaluable ∩ the published gene split; retrieval feature set S, leave-one-out reference means and centred-pearDE masks are recomputed from the evaluation DE.",
     "* **No retraining.** Papalexi: the saved zero-shot predictions (`analyses/untouched/rc_*_preds.npz`) were rescored. Perturb-seq: fine-tuned SCMs were not saved, so the Fig. 5 fine-tunings (obs, FT, 3× label-shuffled, InfoNCE, 3× InfoNCE-shuffled; DeepSEM top-73,728 all arms; obs + control-only pretraining; 3 seeds) were re-run with the identical deterministic code and the ORIGINAL (raw-P) DE targets, predictions for every KO were saved and then scored under each DE. "
     "**Only the evaluation DE changes; fine-tuning targets/features still come from the original DE.** Because re-fine-tuning is cheap (~3 min/model on CPU), a full re-run with fine-tuning targets built from the new DE (train=bh, train=pb) is reported for seed 42 (both lines, obs pretraining).",
     "* DeepSEM fine-tuning was re-run on CPU (published runs: GPU). Inner InfoNCE grids are identical and arm means agree within 2e-6. The published DiD Δ and P are reproduced exactly; the q values differ only because the BH panel here is smaller (no centred-target arms). Duplicate GPU runs (`preds_gpu/`, `runs_gpu/`) are not used.", ""]

# ---------------- harness check
L += ["## Harness reproduction (original DE)", "", "| set | arm | metric | published | rescored |", "|---|---|---|---|---|"]
for k, v in PS["repro"].items():
    if not isinstance(v, dict): L.append(f"| {k} | – | – | {v} | |"); continue
    for a, mm in v.items():
        if isinstance(mm, dict) and "published" in mm and not isinstance(mm["published"], list):
            L.append(f"| {k} | {a} | retrieval | {f3(mm['published'])} | {f3(mm['rescored'])} |")
        elif isinstance(mm, dict) and "published" in mm:
            L.append(f"| {k} | {a} | Δ, P, q | {', '.join(pf(x) if i else f'{x:+.3f}' for i, x in enumerate(mm['published']))} | {', '.join(pf(x) if i else f'{x:+.3f}' for i, x in enumerate(mm['rescored']))} |")
        elif isinstance(mm, dict):
            for m, d in mm.items(): L.append(f"| {k} | {a} | {m.replace('rr_', '')} | {f3(d['published'])} | {f3(d['rescored'])} |")
        else: L.append(f"| {k} | {a} | | | {mm:.1e} |")
L += ["", "Papalexi (seed-mean CMC and baselines; [published, rescored]):", "", "| method | AUROC | centred pearDE | retrieval |", "|---|---|---|---|"]
for k, d in PA["published_check"].items():
    L.append(f"| {k} | " + " | ".join(f"{f3(d[m][0])} / {f3(d[m][1])}" for m in ["rr_auroc", "rr_pe_cent", "rr_retrieval"]) + " |")

# ---------------- DE counts
L += ["", "## DE genes per perturbation", "", "| screen | definition | evaluable KOs | median DE / evaluable KO [IQR] | median DE / all KOs |", "|---|---|---|---|---|"]
for line, s in PS["de_summary"].items():
    for ev in ["orig", "bh", "pb", "pbgem"]:
        d = s[ev]; L.append(f"| {line} | {EVN[ev]} | {d['n_evaluable']} | {d['median_de_evaluable']:.1f} [{d['iqr_evaluable'][0]:.0f}, {d['iqr_evaluable'][1]:.0f}] | {d['median_de_all']:.0f} |")
    L.append(f"| {line} | PB info | median {s['n_pb_median']:.0f} pseudobulks, df {s['df_pb_median']:.0f}; median r(cell LFC, PB LFC) = {s['corr_lfc_cell_vs_pb_median']:.3f}; median Jaccard(orig, bh) = {s['jaccard_orig_bh_median']:.2f} | | |")
for v in ["orig", "bh", "pb"]:
    V = PA["variants"][v]; L.append(f"| Papalexi | {EVN[v]} | {V['n_units']} of 13 | {V['de_per_unit_median']:.0f} | " + ", ".join(f"{g} {n}" for g, n in V["de_per_unit"].items()) + " |")
L.append("")
L.append("Held-out KOs scored (seed-42 obs): " + "; ".join(f"{t.split('_')[0]}: " + ", ".join(f"{ev} {n}" for ev, n in d.items()) for t, d in R["perturbseq"]["n_test"].items() if "_obs_" in t))

# ---------------- Perturb-seq arms
def arm_table(key, title):
    if key not in PS["arms"]: return
    L.extend(["", f"### {title}", "", "| eval DE | " + " | ".join(lb for _, lb in ARMS) + " |", "|---|" + "---|" * len(ARMS)])
    for ev, a in PS["arms"][key].items():
        L.append(f"| {EVN[ev]} | " + " | ".join(f"{a[k]['rr_pe_cent']['mean']:+.3f} / {a[k]['rr_retrieval']['mean']:.3f}" for k, _ in ARMS) + " |")
L += ["", "## Perturb-seq (Fig. 5): held-out centred pearDE / retrieval (mean of seeds)"]
for line in ["jurkat", "k562"]:
    arm_table(f"{line}_obs_trainorig", f"{line}, pretraining = controls + training perturbations, fine-tuned on ORIGINAL DE targets (3 seeds)")
    arm_table(f"{line}_ctrl_trainorig", f"{line}, control-only pretraining, original DE targets (3 seeds)")
    for d in ["bh", "pb"]: arm_table(f"{line}_obs_train{d}", f"{line}, FULL RE-RUN seed 42 with fine-tuning targets from DE = {d}")

def ctab(key, names, title):
    if key not in PS["contrasts"]: return
    L.extend(["", f"### {title}", "", "| eval DE | contrast | Δ | 95% CI (hier.) | per-seed Δ | P (seed-avg Wilcoxon) | q (BH, panel) |", "|---|---|---|---|---|---|---|"])
    for ev, p in PS["contrasts"][key].items():
        for c in names:
            if c in p:
                v = p[c]; L.append(f"| {EVN[ev]} | {c} | {v['mean_diff']:+.3f} | [{v['ci95_hier'][0]:+.3f}, {v['ci95_hier'][1]:+.3f}] | {', '.join(f'{x:+.3f}' for x in v['seed_diffs'])} | {pf(v['p_wilcoxon_seedavg'])} | {pf(v.get('q_bh'))} |")
L += ["", "## Perturb-seq contrasts (seed-averaged per-KO differences; BH within each line × pretraining × eval-DE panel)"]
FT = ["real_vs_shuffle_avg|rr_pe_cent", "real_vs_shuffle_avg|rr_retrieval", "real_vs_obs|rr_pe_cent", "real_vs_obs|rr_retrieval",
      "infonce_vs_infonce_shuffle_avg|rr_pe_cent", "infonce_vs_infonce_shuffle_avg|rr_retrieval"]
DS = ["DiD_CMCgain_minus_DSgain:real|rr_retrieval", "DiD_CMCgain_minus_DSgain:infonce|rr_retrieval", "DiD_CMCgain_minus_DSgain:real|rr_pe_cent", "DiD_CMCgain_minus_DSgain:infonce|rr_pe_cent",
      "DS_real_vs_shuffle_avg|rr_retrieval", "DS_infonce_vs_infonce_shuffle_avg|rr_retrieval"]
for line in ["jurkat", "k562"]:
    ctab(f"{line}_obs_trainorig", FT, f"{line}: fine-tuning gains (Fig. 5c), obs pretraining")
    ctab(f"{line}_obs_trainorig", DS, f"{line}: CMC − DeepSEM(top-73,728, same cells) retrieval-gain difference-in-differences (Fig. 5e)")
    ctab(f"{line}_ctrl_trainorig", FT[:2] + FT[4:], f"{line}: control-only pretraining gains (Fig. 5f)")
    ctab(f"{line}_ctrl_vs_obs", ["ctrl_minus_obs_gain:real|rr_pe_cent", "ctrl_minus_obs_gain:real|rr_retrieval", "ctrl_minus_obs_gain:infonce|rr_pe_cent", "ctrl_minus_obs_gain:infonce|rr_retrieval"],
         f"{line}: control-only minus obs-pretraining gain")
    for d in ["bh", "pb"]:
        ctab(f"{line}_obs_train{d}", FT + DS[:2], f"{line}: full re-run seed 42, fine-tuning targets from DE = {d} (single seed: per-KO Wilcoxon)")
if PS.get("ds_arms"):
    L += ["", "### DeepSEM (top-73,728) arms: centred pearDE / retrieval", "", "| set | eval DE | " + " | ".join(lb for _, lb in ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
    for key, d in PS["ds_arms"].items():
        for ev, a in d.items():
            L.append(f"| {key} | {EVN[ev]} | " + " | ".join(f"{a[k]['rr_pe_cent']['mean']:+.3f} / {a[k]['rr_retrieval']['mean']:.3f}" for k, _ in ARMS) + " |")

# ---------------- Papalexi
L += ["", "## Papalexi ECCITE-seq (Fig. 3c / untouched screen): frozen zero-shot predictions rescored", ""]
MS = ["CMC-VAE (seed mean)", "CMC-VAE sp_dec_gl1000", "CMC-VAE sp_dec_gl1000_s1", "CMC-VAE sp_dec_gl1000_s2", "corr", "totalvi", "deepsem", "zs_ctrl_expr", "zs_ctrl_var", "mean_response_ref"]
for v, V in PA["variants"].items():
    L += [f"### {EVN.get(v, v)} — {V['n_units']} units ({', '.join(V['units'])}); |S| = {V['n_S']}", "",
          "| method | RNA AUROC [95% CI] | centred pearDE [95% CI] | retrieval [95% CI] |", "|---|---|---|---|"]
    for k in MS:
        t = V["table"][k]
        L.append(f"| {k} | " + " | ".join("–" if not t[m].get("n") else f"{t[m]['mean']:.3f} [{t[m]['ci95'][0]:.2f}, {t[m]['ci95'][1]:.2f}]" for m in ["rr_auroc", "rr_pe_cent", "rr_retrieval"]) + " |")
    L += ["", "| CMC-VAE (seed mean) minus | ΔAUROC (q) | Δcentred (q) | Δretrieval (q) |", "|---|---|---|---|"]
    for c, d in V["paired"].items():
        L.append(f"| {c.replace('CMC-VAE minus ', '')} | " + " | ".join("–" if "mean_diff" not in d[m] else f"{d[m]['mean_diff']:+.3f} ({pf(d[m].get('q_bh'))})" for m in ["rr_auroc", "rr_pe_cent", "rr_retrieval"]) + " |")
    vc = V["vs_chance"]["CMC-VAE (seed mean)"]
    L += ["", f"CMC-VAE vs chance: centred pearDE vs 0 P_t = {pf(vc['pe_cent_vs0'].get('p_t'))}; retrieval vs 0.5 P_t = {pf(vc['retrieval_vs05'].get('p_t'))}; AUROC vs 0.5 P_t = {pf(vc['auroc_vs05'].get('p_t'))}", ""]

L += ["## Files", "",
      "* `code/de_perturbseq.py` (Perturb-seq DE variants → `de/{line}_de_variants.npz`), `code/ft_eval_de.py` (copy of `analysis/fig5_interventions/perturbseq/ft_eval.py`: Fig. 5 arms only, PS_DE_TRAIN switch, saves predictions for all KOs), `code/rescore_perturbseq.py`, `code/papalexi_de_rescore.py`, `code/make_report.py`, sbatch wrappers.",
      "* `runs/perturbseq_rescore.json` (+ per-KO `runs/perturbseq_rescore_perko.json.gz`), `runs/papalexi_rescore.json`, `runs/*_s*_{orig,bh,pb}[_ds].json` (fit logs), `preds/` → $CMCVAE_SCRATCH/review2_de_robust/preds (saved predictions), `logs/`.",
      "* Nothing under FINAL_SUBMISSION, FINAL_FIGURES or existing analyses/models was modified."]
open(f"{DER}/REPORT.md", "w").write("\n".join(L) + "\n")
print("WROTE results.json, REPORT.md")
