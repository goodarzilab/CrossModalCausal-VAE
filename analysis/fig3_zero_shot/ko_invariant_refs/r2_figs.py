"""Round-2 figure panels (paper style, analysis/figures/original/style.py). Source data JSON written next to every panel."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import sys, json, numpy as np
sys.path.insert(0, f"{_CMC_ROOT}/analysis/figures/original")
from style import C, plt, panel_label, save
O = f"{_CMC_SCRATCH}/nov_round2/out"; F = f"{_CMC_RESULTS}/paper/figures/round2"
A = json.load(open(f"{O}/main_analysis.json")); S4 = json.load(open(f"{O}/st4_analysis.json")); PP = json.load(open(f"{O}/papalexi_paired_vs_rnaonly.json"))
TAGS = ["frangieh", "iccite", "arce"]; NICE = dict(frangieh="Frangieh\n(melanoma)", iccite="Chen\n(CD4 T)", arce="Arce\n(CD4 T)", all="Pooled")
def mean(pre, st, met, t, src=A): return src["means"].get(f"{pre}|{st}|{met}|{t}")
def fam(name, src=A): return src["families"][name]
def find(name, a, b, met, t, src=A, prefix=None):
    for e in src["families"][name]:
        if e["a"] == a and e["b"] == b and e["metric"] == met and e["screen"] == t and (prefix is None or e["prefix"] == prefix): return e
def dump(obj, name): json.dump(obj, open(f"{F}/{name}_source_data.json", "w"), indent=1)

# ---------------- Panel 1: Fig 3b with knockout-invariant rankings + adjusted AUROC (task 3)
SETS = [("clamp3", "CMC-VAE", C["model"]), ("rnaonly3", "CMC-VAE RNA-only", C["cmc2"]), ("corr", "Corpus correlation", C["base"]),
        ("totalvi_ko", "totalVI", "#f6ad55"), ("deepsem_ko", "DeepSEM", C["genie3"]), ("ctrl_mean", "Control mean\n(KO-invariant)", C["muted"]), ("ctrl_sd", "Control s.d.\n(KO-invariant)", C["shuf"])]
MET = [("rr_auroc", "RNA AUROC"), ("rr_au_strat_ms10", "Expression-stratified AUROC\n(10×10 mean × s.d. strata)"), ("rr_au_resid", "Residualised AUROC\n(|pred| ⊥ mean, s.d.)")]
fig, axes = plt.subplots(len(MET), 3, figsize=(7.2, 6.2), sharey="row")
src = {}
for r, (met, ml) in enumerate(MET):
    for c, t in enumerate(TAGS):
        ax = axes[r, c]; x = np.arange(len(SETS))
        for i, (st, lab, col) in enumerate(SETS):
            e = mean("F3", st, met, t)
            if not e: continue
            ax.bar(i, e["point"], color=col, width=0.72, edgecolor="white", linewidth=0.6)
            if e.get("ci"): ax.errorbar(i, e["point"], yerr=[[e["point"] - e["ci"][0]], [e["ci"][1] - e["point"]]], color=C["ink"], lw=0.8, capsize=1.5)
            src[f"{met}|{st}|{t}"] = e
        ax.axhline(0.5, color=C["muted"], lw=0.7, ls="--"); ax.set_ylim(0.35, 1.0) if r == 0 else ax.set_ylim(0.3, 0.75)
        ax.set_xticks(x); ax.set_xticklabels([s[1] for s in SETS] if r == len(MET) - 1 else [], rotation=60, ha="right", fontsize=6.5)
        if r == 0: ax.set_title(NICE[t].replace("\n", " "))
        if c == 0: ax.set_ylabel(ml, fontsize=7.5)
panel_label(axes[0, 0], "a", x=-0.35); panel_label(axes[1, 0], "b", x=-0.35); panel_label(axes[2, 0], "c", x=-0.35)
fig.tight_layout(); save(fig, f"{F}/R2_Fig3b_koinvariant_stratified.pdf"); dump(src, "R2_Fig3b_koinvariant_stratified")

# ---------------- Panel 2: pairing ablation forest (task 1)
rows = []
for t in TAGS:
    for met, ml in [("rr_auroc", "AUROC"), ("rr_pe", "pearDE"), ("rr_pe_cent", "centred pearDE"), ("rr_retrieval", "retrieval")]:
        e = find("T1_fig3_paired_vs_rnaonly", "clamp3", "rnaonly3", met, t); rows.append((f"Fig 3 · {NICE[t].split(chr(10))[0]} · {ml}", e))
for a, al in [("baseline", "untuned"), ("real", "FT real"), ("rs", "FT real − shuffled")]:
    for met, ml in [("rr_pe_cent", "centred pearDE"), ("rr_retrieval", "retrieval")]:
        rows.append((f"Fig 5b · {al} · {ml}", find("X5_paired_vs_rnaonly", f"paired:{a}3", f"rnaonly:{a}3", met, "all")))
for e in PP["rows"]:
    if e["mode"] == "pb": rows.append((f"Papalexi (pseudobulk) · {e['metric'].replace('rr_', '')}", dict(point=e["diff"], ci=e.get("ci_joint", e["ci_ko"]), q=e.get("q"))))
fig, ax = plt.subplots(figsize=(4.6, 0.19 * len(rows) + 0.8))
for i, (lab, e) in enumerate(rows[::-1]):
    if not e: continue
    col = C["model"] if (e.get("q") is not None and e["q"] < 0.05) else C["muted"]
    ax.errorbar(e["point"], i, xerr=[[e["point"] - e["ci"][0]], [e["ci"][1] - e["point"]]], fmt="o", ms=3.5, color=col, lw=0.9, capsize=1.5)
ax.axvline(0, color=C["ink"], lw=0.7); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.3)
ax.set_xlabel("Paired (RNA + protein) − RNA-only CMC-VAE\n(3 models each; joint KO × control-cell × seed 95% CI)")
ax.set_title("Pairing ablation: what protein adds to RNA-side results", fontsize=8)
save(fig, f"{F}/R2_pairing_ablation_forest.pdf"); dump([(l, e) for l, e in rows], "R2_pairing_ablation_forest")

# ---------------- Panel 3: adapted model (task 5)
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4))
SETS5 = [("clamp3", "Unadapted", C["model"]), ("adapt3", "Adapted (i)", C["accent"]), ("corr", "Correlation", C["base"]), ("deepsem_ko", "DeepSEM", C["genie3"]), ("ctrl_mean", "Control mean", C["muted"])]
src = {}
for k, (met, ml) in enumerate([("rr_auroc", "RNA AUROC"), ("rr_au_strat_ms10", "Stratified AUROC"), ("rr_pe_cent", "Centred pearDE"), ("rr_retrieval", "Retrieval")]):
    ax = axes[k]; w = 0.16
    for i, (st, lab, col) in enumerate(SETS5):
        if st == "ctrl_mean" and met in ("rr_pe_cent", "rr_retrieval"): continue
        for j, t in enumerate(TAGS):
            e = mean("F3", st, met, t)
            if not e: continue
            xx = j + (i - 2) * w
            ax.bar(xx, e["point"], width=w, color=col, edgecolor="white", linewidth=0.4, label=lab if (k == 0 and j == 0) else None)
            if e.get("ci"): ax.errorbar(xx, e["point"], yerr=[[e["point"] - e["ci"][0]], [e["ci"][1] - e["point"]]], color=C["ink"], lw=0.6, capsize=1)
            src[f"{met}|{st}|{t}"] = e
    ax.set_xticks(range(3)); ax.set_xticklabels([NICE[t] for t in TAGS], fontsize=6.3); ax.set_title(ml)
    isa = met.startswith("rr_au") or met == "rr_retrieval"
    ax.axhline(0.5 if isa else 0, color=C["muted"], lw=0.6, ls="--")
    if isa: ax.set_ylim(0.35, 1.0 if met in ("rr_auroc", "rr_retrieval") else 0.75)
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=6.5, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); save(fig, f"{F}/R2_adapted_fig3.pdf"); dump(src, "R2_adapted_fig3")

# ---------------- Panel 4: LFC grid (task 6)
GR = [0.05, 0.10, 0.15, 0.25]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
src = []
for k, (met, ml) in enumerate([("rr_auroc", "RNA AUROC"), ("rr_pe_cent", "Centred pearDE"), ("rr_retrieval", "Retrieval")]):
    ax = axes[k]
    for j, t in enumerate(TAGS):
        xs, ys, lo, hi, ns = [], [], [], [], []
        for fr in GR:
            e = find("T6_lfc_grid", "clamp3", "corr", met, t, prefix=f"LG_r{fr}_p0.05")
            if not e: continue
            xs.append(fr + (j - 1) * 0.006); ys.append(e["point"]); lo.append(e["ci"][0]); hi.append(e["ci"][1]); ns.append(e["n"]); src.append(e)
        col = [C["model"], C["cmc2"], "#90cdf4"][j]
        ax.errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], fmt="-o", ms=3, lw=0.9, color=col, capsize=1.5, label=NICE[t].split("\n")[0])
        for x_, y_, n_ in zip(xs, hi, ns): ax.text(x_, y_ + 0.004, str(n_), fontsize=5, ha="center", color=col)
    ax.axhline(0, color=C["ink"], lw=0.6); ax.set_xticks(GR); ax.set_xlabel("RNA |LFC| floor"); ax.set_title(f"{ml}: clamp − correlation", fontsize=7.5)
axes[0].legend(fontsize=6); fig.tight_layout(); save(fig, f"{F}/R2_lfc_grid.pdf"); dump(src, "R2_lfc_grid")

# ---------------- Panel 5: Fig 5b,c and Supp Table 4 under the joint bootstrap (task 8)
rows = []
for a, b, lab in [("real", "baseline", "real − untuned"), ("real", "shuf", "real − shuffled"), ("real", "mean", "real − screen-mean"), ("shuf", "baseline", "shuffled − untuned")]:
    for met, ml in [("rr_pe_cent", "centred pearDE"), ("rr_retrieval", "retrieval"), ("rr_pe", "raw pearDE")]:
        f_ = "T8_st4_primary" if a == "real" and b in ("baseline", "shuf") else "T8_st4_secondary"
        rows.append((f"Supp T4 · {lab} · {ml}", find(f_, f"st4:{a}3", f"st4:{b}3", met, "all", S4)))
for a, b, lab in [("real", "baseline", "real − untuned"), ("real", "shuf", "real − shuffled")]:
    for met, ml in [("rr_pe_cent", "RNA centred"), ("rr_retrieval", "RNA retrieval"), ("rp_pe", "protein pearDE (n=38)")]:
        rows.append((f"Fig 5b,c · {lab} · {ml}", find("T8_fig5_paired", f"paired:{a}3", f"paired:{b}3", met, "all")))
fig, ax = plt.subplots(figsize=(4.6, 0.19 * len(rows) + 0.8))
for i, (lab, e) in enumerate(rows[::-1]):
    if not e: continue
    col = C["model"] if e.get("q", 1) < 0.05 else C["muted"]
    ax.errorbar(e["point"], i, xerr=[[e["point"] - e["ci"][0]], [e["ci"][1] - e["point"]]], fmt="o", ms=3.5, color=col, lw=0.9, capsize=1.5)
ax.axvline(0, color=C["ink"], lw=0.7); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.3)
ax.set_xlabel("Difference (joint KO × control-cell × seed bootstrap, ≥10,000 replicates)"); ax.set_title("CITE-seq fine-tuning contrasts, one inferential framework", fontsize=8)
save(fig, f"{F}/R2_finetune_jointboot.pdf"); dump([(l, e) for l, e in rows], "R2_finetune_jointboot")
