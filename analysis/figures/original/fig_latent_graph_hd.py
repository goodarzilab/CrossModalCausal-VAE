"""AppendixFigS5 — healthy-only vs disease-only ablation (identical 2-cohort top architecture trained on
hd_split/{healthy_gl,disease_gl}). (A,B) latent-program graphs: the GSEA-significant RNA latents grouped
by immune-program family with their A_rr connections, healthy vs disease. (C) the program shift made
explicit: #significant latents per immune-program family, healthy vs disease -- healthy is
antigen-presentation--dominated; disease expands the type-I/II interferon and apoptosis programs.
Reads results/hd_split/{healthy_gl,disease_gl}/{config.json,final_model.pt} + gsea_latents_immune_*.json."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, math, warnings, numpy as np, torch; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__)); from style import plt, panel_label
sys.path.insert(0, "src"); from model import CrossModalCausalLDVAE
import networkx as nx; from collections import defaultdict, Counter
R = f"{_CMC_ROOT}"
RUNS = [("Healthy-only", "results/hd_split/healthy_gl", "healthy_gl"),
        ("Disease-only", "results/hd_split/disease_gl", "disease_gl")]

def fam(p):
    p = p.upper()
    if "INTERFERON_GAMMA" in p: return "IFN-$\\gamma$"
    if "INTERFERON_ALPHA" in p or "DDX58" in p or "IFIH" in p: return "IFN-$\\alpha/\\beta$"
    if "MHC" in p or "ANTIGEN" in p: return "Antigen/MHC"
    if "TCR" in p or "T_CELL_RECEPTOR" in p: return "TCR"
    if "TNFA" in p or "NFKB" in p: return "TNF$\\alpha$/NF$\\kappa$B"
    if "JAK_STAT" in p: return "JAK-STAT"
    if "APOPTOSIS" in p: return "Apoptosis"
    return "Other"
FCOL = {"IFN-$\\gamma$": "#c53030", "IFN-$\\alpha/\\beta$": "#dd6b20", "Antigen/MHC": "#2b6cb0",
        "TCR": "#2f855a", "TNF$\\alpha$/NF$\\kappa$B": "#6b46c1", "JAK-STAT": "#319795",
        "Apoptosis": "#718096", "Other": "#a0aec0"}
FAM_ORDER = ["Antigen/MHC", "TCR", "JAK-STAT", "IFN-$\\alpha/\\beta$", "IFN-$\\gamma$", "TNF$\\alpha$/NF$\\kappa$B", "Apoptosis", "Other"]

def load(rd):
    c = json.load(open(f"{rd}/config.json")); sd = torch.load(f"{rd}/final_model.pt", map_location="cpu")
    ng, npr = sd["dec_rna.weight"].shape[0], sd["dec_prot.weight"].shape[0]
    m = CrossModalCausalLDVAE(ng, npr, 1, latent_dim_rna=c.get("latent_dim_rna", 128),
                              latent_dim_prot=c.get("latent_dim_prot", 64), recon_likelihood="mse")
    m.load_state_dict(sd, strict=False); m.eval()
    with torch.no_grad(): A_rr, _, _, _ = m.scm.get_block_adjacencies(tau=0.5)
    A = A_rr.float().numpy(); np.fill_diagonal(A, 0); return A

fig = plt.figure(figsize=(12.6, 10.8)); gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.16, wspace=0.05)
counts = {}
for col, (title, rd, tag) in enumerate(RUNS):
    A = load(rd)
    gj = json.load(open(f"{_CMC_RESULTS}/gsea_latents_immune_{tag}.json"))["latent_pathway"]
    sig = {int(k): v for k, v in gj.items() if v[2] < 0.1}
    nodes = sorted(sig); nf = {n: fam(sig[n][0]) for n in nodes}
    counts[title] = Counter(nf.values())
    axA = fig.add_subplot(gs[0, col]); axA.axis("off")
    G = nx.Graph(); G.add_nodes_from(nodes)
    w = sorted([(abs(A[i, j]) + abs(A[j, i]), i, j) for i in nodes for j in nodes if i < j], reverse=True)
    for val, i, j in w[:int(len(nodes) * 1.6)]: G.add_edge(i, j, weight=val)
    famlist = sorted(set(nf[n] for n in G.nodes)); RING = 3.6
    fc = {f: (RING * math.cos(2*math.pi*i/len(famlist)), RING * math.sin(2*math.pi*i/len(famlist))) for i, f in enumerate(famlist)}
    pos = {}
    for f in famlist:
        fn = sorted([n for n in G.nodes if nf[n] == f]); cx, cy = fc[f]
        for k2, n in enumerate(fn):
            if len(fn) == 1: pos[n] = (cx, cy)
            else:
                rr = min(0.55 + 0.13*len(fn), 1.05); ang = 2*math.pi*k2/len(fn); pos[n] = (cx + rr*math.cos(ang), cy + rr*math.sin(ang))
    ews = [G[u][v]["weight"] for u, v in G.edges]; mx = max(ews) if ews else 1
    nx.draw_networkx_edges(G, pos, ax=axA, width=[0.3 + 2.2*e/mx for e in ews], edge_color="#d0d7e2", alpha=0.55)
    nx.draw_networkx_nodes(G, pos, ax=axA, node_size=185, node_color=[FCOL[nf[n]] for n in G.nodes], edgecolors="white", linewidths=0.9)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, ax=axA, font_size=5.0, font_color="white", font_weight="bold")
    for f in famlist:
        cx, cy = fc[f]; nrm = math.hypot(cx, cy) or 1; lx, ly = cx/nrm*(RING+1.85), cy/nrm*(RING+1.85)
        ha = "left" if lx > 0.8 else ("right" if lx < -0.8 else "center")
        axA.text(lx, ly, f, ha=ha, va="center", fontsize=8.2, fontweight="bold", color=FCOL[f], clip_on=False)
    axA.set_xlim(-7.6, 8.2); axA.set_ylim(-6.4, 6.4)
    axA.set_title(f"{title}: latent programs & $A_{{rr}}$ ({len(nodes)} sig. latents)", fontsize=10.5)
    panel_label(axA, "AB"[col])

# ---- C: program-family shift, healthy vs disease ----
axC = fig.add_subplot(gs[1, :])
present = [f for f in FAM_ORDER if counts["Healthy-only"].get(f, 0) or counts["Disease-only"].get(f, 0)]
x = np.arange(len(present)); w = 0.4
h = [counts["Healthy-only"].get(f, 0) for f in present]; d = [counts["Disease-only"].get(f, 0) for f in present]
axC.bar(x - w/2, h, w, color="#2b6cb0", label="Healthy-only")
axC.bar(x + w/2, d, w, color="#c53030", label="Disease-only")
for i, (a, b) in enumerate(zip(h, d)):
    axC.text(i - w/2, a + 0.15, str(a), ha="center", fontsize=7.5, color="#2b6cb0")
    axC.text(i + w/2, b + 0.15, str(b), ha="center", fontsize=7.5, color="#c53030")
axC.set_xticks(x); axC.set_xticklabels(present, fontsize=9)
axC.set_ylabel("# significant latents (FDR $<$ 0.1)"); axC.set_ylim(0, max(h + d) + 2)
axC.legend(fontsize=9, frameon=False, loc="upper right")
axC.set_title("Program shift: healthy is antigen-presentation--dominated; disease expands the interferon programs", fontsize=10.5)
panel_label(axC, "C")

fig.savefig(f"{R}/paper/figures/AppendixFigS5.pdf", bbox_inches="tight")
fig.savefig(f"{R}/paper/figures/AppendixFigS5.png", dpi=150, bbox_inches="tight")
print("WROTE AppendixFigS5.pdf (healthy vs disease)")
print("=== family counts (FDR<0.1) — Healthy | Disease ===")
for f in present: print(f"  {f:22s} {counts['Healthy-only'].get(f,0):2d} | {counts['Disease-only'].get(f,0):2d}")
print(f"  {'TOTAL':22s} {sum(counts['Healthy-only'].values()):2d} | {sum(counts['Disease-only'].values()):2d}")
