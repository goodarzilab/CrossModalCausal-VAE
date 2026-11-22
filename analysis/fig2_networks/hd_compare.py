"""Healthy-vs-disease comparison for the two CMC-VAE models trained on subsets of combined_v4
(shared 2692-gene / 144-protein v4 vocab => node-comparable latent graphs, NO intersection needed).

  HEALTHY  results/hd_split/healthy_gl   (Hao + Stephenson-healthy)
  DISEASE  results/hd_split/disease_gl   (Stephenson Covid/Non_covid/LPS)

Analyses (writes results/healthy_vs_disease.json + paper/figures/_new_healthy_vs_disease.pdf):
  A. GSEA latent-program comparison  (reads results/gsea_latents_immune_{healthy_gl,disease_gl}.json
     produced by analysis/fig2_networks/gsea_latents.py in immune mode): which immune programs are enriched, and which
     are gained / lost in disease.
  B. Latent GRN G_rr similarity (GRN_inferred.csv): edge Pearson/Spearman, per-gene degree corr,
     top-1% edge Jaccard.
  C. Differential connectivity: disease-gained vs healthy-gained hub genes + top differential edges;
     is the IFN/inflammatory sub-network more central in disease? (hypergeometric enrichment).
  D. Cognate translation map G_rp (Transcription_RNA_to_Prot.csv) stability: does the central-dogma
     gene->cognate-protein map persist healthy vs disease?

Usage: python analysis/fig2_networks/hd_compare.py [--figure]
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
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import scipy.stats as ss
from sklearn.metrics import roc_auc_score  # noqa

R = f"{_CMC_ROOT}"
HEALTHY = f"{_CMC_RESULTS}/hd_split/healthy_gl"
DISEASE = f"{_CMC_RESULTS}/hd_split/disease_gl"
GSEA_H = f"{_CMC_RESULTS}/gsea_latents_immune_healthy_gl.json"
GSEA_D = f"{_CMC_RESULTS}/gsea_latents_immune_disease_gl.json"
OUT_JSON = f"{_CMC_RESULTS}/healthy_vs_disease.json"
OUT_FIG = f"{R}/paper/figures/_new_healthy_vs_disease.pdf"

# ---- curated inflammatory / interferon gene sets (same as analysis/fig2_networks/healthy_vs_disease_grn.py) ----
_IFN_A = ("STAT1 STAT2 IRF1 IRF7 IRF9 ISG15 IFIT1 IFIT2 IFIT3 MX1 MX2 OAS1 OAS2 OAS3 OASL RSAD2 "
          "IFI6 IFI27 IFI35 IFI44 IFI44L IFITM1 IFITM2 IFITM3 USP18 BST2 XAF1 SAMD9 DDX58 IFIH1").split()
_IFN_G = ("STAT1 IRF1 GBP1 GBP2 GBP4 GBP5 CXCL9 CXCL10 CXCL11 CIITA TAP1 TAP2 PSMB8 PSMB9 NLRC5 "
          "SOCS1 SOCS3 JAK2 IRF8 IDO1 WARS UBE2L6 ISG20 VCAM1 IFI30").split()
_TNF  = ("TNF NFKB1 NFKB2 RELA RELB REL NFKBIA NFKBIE TNFAIP3 TNFAIP2 TNFAIP6 TRAF1 TRAF2 BIRC3 "
         "CXCL1 CXCL2 CXCL3 CCL2 CCL20 IL1B IL6 CXCL8 PTGS2 ICAM1 SOD2 JUN JUNB FOS FOSB NFKBIB").split()
_STING = ("MB21D1 CGAS TMEM173 STING1 TBK1 IKBKE IRF3 DDX58 IFIH1 MAVS TREX1 IFI16 ZBP1 CCL5 OASL "
          "DDX41 TANK TRAF3 TRAF6").split()
_EXTRA = "C1QA C1QB C1QC IRF5 IFI30 GBP3 OAS1 ISG15".split()
INFLAM = sorted(set(_IFN_A) | set(_IFN_G) | set(_TNF) | set(_STING) | set(_EXTRA))
IFN_ONLY = sorted(set(_IFN_A) | set(_IFN_G) | set(_STING))

# immune programs that are inflammatory/interferon vs baseline lineage/antigen
INFLAM_PROG_KEYS = ("INTERFERON", "TNFA", "JAK_STAT", "STING", "DDX58_IFIH1")


def is_inflam_prog(term):
    t = term.upper()
    return any(k in t for k in INFLAM_PROG_KEYS)


def short_prog(term):
    return (term.replace("HALLMARK_", "").replace("REACTOME_", "R:").replace("KEGG_", "K:")
                .replace("_SIGNALING", "").replace("SIGNALING", ""))


def load_grr(rd):
    """signed projected gene-gene GRN G_rr (diagonal zeroed) + gene order."""
    df = pd.read_csv(f"{rd}/GRN_inferred.csv", index_col=0)
    G = df.values.astype(np.float64).copy()
    np.fill_diagonal(G, 0.0)
    return G, list(map(str, df.index))


def load_grp(rd):
    """signed gene->protein translation map (Transcription_RNA_to_Prot.csv)."""
    df = pd.read_csv(f"{rd}/Transcription_RNA_to_Prot.csv", index_col=0)
    return df.values.astype(np.float64), list(map(str, df.index)), list(map(str, df.columns))


def cognate_pairs(genes, prots):
    """(gene_idx, prot_idx) central-dogma pairs via normalized-name match (same rule as cognate_nulls_csv.py)."""
    def norm(s): return str(s).upper().replace("-", "").replace("_", "").replace(".", "")
    gup = {norm(g): i for i, g in enumerate(genes)}
    cog = []
    for j, pr in enumerate(prots):
        prn = norm(pr)
        for cand in [prn, prn.replace("CD", ""), prn.split("/")[0]]:
            if cand in gup:
                cog.append((gup[cand], j)); break
    return cog


def overlap_enrich(top_genes, annot_set, background):
    bg = set(background); ann = set(annot_set) & bg; top = set(top_genes) & bg
    N, K, n = len(bg), len(ann), len(top); k = len(top & ann)
    exp = n * K / N if N else 0.0
    fold = (k / exp) if exp > 0 else float("nan")
    p = float(ss.hypergeom.sf(k - 1, N, K, n)) if n and K else float("nan")
    return dict(n_top=n, n_hits=k, expected=round(float(exp), 2), fold=round(float(fold), 3),
                pval=p, hits=sorted(top & ann))


# ---------------------------------------------------------------- A. GSEA program comparison
def gsea_program_table():
    if not (os.path.exists(GSEA_H) and os.path.exists(GSEA_D)):
        print(f"WARN: GSEA jsons not found ({GSEA_H} / {GSEA_D}); skipping program comparison.")
        return None

    def summarize(path):
        d = json.load(open(path))["latent_pathway"]  # {j: [term, nes, fdr]}
        prog = {}
        for j, (term, nes, fdr) in d.items():
            if fdr < 0.1 and term not in ("(none)",):
                e = prog.setdefault(term, {"n_latents": 0, "nes": []})
                e["n_latents"] += 1; e["nes"].append(float(nes))
        for term, e in prog.items():
            arr = np.array(e["nes"]); e["best_nes"] = float(arr[np.argmax(np.abs(arr))]); e["mean_nes"] = float(arr.mean())
            del e["nes"]
        n_sig = sum(1 for v in d.values() if v[2] < 0.1 and v[0] != "(none)")
        return prog, n_sig, len(d)

    progH, nsigH, nlat = summarize(GSEA_H)
    progD, nsigD, _ = summarize(GSEA_D)
    terms = sorted(set(progH) | set(progD))
    table = {}
    for t in terms:
        h = progH.get(t, {}); dd = progD.get(t, {})
        table[t] = {
            "inflammatory": is_inflam_prog(t),
            "healthy_n_latents": h.get("n_latents", 0), "disease_n_latents": dd.get("n_latents", 0),
            "healthy_best_nes": round(h.get("best_nes", 0.0), 3), "disease_best_nes": round(dd.get("best_nes", 0.0), 3),
            "delta_n_latents": dd.get("n_latents", 0) - h.get("n_latents", 0),
        }
    gained = sorted([t for t in terms if table[t]["delta_n_latents"] > 0],
                    key=lambda t: -table[t]["delta_n_latents"])
    lost = sorted([t for t in terms if table[t]["delta_n_latents"] < 0],
                  key=lambda t: table[t]["delta_n_latents"])
    inflam_gain = sum(table[t]["delta_n_latents"] for t in terms if table[t]["inflammatory"])
    return {"n_sig_latents": {"healthy": nsigH, "disease": nsigD}, "n_latents_total": nlat,
            "table": table, "programs_gained_in_disease": gained, "programs_lost_in_disease": lost,
            "net_inflammatory_latent_gain_disease": int(inflam_gain)}


# ---------------------------------------------------------------- B/C. GRN comparison
def grn_comparison(GH, GD, genes):
    n = len(genes)
    off = ~np.eye(n, dtype=bool)
    eH, eD = GH[off], GD[off]
    aH, aD = np.abs(eH), np.abs(eD)
    edge = {
        "n_genes": n, "n_offdiag": int(off.sum()),
        "signed_edge_pearson": float(np.corrcoef(eH, eD)[0, 1]),
        "signed_edge_spearman": float(ss.spearmanr(eH, eD).correlation),
        "abs_edge_pearson": float(np.corrcoef(aH, aD)[0, 1]),
        "abs_edge_spearman": float(ss.spearmanr(aH, aD).correlation),
    }
    aGH, aGD = np.abs(GH), np.abs(GD)
    dH = aGH.sum(1) + aGH.sum(0); dD = aGD.sum(1) + aGD.sum(0)
    edge["per_gene_degree_pearson"] = float(np.corrcoef(dH, dD)[0, 1])
    edge["per_gene_degree_spearman"] = float(ss.spearmanr(dH, dD).correlation)
    thH, thD = np.quantile(aH, 0.99), np.quantile(aD, 0.99)
    topH, topD = aH >= thH, aD >= thD
    edge["top1pct_edge_jaccard"] = float((topH & topD).sum() / (topH | topD).sum())
    return edge, dH, dD, aGH, aGD


def differential(genes, dH, dD, aGH, aGD):
    zH = (dH - dH.mean()) / (dH.std() + 1e-12)
    zD = (dD - dD.mean()) / (dD.std() + 1e-12)
    dz = zD - zH
    order = np.argsort(-dz)
    Ntop = 50
    dis_gained = [genes[i] for i in order[:Ntop]]
    heal_gained = [genes[i] for i in order[::-1][:Ntop]]
    top_dis = [{"gene": genes[i], "delta_z": round(float(dz[i]), 3), "z_healthy": round(float(zH[i]), 3),
                "z_disease": round(float(zD[i]), 3), "is_inflammatory": genes[i] in set(INFLAM)}
               for i in order[:25]]
    top_heal = [{"gene": genes[i], "delta_z": round(float(dz[i]), 3), "z_healthy": round(float(zH[i]), 3),
                 "z_disease": round(float(zD[i]), 3), "is_inflammatory": genes[i] in set(INFLAM)}
                for i in order[::-1][:25]]

    # top differential EDGES on scale-normalized |G| (divide each by its mean off-diag |edge|)
    off = ~np.eye(len(genes), dtype=bool)
    sH = aGH / (aGH[off].mean() + 1e-12); sD = aGD / (aGD[off].mean() + 1e-12)
    dE = sD - sH; np.fill_diagonal(dE, 0.0)
    gi = np.array(genes)
    infset = set(INFLAM)

    def top_edges(M, k=20, sign=1):
        flat = (sign * M).ravel(); idx = np.argpartition(-flat, k)[:k]
        idx = idx[np.argsort(-flat[idx])]
        out = []
        for f in idx:
            i, j = divmod(int(f), M.shape[1])
            if i == j:
                continue
            out.append({"src": gi[i], "dst": gi[j], "delta": round(float(sign * flat[f]), 3),
                        "src_inflam": gi[i] in infset, "dst_inflam": gi[j] in infset})
        return out[:15]
    dis_edges = top_edges(dE, k=40, sign=1)
    heal_edges = top_edges(dE, k=40, sign=-1)

    enr = {
        "disease_gained_inflammatory": overlap_enrich(dis_gained, INFLAM, genes),
        "healthy_gained_inflammatory": overlap_enrich(heal_gained, INFLAM, genes),
        "disease_gained_ifn_only": overlap_enrich(dis_gained, IFN_ONLY, genes),
        "healthy_gained_ifn_only": overlap_enrich(heal_gained, IFN_ONLY, genes),
    }

    # inflammatory sub-network centrality: mean |degree| of inflammatory genes vs background
    infmask = np.array([g in infset for g in genes])
    def subnet(d):
        return {"inflam_mean_degree": float(d[infmask].mean()), "bg_mean_degree": float(d[~infmask].mean()),
                "fold_inflam_over_bg": float(d[infmask].mean() / (d[~infmask].mean() + 1e-12))}
    aH_deg = aGH.sum(1) + aGH.sum(0); aD_deg = aGD.sum(1) + aGD.sum(0)
    inflam_central = {"healthy": subnet(aH_deg), "disease": subnet(aD_deg),
                      "n_inflam_in_vocab": int(infmask.sum())}
    # within-inflammatory-subgraph total connectivity
    ii = np.where(infmask)[0]
    subH = aGH[np.ix_(ii, ii)]; subD = aGD[np.ix_(ii, ii)]
    inflam_central["within_subgraph_mean_abs_edge"] = {
        "healthy": float(subH[~np.eye(len(ii), dtype=bool)].mean()),
        "disease": float(subD[~np.eye(len(ii), dtype=bool)].mean())}

    return {"n_top": Ntop, "top25_disease_gained_hubs": top_dis, "top25_healthy_gained_hubs": top_heal,
            "top_disease_gained_edges": dis_edges, "top_healthy_gained_edges": heal_edges,
            "enrichment": enr, "inflammatory_centrality": inflam_central}, zH, zD, dz


# ---------------------------------------------------------------- D. cognate G_rp stability
def cognate_stability(rpH, rpD, genes, prots):
    fullH, fullD = rpH.ravel(), rpD.ravel()
    out = {
        "full_map_pearson": float(np.corrcoef(fullH, fullD)[0, 1]),
        "full_map_spearman": float(ss.spearmanr(fullH, fullD).correlation),
        "abs_full_map_pearson": float(np.corrcoef(np.abs(fullH), np.abs(fullD))[0, 1]),
    }
    cog = cognate_pairs(genes, prots)
    out["n_cognate_pairs"] = len(cog)
    VH, VD = np.abs(rpH), np.abs(rpD)

    def cognate_ranks(V):
        ranks = []
        for g, p in cog:
            col = V[:, p]
            r = int((col > col[g]).sum()) + 1  # 1 = strongest cognate gene in that protein column
            ranks.append(r)
        return np.array(ranks)
    rH, rD = cognate_ranks(VH), cognate_ranks(VD)
    ncog = len(cog)
    for tag, r in [("healthy", rH), ("disease", rD)]:
        out[tag] = {"median_cognate_rank": float(np.median(r)),
                    "frac_cognate_top1": float((r <= 1).mean()),
                    "frac_cognate_top5": float((r <= 5).mean()),
                    "frac_cognate_top10": float((r <= 10).mean())}
    cvH = np.array([VH[g, p] for g, p in cog]); cvD = np.array([VD[g, p] for g, p in cog])
    out["cognate_value_pearson"] = float(np.corrcoef(cvH, cvD)[0, 1])
    out["cognate_value_spearman"] = float(ss.spearmanr(cvH, cvD).correlation)
    out["_cog_idx"] = cog  # for figure (dropped before json dump)
    out["_cvH"] = cvH; out["_cvD"] = cvD
    return out


def main():
    print("Loading G_rr (GRN_inferred.csv) for both models ...", flush=True)
    GH, gH = load_grr(HEALTHY)
    GD, gD = load_grr(DISEASE)
    assert gH == gD, "vocab mismatch between healthy/disease GRN_inferred.csv (must be node-comparable!)"
    genes = gH
    print(f"  shared vocab genes = {len(genes)} (node-comparable, no intersection needed)", flush=True)

    A = gsea_program_table()
    edge, dH, dD, aGH, aGD = grn_comparison(GH, GD, genes)
    C, zH, zD, dz = differential(genes, dH, dD, aGH, aGD)

    print("Loading G_rp (Transcription_RNA_to_Prot.csv) ...", flush=True)
    rpH, gpH, ppH = load_grp(HEALTHY)
    rpD, gpD, ppD = load_grp(DISEASE)
    assert gpH == gpD and ppH == ppD
    D = cognate_stability(rpH, rpD, gpH, ppH)

    summary = {
        "models": {"healthy": HEALTHY, "disease": DISEASE},
        "n_genes": len(genes), "n_prot": len(ppH),
        "A_gsea_programs": A,
        "B_grn_similarity": edge,
        "C_differential_connectivity": C,
        "D_cognate_grp_stability": {k: v for k, v in D.items() if not k.startswith("_")},
    }
    json.dump(summary, open(OUT_JSON, "w"), indent=2, default=str)
    print("WROTE", OUT_JSON, flush=True)

    np.savez(f"{_CMC_RESULTS}/_hd_figdata.npz",
             genes=np.array(genes), zH=zH, zD=zD, dz=dz, dH=dH, dD=dD,
             inflam=np.array(INFLAM),
             cvH=D["_cvH"], cvD=D["_cvD"])

    # ---- console report ----
    print("\n=== A. GSEA immune programs (# latents at FDR<0.1) ===")
    if A:
        print(f'sig latents: healthy={A["n_sig_latents"]["healthy"]}  disease={A["n_sig_latents"]["disease"]}  (of {A["n_latents_total"]})')
        print(f'{"program":52s} {"H":>3s} {"D":>3s} {"dNES_H":>7s} {"dNES_D":>7s}')
        for t, r in sorted(A["table"].items(), key=lambda kv: -(kv[1]["disease_n_latents"] + kv[1]["healthy_n_latents"])):
            star = "*" if r["inflammatory"] else " "
            print(f'{star}{short_prog(t)[:50]:51s} {r["healthy_n_latents"]:3d} {r["disease_n_latents"]:3d} '
                  f'{r["healthy_best_nes"]:7.2f} {r["disease_best_nes"]:7.2f}')
        print(f'net inflammatory latent gain in disease = {A["net_inflammatory_latent_gain_disease"]:+d}')
        print("gained in disease:", ", ".join(short_prog(t) for t in A["programs_gained_in_disease"][:8]))
        print("lost in disease:  ", ", ".join(short_prog(t) for t in A["programs_lost_in_disease"][:8]))
    print("\n=== B. G_rr similarity (shared vocab) ===")
    print(f'signed edge r={edge["signed_edge_pearson"]:.3f} rho={edge["signed_edge_spearman"]:.3f} | '
          f'|edge| r={edge["abs_edge_pearson"]:.3f} | degree rho={edge["per_gene_degree_spearman"]:.3f} | '
          f'top-1% edge Jaccard={edge["top1pct_edge_jaccard"]:.3f}')
    print("\n=== C. Differential connectivity ===")
    print("disease-gained hubs:", ", ".join(f'{h["gene"]}{"*" if h["is_inflammatory"] else ""}' for h in C["top25_disease_gained_hubs"][:12]))
    print("healthy-gained hubs:", ", ".join(f'{h["gene"]}{"*" if h["is_inflammatory"] else ""}' for h in C["top25_healthy_gained_hubs"][:12]))
    ed = C["enrichment"]["disease_gained_inflammatory"]; ei = C["enrichment"]["disease_gained_ifn_only"]
    print(f'disease-gained INFLAM fold={ed["fold"]:.2f} p={ed["pval"]:.1e} ({ed["n_hits"]}/{ed["n_top"]}); '
          f'IFN-only fold={ei["fold"]:.2f} p={ei["pval"]:.1e}')
    ic = C["inflammatory_centrality"]
    print(f'inflam-degree fold-over-bg: healthy={ic["healthy"]["fold_inflam_over_bg"]:.2f} '
          f'disease={ic["disease"]["fold_inflam_over_bg"]:.2f}')
    print("top disease-gained edges:", ", ".join(f'{e["src"]}->{e["dst"]}' for e in C["top_disease_gained_edges"][:8]))
    print("\n=== D. Cognate G_rp stability ===")
    print(f'full map r={D["full_map_pearson"]:.3f} | cognate value r={D["cognate_value_pearson"]:.3f} | '
          f'cognate top-1 frac H={D["healthy"]["frac_cognate_top1"]:.2f} D={D["disease"]["frac_cognate_top1"]:.2f} '
          f'(n_cognate={D["n_cognate_pairs"]})')
    return summary


def make_figure():
    sys.path.insert(0, os.path.join(_CMC_ROOT, "analysis", "figures", "original"))
    from style import plt, C, panel_label, save

    S = json.load(open(OUT_JSON))
    fd = np.load(f"{_CMC_RESULTS}/_hd_figdata.npz", allow_pickle=True)
    genes = list(map(str, fd["genes"])); inflam = set(map(str, fd["inflam"]))
    zH, zD, dz, dH, dD = fd["zH"], fd["zD"], fd["dz"], fd["dH"], fd["dD"]
    cvH, cvD = fd["cvH"], fd["cvD"]
    CH, CD, CINF = C["cmc"], C["neg"], C["accent"]   # healthy=blue disease=red inflam=purple

    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.6))
    axA, axB, axC, axD = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # ---- A: GSEA program bars ----
    A = S["A_gsea_programs"]
    if A:
        tab = A["table"]
        prog = sorted(tab, key=lambda t: -(tab[t]["disease_n_latents"] + tab[t]["healthy_n_latents"]))[:9]
        prog = prog[::-1]
        y = np.arange(len(prog)); h = 0.4
        axA.barh(y + h/2, [tab[t]["healthy_n_latents"] for t in prog], h, color=CH, label="healthy")
        axA.barh(y - h/2, [tab[t]["disease_n_latents"] for t in prog], h, color=CD, label="disease")
        axA.set_yticks(y)
        axA.set_yticklabels([short_prog(t)[:26] for t in prog], fontsize=5.6)
        for tick, t in zip(axA.get_yticklabels(), prog):
            if tab[t]["inflammatory"]:
                tick.set_color(CINF); tick.set_fontweight("bold")
        axA.set_xlabel("# latent programs (FDR<0.1)")
        axA.set_title("Enriched immune programs\n(inflammatory in purple)")
        axA.legend(loc="lower right", fontsize=6.5); axA.tick_params(axis="y", length=0)
    else:
        axA.text(0.5, 0.5, "GSEA jsons\nnot found", ha="center", va="center", transform=axA.transAxes)
    panel_label(axA, "A")

    # ---- B: per-gene degree scatter ----
    eps = min(dH[dH > 0].min(), dD[dD > 0].min()) * 0.5
    xg, yg = np.log10(dH + eps), np.log10(dD + eps)
    is_inf = np.array([g in inflam for g in genes])
    axB.scatter(xg[~is_inf], yg[~is_inf], s=5, c=C["shuf"], alpha=0.5, linewidths=0, rasterized=True)
    axB.scatter(xg[is_inf], yg[is_inf], s=13, c=CINF, alpha=0.9, linewidths=0, label="inflammatory/IFN", rasterized=True)
    lo, hi = min(xg.min(), yg.min()), max(xg.max(), yg.max())
    axB.plot([lo, hi], [lo, hi], ls="--", c=C["muted"], lw=0.8)
    b = S["B_grn_similarity"]
    axB.text(0.03, 0.97, f'degree $\\rho$={b["per_gene_degree_spearman"]:.2f}\n'
                         f'edge $r$={b["signed_edge_pearson"]:.2f}, $\\rho$={b["signed_edge_spearman"]:.2f}\n'
                         f'top-1% edge J={b["top1pct_edge_jaccard"]:.2f}',
             transform=axB.transAxes, va="top", ha="left", fontsize=6.3,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["muted"], lw=0.6))
    axB.set_xlabel("healthy log$_{10}$ degree"); axB.set_ylabel("disease log$_{10}$ degree")
    axB.set_title(f'G$_{{rr}}$ similarity (n={S["n_genes"]} genes)')
    axB.legend(loc="lower right", fontsize=6.0, markerscale=1.3)
    panel_label(axB, "B")

    # ---- C: differential hubs diverging bars ----
    o = np.argsort(-dz)
    disg, healg = o[:9], o[::-1][:9]
    sel = list(healg[::-1]) + list(disg[::-1])
    y = np.arange(len(sel)); vals = dz[sel]
    cols = [CD if v > 0 else CH for v in vals]
    axC.barh(y, vals, color=cols)
    axC.set_yticks(y); axC.set_yticklabels([genes[i] for i in sel], fontsize=5.8)
    for tick, i in zip(axC.get_yticklabels(), sel):
        if genes[i] in inflam:
            tick.set_color(CINF); tick.set_fontweight("bold")
    axC.axvline(0, c=C["ink"], lw=0.7); axC.tick_params(axis="y", length=0)
    ed = S["C_differential_connectivity"]["enrichment"]["disease_gained_inflammatory"]
    axC.set_xlabel("$\\Delta$ connectivity (z$_{disease}$-z$_{healthy}$)")
    axC.set_title("Differential hub genes")
    axC.text(0.97, 0.5, f'disease-gained\ninflammatory\nfold={ed["fold"]:.1f}\np={ed["pval"]:.0e}',
             transform=axC.transAxes, va="center", ha="right", fontsize=6.2, color=CINF,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CINF, lw=0.6))
    xabs = np.abs(vals).max() * 1.05; axC.set_xlim(-xabs, xabs)
    panel_label(axC, "C")

    # ---- D: cognate G_rp stability scatter ----
    d = S["D_cognate_grp_stability"]
    axD.scatter(cvH, cvD, s=14, c=C["pos"], alpha=0.8, linewidths=0)
    lo, hi = 0, max(cvH.max(), cvD.max()) * 1.05
    axD.plot([lo, hi], [lo, hi], ls="--", c=C["muted"], lw=0.8)
    axD.set_xlim(lo, hi); axD.set_ylim(lo, hi)
    axD.text(0.03, 0.97, f'cognate |G$_{{rp}}$| $r$={d["cognate_value_pearson"]:.2f}\n'
                         f'full map $r$={d["full_map_pearson"]:.2f}\n'
                         f'cognate top-1: H={d["healthy"]["frac_cognate_top1"]:.2f} '
                         f'D={d["disease"]["frac_cognate_top1"]:.2f}',
             transform=axD.transAxes, va="top", ha="left", fontsize=6.3,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["muted"], lw=0.6))
    axD.set_xlabel("healthy cognate |G$_{rp}$|"); axD.set_ylabel("disease cognate |G$_{rp}$|")
    axD.set_title(f'Central-dogma map stability\n({d["n_cognate_pairs"]} cognate gene->protein pairs)')
    panel_label(axD, "D")

    fig.tight_layout(w_pad=1.6, h_pad=1.6)
    save(fig, OUT_FIG)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--figure":
        make_figure()
    else:
        main(); make_figure()
