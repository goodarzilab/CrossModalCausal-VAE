"""Compare the projected gene-gene GRNs (G_rr = D_r . A_rr . D_r^T) learned by two CMC-VAE models
trained on different cell populations:
  HEALTHY  results/multi_healthy_nodag   (389,706 healthy PBMCs, vocab 2730)
  DISEASE  results/multi_disease_nodag   (1,766,870 disease PBMCs, vocab 2592)

G_rr is computed from the model exactly as eval/bio_recovery_projected.py / pathway_recovery.py:
load CrossModalCausalLDVAE, A_rr = scm.get_block_adjacencies(tau=0.5), Dr = _dec_r_w(),
G = Dr @ A_rr @ Dr.T (verified to equal each dir's GRN_inferred.csv to 1e-11). The two models have
DIFFERENT vocabularies -> every cross-model comparison is done on the gene-set INTERSECTION.

Analyses (writes results/healthy_vs_disease_grn.json and paper/figures/fig_healthy_vs_disease.pdf/.png):
  1. Inflammatory character / pathway recovery (9 immune pathways, within-pathway |G_rr| AUROC).
  2. Hub analysis (per-gene weighted degree; TF overlap via CollecTRI; inflammatory enrichment; hub Jaccard).
  3. Graph similarity (edge-weight Pearson/Spearman, top-1% edge Jaccard, per-gene degree Spearman).
  4. Differential connectivity (disease-gained vs healthy-gained hubs; interferon/inflammatory enrichment).
  5. Recovery sanity (DoRothEA + CollecTRI AUROC vs ref_nets_v4_2c).

Usage: python analysis/fig2_networks/healthy_vs_disease_grn.py
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
import numpy as np, torch, pandas as pd
import scipy.stats as ss
from sklearn.metrics import roc_auc_score

R = f"{_CMC_ROOT}"
sys.path.insert(0, os.path.join(R, "src"))
from model import CrossModalCausalLDVAE

HEALTHY = f"{_CMC_RESULTS}/multi_healthy_nodag"
DISEASE = f"{_CMC_RESULTS}/multi_disease_nodag"
REF = f"{_CMC_RESULTS}/ref_nets_v4_2c"

# 9 immune pathways in canonical display order
PATHWAY_ORDER = ["MHC-I complex (control)", "TCR signaling", "TNF-alpha / NF-kB signaling",
                 "IFN-alpha (Type-I) response", "IFN-gamma (Type-II) response", "Apoptosis",
                 "Antigen presentation (MHC I & II)", "cGAS-STING", "JAK-STAT signaling"]
# the inflammatory/interferon subset that defines "is disease more inflammatory"
INFLAM_PATHWAYS = ["IFN-alpha (Type-I) response", "IFN-gamma (Type-II) response",
                   "TNF-alpha / NF-kB signaling", "cGAS-STING"]

# ------------------------------------------------------------------ inflammatory / interferon gene set
# union of the curated IFN-alpha, IFN-gamma, TNF/NF-kB, cGAS-STING member lists (from pathway_recovery.py)
# plus the explicit ISG / complement genes called out in the task.
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
# tighter interferon-only (ISG) set, to separate "interferon" from generic inflammation
IFN_ONLY = sorted(set(_IFN_A) | set(_IFN_G) | set(_STING))


def load_absG(rd):
    """|G_rr| with diagonal zeroed + gene list, computed from the model (tau=0.5, eval-mode gate)."""
    c = json.load(open(f"{rd}/config.json"))
    genes = list(pd.read_csv(f"{rd}/GRN_inferred.csv", index_col=0, nrows=0).columns)  # == var_names order
    sd = torch.load(f"{rd}/final_model.pt", map_location="cpu")
    ng, npr = sd["dec_rna.weight"].shape[0], sd["dec_prot.weight"].shape[0]
    assert ng == len(genes), f"gene/dim mismatch {ng} vs {len(genes)}"
    m = CrossModalCausalLDVAE(ng, npr, 1, latent_dim_rna=c["latent_dim_rna"],
                              latent_dim_prot=c["latent_dim_prot"], recon_likelihood="mse",
                              scm_nonlinear=c.get("scm_nonlinear", False), phi=c.get("phi", "tanh"))
    m.load_state_dict(sd, strict=False); m.eval()
    with torch.no_grad():
        A_rr, _, _, _ = m.scm.get_block_adjacencies(tau=0.5)
        Dr = m._dec_r_w().detach().float()
        G = (Dr @ A_rr.float() @ Dr.T).cpu().numpy()
    np.fill_diagonal(G, 0.0)
    return np.abs(G), genes


def degrees(absG):
    """weighted out/in/total degree per gene from |G_rr| (diag already 0)."""
    out = absG.sum(1); inn = absG.sum(0); return out, inn, out + inn


def get_collectri_tfs():
    try:
        import decoupler as dc
        ct = dc.op.collectri(organism="human")
        return set(ct["source"].astype(str)), "decoupler dc.op.collectri"
    except Exception as e:
        z = np.load(f"{REF}.npz", allow_pickle=True)
        rg = list(map(str, z["genes"])); mask = z["CollecTRI"].astype(bool)
        return set(np.array(rg)[mask.any(1)]), f"ref_nets CollecTRI sources (collectri fetch failed: {str(e)[:50]})"


def overlap_enrich(top_genes, annot_set, background):
    """fold enrichment + hypergeometric survival p for annot_set membership among top_genes vs background."""
    bg = set(background); ann = set(annot_set) & bg; top = set(top_genes) & bg
    N, K, n = len(bg), len(ann), len(top)
    k = len(top & ann)
    exp = n * K / N if N else 0.0
    fold = (k / exp) if exp > 0 else float("nan")
    p = float(ss.hypergeom.sf(k - 1, N, K, n)) if n and K else float("nan")
    return dict(n_top=n, n_hits=k, expected=float(exp), fold=float(fold), pval=p,
                hits=sorted(top & ann))


def recovery_auroc(absG, genes):
    """directed CollecTRI + DoRothEA AUROC vs ref_nets_v4_2c on the model-vocab intersection."""
    z = np.load(f"{REF}.npz", allow_pickle=True)
    rg = list(map(str, z["genes"])); ridx = {g: i for i, g in enumerate(rg)}
    gi = {g: i for i, g in enumerate(genes)}
    common = [g for g in rg if g in gi]
    mi = [gi[g] for g in common]; ri = [ridx[g] for g in common]
    Gs = absG[np.ix_(mi, mi)].copy(); np.fill_diagonal(Gs, 0.0)
    off = ~np.eye(len(common), dtype=bool); s = Gs[off]
    out = {"n_common": len(common)}
    for name in ["CollecTRI", "DoRothEA"]:
        lab = z[name].astype(bool)[np.ix_(ri, ri)][off]
        out[name] = float(roc_auc_score(lab, s)) if 0 < lab.sum() < len(lab) else float("nan")
    return out


def main():
    print("Loading models & projecting G_rr ...", flush=True)
    absH, gH = load_absG(HEALTHY)
    absD, gD = load_absG(DISEASE)
    setH, setD = set(gH), set(gD)
    shared = sorted(setH & setD)
    iH = {g: i for i, g in enumerate(gH)}; iD = {g: i for i, g in enumerate(gD)}
    print(f"  healthy vocab={len(gH)}  disease vocab={len(gD)}  shared={len(shared)}", flush=True)

    tfs, tf_src = get_collectri_tfs()
    inflam_all = sorted(set(INFLAM))
    summary = {"models": {"healthy": {"dir": HEALTHY, "n_genes": len(gH)},
                          "disease": {"dir": DISEASE, "n_genes": len(gD)}},
               "n_shared_genes": len(shared), "tf_source": tf_src, "n_collectri_tfs": len(tfs),
               "inflammatory_geneset_size": len(inflam_all)}

    # ---------------------------------------------------------------- 1. pathway recovery
    pjH = json.load(open(f"{HEALTHY}/pathway_recovery.json"))["pathways"]
    pjD = json.load(open(f"{DISEASE}/pathway_recovery.json"))["pathways"]
    ptab = {}
    for nm in PATHWAY_ORDER:
        ptab[nm] = {"healthy_auroc": pjH.get(nm, {}).get("auroc"),
                    "disease_auroc": pjD.get(nm, {}).get("auroc"),
                    "healthy_fold": pjH.get(nm, {}).get("fold_enrichment"),
                    "disease_fold": pjD.get(nm, {}).get("fold_enrichment"),
                    "delta_auroc": (pjD.get(nm, {}).get("auroc", np.nan) - pjH.get(nm, {}).get("auroc", np.nan))}
    inflam_delta = float(np.nanmean([ptab[n]["delta_auroc"] for n in INFLAM_PATHWAYS]))
    noninflam = [n for n in PATHWAY_ORDER if n not in INFLAM_PATHWAYS]
    noninflam_delta = float(np.nanmean([ptab[n]["delta_auroc"] for n in noninflam]))
    summary["1_pathway_recovery"] = {"table": ptab,
        "inflammatory_pathways": INFLAM_PATHWAYS,
        "mean_delta_inflammatory_disease_minus_healthy": inflam_delta,
        "mean_delta_noninflammatory": noninflam_delta}

    # ---------------------------------------------------------------- 2. hub analysis
    outH, innH, totH = degrees(absH)
    outD, innD, totD = degrees(absD)
    def hub_table(genes, out, inn, tot, k=20):
        order = np.argsort(-tot)[:k]
        return [{"gene": genes[i], "total_degree": float(tot[i]), "out_degree": float(out[i]),
                 "in_degree": float(inn[i]), "is_TF": genes[i] in tfs,
                 "is_inflammatory": genes[i] in set(inflam_all)} for i in order]
    hubH = hub_table(gH, outH, innH, totH)
    hubD = hub_table(gD, outD, innD, totD)
    top50H = set(g for g in np.array(gH)[np.argsort(-totH)[:50]])
    top50D = set(g for g in np.array(gD)[np.argsort(-totD)[:50]])
    hub_jacc = len(top50H & top50D) / len(top50H | top50D)
    def frac_tf_inflam(hublist):
        tf = np.mean([h["is_TF"] for h in hublist]); inf = np.mean([h["is_inflammatory"] for h in hublist])
        return float(tf), float(inf)
    tfH, infH = frac_tf_inflam(hubH); tfD, infD = frac_tf_inflam(hubD)
    # enrichment of inflammatory genes among top-50 hubs vs each model's own vocab background
    enrH_top = overlap_enrich(top50H, inflam_all, gH)
    enrD_top = overlap_enrich(top50D, inflam_all, gD)
    summary["2_hub_analysis"] = {
        "top20_hubs_healthy": hubH, "top20_hubs_disease": hubD,
        "frac_TF_top20": {"healthy": tfH, "disease": tfD},
        "frac_inflammatory_top20": {"healthy": infH, "disease": infD},
        "inflammatory_enrichment_top50_hubs": {"healthy": enrH_top, "disease": enrD_top},
        "hub_jaccard_top50_shared_union": float(hub_jacc),
        "top50_hubs_shared_both": sorted(top50H & top50D)}

    # ---------------------------------------------------------------- 3. graph similarity (shared genes)
    idxH = [iH[g] for g in shared]; idxD = [iD[g] for g in shared]
    sH = absH[np.ix_(idxH, idxH)].copy(); sD = absD[np.ix_(idxD, idxD)].copy()
    np.fill_diagonal(sH, 0.0); np.fill_diagonal(sD, 0.0)
    off = ~np.eye(len(shared), dtype=bool)
    eH = sH[off]; eD = sD[off]
    pear = float(np.corrcoef(eH, eD)[0, 1])
    rH = ss.rankdata(eH); rD = ss.rankdata(eD)
    spear = float(np.corrcoef(rH, rD)[0, 1])
    # top-1% directed-edge Jaccard
    thH = np.quantile(eH, 0.99); thD = np.quantile(eD, 0.99)
    topH = eH >= thH; topD = eD >= thD
    edge_jacc = float((topH & topD).sum() / (topH | topD).sum())
    # per-gene total-degree Spearman on shared submatrix
    dH = sH.sum(1) + sH.sum(0); dD = sD.sum(1) + sD.sum(0)
    deg_spear = float(ss.spearmanr(dH, dD).correlation)
    summary["3_graph_similarity"] = {
        "n_shared_genes": len(shared), "n_offdiag_pairs": int(off.sum()),
        "edge_pearson": pear, "edge_spearman": spear,
        "topk_edge_jaccard_k1pct": edge_jacc,
        "per_gene_total_degree_spearman": deg_spear}

    # ---------------------------------------------------------------- 4. differential connectivity
    zH = (dH - dH.mean()) / dH.std(); zD = (dD - dD.mean()) / dD.std()
    dz = zD - zH  # disease - healthy, on shared genes
    order = np.argsort(-dz)
    Ntop = 50
    dis_gained = [shared[i] for i in order[:Ntop]]
    heal_gained = [shared[i] for i in order[::-1][:Ntop]]
    enr_dis_inflam = overlap_enrich(dis_gained, inflam_all, shared)
    enr_heal_inflam = overlap_enrich(heal_gained, inflam_all, shared)
    enr_dis_ifn = overlap_enrich(dis_gained, IFN_ONLY, shared)
    enr_heal_ifn = overlap_enrich(heal_gained, IFN_ONLY, shared)
    top_dis_tbl = [{"gene": shared[i], "delta_z": float(dz[i]), "z_healthy": float(zH[i]),
                    "z_disease": float(zD[i]), "is_inflammatory": shared[i] in set(inflam_all),
                    "is_TF": shared[i] in tfs} for i in order[:25]]
    top_heal_tbl = [{"gene": shared[i], "delta_z": float(dz[i]), "z_healthy": float(zH[i]),
                     "z_disease": float(zD[i]), "is_inflammatory": shared[i] in set(inflam_all),
                     "is_TF": shared[i] in tfs} for i in order[::-1][:25]]
    summary["4_differential_connectivity"] = {
        "n_top": Ntop, "top25_disease_gained": top_dis_tbl, "top25_healthy_gained": top_heal_tbl,
        "inflammatory_enrichment": {"disease_gained": enr_dis_inflam, "healthy_gained": enr_heal_inflam},
        "interferon_only_enrichment": {"disease_gained": enr_dis_ifn, "healthy_gained": enr_heal_ifn}}

    # ---------------------------------------------------------------- 5. recovery sanity
    recH = recovery_auroc(absH, gH); recD = recovery_auroc(absD, gD)
    summary["5_recovery_sanity"] = {"ref": "ref_nets_v4_2c", "healthy": recH, "disease": recD}

    outp = f"{_CMC_RESULTS}/healthy_vs_disease_grn.json"
    json.dump(summary, open(outp, "w"), indent=2)
    print("WROTE", outp, flush=True)

    # ---- console report ----
    print("\n=== 1. PATHWAY AUROC (healthy vs disease) ===")
    print(f'{"pathway":36s} {"healthy":>8s} {"disease":>8s} {"delta":>7s}')
    for nm in PATHWAY_ORDER:
        t = ptab[nm]; star = "  *INFLAM" if nm in INFLAM_PATHWAYS else ""
        print(f'{nm:36s} {t["healthy_auroc"]:8.3f} {t["disease_auroc"]:8.3f} {t["delta_auroc"]:+7.3f}{star}')
    print(f'mean delta inflammatory = {inflam_delta:+.3f}   non-inflammatory = {noninflam_delta:+.3f}')
    print("\n=== 2. TOP-10 HUBS ===")
    print("HEALTHY:", ", ".join(f'{h["gene"]}{"*" if h["is_inflammatory"] else ""}' for h in hubH[:10]))
    print("DISEASE:", ", ".join(f'{h["gene"]}{"*" if h["is_inflammatory"] else ""}' for h in hubD[:10]))
    print(f'frac TF top20 H={tfH:.2f} D={tfD:.2f} | frac inflam top20 H={infH:.2f} D={infD:.2f} | hub Jaccard(top50)={hub_jacc:.3f}')
    print("\n=== 3. GRAPH SIMILARITY (shared genes) ===")
    print(f'edge Pearson={pear:.3f} Spearman={spear:.3f} | top-1% edge Jaccard={edge_jacc:.3f} | degree Spearman={deg_spear:.3f}')
    print("\n=== 4. DIFFERENTIAL CONNECTIVITY ===")
    print("disease-gained top10:", ", ".join(f'{t["gene"]}{"*" if t["is_inflammatory"] else ""}' for t in top_dis_tbl[:10]))
    print("healthy-gained top10:", ", ".join(f'{t["gene"]}{"*" if t["is_inflammatory"] else ""}' for t in top_heal_tbl[:10]))
    print(f'INFLAM enrich disease-gained fold={enr_dis_inflam["fold"]:.2f} p={enr_dis_inflam["pval"]:.1e} '
          f'({enr_dis_inflam["n_hits"]}/{enr_dis_inflam["n_top"]}) | healthy-gained fold={enr_heal_inflam["fold"]:.2f} p={enr_heal_inflam["pval"]:.1e}')
    print(f'IFN-only enrich disease-gained fold={enr_dis_ifn["fold"]:.2f} p={enr_dis_ifn["pval"]:.1e}')
    print("\n=== 5. RECOVERY (AUROC vs ref_nets_v4_2c) ===")
    print(f'HEALTHY CollecTRI={recH["CollecTRI"]:.3f} DoRothEA={recH["DoRothEA"]:.3f} (n={recH["n_common"]})')
    print(f'DISEASE CollecTRI={recD["CollecTRI"]:.3f} DoRothEA={recD["DoRothEA"]:.3f} (n={recD["n_common"]})')

    # ---- stash arrays the figure needs (avoid recompute) ----
    np.savez(f"{_CMC_RESULTS}/_hvd_figdata.npz",
             shared=np.array(shared), eH=eH.astype(np.float32), eD=eD.astype(np.float32),
             dH=dH, dD=dD, zH=zH, zD=zD, dz=dz,
             inflam=np.array(sorted(set(inflam_all))))
    return summary


def make_figure():
    """Four-panel comparison figure from results/healthy_vs_disease_grn.json + _hvd_figdata.npz."""
    sys.path.insert(0, os.path.join(_CMC_ROOT, "analysis", "figures", "original"))
    from style import plt, C, panel_label, save
    import matplotlib.ticker as mticker

    S = json.load(open(f"{_CMC_RESULTS}/healthy_vs_disease_grn.json"))
    fd = np.load(f"{_CMC_RESULTS}/_hvd_figdata.npz", allow_pickle=True)
    shared = list(map(str, fd["shared"])); inflam = set(map(str, fd["inflam"]))
    zH, zD, dz = fd["zH"], fd["zD"], fd["dz"]
    dH, dD = fd["dH"], fd["dD"]
    pjH = json.load(open(f"{HEALTHY}/pathway_recovery.json"))["pathways"]
    pjD = json.load(open(f"{DISEASE}/pathway_recovery.json"))["pathways"]

    CH, CD = C["cmc"], C["neg"]           # healthy = blue, disease = red
    CINF = C["accent"]                    # inflammatory highlight (purple)
    short = {"MHC-I complex (control)": "MHC-I\n(ctrl)", "TCR signaling": "TCR",
             "TNF-alpha / NF-kB signaling": "TNF/\nNF-κB", "IFN-alpha (Type-I) response": "IFN-α",
             "IFN-gamma (Type-II) response": "IFN-γ", "Apoptosis": "Apop.",
             "Antigen presentation (MHC I & II)": "Antigen\npres.", "cGAS-STING": "cGAS-\nSTING",
             "JAK-STAT signaling": "JAK-\nSTAT"}

    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.4))
    axA, axB, axC, axD = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # -------- A: pathway AUROC grouped bars --------
    order = PATHWAY_ORDER
    x = np.arange(len(order)); w = 0.40
    vH = [pjH.get(n, {}).get("auroc", np.nan) for n in order]
    vD = [pjD.get(n, {}).get("auroc", np.nan) for n in order]
    nD = [pjD.get(n, {}).get("n_members", 0) for n in order]
    axA.bar(x - w/2, vH, w, color=CH, label="healthy")
    axA.bar(x + w/2, vD, w, color=CD, label="disease")
    for xi, n in zip(x, nD):
        axA.text(xi, 0.455, f"n={n}", ha="center", va="bottom", fontsize=5.2, color=C["muted"])
    axA.axhline(0.5, ls="--", c=C["muted"], lw=0.8)
    axA.set_xticks(x); axA.set_xticklabels([short[n] for n in order], fontsize=6.2)
    for tick, nm in zip(axA.get_xticklabels(), order):   # inflammatory/IFN pathways -> purple bold
        if nm in INFLAM_PATHWAYS:
            tick.set_color(CINF); tick.set_fontweight("bold")
    axA.set_ylim(0.45, 1.0); axA.set_ylabel("within-pathway |G$_{rr}$| AUROC")
    axA.set_title("Immune-pathway recovery (inflammatory in purple)"); axA.legend(loc="upper right", ncol=2, fontsize=6.5)
    panel_label(axA, "A")

    # -------- B: top hubs paired horizontal bars (shared-gene z-degree) --------
    comb = np.minimum(zH, zD)                     # hubs strong in BOTH models = shared skeleton
    topb = np.argsort(-comb)[:15][::-1]           # ascending for barh
    y = np.arange(len(topb)); h = 0.40
    axB.barh(y + h/2, zH[topb], h, color=CH, label="healthy")
    axB.barh(y - h/2, zD[topb], h, color=CD, label="disease")
    axB.set_yticks(y)
    labels = [shared[i] for i in topb]
    axB.set_yticklabels(labels, fontsize=6.0)
    for tick, i in zip(axB.get_yticklabels(), topb):
        if shared[i] in inflam:
            tick.set_color(CINF); tick.set_fontweight("bold")
    axB.set_xlabel("connectivity (z-scored degree)")
    axB.set_title("Top shared hub genes"); axB.legend(loc="lower right", fontsize=6.5)
    axB.tick_params(axis="y", length=0)
    panel_label(axB, "B")

    # -------- C: per-gene degree scatter healthy vs disease --------
    eps = min(dH[dH > 0].min(), dD[dD > 0].min()) * 0.5
    xg = np.log10(dH + eps); yg = np.log10(dD + eps)
    is_inf = np.array([g in inflam for g in shared])
    axC.scatter(xg[~is_inf], yg[~is_inf], s=5, c=C["shuf"], alpha=0.5, linewidths=0, rasterized=True)
    axC.scatter(xg[is_inf], yg[is_inf], s=12, c=CINF, alpha=0.9, linewidths=0,
                label="inflammatory/IFN gene", rasterized=True)
    lo, hi = min(xg.min(), yg.min()), max(xg.max(), yg.max())
    axC.plot([lo, hi], [lo, hi], ls="--", c=C["muted"], lw=0.8)
    sim = S["3_graph_similarity"]
    axC.text(0.03, 0.97, f'degree $\\rho$={sim["per_gene_total_degree_spearman"]:.2f}\n'
                         f'edge $r$={sim["edge_pearson"]:.2f}, $\\rho$={sim["edge_spearman"]:.2f}\n'
                         f'top-1% edge J={sim["topk_edge_jaccard_k1pct"]:.2f}',
             transform=axC.transAxes, va="top", ha="left", fontsize=6.3,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["muted"], lw=0.6))
    axC.set_xlabel("healthy log$_{10}$ degree"); axC.set_ylabel("disease log$_{10}$ degree")
    axC.set_title(f"Graph similarity (n={len(shared)} shared genes)")
    axC.legend(loc="lower right", fontsize=6.0, markerscale=1.3)
    panel_label(axC, "C")

    # -------- D: differential connectivity diverging bars --------
    o = np.argsort(-dz)
    disg = o[:9]; healg = o[::-1][:9]
    sel = list(healg[::-1]) + list(disg[::-1])     # healthy(neg) bottom -> disease(pos) top
    y = np.arange(len(sel))
    vals = dz[sel]
    cols = [CD if v > 0 else CH for v in vals]
    axD.barh(y, vals, color=cols)
    axD.set_yticks(y); axD.set_yticklabels([shared[i] for i in sel], fontsize=5.8)
    for tick, i in zip(axD.get_yticklabels(), sel):
        if shared[i] in inflam:
            tick.set_color(CINF); tick.set_fontweight("bold")
    axD.axvline(0, c=C["ink"], lw=0.7)
    axD.tick_params(axis="y", length=0)
    diff = S["4_differential_connectivity"]["inflammatory_enrichment"]
    ed = diff["disease_gained"]
    axD.set_xlabel("$\\Delta$ connectivity  (z$_{disease}$ - z$_{healthy}$)")
    axD.set_title("Differential hubs")
    axD.text(0.97, 0.5, f'disease-gained\ninflammatory\nfold={ed["fold"]:.1f}\np={ed["pval"]:.0e}',
             transform=axD.transAxes, va="center", ha="right", fontsize=6.3, color=CINF,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CINF, lw=0.6))
    xabs = np.abs(vals).max() * 1.05; axD.set_xlim(-xabs, xabs)
    panel_label(axD, "D")

    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    outp = f"{R}/paper/figures/fig_healthy_vs_disease.pdf"
    save(fig, outp)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--figure":
        make_figure()
    else:
        main(); make_figure()
