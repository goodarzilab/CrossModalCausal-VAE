"""SIGNALING-PATHWAY-RECOVERY eval for the projected gene-gene GRN.

Question: does the model's projected gene-space network  G_rr = D_r A_rr D_r^T  (the SAME
projection used by causal_eval_suite.py) recover known SIGNALING-PATHWAY structure -- i.e. are
within-pathway gene pairs assigned larger |G_rr| edge weights than background gene pairs?

Pathways (restricted to the model's vocab, >=8 members kept): cGAS-STING, Type-I IFN / IFN-alpha
response, Type-II / IFN-gamma response, JAK-STAT signaling, TNF-alpha / NF-kB signaling, Antigen
presentation (MHC class I & II), TCR signaling, Apoptosis, and a tight MHC-I complex positive-control.

Gene sets come from MSigDB Hallmark / Reactome / KEGG via `decoupler` (cached to
eval/_pathway_genesets_cache.json so runs are deterministic & work offline). cGAS-STING and the
MHC-I positive control use CURATED canonical member lists (the Reactome STING set is too sparse in
this vocab). Per-pathway provenance is recorded in the "source" field.

Per pathway: positives = all within-pathway ordered gene pairs (i,j both in pathway, i!=j);
negatives = pairs where >=1 gene is outside. Reports AUROC, AUPRC, AUPRC-over-random, and
enrichment fold = median(|G_rr| within) / median(|G_rr| background) of |G_rr| separating the two.

Usage:
  python analysis/fig2_networks/pathway_recovery.py results/<run>        # -> results/<run>/pathway_recovery.json
  python analysis/fig2_networks/pathway_recovery.py --figure             # -> results/figures/slides/fig_pathway.png
                                                       #    (needs both cohort JSONs present)
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
import numpy as np, torch, scanpy as sc, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_CMC_ROOT, "src"))
RES = _CMC_RESULTS
CACHE = os.path.join(HERE, "_pathway_genesets_cache.json")

# the two cohorts wired into the deck figure
RUN_2C = "linfb_2c_fb10_b20_v4"
RUN_COMBAT = "linfb_2ccombat_fb10_b20_v4"

# ---------------------------------------------------------------- curated canonical member lists
# used when a database set is too sparse in-vocab (cGAS-STING) or as an offline fallback.
CURATED = {
    "cGAS-STING": ("MB21D1 CGAS TMEM173 STING1 TBK1 IKBKE IRF3 IRF7 IRF1 IRF9 NFKB1 NFKB2 RELA "
                   "RELB DDX58 IFIH1 MAVS TREX1 ENPP1 TRAF2 TRAF3 TRAF6 TANK TBKBP1 AZI2 NLRC3 "
                   "XRCC5 XRCC6 PRKDC IFI16 ZBP1 STAT1 STAT2 IFNB1 CXCL10 CCL5 ISG15 IFIT1 IFIT3 "
                   "OASL DDX41").split(),
    # tight peptide-loading / MHC-I complex -- positive control (co-regulated, biologically tight)
    "MHC-I complex (control)": ("HLA-A HLA-B HLA-C HLA-E HLA-F HLA-G B2M TAP1 TAP2 TAPBP TAPBPL "
                                "PSMB8 PSMB9 PSMB10 CALR CANX ERAP1 ERAP2 NLRC5").split(),
    # offline fallbacks for the database-sourced pathways (canonical cores)
    "IFN-alpha (Type-I) response": ("STAT1 STAT2 IRF1 IRF7 IRF9 ISG15 IFIT1 IFIT2 IFIT3 MX1 MX2 "
                                    "OAS1 OAS2 OAS3 OASL RSAD2 IFI6 IFI27 IFI35 IFI44 IFI44L "
                                    "IFITM1 IFITM2 IFITM3 USP18 BST2 XAF1 SAMD9 DDX58 IFIH1").split(),
    "IFN-gamma (Type-II) response": ("STAT1 IRF1 GBP1 GBP2 GBP4 GBP5 CXCL9 CXCL10 CXCL11 CIITA "
                                     "TAP1 TAP2 PSMB8 PSMB9 HLA-A HLA-B HLA-C HLA-DRA B2M NLRC5 "
                                     "SOCS1 SOCS3 JAK2 IRF8 IDO1 WARS UBE2L6 ISG20 VCAM1 IFI30").split(),
    "JAK-STAT signaling": ("JAK1 JAK2 JAK3 TYK2 STAT1 STAT2 STAT3 STAT4 STAT5A STAT5B STAT6 "
                           "SOCS1 SOCS2 SOCS3 CISH IL2RA IL2RB IL2RG IL4R IL6R IL7R IL10RA "
                           "IL10RB IFNGR1 IFNGR2 IFNAR1 IFNAR2 PIM1 BCL2L1 MYC").split(),
    "TNF-alpha / NF-kB signaling": ("TNF NFKB1 NFKB2 RELA RELB REL NFKBIA NFKBIE TNFAIP3 TNFAIP2 "
                                    "TNFAIP6 TRAF1 TRAF2 BIRC3 CXCL1 CXCL2 CXCL3 CCL2 CCL20 IL1B "
                                    "IL6 CXCL8 PTGS2 ICAM1 VCAM1 SOD2 JUN JUNB FOS FOSB NFKBIB").split(),
    "Antigen presentation (MHC I & II)": ("HLA-A HLA-B HLA-C HLA-E HLA-F HLA-G HLA-DRA HLA-DRB1 "
                                          "HLA-DPA1 HLA-DPB1 HLA-DQA1 HLA-DQB1 HLA-DMA HLA-DMB "
                                          "B2M TAP1 TAP2 TAPBP CALR CANX PDIA3 CD74 CTSS CTSL "
                                          "PSMB8 PSMB9 ERAP1 ERAP2 NLRC5 CIITA").split(),
    "TCR signaling": ("CD3D CD3E CD3G CD247 LCK ZAP70 LAT PLCG1 FYN ITK VAV1 GRAP2 PRKCQ "
                      "CARD11 BCL10 MALT1 NFATC1 NFATC2 CD4 CD8A PTPRC PDPK1 GRB2 SOS1 "
                      "MAPK1 MAPK3 NCK1 WAS PAK1 RHOA").split(),
    "Apoptosis": ("CASP3 CASP7 CASP8 CASP9 CASP6 CASP10 BAX BAK1 BID BAD BCL2 BCL2L1 MCL1 "
                  "BCL2L11 PMAIP1 BBC3 APAF1 CYCS FAS FASLG FADD TNFRSF10A TNFRSF10B DIABLO "
                  "XIAP BIRC2 BIRC3 CFLAR TP53 PARP1 CASP2").split(),
}

# ------------------------------------------------------------- pathway -> gene-set spec (in order)
#   mode="cache": union of one/more MSigDB(Hallmark/Reactome/KEGG) sets loaded from decoupler-cache
#   mode="curated": use CURATED[name] directly
#   each cache pathway also has a curated fallback (same key in CURATED) if the cache is missing.
PATHWAYS = [
    dict(name="cGAS-STING",                          mode="curated",
         source="curated canonical (Reactome STING too sparse in vocab)"),
    dict(name="IFN-alpha (Type-I) response",         mode="cache",
         keys=["HALLMARK_INTERFERON_ALPHA_RESPONSE"], source="MSigDB Hallmark"),
    dict(name="IFN-gamma (Type-II) response",        mode="cache",
         keys=["HALLMARK_INTERFERON_GAMMA_RESPONSE"], source="MSigDB Hallmark"),
    dict(name="JAK-STAT signaling",                  mode="cache",
         keys=["KEGG_JAK_STAT_SIGNALING_PATHWAY"],    source="MSigDB KEGG"),
    dict(name="TNF-alpha / NF-kB signaling",         mode="cache",
         keys=["HALLMARK_TNFA_SIGNALING_VIA_NFKB"],   source="MSigDB Hallmark"),
    dict(name="Antigen presentation (MHC I & II)",   mode="cache",
         keys=["REACTOME_CLASS_I_MHC_MEDIATED_ANTIGEN_PROCESSING_PRESENTATION",
               "REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION"], source="MSigDB Reactome"),
    dict(name="TCR signaling",                       mode="cache",
         keys=["REACTOME_TCR_SIGNALING"],             source="MSigDB Reactome"),
    dict(name="Apoptosis",                           mode="cache",
         keys=["HALLMARK_APOPTOSIS"],                 source="MSigDB Hallmark"),
    dict(name="MHC-I complex (control)",             mode="curated",
         source="curated canonical (positive control)"),
]
MIN_MEMBERS = 8


# ------------------------------------------------------------------------ gene-set loading helpers
def _load_cache():
    if os.path.exists(CACHE):
        try:
            return json.load(open(CACHE))
        except Exception:
            pass
    return None

def _fetch_cache():
    """Fetch the needed MSigDB/Hallmark sets via decoupler and cache them (interactive node only)."""
    import decoupler as dc
    need_msig = sorted({k for p in PATHWAYS if p["mode"] == "cache" for k in p.get("keys", [])
                        if not k.startswith("HALLMARK_")})
    out = {}
    if need_msig:
        m = dc.op.resource("MSigDB", organism="human"); m["geneset"] = m["geneset"].astype(str)
        sub = m[m["geneset"].isin(need_msig)]
        for gs in need_msig:
            out[gs] = sorted(set(sub.loc[sub["geneset"] == gs, "genesymbol"].astype(str)))
    H = dc.op.hallmark(organism="human")
    for p in PATHWAYS:
        for k in p.get("keys", []):
            if k.startswith("HALLMARK_"):
                hs = k[len("HALLMARK_"):]
                out[k] = sorted(set(H.loc[H["source"] == hs, "target"].astype(str)))
    json.dump(out, open(CACHE, "w"), indent=1)
    return out

def resolve_members(vocab_set):
    """Return list of (pathway_name, members_in_vocab, source) restricted to vocab & >=MIN_MEMBERS."""
    cache = _load_cache()
    if cache is None:
        try:
            cache = _fetch_cache()
        except Exception as e:
            print(f"[warn] decoupler fetch failed ({str(e)[:80]}); using curated fallbacks", flush=True)
            cache = {}
    resolved = []
    for p in PATHWAYS:
        name, src = p["name"], p["source"]
        if p["mode"] == "curated":
            members = set(CURATED[name])
        else:
            members = set()
            for k in p["keys"]:
                if cache.get(k):
                    members |= set(cache[k])
            if not members:                       # cache miss -> curated offline fallback
                members = set(CURATED.get(name, []))
                src = "curated fallback (decoupler unavailable)"
        inv = sorted(members & vocab_set)
        resolved.append((name, inv, src, len(members)))
    return resolved


# ------------------------------------------------------------------------------- model / G_rr load
def _ckpt(rd):
    """FINAL checkpoint by default (fully-regularized); fallback best. Override CMC_CKPT=best."""
    pref = os.environ.get("CMC_CKPT", "final")
    for n in [f"{pref}_model.pt", "final_model.pt", "best_model.pt"]:
        p = os.path.join(rd, n)
        if os.path.exists(p):
            return p
    return os.path.join(rd, "best_model.pt")

def load_Grr(rd):
    """Load model & compute |G_rr| (diagonal zeroed) + gene names -- SAME projection as causal_eval_suite."""
    from model import CrossModalCausalLDVAE
    cfg = json.load(open(f"{rd}/config.json"))
    ar = sc.read_h5ad(cfg["rna_path"], backed="r"); ap = sc.read_h5ad(cfg["prot_path"], backed="r")
    genes = list(ar.var_names.astype(str)); ng, npr = len(genes), ap.n_vars
    _dm = None
    if cfg.get("dec_mask_rna_path"):
        _mk = torch.load(cfg["dec_mask_rna_path"], map_location="cpu")
        _dm = _mk["mask"] if isinstance(_mk, dict) else _mk
    md = CrossModalCausalLDVAE(ng, npr, 1, latent_dim_rna=cfg["latent_dim_rna"],
        latent_dim_prot=cfg["latent_dim_prot"], recon_likelihood="mse",
        scm_nonlinear=cfg.get("scm_nonlinear", False), phi=cfg.get("phi", "tanh"),
        dec_mask_rna=_dm, dec_mask_lambda=cfg.get("dec_mask_lambda", 0.1),
        dec_mask_freeze=cfg.get("dec_mask_freeze", False))
    md.load_state_dict(torch.load(_ckpt(rd), map_location="cpu"), strict=False); md.eval()
    with torch.no_grad():
        A_rr, A_pp, W_rp, W_pr = md.scm.get_block_adjacencies(tau=0.5)
        Dr = md._dec_r_w().detach().float()
        G_rr = (Dr @ A_rr.float() @ Dr.T).cpu().numpy()
    np.fill_diagonal(G_rr, 0.0)
    return np.abs(G_rr), genes, cfg


# ------------------------------------------------------------------------------------ scoring core
def score_pathway(absG, off_scores, off_mask, inpath):
    """AUROC / AUPRC / AUPRC-over-random / fold for within-pathway vs background off-diagonal pairs."""
    pos_mask = np.outer(inpath, inpath)          # both members (diagonal handled by off_mask)
    labels_off = pos_mask[off_mask]
    npos = int(labels_off.sum())
    if npos < 3:
        return None
    prevalence = float(labels_off.mean())
    within = off_scores[labels_off]; background = off_scores[~labels_off]
    auroc = float(roc_auc_score(labels_off, off_scores))
    auprc = float(average_precision_score(labels_off, off_scores))
    fold = float(np.median(within) / (np.median(background) + 1e-12))
    return dict(auroc=auroc, auprc=auprc, auprc_over_random=float(auprc / prevalence),
                fold_enrichment=fold, within_median=float(np.median(within)),
                background_median=float(np.median(background)),
                n_within_pairs=npos, prevalence=prevalence)


def run(rd):
    absG, genes, cfg = load_Grr(rd)
    ng = len(genes); gidx = {g: i for i, g in enumerate(genes)}
    vocab_set = set(genes)
    off_mask = ~np.eye(ng, dtype=bool)
    off_scores = absG[off_mask]                  # computed once, reused per pathway
    resolved = resolve_members(vocab_set)

    run_name = os.path.basename(rd)
    out = {"run": run_name, "n_genes": ng,
           "G_rr_projection": "|D_r A_rr D_r^T| (diag zeroed) -- same as causal_eval_suite",
           "min_members": MIN_MEMBERS, "pathways": {}, "skipped": {}}
    for name, inv, src, n_total in resolved:
        if len(inv) < MIN_MEMBERS:
            out["skipped"][name] = dict(n_members=len(inv), n_total_geneset=n_total, source=src,
                                        reason=f"<{MIN_MEMBERS} members in vocab")
            continue
        inpath = np.zeros(ng, dtype=bool)
        for g in inv:
            inpath[gidx[g]] = True
        res = score_pathway(absG, off_scores, off_mask, inpath)
        if res is None:
            out["skipped"][name] = dict(n_members=len(inv), reason="too few pairs")
            continue
        res.update(n_members=len(inv), n_total_geneset=n_total, source=src, members=inv)
        out["pathways"][name] = res

    outp = f"{rd}/pathway_recovery.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"=== pathway_recovery: {run_name}  (n_genes={ng}) ===")
    hdr = f'{"pathway":36s} {"n":>4s} {"AUROC":>7s} {"AUPRC":>7s} {"AUPRCx":>7s} {"fold":>6s}  source'
    print(hdr)
    for name, r in out["pathways"].items():
        print(f'{name:36s} {r["n_members"]:4d} {r["auroc"]:7.3f} {r["auprc"]:7.4f} '
              f'{r["auprc_over_random"]:7.2f} {r["fold_enrichment"]:6.2f}  {r["source"]}')
    for name, s in out["skipped"].items():
        print(f'[skip] {name}: {s.get("reason","")} (n_members={s.get("n_members","?")})')
    print("WROTE", outp)
    return out


# ------------------------------------------------------------------------------------ deck figure
def make_figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
    C_2C, C_CB = "#1f4e79", "#b5651d"
    j2 = json.load(open(f"{RES}/{RUN_2C}/pathway_recovery.json"))
    jc = json.load(open(f"{RES}/{RUN_COMBAT}/pathway_recovery.json"))
    order = [p["name"] for p in PATHWAYS if p["name"] in j2["pathways"] or p["name"] in jc["pathways"]]
    short = {"cGAS-STING": "cGAS-\nSTING", "IFN-alpha (Type-I) response": "IFN-α\n(Type-I)",
             "IFN-gamma (Type-II) response": "IFN-γ\n(Type-II)", "JAK-STAT signaling": "JAK-\nSTAT",
             "TNF-alpha / NF-kB signaling": "TNF/\nNF-κB", "Antigen presentation (MHC I & II)": "Antigen\npres.",
             "TCR signaling": "TCR", "Apoptosis": "Apop-\ntosis", "MHC-I complex (control)": "MHC-I\n(ctrl)"}
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    for i, (j, c, lab) in enumerate([(j2, C_2C, "2-cohort"), (jc, C_CB, "2-cohort + COMBAT")]):
        vals = [j["pathways"].get(nm, {}).get("auroc", np.nan) for nm in order]
        ns = [j["pathways"].get(nm, {}).get("n_members", 0) for nm in order]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=c, label=lab)
        for b, v, n in zip(bars, vals, ns):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"n={n}", ha="center", va="bottom",
                        fontsize=8.5, color=c)
    ax.axhline(0.5, ls="--", c="k", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([short.get(nm, nm) for nm in order])
    ax.set_ylabel("AUROC  (within-pathway vs background |G_rr| pairs)")
    ymax = max([j["pathways"].get(nm, {}).get("auroc", 0.5) for j in (j2, jc) for nm in order] + [0.6])
    ax.set_ylim(0.45, min(1.0, ymax + 0.07))
    ax.set_title("Signaling-pathway recovery in the projected GRN (G_rr)")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=12)
    outp = f"{RES}/figures/slides/fig_pathway.png"; os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.tight_layout(); fig.savefig(outp); plt.close(fig)
    print("WROTE", outp)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--figure":
        make_figure()
    else:
        run(os.path.abspath(sys.argv[1]))
        j2 = f"{RES}/{RUN_2C}/pathway_recovery.json"; jc = f"{RES}/{RUN_COMBAT}/pathway_recovery.json"
        if os.path.exists(j2) and os.path.exists(jc):
            make_figure()
