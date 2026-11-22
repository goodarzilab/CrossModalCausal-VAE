"""HELD-OUT-TF evaluation of the signed directed prior — the key generalization-vs-memorization test.

For each perturbation dataset (frangieh/iccite/arce) and each model (baseline prior-off vs signed-prior),
compute per-KO RR read-outs (mutil + std do-op, and the direct projected-graph gproj = -G_rr[TF,:]):
  - pearDE   : signed Pearson(predicted Δ, observed LFC) on DE genes
  - sign_acc : fraction of DE genes with sign(pred Δ)==sign(obs LFC)
Each perturbed gene is tagged by its membership in the signed prior's TF split:
  heldout  (TF whose edges were NEVER in the training loss  -> tests GENERALIZATION)
  train    (TF whose edges WERE in the training loss        -> tests MEMORIZATION)
  other    (perturbed gene that is not a prior TF)
Also: TF->gene ASYMMETRY of |G_rr| on the prior's heldout-TF vs train-TF edges, per model.

The comparison the paper needs: does the signed prior lift held-out-TF pearDE/sign_acc over the
prior-off baseline? If it only helps train (in-prior) TFs, that is the (expected) memorization result.

Usage:
  python analysis/supplementary/objectives/signed_prior_heldout_eval.py <baseline_dir> <signedprior_dir>
  (loops all 3 datasets internally; writes results/signed_prior_eval.json)
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
import os, sys, json
import numpy as np, torch, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE
P.MAX_CONTROL_CELLS = 4000

BASELINE = os.path.abspath(sys.argv[1])
SIGNED = os.path.abspath(sys.argv[2])
SP_PT = f"{_CMC_ROOT}/priors/signed_prior_2c_v4.pt"
OUT = f"{_CMC_RESULTS}/signed_prior_eval.json"
DATA = f"{_CMC_DATA}"
DATASETS = {
    "frangieh": (f"{DATA}/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad",
                 f"{DATA}/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad"),
    "iccite":   (f"{DATA}/chen_perturb_icciteseq/export/iccite_rna.h5ad",
                 f"{DATA}/chen_perturb_icciteseq/export/iccite_prot.h5ad"),
    "arce":     (f"{DATA}/arce_perturb_citeseq/arce_rna.h5ad",
                 f"{DATA}/arce_perturb_citeseq/arce_prot.h5ad"),
}
# locked DE filter (same as rerun_de_evals.sh)
os.environ.setdefault("CMC_DE_LFC_RNA", "0.10")
os.environ.setdefault("CMC_DE_LFC_PROT", "0.05")
os.environ.setdefault("CMC_CKPT", "final")

sp = torch.load(SP_PT, map_location="cpu")
HELD = set(map(str, sp["heldout_tf_names"]))
TRAIN = set(map(str, sp["train_tf_names"]))
sign_np = (sp["sign"].numpy() if torch.is_tensor(sp["sign"]) else np.asarray(sp["sign"]))
sp_genes = list(map(str, sp["gene_names"]))
sp_gi = {g: i for i, g in enumerate(sp_genes)}


def split_of(g):
    return "heldout" if g in HELD else ("train" if g in TRAIN else "other")


def pe(pred, obs):
    if len(pred) < 5 or np.std(pred) == 0 or np.std(obs) == 0:
        return np.nan
    return float(np.corrcoef(pred, obs)[0, 1])


def signacc(pred, obs):
    m = obs != 0
    if m.sum() < 5:
        return np.nan
    return float((np.sign(pred[m]) == np.sign(obs[m])).mean())


def build_predictor(rd, cfg, align, pr, pp):
    """Return (predict_fn, grn, tgi, reti, g2i, common, corr_rr) for one model on this dataset."""
    model = P.load_ctx_model(CrossModalCausalLDVAE, rd, len(align["train_gene_names"]),
                             len(align["train_prot_names"]), cfg)
    eps_r, eps_p, *_ = P.encode_control_cells(model, pr, pp, align)
    eps = torch.cat([eps_r, eps_p], 1).to(P.DEVICE)[:4000]
    lr, lp = model.latent_r, model.latent_p
    with torch.no_grad():
        A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
        A_full = torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0).to(P.DEVICE)
        Wr = model._dec_r_w().detach().to(P.DEVICE)
        grn = model.get_projected_graphs()[0].cpu().numpy()      # signed G_rr
    solve = lambda Amat, e: model.scm._scm_solve(Amat, e)

    def predict(gidx, mutilate):
        with torch.no_grad():
            z_c = solve(A_full, eps); zr_c, zp_c = torch.split(z_c, [lr, lp], 1)
            rec_r_c = model.dec_rna(zr_c) + model.bias_rna
            dx = torch.zeros_like(rec_r_c); dx[:, gidx] = -rec_r_c[:, gidx]
            deps = torch.cat([dx @ Wr, torch.zeros_like(zp_c)], 1)
            Amat = A_full
            if mutilate:
                s = (Wr[gidx].abs() / (Wr[gidx].abs().max() + 1e-8)).clamp(0, 1)
                Amat = A_full.clone(); Amat[:, :lr] = Amat[:, :lr] * (1 - s).unsqueeze(0)
            z_p = solve(Amat, eps + deps); zr_p, _ = torch.split(z_p, [lr, lp], 1)
            d_r = ((model.dec_rna(zr_p) + model.bias_rna) - rec_r_c).mean(0).cpu().numpy()
        return d_r

    return predict, grn


results = {"baseline_dir": os.path.basename(BASELINE), "signedprior_dir": os.path.basename(SIGNED),
           "n_heldout_tfs": len(HELD), "n_train_tfs": len(TRAIN), "per_dataset": {}, "perpert": []}

for tag, (rna_p, prot_p) in DATASETS.items():
    if not os.path.exists(rna_p):
        print(f"[skip {tag}] missing {rna_p}"); continue
    os.environ["PERTURB_RNA_PATH"] = rna_p; os.environ["PERTURB_PROT_PATH"] = prot_p
    P.PERTURB_RNA_PATH = rna_p; P.PERTURB_PROT_PATH = prot_p
    print(f"\n===== {tag} =====")
    cfg = json.load(open(f"{BASELINE}/config.json"))
    pr, pp, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
    de = P.compute_observed_de(pr, pp, align)
    common = np.array(align["common_genes"]); g2i = align["train_gene_to_idx_common"]
    tgi = align["train_gene_idx"]; reti = align["rna_eval_train_indices"]
    corr_rr = align.get("corr_train")

    preds = {}
    for label, rd in [("baseline", BASELINE), ("signed", SIGNED)]:
        cfg_m = json.load(open(f"{rd}/config.json"))
        preds[label] = build_predictor(rd, cfg_m, align, pr, pp)

    for g, d in de.items():
        if g not in tgi or g not in g2i:
            continue
        keep = common != g
        lab = d["sig_rna"][keep].astype(bool)
        if lab.sum() < 5:
            continue
        lfc = np.asarray(d["lfc_rna"])[keep][lab]
        row = {"pert": g, "dataset": tag, "split": split_of(g), "n_de": int(lab.sum())}
        for label in ("baseline", "signed"):
            predict, grn = preds[label]
            for nm, mut in [("std", False), ("mutil", True)]:
                dr = predict(tgi[g], mut)[reti][keep][lab]
                row[f"{label}_{nm}_pearDE"] = pe(dr, lfc)
                row[f"{label}_{nm}_signacc"] = signacc(dr, lfc)
            gv = grn[tgi[g], :][reti][keep][lab]
            row[f"{label}_gproj_pearDE"] = pe(-gv, lfc)
            row[f"{label}_gproj_signacc"] = signacc(-gv, lfc)
        # data-derived correlation baseline (model-independent)
        if corr_rr is not None:
            cv = corr_rr[g2i[g], :][keep][lab]
            row["corr_pearDE"] = pe(-cv, lfc)
            row["corr_signacc"] = signacc(-cv, lfc)
        results["perpert"].append(row)
        print(f"  {g:10s} [{row['split']:7s}] nDE={row['n_de']:4d} | "
              f"base mutil pearDE={row.get('baseline_mutil_pearDE',float('nan')):+.3f} "
              f"signed mutil pearDE={row.get('signed_mutil_pearDE',float('nan')):+.3f} | "
              f"base gproj={row.get('baseline_gproj_pearDE',float('nan')):+.3f} "
              f"signed gproj={row.get('signed_gproj_pearDE',float('nan')):+.3f}")

    # release perturb data before next dataset
    del pr, pp, align, de, preds
    import gc; gc.collect()

# ---------------- aggregate by split ----------------
df = pd.DataFrame(results["perpert"])
agg = {}
if len(df):
    metrics = [c for c in df.columns if c.endswith("_pearDE") or c.endswith("_signacc")]
    for split in ["heldout", "train", "other", "ALL"]:
        sub = df if split == "ALL" else df[df["split"] == split]
        if not len(sub):
            continue
        agg[split] = {"n_perts": int(len(sub)),
                      "perts": sorted(sub["pert"].unique().tolist())}
        for m in metrics:
            vals = sub[m].dropna()
            if len(vals):
                agg[split][m] = {"mean": round(float(vals.mean()), 4),
                                 "median": round(float(vals.median()), 4), "n": int(len(vals))}
results["aggregate_by_split"] = agg

# ---------------- TF->gene asymmetry on prior edges (per model) ----------------
def asymmetry(grn, tf_names):
    """median |G_rr[tf,tgt]| / median |G_rr[tgt,tf]| over the signed prior's edges of these TFs."""
    fwd, rev = [], []
    for tf in tf_names:
        if tf not in sp_gi:
            continue
        i = sp_gi[tf]
        tgts = np.nonzero(sign_np[i])[0]
        for j in tgts:
            g_tf, g_tg = sp_genes[i], sp_genes[j]
            # grn is in training-gene order == vocab order == sp order (all combined_v4)
            fwd.append(abs(grn[i, j])); rev.append(abs(grn[j, i]))
    if not fwd:
        return None
    fwd, rev = np.array(fwd), np.array(rev)
    return {"fwd_median": float(np.median(fwd)), "rev_median": float(np.median(rev)),
            "fwd_over_rev": float(np.median(fwd) / (np.median(rev) + 1e-12)), "n_edges": int(len(fwd))}

# recompute grn once per model (dataset-independent) from a light load
asym = {}
for label, rd in [("baseline", BASELINE), ("signed", SIGNED)]:
    cfg_m = json.load(open(f"{rd}/config.json"))
    import scanpy as sc
    ar = sc.read_h5ad(cfg_m["rna_path"], backed="r"); ng = ar.n_vars
    ap = sc.read_h5ad(cfg_m["prot_path"], backed="r"); npr = ap.n_vars
    ar.file.close(); ap.file.close()
    m = CrossModalCausalLDVAE(ng, npr, 1, latent_dim_rna=cfg_m["latent_dim_rna"],
                              latent_dim_prot=cfg_m["latent_dim_prot"], recon_likelihood="mse",
                              scm_nonlinear=cfg_m.get("scm_nonlinear", False), phi=cfg_m.get("phi", "tanh"))
    m.load_state_dict(torch.load(P.resolve_ckpt(rd), map_location="cpu"), strict=False); m.eval()
    grn = m.get_projected_graphs()[0].cpu().numpy()
    asym[label] = {"heldout_TFs": asymmetry(grn, HELD), "train_TFs": asymmetry(grn, TRAIN)}
results["asymmetry"] = asym

json.dump(results, open(OUT, "w"), indent=2)
print(f"\nSAVED {OUT}")
print("=== AGGREGATE (mean pearDE / sign_acc by split) ===")
for split in ["heldout", "train", "other"]:
    if split not in agg:
        continue
    a = agg[split]
    def g(k):
        return a.get(k, {}).get("mean", float("nan"))
    print(f"[{split}] n={a['n_perts']:3d} | "
          f"mutil pearDE base={g('baseline_mutil_pearDE'):+.3f} signed={g('signed_mutil_pearDE'):+.3f} | "
          f"gproj pearDE base={g('baseline_gproj_pearDE'):+.3f} signed={g('signed_gproj_pearDE'):+.3f} | "
          f"gproj signacc base={g('baseline_gproj_signacc'):.3f} signed={g('signed_gproj_signacc'):.3f} | "
          f"corr pearDE={g('corr_pearDE'):+.3f}")
print("ASYMMETRY (fwd/rev on prior edges):")
for label in ("baseline", "signed"):
    for grp in ("heldout_TFs", "train_TFs"):
        v = asym[label].get(grp)
        if v:
            print(f"  {label} {grp}: fwd/rev={v['fwd_over_rev']:.3f} (n={v['n_edges']})")
print("DONE")
