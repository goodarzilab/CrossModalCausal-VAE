#!/usr/bin/env python
"""BLOCK-ISOLATED DAG ablation. Unlike dag_shuffle_ablation.py (which shuffles/zeros the WHOLE joint
A_full), this perturbs ONLY the block relevant to each readout, holding the other three blocks at their
trained values:
    RNA->RNA (RR)  depends on  A_rr    -> shuffle/zero ONLY A_rr  (keep W_rp, W_pr, A_pp)
    RNA->Prot (RP) depends on  W_rp    -> shuffle/zero ONLY W_rp  (keep A_rr, W_pr, A_pp)

This directly answers "is block X load-bearing for its own readout?" without cross-block confounding.
Usage: python analysis/common/block_isolated_ablation.py <results_dir>
"""
import sys, os, json, warnings
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perturb_validation_analysis as P
from dag_shuffle_ablation import auroc, shuffle_offdiag, do_op
DEVICE = P.DEVICE; K = 5


def put_block(A_full, block, r0, r1, c0, c1):
    """Return a copy of A_full with the [r0:r1, c0:c1] sub-block replaced by `block`."""
    M = A_full.clone(); M[r0:r1, c0:c1] = block; return M


def shuffle_all(M, gen):
    """Permute ALL entries of a (possibly rectangular) matrix — for the cross-modal W_rp block, which
    has no diagonal to preserve (shuffle_offdiag assumes square)."""
    M = M.clone(); flat = M.flatten()
    perm = (torch.randperm(flat.numel(), generator=gen) if M.device.type == "cpu"
            else torch.randperm(flat.numel()).to(M.device))
    return flat[perm].reshape(M.shape)


def auprc(lab, score):
    """AUPRC-over-random (average precision / prevalence); 1.0 = random."""
    lab = lab.astype(int)
    if lab.sum() < 3 or lab.sum() >= len(lab): return np.nan
    try: return float(average_precision_score(lab, score) / lab.mean())
    except Exception: return np.nan


def main():
    rd = os.path.abspath(sys.argv[1]); cfg = json.load(open(os.path.join(rd, "config.json")))
    MUT = os.environ.get("CMC_MUTILATE", "0") == "1"; SFX = "_mut" if MUT else ""   # mutilation read-out for the load-bearing
    P.MAX_CONTROL_CELLS = 4000   # CONSISTENT with mutilation_directed_eval (was 2000 here -> now 4000 everywhere)
    perturb_rna, perturb_prot, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
    de = P.compute_observed_de(perturb_rna, perturb_prot, align)
    from model import CrossModalCausalLDVAE
    model = P.load_ctx_model(CrossModalCausalLDVAE, rd, len(align["train_gene_names"]), len(align["train_prot_names"]), cfg)
    _ovr = os.environ.get("CMC_SCM_OVERRIDE")
    if _ovr:
        model.scm.load_state_dict(torch.load(_ovr, map_location=DEVICE)); print(f"[scm override] {_ovr}", flush=True)
    eps_r, eps_p, _, _, _ = P.encode_control_cells(model, perturb_rna, perturb_prot, align)
    eps = torch.cat([eps_r, eps_p], 1).to(DEVICE)[:4000]
    lr, lp = model.latent_r, model.latent_p
    with torch.no_grad():
        A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
        A_full = torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0).to(DEVICE)
    g2c = align["train_gene_to_idx_common"]; common = align["common_genes"]
    reti = align["rna_eval_train_indices"]; tgi = align["train_gene_idx"]
    gen = torch.Generator().manual_seed(0)
    # block-isolated variants: only A_rr perturbed (RR), only W_rp perturbed (RP)
    Arr_true = A_full[:lr, :lr].cpu(); Wrp_true = A_full[:lr, lr:].cpu()
    rr_shuf_As = [put_block(A_full, shuffle_offdiag(Arr_true, gen).to(DEVICE), 0, lr, 0, lr) for _ in range(K)]
    rr_zero_A  = put_block(A_full, torch.zeros_like(A_full[:lr, :lr]), 0, lr, 0, lr)
    rp_shuf_As = [put_block(A_full, shuffle_all(Wrp_true, gen).to(DEVICE), 0, lr, lr, lr + lp) for _ in range(K)]
    rp_zero_A  = put_block(A_full, torch.zeros_like(A_full[:lr, lr:]), 0, lr, lr, lr + lp)

    rows = []
    with torch.no_grad():
        for g, d in de.items():
            if g not in tgi or g not in g2c: continue
            keep = np.array([cg != g for cg in common]); lab = d["sig_rna"][keep].astype(int)
            if lab.sum() < 3: continue
            labp = d.get("sig_prot", np.array([]))
            has_p = len(labp) and labp.sum() >= 3
            def rr(dr): return np.abs(dr.mean(0).cpu().numpy()[reti])[keep]
            def rp(dp): return np.abs(dp.mean(0).cpu().numpy()[align["prot_eval_train_indices"]])
            # continuous ground-truth effect sizes (full dynamic range; not floor-compressed like near-
            # chance AUROC). Spearman(|predicted Δ|, observed effect) — a more sensitive load-bearing metric.
            obs_eff_rr = np.asarray(d["effect_rna"])[keep]
            obs_eff_rp = np.asarray(d.get("effect_prot", []))
            obs_lfc_rr = np.asarray(d["lfc_rna"])[keep]                 # signed ground-truth Δ (ex-KO)
            obs_lfc_rp = np.asarray(d.get("lfc_prot", []))
            def sp(pred, obs):
                if len(pred) < 5 or np.nanstd(obs) == 0 or np.nanstd(pred) == 0: return np.nan
                return spearmanr(pred, obs, nan_policy="omit").correlation
            def rr_s(dr): return dr.mean(0).cpu().numpy()[reti][keep]    # SIGNED predicted Δ (ex-KO)
            def rp_s(dp): return dp.mean(0).cpu().numpy()[align["prot_eval_train_indices"]]
            def pear(pred, obs, mask=None):                             # signed Pearson (continuous, directional)
                p, o = (pred, obs) if mask is None else (pred[mask], obs[mask])
                if len(p) < 5 or np.nanstd(p) == 0 or np.nanstd(o) == 0: return np.nan
                return float(np.corrcoef(p, o)[0, 1])
            dr_t, dp_t = do_op(model, eps, A_full, tgi[g], lr, lp, mutilate=MUT)
            dr_0 = do_op(model, eps, rr_zero_A, tgi[g], lr, lp, mutilate=MUT)[0]
            dp_0 = do_op(model, eps, rp_zero_A, tgi[g], lr, lp, mutilate=MUT)[1] if has_p else None
            # RR: perturb ONLY A_rr
            rr_true = auroc(lab, rr(dr_t))
            rr_zero = auroc(lab, rr(dr_0))
            rr_shuf_preds = [do_op(model, eps, Ap, tgi[g], lr, lp, mutilate=MUT)[0] for Ap in rr_shuf_As]
            rr_shuf = np.nanmean([auroc(lab, rr(p)) for p in rr_shuf_preds])
            rr_shuf_ap = np.nanmean([auprc(lab, rr(p)) for p in rr_shuf_preds])
            rr_rho_true = sp(rr(dr_t), obs_eff_rr); rr_rho_zero = sp(rr(dr_0), obs_eff_rr)  # magnitude Spearman
            de_r = lab.astype(bool)                                     # DE-gene mask (ex-KO)
            rr_pear_true = pear(rr_s(dr_t), obs_lfc_rr); rr_pear_zero = pear(rr_s(dr_0), obs_lfc_rr)          # signed Pearson, all genes
            rr_pearDE_true = pear(rr_s(dr_t), obs_lfc_rr, de_r); rr_pearDE_zero = pear(rr_s(dr_0), obs_lfc_rr, de_r)  # signed Pearson, DE only
            rr_pearDE_shuf = np.nanmean([pear(rr_s(p), obs_lfc_rr, de_r) for p in rr_shuf_preds])
            # RP: perturb ONLY W_rp
            rp_true = auroc(labp.astype(int), rp(dp_t)) if has_p else np.nan
            rp_zero = auroc(labp.astype(int), rp(dp_0)) if has_p else np.nan
            rp_shuf_preds = [do_op(model, eps, Ap, tgi[g], lr, lp, mutilate=MUT)[1] for Ap in rp_shuf_As] if has_p else []
            rp_shuf = np.nanmean([auroc(labp.astype(int), rp(p)) for p in rp_shuf_preds]) if has_p else np.nan
            rp_shuf_ap = np.nanmean([auprc(labp.astype(int), rp(p)) for p in rp_shuf_preds]) if has_p else np.nan
            rp_rho_true = sp(rp(dp_t), obs_eff_rp) if has_p else np.nan
            rp_rho_zero = sp(rp(dp_0), obs_eff_rp) if has_p else np.nan
            de_p = labp.astype(bool) if has_p else None
            rp_pear_true = pear(rp_s(dp_t), obs_lfc_rp) if has_p else np.nan
            rp_pear_zero = pear(rp_s(dp_0), obs_lfc_rp) if has_p else np.nan
            rp_pearDE_true = pear(rp_s(dp_t), obs_lfc_rp, de_p) if has_p else np.nan
            rp_pearDE_zero = pear(rp_s(dp_0), obs_lfc_rp, de_p) if has_p else np.nan
            rp_pearDE_shuf = np.nanmean([pear(rp_s(p), obs_lfc_rp, de_p) for p in rp_shuf_preds]) if has_p else np.nan
            rr_true_ap = auprc(lab, rr(dr_t)); rr_zero_ap = auprc(lab, rr(dr_0))   # AUPRC-over-random
            rp_true_ap = auprc(labp.astype(int), rp(dp_t)) if has_p else np.nan
            rp_zero_ap = auprc(labp.astype(int), rp(dp_0)) if has_p else np.nan
            rows.append(dict(g=g, rr_true=rr_true, rr_shufArr=rr_shuf, rr_zeroArr=rr_zero,
                             rr_true_ap=rr_true_ap, rr_shufArr_ap=rr_shuf_ap, rr_zeroArr_ap=rr_zero_ap,
                             rr_rho_true=rr_rho_true, rr_rho_zeroArr=rr_rho_zero,
                             rr_pear_true=rr_pear_true, rr_pear_zeroArr=rr_pear_zero,
                             rr_pearDE_true=rr_pearDE_true, rr_pearDE_shufArr=rr_pearDE_shuf, rr_pearDE_zeroArr=rr_pearDE_zero,
                             rp_true=rp_true, rp_shufWrp=rp_shuf, rp_zeroWrp=rp_zero,
                             rp_true_ap=rp_true_ap, rp_shufWrp_ap=rp_shuf_ap, rp_zeroWrp_ap=rp_zero_ap,
                             rp_rho_true=rp_rho_true, rp_rho_zeroWrp=rp_rho_zero,
                             rp_pear_true=rp_pear_true, rp_pear_zeroWrp=rp_pear_zero,
                             rp_pearDE_true=rp_pearDE_true, rp_pearDE_shufWrp=rp_pearDE_shuf, rp_pearDE_zeroWrp=rp_pearDE_zero))
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = {f"mean_{c}": float(df[c].mean()) for c in df.columns if c != "g"}
    summ |= {f"sem_{c}": float(df[c].std(ddof=1) / np.sqrt(df[c].notna().sum())) for c in df.columns if c != "g"}
    summ["n"] = len(df)
    # non-mut keeps the legacy name (Frangieh, read by existing tables); mut is tagged per dataset
    _tag = os.environ.get("CMC_DS_TAG", "frangieh")
    _out = f"block_isolated_mut_{_tag}_ablation{os.environ.get("CMC_OUT_SFX","")}.json" if MUT else "block_isolated_ablation.json"
    json.dump({"summary": summ, "per_perturbation": rows}, open(os.path.join(rd, _out), "w"), indent=2)
    f = lambda k: (summ.get(f"mean_{k}", np.nan), summ.get(f"sem_{k}", np.nan))
    print(f"\n=== BLOCK-ISOLATED ablation: {os.path.basename(rd)}  (n={len(df)}) ===")
    print("  RR (perturb ONLY A_rr):   true {:.3f}±{:.3f}   shufArr {:.3f}±{:.3f}   zeroArr {:.3f}±{:.3f}".format(
        *f('rr_true'), *f('rr_shufArr'), *f('rr_zeroArr')))
    print("     -> A_rr load-bearing Delta(true-zeroArr) = {:+.3f}".format(f('rr_true')[0] - f('rr_zeroArr')[0]))
    print("  RP (perturb ONLY W_rp):   true {:.3f}±{:.3f}   shufWrp {:.3f}±{:.3f}   zeroWrp {:.3f}±{:.3f}".format(
        *f('rp_true'), *f('rp_shufWrp'), *f('rp_zeroWrp')))
    print("     -> W_rp load-bearing Delta(true-zeroWrp) = {:+.3f}".format(f('rp_true')[0] - f('rp_zeroWrp')[0]))
    print("  -- SIGNED PEARSON (predicted Δ vs observed LFC, ex-KO; continuous+directional = most sensitive) --")
    print("    RR pear(DE)  true {:.3f}  zeroArr {:.3f}  -> Delta = {:+.3f}   | pear(all) Delta = {:+.3f}".format(
        f('rr_pearDE_true')[0], f('rr_pearDE_zeroArr')[0], f('rr_pearDE_true')[0] - f('rr_pearDE_zeroArr')[0],
        f('rr_pear_true')[0] - f('rr_pear_zeroArr')[0]))
    print("    RP pear(DE)  true {:.3f}  zeroWrp {:.3f}  -> Delta = {:+.3f}   | pear(all) Delta = {:+.3f}".format(
        f('rp_pearDE_true')[0], f('rp_pearDE_zeroWrp')[0], f('rp_pearDE_true')[0] - f('rp_pearDE_zeroWrp')[0],
        f('rp_pear_true')[0] - f('rp_pear_zeroWrp')[0]))
    print("  -- (secondary: magnitude-Spearman) RR Delta_rho={:+.3f}  RP Delta_rho={:+.3f}".format(
        f('rr_rho_true')[0] - f('rr_rho_zeroArr')[0], f('rp_rho_true')[0] - f('rp_rho_zeroWrp')[0]))
    print("DONE")


if __name__ == "__main__":
    main()
