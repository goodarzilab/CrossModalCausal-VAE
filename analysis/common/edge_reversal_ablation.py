#!/usr/bin/env python
"""EDGE-REVERSAL ablation (derived from analysis/common/block_isolated_ablation.py).

Tests whether the learned latent causal graph is genuinely DIRECTIONAL. On top of the block-isolated
REAL / SHUFFLED / ZEROED variants, this adds an edge-REVERSED variant per block and asks whether
reversing edge orientation degrades held-out perturbation prediction:

  RR (RNA->RNA, A_rr):
     REAL      : trained A_rr                                     (reference)
     REVERSED  : A_rr -> A_rr^T          (swap every edge i->j with j->i)
     SHUFFLED  : off-diagonal re-paired  (same weights/density, structure destroyed) = random-edge-swap CONTROL
     ZEROED    : A_rr = 0
  RP (RNA->Prot, W_rp):
     REAL      : trained W_rp                                     (reference)
     REVERSED  : W_rp -> W_pr^T          (use the transposed feedback block = the protein->RNA direction)
     SHUFFLED  : all entries permuted    (control)
     ZEROED    : W_rp = 0

Only the target block is altered; the other three blocks stay at their trained values (block-isolated).
Propagation is the model's mutilation do-op read-out (do_op(..., mutilate=True)), z=(I-A^T)^-1 eps solver,
so reversal (transpose) is the exact edge-orientation flip.

Same env-var / invocation contract as block_isolated_ablation.py. Writes a per-cohort summary into the
results dir AND merges its summary+per-perturbation rows into results/edge_reversal.json under the
CMC_DS_TAG cohort key.

Usage: python analysis/common/edge_reversal_ablation.py <results_dir>
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
import sys, os, json, warnings
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perturb_validation_analysis as P
from dag_shuffle_ablation import auroc, shuffle_offdiag, do_op
from block_isolated_ablation import put_block, shuffle_all, auprc
DEVICE = P.DEVICE; K = 5


def main():
    rd = os.path.abspath(sys.argv[1]); cfg = json.load(open(os.path.join(rd, "config.json")))
    MUT = os.environ.get("CMC_MUTILATE", "1") == "1"   # mutilation read-out ON by default for the load-bearing/reversal test
    P.MAX_CONTROL_CELLS = 4000
    perturb_rna, perturb_prot, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
    de = P.compute_observed_de(perturb_rna, perturb_prot, align)
    from model import CrossModalCausalLDVAE
    model = P.load_ctx_model(CrossModalCausalLDVAE, rd, len(align["train_gene_names"]), len(align["train_prot_names"]), cfg)
    eps_r, eps_p, _, _, _ = P.encode_control_cells(model, perturb_rna, perturb_prot, align)
    eps = torch.cat([eps_r, eps_p], 1).to(DEVICE)[:4000]
    lr, lp = model.latent_r, model.latent_p
    with torch.no_grad():
        A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
        A_full = torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0).to(DEVICE)
    g2c = align["train_gene_to_idx_common"]; common = align["common_genes"]
    reti = align["rna_eval_train_indices"]; tgi = align["train_gene_idx"]
    gen = torch.Generator().manual_seed(0)

    # true trained blocks
    Arr_true = A_full[:lr, :lr].cpu()          # [lr, lr]  RNA->RNA
    Wrp_true = A_full[:lr, lr:].cpu()          # [lr, lp]  RNA->Prot (translation)
    Wpr_true = A_full[lr:, :lr].cpu()          # [lp, lr]  Prot->RNA (feedback)

    # ---- RR variants (perturb ONLY A_rr) ----
    rr_rev_A   = put_block(A_full, Arr_true.T.to(DEVICE),           0, lr, 0, lr)   # REVERSED = transpose (i->j swap j->i)
    rr_shuf_As = [put_block(A_full, shuffle_offdiag(Arr_true, gen).to(DEVICE), 0, lr, 0, lr) for _ in range(K)]  # random-edge-swap CONTROL
    rr_zero_A  = put_block(A_full, torch.zeros_like(A_full[:lr, :lr]), 0, lr, 0, lr)
    # ---- RP variants (perturb ONLY W_rp) ----
    rp_rev_A   = put_block(A_full, Wpr_true.T.to(DEVICE),           0, lr, lr, lr + lp)  # REVERSED = W_pr^T (feedback dir transposed)
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
            obs_lfc_rr = np.asarray(d["lfc_rna"])[keep]
            obs_lfc_rp = np.asarray(d.get("lfc_prot", []))
            def rr_s(dr): return dr.mean(0).cpu().numpy()[reti][keep]      # signed predicted Δ
            def rp_s(dp): return dp.mean(0).cpu().numpy()[align["prot_eval_train_indices"]]
            def pear(pred, obs, mask=None):
                p, o = (pred, obs) if mask is None else (pred[mask], obs[mask])
                if len(p) < 5 or np.nanstd(p) == 0 or np.nanstd(o) == 0: return np.nan
                return float(np.corrcoef(p, o)[0, 1])
            de_r = lab.astype(bool)
            de_p = labp.astype(bool) if has_p else None

            # ---------- RR: perturb ONLY A_rr ----------
            dr_t = do_op(model, eps, A_full,   tgi[g], lr, lp, mutilate=MUT)[0]
            dr_r = do_op(model, eps, rr_rev_A, tgi[g], lr, lp, mutilate=MUT)[0]
            dr_0 = do_op(model, eps, rr_zero_A, tgi[g], lr, lp, mutilate=MUT)[0]
            rr_shuf_preds = [do_op(model, eps, Ap, tgi[g], lr, lp, mutilate=MUT)[0] for Ap in rr_shuf_As]
            rr_true = auroc(lab, rr(dr_t)); rr_rev = auroc(lab, rr(dr_r)); rr_zero = auroc(lab, rr(dr_0))
            rr_shuf = np.nanmean([auroc(lab, rr(p)) for p in rr_shuf_preds])
            rr_true_ap = auprc(lab, rr(dr_t)); rr_rev_ap = auprc(lab, rr(dr_r)); rr_zero_ap = auprc(lab, rr(dr_0))
            rr_shuf_ap = np.nanmean([auprc(lab, rr(p)) for p in rr_shuf_preds])
            rr_pearDE_true = pear(rr_s(dr_t), obs_lfc_rr, de_r)
            rr_pearDE_rev  = pear(rr_s(dr_r), obs_lfc_rr, de_r)
            rr_pearDE_zero = pear(rr_s(dr_0), obs_lfc_rr, de_r)
            rr_pearDE_shuf = np.nanmean([pear(rr_s(p), obs_lfc_rr, de_r) for p in rr_shuf_preds])
            # ---------- RR strict gating: top-k strongest DEGs by |LFC| (AUPRCx + signed pearson) ----------
            rr_topk = {}
            _order = np.argsort(-np.abs(obs_lfc_rr))
            for _k in (20, 50):
                tm = np.zeros(len(obs_lfc_rr), bool); tm[_order[:min(_k, len(obs_lfc_rr))]] = True
                tmi = tm.astype(int)
                rr_topk[f"rr_top{_k}_ap_true"] = auprc(tmi, rr(dr_t))
                rr_topk[f"rr_top{_k}_ap_rev"]  = auprc(tmi, rr(dr_r))
                rr_topk[f"rr_top{_k}_ap_shuf"] = float(np.nanmean([auprc(tmi, rr(p)) for p in rr_shuf_preds]))
                rr_topk[f"rr_top{_k}_pearDE_true"] = pear(rr_s(dr_t), obs_lfc_rr, tm)
                rr_topk[f"rr_top{_k}_pearDE_rev"]  = pear(rr_s(dr_r), obs_lfc_rr, tm)
                rr_topk[f"rr_top{_k}_pearDE_shuf"] = float(np.nanmean([pear(rr_s(p), obs_lfc_rr, tm) for p in rr_shuf_preds]))

            # ---------- RP: perturb ONLY W_rp ----------
            if has_p:
                dp_t = do_op(model, eps, A_full,   tgi[g], lr, lp, mutilate=MUT)[1]
                dp_r = do_op(model, eps, rp_rev_A, tgi[g], lr, lp, mutilate=MUT)[1]
                dp_0 = do_op(model, eps, rp_zero_A, tgi[g], lr, lp, mutilate=MUT)[1]
                rp_shuf_preds = [do_op(model, eps, Ap, tgi[g], lr, lp, mutilate=MUT)[1] for Ap in rp_shuf_As]
                rp_true = auroc(labp.astype(int), rp(dp_t)); rp_rev = auroc(labp.astype(int), rp(dp_r))
                rp_zero = auroc(labp.astype(int), rp(dp_0))
                rp_shuf = np.nanmean([auroc(labp.astype(int), rp(p)) for p in rp_shuf_preds])
                rp_true_ap = auprc(labp.astype(int), rp(dp_t)); rp_rev_ap = auprc(labp.astype(int), rp(dp_r))
                rp_zero_ap = auprc(labp.astype(int), rp(dp_0))
                rp_shuf_ap = np.nanmean([auprc(labp.astype(int), rp(p)) for p in rp_shuf_preds])
                rp_pearDE_true = pear(rp_s(dp_t), obs_lfc_rp, de_p)
                rp_pearDE_rev  = pear(rp_s(dp_r), obs_lfc_rp, de_p)
                rp_pearDE_zero = pear(rp_s(dp_0), obs_lfc_rp, de_p)
                rp_pearDE_shuf = np.nanmean([pear(rp_s(p), obs_lfc_rp, de_p) for p in rp_shuf_preds])
            else:
                rp_true = rp_rev = rp_zero = rp_shuf = np.nan
                rp_true_ap = rp_rev_ap = rp_zero_ap = rp_shuf_ap = np.nan
                rp_pearDE_true = rp_pearDE_rev = rp_pearDE_zero = rp_pearDE_shuf = np.nan

            rows.append(dict(
                g=g,
                # RR AUROC
                rr_true=rr_true, rr_rev=rr_rev, rr_shuf=rr_shuf, rr_zero=rr_zero,
                rr_true_ap=rr_true_ap, rr_rev_ap=rr_rev_ap, rr_shuf_ap=rr_shuf_ap, rr_zero_ap=rr_zero_ap,
                # RR pearDE
                rr_pearDE_true=rr_pearDE_true, rr_pearDE_rev=rr_pearDE_rev,
                rr_pearDE_shuf=rr_pearDE_shuf, rr_pearDE_zero=rr_pearDE_zero,
                # RP AUROC
                rp_true=rp_true, rp_rev=rp_rev, rp_shuf=rp_shuf, rp_zero=rp_zero,
                rp_true_ap=rp_true_ap, rp_rev_ap=rp_rev_ap, rp_shuf_ap=rp_shuf_ap, rp_zero_ap=rp_zero_ap,
                # RP pearDE
                rp_pearDE_true=rp_pearDE_true, rp_pearDE_rev=rp_pearDE_rev,
                rp_pearDE_shuf=rp_pearDE_shuf, rp_pearDE_zero=rp_pearDE_zero,
                # RR strict-gating (top-k DEGs)
                **rr_topk,
            ))

    import pandas as pd
    df = pd.DataFrame(rows)
    summ = {f"mean_{c}": float(df[c].mean()) for c in df.columns if c != "g"}
    summ |= {f"sem_{c}": float(df[c].std(ddof=1) / np.sqrt(df[c].notna().sum())) for c in df.columns if c != "g"}
    summ["n"] = len(df)
    summ["mutilate"] = bool(MUT)

    tag = os.environ.get("CMC_DS_TAG", "frangieh")
    # per-cohort file next to the model
    json.dump({"summary": summ, "per_perturbation": rows},
              open(os.path.join(rd, f"edge_reversal_{tag}.json"), "w"), indent=2)
    # merge into the aggregate deliverable results/edge_reversal.json under the cohort key
    repo = _CMC_ROOT
    agg_path = os.path.join(repo, "results", "edge_reversal.json")
    agg = {}
    if os.path.exists(agg_path):
        try: agg = json.load(open(agg_path))
        except Exception: agg = {}
    agg.setdefault("_meta", {"model": os.path.basename(rd), "K_shuffles": K,
                             "variants": ["true", "rev(reversed)", "shuf(random-edge-swap control)", "zero"],
                             "note": "REAL vs REVERSED vs SHUFFLED vs ZEROED, block-isolated mutilation do-op read-out."})
    agg[tag] = {"model": os.path.basename(rd), "n": len(df), "summary": summ, "per_perturbation": rows}
    json.dump(agg, open(agg_path, "w"), indent=2)

    f = lambda k: (summ.get(f"mean_{k}", np.nan), summ.get(f"sem_{k}", np.nan))
    print(f"\n=== EDGE-REVERSAL ablation: {os.path.basename(rd)} / {tag}  (n={len(df)}, mut={MUT}) ===")
    print("  RR (A_rr) AUROC:   REAL {:.3f}±{:.3f}   REVERSED {:.3f}±{:.3f}   SHUFFLED {:.3f}±{:.3f}   ZEROED {:.3f}±{:.3f}".format(
        *f('rr_true'), *f('rr_rev'), *f('rr_shuf'), *f('rr_zero')))
    print("     -> Delta(REAL-REVERSED)={:+.3f}   Delta(REAL-SHUFFLED)={:+.3f}".format(
        f('rr_true')[0] - f('rr_rev')[0], f('rr_true')[0] - f('rr_shuf')[0]))
    print("  RR (A_rr) pearDE:  REAL {:.3f}±{:.3f}   REVERSED {:.3f}±{:.3f}   SHUFFLED {:.3f}±{:.3f}   ZEROED {:.3f}±{:.3f}".format(
        *f('rr_pearDE_true'), *f('rr_pearDE_rev'), *f('rr_pearDE_shuf'), *f('rr_pearDE_zero')))
    print("     -> Delta(REAL-REVERSED)={:+.3f}   Delta(REAL-SHUFFLED)={:+.3f}".format(
        f('rr_pearDE_true')[0] - f('rr_pearDE_rev')[0], f('rr_pearDE_true')[0] - f('rr_pearDE_shuf')[0]))
    print("  RP (W_rp) AUROC:   REAL {:.3f}±{:.3f}   REVERSED {:.3f}±{:.3f}   SHUFFLED {:.3f}±{:.3f}   ZEROED {:.3f}±{:.3f}".format(
        *f('rp_true'), *f('rp_rev'), *f('rp_shuf'), *f('rp_zero')))
    print("     -> Delta(REAL-REVERSED)={:+.3f}   Delta(REAL-SHUFFLED)={:+.3f}".format(
        f('rp_true')[0] - f('rp_rev')[0], f('rp_true')[0] - f('rp_shuf')[0]))
    print("  RP (W_rp) pearDE:  REAL {:.3f}±{:.3f}   REVERSED {:.3f}±{:.3f}   SHUFFLED {:.3f}±{:.3f}   ZEROED {:.3f}±{:.3f}".format(
        *f('rp_pearDE_true'), *f('rp_pearDE_rev'), *f('rp_pearDE_shuf'), *f('rp_pearDE_zero')))
    print("     -> Delta(REAL-REVERSED)={:+.3f}   Delta(REAL-SHUFFLED)={:+.3f}".format(
        f('rp_pearDE_true')[0] - f('rp_pearDE_rev')[0], f('rp_pearDE_true')[0] - f('rp_pearDE_shuf')[0]))
    print("DONE")


if __name__ == "__main__":
    main()
