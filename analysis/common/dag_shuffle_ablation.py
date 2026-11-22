#!/usr/bin/env python
"""DAG-shuffle ablation (eval-time, no retraining) on a trained model.
Tests what the perturbation DEG-AUROC depends on:

  do-operation  Δx̂ = D (I-A)^-1 Dᵀ δx   uses the LATENT DAG A (and decoder D), NOT the projected
  feature-space graph G = D A Dᵀ (which is a derived readout).

So we compare, per held-out Frangieh perturbation (RR and RP DE-ranking AUROC):
  1. true_A       : the real do-operation (reference)
  2. shuffled_A   : permute A's off-diagonal (K perms) -> do-op with a randomized LATENT DAG
  3. zero_A       : A=0 (decoder-only, (I-A)^-1=I)
And RR only, two predictors built FROM the projected graph G (to show G-shuffle's effect):
  4. G_predict    : rank responders of gene g by |G[g, :]|  (uses the projected DAG)
  5. G_shuffled   : same but permute G's off-diagonal (K perms)

Expectation: shuffling the LATENT A changes the do-op AUROC; shuffling the PROJECTED G does NOT
affect the do-op (it's outside the prediction path) but DOES destroy the G-based predictor.

Usage: python dag_shuffle_ablation.py <results_dir>
"""
import sys, os, json, warnings
import numpy as np, torch
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perturb_validation_analysis as P
DEVICE = P.DEVICE; K = 5


def auroc(lab, score):
    lab = lab.astype(int)
    if lab.sum() < 3 or lab.sum() == len(lab): return np.nan
    try: return roc_auc_score(lab, score)
    except Exception: return np.nan


def shuffle_offdiag(M, gen):
    """Permute the off-diagonal entries of a square matrix (preserve count/value dist, randomize positions)."""
    M = M.clone(); n = M.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=M.device)
    vals = M[off]; perm = torch.randperm(vals.numel(), generator=gen, device=M.device if M.device.type == "cpu" else None)
    if M.device.type != "cpu":
        perm = torch.randperm(vals.numel()).to(M.device)
    M[off] = vals[perm]
    return M


def do_op(model, eps_joint, A_full, target_gene_idx, lr, lp, mutilate=False):
    """Projected do-operation with an injected A_full. Returns (Δ_rna, Δ_prot) over all features.
    Propagation uses the model's structural solver (_scm_solve): linear (I-A^T)^-1 or nonlinear
    z=phi(A^T z)+eps fixed-point, so the shuffle/zero-A test is exact for nonlinear models too.
    mutilate=True: proper soft do() — scale the incoming edges to the gene-g latents by (1-s_j),
    s_j=|Wr[g,j]|/max, before the PERTURBED solve (control keeps the full graph)."""
    Wr = model.dec_rna.weight
    z_ctrl = model.scm._scm_solve(A_full, eps_joint)
    zr_c, zp_c = torch.split(z_ctrl, [lr, lp], 1)
    rec_r_c = model.dec_rna(zr_c) + model.bias_rna
    dx = torch.zeros_like(rec_r_c); dx[:, target_gene_idx] = -rec_r_c[:, target_gene_idx]
    deps = torch.cat([dx @ Wr, torch.zeros_like(zp_c)], 1)
    # ε-bottleneck: the KO perturbation enters via the (gated) exogenous input, so gate deps too — else
    # it would leak into endogenous (gate-closed) latents WITHOUT A, falsely showing A as non-load-bearing.
    if getattr(model, 'eps_bottleneck', False):
        with torch.no_grad():
            _g, _ = model._eps_gate()
        deps = deps * _g.to(deps.device).unsqueeze(0)
    A_pert = A_full
    if mutilate:
        s = (Wr[target_gene_idx].abs() / (Wr[target_gene_idx].abs().max() + 1e-8)).clamp(0, 1)  # [lr]
        A_pert = A_full.clone(); A_pert[:, :lr] = A_pert[:, :lr] * (1 - s).unsqueeze(0)  # cut incoming to gene-g latents
    z_p = model.scm._scm_solve(A_pert, eps_joint + deps)
    zr_p, zp_p = torch.split(z_p, [lr, lp], 1)
    d_r = (model.dec_rna(zr_p) + model.bias_rna) - rec_r_c
    d_p = (model.dec_prot(zp_p) + model.bias_prot) - (model.dec_prot(zp_c) + model.bias_prot)
    return d_r, d_p


def main():
    rd = os.path.abspath(sys.argv[1]); cfg = json.load(open(os.path.join(rd, "config.json")))
    perturb_rna, perturb_prot, align = P.load_shared_data(cfg["rna_path"], cfg["prot_path"])
    de = P.compute_observed_de(perturb_rna, perturb_prot, align)
    from model import CrossModalCausalLDVAE
    model = P.load_ctx_model(CrossModalCausalLDVAE, rd, len(align["train_gene_names"]), len(align["train_prot_names"]), cfg)
    eps_r, eps_p, _, _, _ = P.encode_control_cells(model, perturb_rna, perturb_prot, align)
    eps = torch.cat([eps_r, eps_p], 1).to(DEVICE)[:2000]
    lr, lp = model.latent_r, model.latent_p
    with torch.no_grad():
        A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
        A_full = torch.cat([torch.cat([A_rr, W_rp], 1), torch.cat([W_pr, A_pp], 1)], 0).to(DEVICE)
        Dr = model.dec_rna.weight.detach()
        G = (Dr @ A_rr @ Dr.T).cpu().numpy()                         # projected GRN [n_genes x n_genes]
    g2c = align["train_gene_to_idx_common"]; common = align["common_genes"]
    reti = align["rna_eval_train_indices"]; tgi = align["train_gene_idx"]
    gen = torch.Generator().manual_seed(0); rng = np.random.default_rng(0)
    A_perms = [shuffle_offdiag(A_full.cpu(), gen).to(DEVICE) for _ in range(K)]
    Goff = ~np.eye(G.shape[0], dtype=bool)

    rows = []
    with torch.no_grad():
        for g, d in de.items():
            if g not in tgi or g not in g2c: continue
            keep = np.array([cg != g for cg in common]); lab = d["sig_rna"][keep].astype(int)
            if lab.sum() < 3: continue
            labp = d.get("sig_prot", np.array([]));
            def rr(dr): return np.abs(dr.mean(0).cpu().numpy()[reti])[keep]
            def rp(dp): return np.abs(dp.mean(0).cpu().numpy()[align["prot_eval_train_indices"]])
            dr_t, dp_t = do_op(model, eps, A_full, tgi[g], lr, lp)
            dr_0, dp_0 = do_op(model, eps, torch.zeros_like(A_full), tgi[g], lr, lp)
            rr_sh = np.nanmean([auroc(lab, rr(do_op(model, eps, Ap, tgi[g], lr, lp)[0])) for Ap in A_perms])
            rp_sh = np.nanmean([auroc(labp.astype(int), rp(do_op(model, eps, Ap, tgi[g], lr, lp)[1])) for Ap in A_perms]) if len(labp) and labp.sum() >= 3 else np.nan
            # G-based predictor (uses projected DAG)
            gG = np.abs(G[tgi[g], :][reti])[keep]
            gG_sh = []
            for _ in range(K):
                Gs = G.copy(); v = Gs[Goff]; Gs[Goff] = rng.permutation(v); gG_sh.append(auroc(lab, np.abs(Gs[tgi[g], :][reti])[keep]))
            r = {"g": g,
                 "rr_true": auroc(lab, rr(dr_t)), "rr_shuffA": rr_sh, "rr_zeroA": auroc(lab, rr(dr_0)),
                 "rr_Gpredict": auroc(lab, gG), "rr_Gshuffle": float(np.nanmean(gG_sh)),
                 "rp_true": auroc(labp.astype(int), rp(dp_t)) if len(labp) and labp.sum() >= 3 else np.nan,
                 "rp_shuffA": rp_sh, "rp_zeroA": auroc(labp.astype(int), rp(dp_0)) if len(labp) and labp.sum() >= 3 else np.nan}
            rows.append(r)
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = {f"mean_{c}": float(df[c].mean()) for c in df.columns if c != "g"}
    summ |= {f"sem_{c}": float(df[c].std(ddof=1) / np.sqrt(df[c].notna().sum())) for c in df.columns if c != "g"}
    summ["n"] = len(df)
    json.dump({"summary": summ, "per_perturbation": rows}, open(os.path.join(rd, "dag_shuffle_ablation.json"), "w"), indent=2)
    f = lambda k: (summ.get(f"mean_{k}", np.nan), summ.get(f"sem_{k}", np.nan))
    print(f"\n=== DAG-shuffle ablation: {os.path.basename(rd)}  (n={len(df)}) ===")
    print("  -- do-operation (uses LATENT A) --                  RR              RP")
    for k, lab in [("true", "true A (reference)"), ("shuffA", "shuffled latent A"), ("zeroA", "A=0 (decoder only)")]:
        print(f"    {lab:24s} RR {f('rr_'+k)[0]:.3f}±{f('rr_'+k)[1]:.3f}   RP {f('rp_'+k)[0]:.3f}±{f('rp_'+k)[1]:.3f}")
    print("  -- predictor FROM projected G (RR) --")
    print(f"    G_predict (|G[g,:]|)     {f('rr_Gpredict')[0]:.3f}±{f('rr_Gpredict')[1]:.3f}")
    print(f"    G_shuffled               {f('rr_Gshuffle')[0]:.3f}±{f('rr_Gshuffle')[1]:.3f}")
    print("\nNOTE: do-op AUROC is invariant to shuffling the PROJECTED G (G is not in the do-op path);")
    print("      shuffling the LATENT A changes it. The G_predict vs G_shuffle contrast shows G itself carries the signal.")


if __name__ == "__main__":
    main()
