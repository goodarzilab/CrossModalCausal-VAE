"""Archived runs with real latent-graph sparsity: does sparsity make A_rr/G_rr 'directed', and is that
direction correct on CollecTRI/DoRothEA?  Compared with the headline model and DeepSEM."""
import json, glob, os, numpy as np, torch
REF = np.load("results/ref_nets_v4_2c.npz", allow_pickle=True)
REFS = {k: REF[k].astype(bool) for k in ["CollecTRI", "DoRothEA"]}
def orient(G):
    out = {}
    for nm, R in REFS.items():
        ii, jj = np.where(R); k = ii != jj; f, r = np.abs(G[ii[k], jj[k]]), np.abs(G[jj[k], ii[k]]); m = (f + r) > 1e-12
        out[nm] = (float(np.mean(f[m] > r[m])), float(np.median(f[m]) / (np.median(r[m]) + 1e-12)))
    return out
def onesided(M, q=0.99):
    a = np.abs(M); iu = np.triu_indices(len(M), 1); s = a[iu] + a.T[iu]; m = s >= np.quantile(s, q)
    return float(np.mean(np.minimum(a[iu], a.T[iu])[m] / np.maximum(a[iu], a.T[iu])[m] < 0.1))
def perturb(d):
    vals = {}
    for c in ["frangieh", "iccite", "arce"]:
        for f in [f"{d}/block_isolated_mut_{c}_ablation.json", f"{d}/full_perturb_eval_{c}.json"]:
            if os.path.exists(f):
                s = json.load(open(f)); s = s.get("summary", s)
                vals[c] = (s.get("mean_rr_true", s.get("rr_auroc")), s.get("mean_rr_pearDE_true", s.get("rr_pearDE"))); break
    return vals
runs = ["results/sweep_sparse/sp_dec_gl1000"] + sorted(glob.glob("results/sweep_sparse/sp_lat_*")) + sorted(glob.glob("results/sweep_sparse/so_latonly_*")) + sorted(glob.glob("results/sweep_sparse/sp_comb_*")) + ["results/sweep_sparse/sp_lat_dec"]
seen = set(); out = {}
for d in runs:
    if d in seen or not os.path.exists(f"{d}/final_model.pt"): continue
    seen.add(d); c = json.load(open(f"{d}/config.json")); sd = torch.load(f"{d}/final_model.pt", map_location="cpu")
    A = (torch.sigmoid(sd["scm.logits_rr"]) * sd["scm.weights_rr"] * sd["scm.mask_rr"]).float().numpy()
    D = sd["dec_rna.weight"].float().numpy(); G = D @ A @ D.T; np.fill_diagonal(G, 0)
    off = ~np.eye(len(A), dtype=bool); gi = np.triu_indices(len(G), 1)
    g = torch.sigmoid(sd["scm.logits_rr"]).float().numpy()[off]
    rec = dict(lam=c.get("lambda_sparse_intra"), target=c.get("scm_sparse_target"), red=c.get("scm_sparse_reduction"),
               corr_A=float(np.corrcoef(A[off], A.T[off])[0, 1]), corr_G=float(np.corrcoef(G[gi], G.T[gi])[0, 1]),
               onesided_A=onesided(A), frac_gate_lt_0p1=float(np.mean(g < 0.1)), orient_G=orient(G), perturb=perturb(d))
    out[os.path.basename(d)] = rec
W = np.load("results/baselines/deepsem_W_809k_clamp09.npz", allow_pickle=True)["W"].astype(float); np.fill_diagonal(W, 0)
off = ~np.eye(len(W), dtype=bool)
out["DeepSEM"] = dict(corr_A=float(np.corrcoef(W[off], W.T[off])[0, 1]), onesided_A=onesided(W), orient_G=orient(W))
json.dump(out, open("results/reviewer_controls/latent_sparsity_orientation.json", "w"), indent=1)
print(f"{'run':24s} {'lam':>6s} {'corrA':>6s} {'corrG':>6s} {'1sided':>6s} {'gate<.1':>7s}  CollecTRI fwd>rev(ratio)  DoRothEA fwd>rev(ratio)  RR AUROC / pearDE (F,C,A)")
for k, r in out.items():
    o = r["orient_G"]; pt = r.get("perturb", {})
    pts = " ".join(f"{(v[0] or float('nan')):.3f}/{(v[1] or float('nan')):+.3f}" for v in pt.values()) if pt else ""
    print(f"{k:24s} {str(r.get('lam','')):>6s} {r['corr_A']:6.3f} {r.get('corr_G',float('nan')):6.3f} {r['onesided_A']:6.2f} {r.get('frac_gate_lt_0p1',float('nan')):7.2f}  "
          f"{o['CollecTRI'][0]:.3f} ({o['CollecTRI'][1]:.2f})            {o['DoRothEA'][0]:.3f} ({o['DoRothEA'][1]:.2f})           {pts}")
