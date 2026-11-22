"""Projected-GRN topology: does G_rr = D_r A_rr D_r^T look like a real DIRECTED regulatory network?
Density, in/out-degree, per-node degree asymmetry, reciprocity vs degree-preserving shuffle, hub genes,
and whether TFs (CollecTRI) are out-hubs (out/in ratio) -- evidence for/against recovered direction.
CPU. Writes results/graph_topology.json + results/figures/graph_topology.png."""
import os, sys, json, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "analysis/common")
import torch, scanpy as sc
from scipy.stats import hypergeom
import perturb_validation_analysis as P
from model import CrossModalCausalLDVAE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RD = "results/linfb_2c_fb10_b20_v4"; cfg = json.load(open(f"{RD}/config.json"))

# ---- load model (CPU) + projected GRN ----
sd = torch.load(P.resolve_ckpt(RD), map_location="cpu"); NG, NP = sd["dec_rna.weight"].shape[0], sd["dec_prot.weight"].shape[0]; del sd
model = P.load_ctx_model(CrossModalCausalLDVAE, RD, NG, NP, cfg); model.eval()
with torch.no_grad():
    A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies(tau=0.5)
    Dr = model._dec_r_w().detach().float()
    G = (Dr @ A_rr.float() @ Dr.T).cpu().numpy()
np.fill_diagonal(G, 0.0)
ng = G.shape[0]
genes = list(sc.read_h5ad(cfg["rna_path"], backed="r").var_names.astype(str))
ref = np.load("results/ref_nets_v4_2c.npz", allow_pickle=True)
assert list(ref["genes"]) == genes, "gene order mismatch"
coexpr = ref["coexpr"]  # symmetric bool
tf_mask = np.asarray(ref["CollecTRI"]).any(axis=1)   # TFs = source nodes in CollecTRI directed prior
n_tf = int(tf_mask.sum())
Drn = np.linalg.norm(Dr.cpu().numpy(), axis=1)       # decoder row norm per gene (magnitude confound proxy)

def topk_binary(M, frac):
    a = np.abs(M).copy(); np.fill_diagonal(a, 0.0)
    thr = np.quantile(a[a > 0], 1 - frac)
    return a >= thr  # directed binary (row=source, col=target)

rng = np.random.RandomState(0)
res = {"n_genes": ng, "n_TF": n_tf, "levels": {}}
for frac in [0.001, 0.005, 0.01]:
    B = topk_binary(G, frac)                      # B[i,j]=1 -> edge i->j
    outdeg = B.sum(1); indeg = B.sum(0)
    dens = float(B.mean())
    recip = float((B & B.T).sum() / max(B.sum(), 1))            # fraction of edges that are reciprocated
    # degree-preserving shuffle reciprocity (configuration-ish: shuffle targets within row)
    recips_null = []
    for _ in range(20):
        Bs = np.zeros_like(B)
        for i in range(ng):
            k = int(B[i].sum())
            if k: Bs[i, rng.choice(ng, k, replace=False)] = 1
        recips_null.append((Bs & Bs.T).sum() / max(Bs.sum(), 1))
    recip_null = float(np.mean(recips_null))
    # per-node degree asymmetry
    tot = outdeg + indeg; asym = np.abs(outdeg - indeg) / np.maximum(tot, 1)
    # TF out/in
    oi = outdeg / np.maximum(indeg, 1)
    tf_oi = float(np.median(oi[tf_mask & (tot > 0)])); nt_oi = float(np.median(oi[~tf_mask & (tot > 0)]))
    # hub-TF enrichment: top-N out-degree hubs, how many are TFs vs hypergeom
    N = 50; out_hubs = np.argsort(outdeg)[::-1][:N]; in_hubs = np.argsort(indeg)[::-1][:N]
    k_tf_out = int(tf_mask[out_hubs].sum()); k_tf_in = int(tf_mask[in_hubs].sum())
    p_out = float(hypergeom.sf(k_tf_out - 1, ng, n_tf, N)); p_in = float(hypergeom.sf(k_tf_in - 1, ng, n_tf, N))
    # coexpr (symmetric) comparison at matched density
    Bc = topk_binary(coexpr.astype(float) + 1e-9 * rng.rand(ng, ng), frac)  # coexpr is bool; break ties
    recip_coexpr = float((Bc & Bc.T).sum() / max(Bc.sum(), 1))
    # magnitude confound: is out-degree (or out/in) just explained by decoder-row-norm?
    m = tot > 0
    conf_out = float(np.corrcoef(outdeg[m], Drn[m])[0, 1])
    conf_oi = float(np.corrcoef(np.log2(oi[m] + 1e-3), np.log2(Drn[m] + 1e-9))[0, 1])
    res["levels"][f"{frac}"] = dict(
        density=dens, reciprocity=recip, reciprocity_shuffle_null=recip_null, reciprocity_coexpr=recip_coexpr,
        median_node_asymmetry=float(np.median(asym[tot > 0])),
        TF_out_in_ratio=tf_oi, nonTF_out_in_ratio=nt_oi,
        corr_outdeg_decnorm=conf_out, corr_outin_decnorm=conf_oi,
        top50_out_TF_count=k_tf_out, top50_out_TF_p=p_out, expected_TF_in_top50=float(N * n_tf / ng),
        top50_in_TF_count=k_tf_in, top50_in_TF_p=p_in,
        top_out_hubs=[genes[i] for i in out_hubs[:15]], top_in_hubs=[genes[i] for i in in_hubs[:15]])
    print(f"[frac {frac}] dens {dens:.4f} recip {recip:.3f} (null {recip_null:.3f}, coexpr {recip_coexpr:.3f}) "
          f"TF out/in {tf_oi:.2f} vs nonTF {nt_oi:.2f} | top50-out TFs {k_tf_out} (exp {N*n_tf/ng:.1f}, p {p_out:.1e})", flush=True)

# ---- figure ----
fr = "0.005"; B = topk_binary(G, 0.005); outdeg = B.sum(1); indeg = B.sum(0)
fig, ax = plt.subplots(1, 3, figsize=(9, 2.8))
ax[0].hist(np.log10(outdeg[outdeg > 0]), bins=30, color="#2b6cb0", alpha=0.7, label="out")
ax[0].hist(np.log10(indeg[indeg > 0]), bins=30, color="#dd6b20", alpha=0.6, label="in")
ax[0].set_xlabel("log10 degree"); ax[0].set_ylabel("# genes"); ax[0].legend(); ax[0].set_title("degree distribution")
oi = outdeg / np.maximum(indeg, 1); tot = outdeg + indeg
ax[1].boxplot([np.log2(oi[tf_mask & (tot > 0)] + 1e-3), np.log2(oi[~tf_mask & (tot > 0)] + 1e-3)],
              tick_labels=["TF", "non-TF"], showfliers=False)
ax[1].set_ylabel("log2 out/in degree"); ax[1].set_title("TF out/in vs non-TF"); ax[1].axhline(0, ls=":", c="gray")
lv = res["levels"]; xs = list(lv.keys())
ax[2].plot(range(len(xs)), [lv[k]["reciprocity"] for k in xs], "-o", label="G_rr", color="#2b6cb0")
ax[2].plot(range(len(xs)), [lv[k]["reciprocity_shuffle_null"] for k in xs], "-s", label="shuffle null", color="#a0aec0")
ax[2].plot(range(len(xs)), [lv[k]["reciprocity_coexpr"] for k in xs], "-^", label="coexpr", color="#dd6b20")
ax[2].set_xticks(range(len(xs))); ax[2].set_xticklabels(xs); ax[2].set_xlabel("edge fraction")
ax[2].set_ylabel("reciprocity"); ax[2].set_title("reciprocity"); ax[2].legend(fontsize=6)
plt.tight_layout(); os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/graph_topology.png", dpi=200, bbox_inches="tight")
json.dump(res, open("results/graph_topology.json", "w"), indent=2)
print("WROTE results/graph_topology.json + figure")
