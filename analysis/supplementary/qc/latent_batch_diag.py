"""Does batch structure survive into the MODEL LATENT space? The input-space PCA (pca_umap_cohorts.py)
shows datasets separate — that is expected (different CITE-seq experiments). The question that matters
for A_rr / decoder-Gram is whether the encoder embedding (eps = [mu_r, mu_p]) is still batch-separated.
We quantify dataset/donor silhouette in INPUT PCA vs LATENT PCA for both cohorts (final checkpoints,
Lin+fb) and plot UMAPs. Lower latent silhouette => the model mixes batches despite no ComBat."""
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
sys.path.insert(0, "src"); sys.path.insert(0, "analysis/common")
import numpy as np, torch, scanpy as sc, anndata as ad
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from model import CrossModalCausalLDVAE
from perturb_validation_analysis import load_ctx_model, DEVICE

DATA = f"{_CMC_DATA}"
COH = [("2c (Hao+Steph)", "linfb_2c_fb10_b20_v4", f"{DATA}/combined_v4"),
       ("2c+COMBAT (3 datasets)", "linfb_2ccombat_fb10_b20_v4", f"{DATA}/combined_combat_v4")]
N = 15000; SEED = 42; SIL_SAMPLE = 5000

def sil(P, lab):
    return silhouette_score(P[:, :10], lab, sample_size=min(SIL_SAMPLE, len(lab)), random_state=0)

fig, axs = plt.subplots(2, 3, figsize=(18, 11))
summary = []
for ri, (coh, mdir, d) in enumerate(COH):
    cfg = json.load(open(f"results/{mdir}/config.json"))
    ar = sc.read_h5ad(f"{d}/rna_combined.h5ad", backed="r")
    ap = sc.read_h5ad(f"{d}/citeseq_combined.h5ad", backed="r")
    rng = np.random.RandomState(SEED); idx = np.sort(rng.choice(ar.n_obs, min(N, ar.n_obs), replace=False))
    sr = ar[idx].to_memory(); sp = ap[idx].to_memory(); ar.file.close(); ap.file.close()
    Xr = sr.X.toarray() if hasattr(sr.X, "toarray") else np.asarray(sr.X)
    Xp = sp.X.toarray() if hasattr(sp.X, "toarray") else np.asarray(sp.X)
    ds = sr.obs["dataset"].astype(str).values; dn = sr.obs["donor"].astype(str).values
    # global z-score = encoder input convention used in training
    Zr = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-6); Zp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-6)
    m = load_ctx_model(CrossModalCausalLDVAE, f"results/{mdir}", Xr.shape[1], Xp.shape[1], cfg)
    with torch.no_grad():
        mu_r, _ = m.enc_rna(torch.FloatTensor(Zr).to(DEVICE))
        mu_p, _ = m.enc_prot(torch.FloatTensor(Zp).to(DEVICE))
    eps = torch.cat([mu_r, mu_p], 1).cpu().numpy()
    Pin = PCA(30, random_state=0).fit_transform(Zr)
    Plat = PCA(min(30, eps.shape[1]), random_state=0).fit_transform(eps)
    s_in, s_lat = sil(Pin, ds), sil(Plat, ds)
    s_in_dn, s_lat_dn = sil(Pin, dn), sil(Plat, dn)
    summary.append((coh, s_in, s_lat, s_in_dn, s_lat_dn))
    print(f"{coh}: INPUT dataset-sil={s_in:.3f} -> LATENT dataset-sil={s_lat:.3f} | "
          f"INPUT donor-sil={s_in_dn:.3f} -> LATENT donor-sil={s_lat_dn:.3f}", flush=True)
    A = ad.AnnData(eps.astype(np.float32)); A.obs["dataset"] = ds
    sc.pp.neighbors(A, use_rep="X", n_neighbors=15); sc.tl.umap(A); U = A.obsm["X_umap"]
    for g in np.unique(ds):
        mk = ds == g
        axs[ri, 0].scatter(Pin[mk, 0], Pin[mk, 1], s=3, alpha=.4, label=g)
        axs[ri, 1].scatter(Plat[mk, 0], Plat[mk, 1], s=3, alpha=.4, label=g)
        axs[ri, 2].scatter(U[mk, 0], U[mk, 1], s=3, alpha=.4, label=g)
    axs[ri, 0].set_title(f"{coh}\nINPUT PCA by dataset (sil={s_in:.2f})", fontweight="bold", fontsize=11)
    axs[ri, 1].set_title(f"{coh}\nLATENT PCA by dataset (sil={s_lat:.2f})", fontweight="bold", fontsize=11)
    axs[ri, 2].set_title(f"{coh}\nLATENT UMAP by dataset", fontweight="bold", fontsize=11)
    for c in range(3):
        axs[ri, c].legend(markerscale=3, fontsize=8); axs[ri, c].set_xticks([]); axs[ri, c].set_yticks([])
fig.suptitle("Batch structure — INPUT space vs MODEL LATENT (encoder eps).  silhouette>0 = datasets separate; "
             "lower LATENT silhouette = the encoder mixes batches (no ComBat used)", fontweight="bold", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/latent_batch_diag.png", dpi=130)
print("\n=== SUMMARY (dataset silhouette: input -> latent) ===")
for coh, si, sl, sidn, sldn in summary:
    print(f"  {coh:26s} dataset {si:+.3f} -> {sl:+.3f}   donor {sidn:+.3f} -> {sldn:+.3f}")
print("WROTE results/figures/latent_batch_diag.png")
