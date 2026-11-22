"""How much do per-feature standardization stats change if fit on the 80% training cells only
(the model fits them on all cells before the 80/10/10 split)? Same split call as src/train.py."""
import json, sys, numpy as np, scanpy as sc
from sklearn.model_selection import train_test_split
cfg = json.load(open(sys.argv[1] + "/config.json"))
out = {}
for mod, path in [("rna", cfg["rna_path"]), ("prot", cfg["prot_path"])]:
    a = sc.read_h5ad(path); X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X); X = X.astype(np.float64)
    n = X.shape[0]
    itr, itmp = train_test_split(np.arange(n), test_size=cfg["test_size"], random_state=cfg["seed"])
    mu_a, sd_a = X.mean(0), X.std(0) + 1e-6; mu_t, sd_t = X[itr].mean(0), X[itr].std(0) + 1e-6
    dmu = np.abs(mu_t - mu_a) / sd_a; dsd = np.abs(sd_t / sd_a - 1)
    # effect on standardized values of held-out cells
    _, ite = train_test_split(itmp, test_size=0.5, random_state=cfg["seed"])
    za = (X[ite] - mu_a) / sd_a; zt = (X[ite] - mu_t) / sd_t
    out[mod] = dict(n_cells=n, n_train=len(itr), n_features=X.shape[1],
                    max_abs_dmean_in_sd=float(dmu.max()), median_abs_dmean_in_sd=float(np.median(dmu)),
                    max_rel_dsd=float(dsd.max()), median_rel_dsd=float(np.median(dsd)),
                    test_z_max_abs_diff=float(np.abs(za - zt).max()), test_z_rmse_diff=float(np.sqrt(((za - zt) ** 2).mean())))
    print(mod, out[mod], flush=True); del X, a
json.dump(out, open("results/reviewer_controls/std_leakage_check.json", "w"), indent=1)
