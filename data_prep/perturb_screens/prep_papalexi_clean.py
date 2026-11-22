#!/usr/bin/env python
"""Labelled deviation (post-registration, requested by the orchestrator after the Frangieh control-pool issue):
'clean-control' Papalexi set. Removes cells whose guide (GDO) counts indicate a second guide: second-highest guide
>=5 UMIs AND >=10% of the top guide's UMIs. Controls = NT-guide cells without a second guide; KO cells = single
targeting guide. All cells are IFN-gamma-treated ('tx' HTO), so controls are condition-matched.
Usage: python -I prep_clean.py <in.h5mu> <proc_dir>"""
import sys, json, numpy as np, mudata as md, anndata as ad
IN, PD = sys.argv[1], sys.argv[2]
m = md.read_h5mu(IN); g = m["gdo"]; X = g.X.toarray(); o = m["rna"].obs
names = np.array(g.var_names); s = np.sort(X, 1); top = names[X.argmax(1)]
assert (top == o["guide_ID"].astype(str).values).all()
multi = (s[:, -2] >= 5) & (s[:, -2] >= 0.1 * s[:, -1])
keep_bc = set(np.array(m["rna"].obs_names)[~multi])
assert (o["HTO_classification"].astype(str).str.endswith("-tx")).all()
info = dict(n_total=int(len(multi)), n_multi_removed=int(multi.sum()),
            n_NT_kept=int(((o.gene_target.astype(str) == "NT").values & ~multi).sum()))
for k in ["rna", "prot"]:
    a = ad.read_h5ad(f"{PD}/papalexi_{k}.h5ad"); a = a[[b in keep_bc for b in a.obs_names]].copy(); a.write(f"{PD}/papalexi_clean_{k}.h5ad")
    info[f"n_{k}"] = int(a.n_obs)
json.dump(info, open(f"{PD}/prep_clean_info.json", "w"), indent=1); print(info)
