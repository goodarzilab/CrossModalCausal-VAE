"""Build HEALTHY vs DISEASE subsets of the 2-cohort combined_v4 CITE-seq data for the
healthy-vs-disease CMC-VAE comparison (APPENDIX).

CRITICAL: both subsets are built by SUBSETTING combined_v4 (shared 2692-gene / 144-protein v4
vocab) so the two trained models have a node-comparable latent graph. Var order is preserved
verbatim; only cells are subset.

Split definition:
  donor->Status map is built from the RAW Stephenson h5ad (haniffa21.processed.h5ad), keyed by
  donor = "steph_" + patient_id (matching how combined_v4 donors were constructed).
  HEALTHY = all Hao cells  +  Stephenson cells whose donor Status == "Healthy".
  DISEASE = Stephenson cells whose donor Status != "Healthy" (Covid / Non_covid / LPS).

Outputs (RNA + protein h5ads, same var order as combined_v4):
  results/hd_split/healthy/{rna_combined.h5ad, citeseq_combined.h5ad}
  results/hd_split/disease/{rna_combined.h5ad, citeseq_combined.h5ad}
  results/hd_split/split_summary.json

Usage: python data_prep/vocab/hd_split_build.py
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
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad

R = f"{_CMC_ROOT}"
DATA = f"{_CMC_DATA}"
CV = f"{DATA}/combined_v4"
STEPH_RAW = f"{DATA}/stephenson_citeseq/haniffa21.processed.h5ad"
OUT = f"{_CMC_RESULTS}/hd_split"


def build_donor_status_map():
    """donor -> Status (mode per donor) from the raw Stephenson h5ad. donor = 'steph_'+patient_id."""
    print("Reading raw Stephenson obs (patient_id, Status) ...", flush=True)
    a = ad.read_h5ad(STEPH_RAW, backed="r")
    donor = "steph_" + a.obs["patient_id"].astype(str)
    status = a.obs["Status"].astype(str)
    df = pd.DataFrame({"donor": donor.values, "Status": status.values})
    a.file.close()
    dmap = df.groupby("donor")["Status"].agg(lambda s: s.value_counts().index[0])
    nun = df.groupby("donor")["Status"].nunique()
    ambiguous = sorted(nun.index[nun > 1])
    print(f"  {dmap.shape[0]} Stephenson donors; ambiguous (>1 Status) = {len(ambiguous)}", flush=True)
    print(f"  raw cell counts by Status: {df['Status'].value_counts().to_dict()}", flush=True)
    return dmap.to_dict(), ambiguous


def main():
    os.makedirs(f"{OUT}/healthy", exist_ok=True)
    os.makedirs(f"{OUT}/disease", exist_ok=True)

    dmap, ambiguous = build_donor_status_map()

    print("Loading combined_v4 RNA + protein (full) ...", flush=True)
    rna = ad.read_h5ad(f"{CV}/rna_combined.h5ad")
    prot = ad.read_h5ad(f"{CV}/citeseq_combined.h5ad")
    assert list(rna.obs_names) == list(prot.obs_names), "RNA/prot obs_names mismatch"
    print(f"  RNA {rna.shape}  PROT {prot.shape}", flush=True)

    obs = rna.obs
    dataset = obs["dataset"].astype(str).values
    donor = obs["donor"].astype(str).values
    is_hao = dataset == "hao"
    is_steph = dataset == "stephenson"

    # map every Stephenson cell to a Status; report unmapped
    status = np.array(["<unmapped>"] * len(donor), dtype=object)
    status[is_hao] = "Hao"
    steph_mapped = np.array([dmap.get(d, None) for d in donor], dtype=object)
    have = is_steph & np.array([d in dmap for d in donor])
    status[have] = steph_mapped[have]
    n_unmapped = int((is_steph & ~have).sum())
    unmapped_donors = sorted(set(donor[is_steph & ~have]))
    print(f"  Stephenson cells unmapped to a Status: {n_unmapped} "
          f"(donors: {unmapped_donors[:10]}{'...' if len(unmapped_donors) > 10 else ''})", flush=True)

    healthy_mask = is_hao | (is_steph & have & (status == "Healthy"))
    disease_mask = is_steph & have & (status != "Healthy")
    # unmapped Stephenson cells are dropped from BOTH splits

    steph_status_series = pd.Series(status[is_steph])
    summary = {
        "combined_v4": {"n_cells": int(rna.n_obs), "n_genes": int(rna.n_vars), "n_prot": int(prot.n_vars),
                        "hao_cells": int(is_hao.sum()), "steph_cells": int(is_steph.sum())},
        "donor_status_map": {"n_stephenson_donors": len(dmap), "ambiguous_donors": ambiguous,
                             "raw_status_counts": None},
        "unmapped": {"n_cells": n_unmapped, "donors": unmapped_donors},
        "stephenson_status_breakdown": steph_status_series.value_counts().to_dict(),
        "healthy": {}, "disease": {},
    }

    for name, mask in [("healthy", healthy_mask), ("disease", disease_mask)]:
        rsub = rna[mask].copy()
        psub = prot[mask].copy()
        rsub.obs["Status"] = pd.Categorical(status[mask])
        psub.obs["Status"] = pd.Categorical(status[mask])
        # sanity: var order identical to combined_v4
        assert list(rsub.var_names) == list(rna.var_names)
        assert list(psub.var_names) == list(prot.var_names)
        rpath = f"{OUT}/{name}/rna_combined.h5ad"
        ppath = f"{OUT}/{name}/citeseq_combined.h5ad"
        rsub.write_h5ad(rpath)
        psub.write_h5ad(ppath)
        info = {
            "n_cells": int(mask.sum()),
            "rna_path": rpath, "prot_path": ppath,
            "n_donors": int(pd.Series(donor[mask]).nunique()),
            "dataset_counts": pd.Series(dataset[mask]).value_counts().to_dict(),
            "status_counts": pd.Series(status[mask]).value_counts().to_dict(),
        }
        summary[name] = info
        print(f"\n[{name.upper()}] {info['n_cells']} cells, {info['n_donors']} donors")
        print(f"  datasets: {info['dataset_counts']}")
        print(f"  status:   {info['status_counts']}")
        print(f"  wrote {rpath}\n        {ppath}", flush=True)

    json.dump(summary, open(f"{OUT}/split_summary.json", "w"), indent=2, default=str)
    print(f"\nWROTE {OUT}/split_summary.json", flush=True)


if __name__ == "__main__":
    main()
