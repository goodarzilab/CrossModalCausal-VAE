"""Shared loaders for review2/fig3. DE realisation copied verbatim (logic) from analysis/fig3_zero_shot/stats/boot.py::realise."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, pickle
import numpy as np, scipy.sparse as sp
SD = f"{_CMC_ROOT}/analysis/fig3_zero_shot/stats"
sys.path.insert(0, SD)
from pblimma import limma_ko, bh
import score_lib as SL
ST = f"{_CMC_RESULTS}/paper/analyses/stats"
RUNS = f"{_CMC_SCRATCH}/nov_revisions_offload/stats/runs"
OUTD = f"{_CMC_RESULTS}/paper/analyses/review2/fig3"
TAGS = ["frangieh", "iccite", "arce"]
SEEDS = {"s42": f"{RUNS}/rc_sp_dec_gl1000_pb_bh_preds.npz", "s1": f"{RUNS}/rc_sp_dec_gl1000_s1_pb_bh_preds.npz",
         "s2": f"{RUNS}/rc_sp_dec_gl1000_s2_pb_bh_preds.npz"}
PB = {t: pickle.load(open(f"{ST}/de/pb_{t}.pkl", "rb")) for t in TAGS}
NPZ = {s: dict(np.load(f, allow_pickle=True)) for s, f in SEEDS.items()}
COG = json.load(open(f"{_CMC_RESULTS}/baselines/_cmcvae_cognate.json"))
cog_map = {p: g for p, g in zip(COG["protein_names"], COG["cognate_gene"]) if g}

def getset(npz, name, t):
    k = f"{name}|{t}|genes"
    if k not in npz: return None
    return {g: (r.astype(float), p.astype(float)) for g, r, p in zip(npz[k], npz[f"{name}|{t}|rna"], npz[f"{name}|{t}|prot"])}

M0 = NPZ["s42"]
META = {t: dict(sd_r=M0[f"meta|{t}|sd_r"], sd_p=M0[f"meta|{t}|sd_p"], common=M0[f"meta|{t}|common"], pnames=list(M0[f"meta|{t}|pnames"]),
                train=list(M0[f"meta|{t}|train"]), test=list(M0[f"meta|{t}|test"])) for t in TAGS}
for t in TAGS:
    assert list(META[t]["common"]) == list(PB[t]["common_genes"])
    for s in ["s1", "s2"]:
        assert list(NPZ[s][f"meta|{t}|common"]) == list(META[t]["common"])

# prediction sets: name -> {screen -> [rep dicts]}  (single-rep sets are lists of length 1)
SETS = {}
for s, npz in NPZ.items():
    for t in TAGS:
        SETS.setdefault(f"cmc_{s}", {})[t] = [getset(npz, "obs", t)]
        SETS.setdefault(f"Arr_shuffled_{s}", {})[t] = [getset(npz, f"fig4_Arr_shuffled_{k}", t) for k in range(3)]
        SETS.setdefault(f"Arr_zeroed_{s}", {})[t] = [getset(npz, "fig4_Arr_zeroed_0", t)]
for t in TAGS:
    for b in ["corr", "totalvi_ko", "deepsem_ko"]:
        SETS.setdefault(b, {})[t] = [getset(M0, f"base_{b}", t)]

def realise(rng):
    C = {}
    for t in TAGS:
        pb = PB[t]; ct = pb["ct"]; strata = np.asarray(ct["strata"]); n_ct = len(strata)
        if rng is None:
            Yr, Yp, sc = ct["Y_r"], ct["Y_p"], strata
        else:
            sel = np.arange(n_ct); rows, cols, vals = [], [], []
            cell_ids = [np.where(ct["cell_pb"] == k)[0] for k in range(n_ct)]
            for r_, k in enumerate(sel):
                idx = cell_ids[k]; w = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)))
                nz = w > 0; rows.append(np.full(nz.sum(), r_)); cols.append(idx[nz]); vals.append(w[nz] / len(idx))
            W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(sel), len(ct["cell_pb"])))
            Yr = np.asarray(W @ ct["cells_r"]); Yp = np.asarray(W @ ct["cells_p"]) if ct["cells_p"].shape[1] else np.zeros((len(sel), 0)); sc = strata[sel]
        common = np.asarray(pb["common_genes"]); genes = [g for g in pb["de"] if g in set(common)]
        F = len(common); Pn = len(pb["prot_names"])
        L = np.zeros((len(genes), F)); Sg = np.zeros((len(genes), F), bool); Lp = np.zeros((len(genes), Pn)); Sp_ = np.zeros((len(genes), Pn), bool)
        for i, g in enumerate(genes):
            kp = pb["ko_pb"][g]
            if rng is None:
                d = pb["de"][g]; b, q, bp_, qp = d["lfc_rna"], d["q_rna"], d["lfc_prot"], d["q_prot"]
            else:
                b, p, _ = limma_ko(kp["Y_r"], np.asarray(kp["strata"]), Yr, sc); q = bh(p)
                if Pn: bp_, pp_, _ = limma_ko(kp["Y_p"], np.asarray(kp["strata"]), Yp, sc); qp = bh(pp_)
                else: bp_, qp = np.zeros(0), np.zeros(0)
            b = np.nan_to_num(b); L[i] = b
            Sg[i] = (q < 0.05) & (np.abs(b) >= SL.LFCR)
            if Pn:
                bp_ = np.nan_to_num(bp_); Lp[i] = bp_; Sp_[i] = (qp < 0.05) & (np.abs(bp_) >= SL.LFCP)
        pn = META[t]["pnames"]
        cogm = {g: np.array([cog_map.get(x) == g for x in pn], bool) for g in genes}
        C[t] = SL.Cohort(genes, common, L, Sg, Lp, Sp_, META[t]["sd_r"], META[t]["sd_p"], cogm, META[t]["train"])
    return C

def rr_auroc_set(C, reps, ev):
    """per-KO RNA AUROC (headline scorer), averaged over replicate prediction sets (as boot.py does for Fig-4 sets)."""
    out = {}
    for g in ev:
        i = C.gi[g]; k = C.keep[i]; vals = [SL.auroc(C.sig[i][k], np.abs(rp[g][0][k])) for rp in reps if g in rp]
        vals = [v for v in vals if np.isfinite(v)]
        out[g] = float(np.mean(vals)) if vals else np.nan
    return out
