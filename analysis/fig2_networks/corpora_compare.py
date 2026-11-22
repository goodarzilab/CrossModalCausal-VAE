"""CMC-VAE vs correlation computed on the SAME training cells, across the 20 training corpora of the cohort sweep
(Supp Note 2; results/combo_*_nodag, multi_*_nodag, linfb_2c_nodag_v4, combat_only_nodag, linfb_2c_nodag_100k) + headline.
Paired comparison across corpora of regulon/STRING AUROC and cognate enrichment / top-1 specificity.
Writes analyses/baselines/corpora_compare.json"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, glob, time
import numpy as np, torch, anndata as ad
from sklearn.metrics import roc_auc_score
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
R = f"{_CMC_ROOT}"; sys.path.insert(0, f"{R}/src"); sys.path.insert(0, f"{_CMC_ROOT}/analysis/common")
from model import CrossModalCausalLDVAE
import perturb_validation_analysis as P
OUTP = f"{_CMC_RESULTS}/paper/analyses/baselines/corpora_compare.json"
ref = np.load(f"{_CMC_RESULTS}/ref_nets_v3_2c.npz", allow_pickle=True); meta = json.load(open(f"{_CMC_RESULTS}/ref_nets_v3_2c.json"))
rg = list(map(str, ref["genes"])); ridx = {g: i for i, g in enumerate(rg)}
cur = json.load(open(f"{_CMC_RESULTS}/baselines/_cmcvae_cognate.json")); cogmap = {p: g for p, g in zip(cur["protein_names"], cur["cognate_gene"]) if g}

RUNS = sorted(glob.glob(f"{_CMC_RESULTS}/combo_*_nodag")) + [f"{_CMC_RESULTS}/multi_{s}_nodag" for s in ["healthy", "disease", "pooled"]] + \
       [f"{_CMC_RESULTS}/linfb_2c_nodag_v4", f"{_CMC_RESULTS}/combat_only_nodag", f"{_CMC_RESULTS}/linfb_2c_nodag_100k", f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000"]
res = json.load(open(OUTP)) if os.path.exists(OUTP) else {}

for rd in RUNS:
    tag = os.path.basename(rd)
    if tag in res or tag == "summary": continue
    try:
        cfg = json.load(open(f"{rd}/config.json"))
        ar = ad.read_h5ad(cfg["rna_path"], backed="r"); ap = ad.read_h5ad(cfg["prot_path"], backed="r")
        genes = list(ar.var_names.astype(str)); prots = list(ap.var_names.astype(str)); NG, NP = len(genes), len(prots)
        same = ar.n_obs == ap.n_obs and (ar.obs_names == ap.obs_names).all()
        if not same:
            common = np.array(sorted(set(ar.obs_names) & set(ap.obs_names))); ri_ = ar.obs_names.get_indexer(common); pi_ = ap.obs_names.get_indexer(common)
        else:
            ri_ = pi_ = np.arange(ar.n_obs)
        n = len(ri_); s1 = np.zeros(NG + NP); S = np.zeros((NG + NP, NG + NP)); CH = 40000
        o = np.argsort(ri_); ri_s, pi_s = ri_[o], pi_[o]
        for k in range(0, n, CH):
            a_idx = ri_s[k:k + CH]; b_idx = pi_s[k:k + CH]
            xr = ar.X[a_idx]
            xr = xr.toarray() if hasattr(xr, "toarray") else np.asarray(xr)
            bo = np.argsort(b_idx); xp = ap.X[b_idx[bo]]; xp = xp.toarray() if hasattr(xp, "toarray") else np.asarray(xp)
            xpp = np.empty_like(xp); xpp[bo] = xp
            X = np.concatenate([xr, xpp], 1).astype(np.float64); s1 += X.sum(0); S += X.T @ X
        cov = (S - np.outer(s1, s1) / n) / (n - 1); d = np.sqrt(np.clip(np.diag(cov), 1e-18, None)); C = np.nan_to_num(cov / np.outer(d, d))
        m = P.load_ctx_model(CrossModalCausalLDVAE, rd, NG, NP, cfg); m.eval()
        with torch.no_grad():
            A_rr, A_pp, W_rp, W_pr = m.scm.get_block_adjacencies(tau=0.5)
            Dr = m._dec_r_w().detach().double().numpy(); Dp = m.dec_prot.weight.detach().double().numpy()
        Grr = Dr @ A_rr.double().numpy() @ Dr.T; Grp = Dr @ W_rp.double().numpy() @ Dp.T
        gi = {g: i for i, g in enumerate(genes)}; cm = [g for g in rg if g in gi]; ci = np.array([gi[g] for g in cm]); rix = np.array([ridx[g] for g in cm])
        off = ~np.eye(len(cm), dtype=bool); out = dict(n_cells=int(n), n_genes=NG, n_prots=NP, n_common_ref=len(cm))
        for name, M in [("CMC", np.abs(Grr)), ("corr", np.abs(C[:NG, :NG]))]:
            Ms = M[np.ix_(ci, ci)].copy(); np.fill_diagonal(Ms, 0)
            for rn in ["STRING", "DoRothEA", "CollecTRI"]:
                mk = ref[rn][np.ix_(rix, rix)].astype(bool)
                if meta[rn]["directed"]: sc, lab = Ms[off], mk[off]
                else: sc, lab = (Ms + Ms.T)[off], (mk | mk.T)[off]
                out[f"{name}_{rn}"] = float(roc_auc_score(lab, sc))
        cog = [(gi[cogmap[p]], j) for j, p in enumerate(prots) if p in cogmap and cogmap[p] in gi]
        cbp = {}
        for g, p in cog: cbp.setdefault(p, set()).add(g)
        for name, V in [("CMC", np.abs(Grp)), ("corr", np.abs(C[:NG, NG:]))]:
            mt = np.array([V[g, p] for g, p in cog]); bg = np.concatenate([np.delete(V[:, p], list(cbp[p])) for g, p in cog])
            colrank = np.array([(V[:, p] > V[g, p]).sum() for g, p in cog]); rowrank = np.array([(V[g, :] > V[g, p]).sum() for g, p in cog])
            out[f"{name}_cog_fold"] = float(np.median(mt) / np.median(bg)); out[f"{name}_cog_top1_col"] = float(np.mean(colrank == 0))
            out[f"{name}_cog_top1_row"] = float(np.mean(rowrank == 0)); out[f"{name}_cog_median_colrank_frac"] = float(np.median(colrank) / NG)
        out["n_cognate"] = len(cog)
        res[tag] = out; log(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
        json.dump(res, open(OUTP, "w"), indent=1)
    except Exception as e:
        log("ERR", tag, repr(e)[:300])
from scipy.stats import wilcoxon
summ = {}
tags = [t for t in res if t != "summary"]
for k in ["STRING", "DoRothEA", "CollecTRI", "cog_fold", "cog_top1_col", "cog_top1_row", "cog_median_colrank_frac"]:
    a = np.array([res[t][f"CMC_{k}"] for t in tags]); b = np.array([res[t][f"corr_{k}"] for t in tags]); d = a - b
    summ[k] = dict(n=len(d), mean_cmc=float(a.mean()), mean_corr=float(b.mean()), mean_diff=float(d.mean()), n_cmc_better=int((d > 0).sum()),
                   p_wilcoxon=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
res["summary"] = summ; json.dump(res, open(OUTP, "w"), indent=1); log("summary", summ); log("DONE")
