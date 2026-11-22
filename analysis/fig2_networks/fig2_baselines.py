"""Fig. 2 / Supp. Fig. 1c / Supp. Fig. 3 / Supp. Note 3-4-12 re-analysis with information-matched baselines (review M1, M9, M10).

Comparators (all from the same 809,130 training cells; paper_nov_revisions/analyses/baselines/train_stats/joint_stats.npz):
  corr            |Pearson|                                   (the paper's correlation baseline)
  partial_corr    |Ledoit-Wolf partial correlation| (joint RNA+protein precision)
  corr_within_ct  |Pearson| after removing cell-type x cohort means (de-confounds cell identity)
  partial_within_ct
CMC-VAE: projected maps of the 3 independently trained models (seeds 42, 1, 2).

Paired inference: every AUROC is written as the mean over positives of the positive's percentile among the (fixed) negatives;
paired bootstrap resamples the positives' grouping unit (TF / source gene / pathway member / protein / edge) and recomputes
CMC - baseline on the same resample (2,000 resamples; two-sided bootstrap P; BH within panel).
Writes analyses/baselines/fig2_baselines.json (+ csv source data).
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
import os, sys, json, re, time
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact, pearsonr

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
R = f"{_CMC_ROOT}"; BASE = f"{_CMC_RESULTS}/paper/analyses/baselines"
sys.path.insert(0, f"{R}/src"); sys.path.insert(0, f"{_CMC_ROOT}/analysis/common")
RUNS = {"42": f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000", "1": f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000_s1",
        "2": f"{_CMC_RESULTS}/sweep_sparse/sp_dec_gl1000_s2"}
NB = int(os.environ.get("NBOOT", "2000")); rng = np.random.RandomState(0)
OUT = {}

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q
def boot_p(d):
    d = np.asarray(d); return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)) + 1.0 / len(d)))
def gini(x):
    x = np.sort(np.abs(np.asarray(x, float)).ravel()); n = x.size; i = np.arange(1, n + 1)
    return float(2 * (i * x).sum() / (n * x.sum()) - (n + 1) / n)
def conc(a):
    a = np.sort(np.abs(a).ravel())[::-1]; s = a.sum()
    return dict(gini=gini(a), mass_top1pct=float(a[:max(1, int(0.01 * a.size))].sum() / s),
                mass_top01pct=float(a[:max(1, int(0.001 * a.size))].sum() / s), mass_top10pct=float(a[:max(1, int(0.1 * a.size))].sum() / s),
                frac_lt_1e3_max=float((a < 1e-3 * a[0]).mean()))

# ------------------------------------------------------------------ load matrices
JS = np.load(f"{BASE}/train_stats/joint_stats.npz", allow_pickle=True)
genes = list(map(str, JS["genes"])); prots = list(map(str, JS["prots"])); NG, NP = len(genes), len(prots)
gi = {g: i for i, g in enumerate(genes)}
BL = {"corr": JS["C"].astype(np.float64), "partial_corr": JS["P_lw"].astype(np.float64),
      "corr_within_ct": JS["C_ct"].astype(np.float64), "partial_within_ct": JS["P_ct"].astype(np.float64)}
for k in BL: np.fill_diagonal(BL[k], 0.0)

from model import CrossModalCausalLDVAE
import perturb_validation_analysis as P
def load_model(rd):
    cfg = json.load(open(f"{rd}/config.json")); m = P.load_ctx_model(CrossModalCausalLDVAE, rd, NG, NP, cfg); m.eval()
    with torch.no_grad():
        A_rr, A_pp, W_rp, W_pr = m.scm.get_block_adjacencies(tau=0.5)
        Dr = m._dec_r_w().detach().double().numpy(); Dp = m.dec_prot.weight.detach().double().numpy()
    return dict(A_rr=A_rr.double().numpy(), A_pp=A_pp.double().numpy(), W_rp=W_rp.double().numpy(), Dr=Dr, Dp=Dp)
MOD = {s: load_model(rd) for s, rd in RUNS.items()}
G = {}
for s, M in MOD.items():
    Grr = M["Dr"] @ M["A_rr"] @ M["Dr"].T; np.fill_diagonal(Grr, 0.0)
    Grp = M["Dr"] @ M["W_rp"] @ M["Dp"].T; Gpp = M["Dp"] @ M["A_pp"] @ M["Dp"].T; np.fill_diagonal(Gpp, 0.0)
    G[s] = dict(rr=Grr, rp=Grp, pp=Gpp)
# consistency with the saved CSV used by the paper (seed 42)
_csv = pd.read_csv(f"{RUNS['42']}/Transcription_RNA_to_Prot.csv", index_col=0)
assert list(map(str, _csv.index)) == genes and list(map(str, _csv.columns)) == prots
OUT["csv_check_rp_maxrel"] = float(np.abs(_csv.values - G["42"]["rp"]).max() / np.abs(_csv.values).max())
log("models loaded; G_rp csv rel diff", OUT["csv_check_rp_maxrel"])

def RR(name, s=None):
    return np.abs(G[s]["rr"]) if name == "CMC-VAE" else np.abs(BL[name][:NG, :NG])
METHODS = ["CMC-VAE", "corr", "partial_corr", "corr_within_ct", "partial_within_ct"]

# ------------------------------------------------------------------ (a) reference-network recovery with paired bootstrap
ref = np.load(f"{_CMC_RESULTS}/ref_nets_v3_2c.npz", allow_pickle=True); meta = json.load(open(f"{_CMC_RESULTS}/ref_nets_v3_2c.json"))
rg = list(map(str, ref["genes"])); ridx = {g: i for i, g in enumerate(rg)}
common = [g for g in rg if g in gi]; ci = np.array([gi[g] for g in common]); ri = np.array([ridx[g] for g in common]); nc = len(common)
off = ~np.eye(nc, dtype=bool); II, JJ = np.where(off)

def percentiles(score, lab):
    neg = np.sort(score[~lab]); pos = score[lab]
    lo = np.searchsorted(neg, pos, "left"); hi = np.searchsorted(neg, pos, "right")
    return (lo + 0.5 * (hi - lo)) / len(neg)

rec = {}
for rn in ["STRING", "DoRothEA", "CollecTRI"]:
    mk = ref[rn][np.ix_(ri, ri)].astype(bool); directed = meta[rn]["directed"]
    lab = (mk if directed else (mk | mk.T))[off]
    grp = II[lab]                                         # grouping unit: source TF (directed) / first gene (undirected)
    U = {}
    for m in METHODS:
        for s in (RUNS if m == "CMC-VAE" else [None]):
            Ms = RR(m, s)[np.ix_(ci, ci)]
            sc = Ms[off] if directed else (Ms + Ms.T)[off]
            U[m if s is None else f"CMC-VAE_s{s}"] = percentiles(sc, lab)
    ug, ginv = np.unique(grp, return_inverse=True); nG = len(ug)
    CNT = np.stack([np.bincount(rng.randint(0, nG, nG), minlength=nG) for _ in range(NB)]).astype(float)   # NB x groups
    nper = np.bincount(ginv, minlength=nG).astype(float)
    def bmean(u): return (CNT @ np.bincount(ginv, weights=u, minlength=nG)) / (CNT @ nper)
    res = dict(n_pos=int(lab.sum()), n_groups=int(nG), auroc={k: float(v.mean()) for k, v in U.items()}, paired={})
    for s in RUNS:
        for m in METHODS[1:]:
            dvec = U[f"CMC-VAE_s{s}"] - U[m]; d = bmean(dvec)
            de = np.array([dvec[rng.randint(0, len(dvec), len(dvec))].mean() for _ in range(500)])
            res["paired"][f"s{s}_vs_{m}"] = dict(diff=float(dvec.mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                                 p_boot_group=boot_p(d), ci95_edge=[float(np.percentile(de, 2.5)), float(np.percentile(de, 97.5))], p_boot_edge=boot_p(de))
    rec[rn] = res; log(rn, {k: round(v, 3) for k, v in res["auroc"].items()})
keys = [(rn, k) for rn in rec for k in rec[rn]["paired"]]
q = bh([rec[rn]["paired"][k]["p_boot_group"] for rn, k in keys])
for (rn, k), qq in zip(keys, q): rec[rn]["paired"][k]["q_bh"] = float(qq)
OUT["fig2a_reference_recovery"] = rec

# ------------------------------------------------------------------ (b) immune programs, with correlation baselines
pw = json.load(open(f"{RUNS['42']}/pathway_recovery.json"))["pathways"]
offG = ~np.eye(NG, dtype=bool)
prog = {}
SC_off = {}
for m in METHODS:
    for s in (RUNS if m == "CMC-VAE" else [None]):
        SC_off[m if s is None else f"CMC-VAE_s{s}"] = RR(m, s)
for name, d in pw.items():
    mem = np.array([gi[g] for g in d["members"] if g in gi]); inp = np.zeros(NG, bool); inp[mem] = True
    pos = np.outer(inp, inp) & offG; neg = (~np.outer(inp, inp)) & offG
    Umat = {}
    for k, M in SC_off.items():
        ng_ = np.sort(M[neg]); sub = M[np.ix_(mem, mem)]
        lo = np.searchsorted(ng_, sub, "left"); hi = np.searchsorted(ng_, sub, "right"); Umat[k] = (lo + 0.5 * (hi - lo)) / len(ng_)
    offm = ~np.eye(len(mem), dtype=bool)
    au_ = {k: float(v[offm].mean()) for k, v in Umat.items()}
    res = dict(n_members=int(len(mem)), source=d.get("source"), auroc=au_, paired={})
    Bi = [rng.randint(0, len(mem), len(mem)) for _ in range(NB)]
    for s in RUNS:
        for m in METHODS[1:]:
            dd = []
            for pick in Bi:
                o2 = pick[:, None] != pick[None, :]
                dd.append(Umat[f"CMC-VAE_s{s}"][np.ix_(pick, pick)][o2].mean() - Umat[m][np.ix_(pick, pick)][o2].mean())
            dd = np.array(dd)
            res["paired"][f"s{s}_vs_{m}"] = dict(diff=au_[f"CMC-VAE_s{s}"] - au_[m], ci95=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))], p_boot=boot_p(dd))
    prog[name] = res
    log(name, {k: round(v, 3) for k, v in au_.items()})
keys = [(nm, k) for nm in prog for k in prog[nm]["paired"]]
q = bh([prog[nm]["paired"][k]["p_boot"] for nm, k in keys])
for (nm, k), qq in zip(keys, q): prog[nm]["paired"][k]["q_bh"] = float(qq)
OUT["fig2b_programs"] = prog

# ------------------------------------------------------------------ (c) cognate map: enrichment + specificity, vs corr / partial corr
cur = json.load(open(f"{_CMC_RESULTS}/baselines/_cmcvae_cognate.json"))
assert cur["protein_names"] == prots
cog = [(gi[g], j) for j, g in enumerate(cur["cognate_gene"]) if g and g in gi]
cbp = {}
for g, p in cog: cbp.setdefault(p, set()).add(g)
decod = {j: cur["cognate_mrna_pearson"][j] for _, j in cog}
def RP(name, s=None): return G[s]["rp"] if name == "CMC-VAE" else BL[name][:NG, NG:]
def cognate_metrics(V):
    A = np.abs(V)
    matched = np.array([A[g, p] for g, p in cog]); bg = np.concatenate([np.delete(A[:, p], list(cbp[p])) for g, p in cog])
    colrank = np.array([float((A[:, p] > A[g, p]).sum()) for g, p in cog])          # #transcripts ranked above the cognate (protein column)
    rowrank = np.array([float((A[g, :] > A[g, p]).sum()) for g, p in cog])          # #proteins ranked above the cognate (transcript row)
    spec = np.array([A[g, p] / (A[:, p].sum() + 1e-30) for g, p in cog])            # share of the column's mass on the cognate
    rho = spearmanr(matched, [decod[p] for _, p in cog]).correlation
    out = dict(n=len(cog), fold=float(np.median(matched) / np.median(bg)), mwu_p=float(mannwhitneyu(matched, bg, alternative="greater").pvalue),
               median_colrank_frac=float(np.median(colrank) / NG), top1_col=float(np.mean(colrank == 0)), top5_col=float(np.mean(colrank < 5)),
               top10_col=float(np.mean(colrank < 10)), mrr_col=float(np.mean(1 / (colrank + 1))),
               top1_row=float(np.mean(rowrank == 0)), top5_row=float(np.mean(rowrank < 5)), median_rowrank=float(np.median(rowrank)),
               median_colmass_share=float(np.median(spec)), decodability_spearman=float(rho), **conc(A))
    return out, colrank, rowrank, spec
cogres = {}; PER = {}
for m in METHODS:
    for s in (RUNS if m == "CMC-VAE" else [None]):
        k = m if s is None else f"CMC-VAE_s{s}"; o, cr, rr_, sp = cognate_metrics(RP(m, s)); cogres[k] = o; PER[k] = (cr, rr_, sp)
log("cognate", {k: (round(v["fold"], 2), round(v["top1_col"], 2), round(v["top1_row"], 2), round(v["gini"], 2)) for k, v in cogres.items()})
pair = {}
for s in RUNS:
    for m in METHODS[1:]:
        a, b = PER[f"CMC-VAE_s{s}"], PER[m]; dd = {}
        for nm, ia, f in [("top1_col", 0, lambda x: np.mean(x == 0)), ("mrr_col", 0, lambda x: np.mean(1 / (x + 1))), ("top1_row", 1, lambda x: np.mean(x == 0)),
                          ("median_colmass_share", 2, np.median)]:
            bs = []
            for _ in range(NB):
                pk = rng.randint(0, len(cog), len(cog)); bs.append(f(a[ia][pk]) - f(b[ia][pk]))
            bs = np.array(bs); dd[nm] = dict(diff=float(f(a[ia]) - f(b[ia])), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p_boot=boot_p(bs))
        pair[f"s{s}_vs_{m}"] = dd
OUT["fig2c_cognate"] = dict(metrics=cogres, paired=pair)

# ------------------------------------------------------------------ (d) exceedance counts for the three cognate nulls (seed 42; 74 pairs)
def fold_of(A):
    matched = np.array([A[g, p] for g, p in cog]); bg = np.concatenate([np.delete(A[:, p], list(cbp[p])) for g, p in cog])
    return float(np.median(matched) / (np.median(bg) + 1e-12))
A42 = np.abs(G["42"]["rp"]); obs = fold_of(A42); r2 = np.random.RandomState(0); NN = int(os.environ.get("NNULL", "10000"))
nulls = {"permute_Grp_entries": [], "shuffle_Wrp_latent_then_project": [], "permute_protein_columns": [], "column_norm_preserving": []}
M42 = MOD["42"]
for i in range(NN):
    v = A42.ravel().copy(); r2.shuffle(v); nulls["permute_Grp_entries"].append(fold_of(v.reshape(A42.shape)))
    w = M42["W_rp"].ravel().copy(); r2.shuffle(w); nulls["shuffle_Wrp_latent_then_project"].append(fold_of(np.abs(M42["Dr"] @ w.reshape(M42["W_rp"].shape) @ M42["Dp"].T)))
    nulls["permute_protein_columns"].append(fold_of(A42[:, r2.permutation(NP)]))
    Mc = A42[r2.rand(NG, NP).argsort(0), np.arange(NP)[None, :]]; nulls["column_norm_preserving"].append(fold_of(Mc))
nd = {}
for k, v in nulls.items():
    v = np.array(v)
    for N_ in [1000, NN]:
        vv = v[:N_]; ex = int((vv >= obs).sum())
        nd[f"{k}_N{N_}"] = dict(N=N_, exceed=ex, p_plus1=(ex + 1) / (N_ + 1), null_mean=float(vv.mean()), null_sd=float(vv.std()), z=float((obs - vv.mean()) / vv.std()), null_max=float(vv.max()))
OUT["fig2d_nulls"] = dict(observed_fold=obs, nulls=nd)
log("nulls", {k: (v["exceed"], round(v["z"], 1), round(v["null_max"], 2)) for k, v in nd.items()})

# ------------------------------------------------------------------ (e) concentration (instead of max-relative sparsity), G_rr off-diagonal
concr = {}
for s in RUNS: concr[f"CMC-VAE_s{s}"] = conc(G[s]["rr"][offG])
for m in METHODS[1:]: concr[m] = conc(BL[m][:NG, :NG][offG])
for nm, path, key in [("GRNBoost2", f"{_CMC_RESULTS}/baselines/genie3_matrix_grnboost2_all_809k.npz", "matrix"),
                      ("DeepSEM", f"{_CMC_RESULTS}/baselines/deepsem_W_809k_clamp09.npz", "W"), ("totalVI", f"{_CMC_RESULTS}/baselines/totalvi_raw.npz", "G_rr_tv")]:
    z = np.load(path, allow_pickle=True); Mx = np.asarray(z[key], float).copy(); np.fill_diagonal(Mx, 0.0)
    concr[nm] = conc(Mx[~np.eye(Mx.shape[0], dtype=bool)])
OUT["fig2e_concentration"] = concr
log("conc", {k: (round(v["gini"], 3), round(v["mass_top1pct"], 3)) for k, v in concr.items()})

# ------------------------------------------------------------------ (f) surface complexes, paired bootstrap over positive edges
import importlib.util
src = open(f"{_CMC_ROOT}/analysis/fig2_networks/ppi_surface_complexes.py").read()
cc_txt = src[src.index("CURATED_COMPLEXES = {"):src.index("# ---------------------------------------------------------------- P3")]
ns = {}; exec(cc_txt, ns); CCX = ns["CURATED_COMPLEXES"]
pidx = {p: i for i, p in enumerate(prots)}
def parse_edge(e):
    for a in sorted(prots, key=len, reverse=True):
        if e.startswith(a + "-") and e[len(a) + 1:] in pidx: return a, e[len(a) + 1:]
pos = np.zeros((NP, NP), bool); cis = np.zeros((NP, NP), bool)
for nm, (edges, kind) in CCX.items():
    for e in edges:
        a, b = parse_edge(e); pos[pidx[a], pidx[b]] = pos[pidx[b], pidx[a]] = True
        if kind == "cis": cis[pidx[a], pidx[b]] = cis[pidx[b], pidx[a]] = True
iu = np.triu_indices(NP, 1); labA = pos[iu]; labC = cis[iu]
PP = {f"CMC-VAE_s{s}": G[s]["pp"] for s in RUNS}
for m in ["corr", "corr_within_ct", "partial_corr", "partial_within_ct"]: PP[m] = BL[m][NG:, NG:]
Pg = JS["P_prot_glasso"].astype(float)
if np.isfinite(Pg).all(): PP["partial_corr_protein_glasso"] = Pg
# protein-only Ledoit-Wolf partial correlation (no conditioning on RNA)
Cpp = JS["C"][NG:, NG:].astype(float); dl = float(JS["lw_delta"]); Pi = np.linalg.inv((1 - dl) * Cpp + dl * np.eye(NP)); dg = np.sqrt(np.diag(Pi))
PP["partial_corr_protein_lw"] = -Pi / np.outer(dg, dg)
ppi = {}
for setname, lab in [("all36", labA), ("cis15", labC)]:
    U = {k: percentiles((np.abs(M) + np.abs(M).T)[iu], lab) for k, M in PP.items()}
    res = dict(n_pos=int(lab.sum()), auroc={k: float(v.mean()) for k, v in U.items()}, paired={})
    for s in RUNS:
        for m in [k for k in PP if not k.startswith("CMC")]:
            d = np.array([(U[f"CMC-VAE_s{s}"][ix] - U[m][ix]).mean() for ix in [rng.randint(0, lab.sum(), lab.sum()) for _ in range(NB)]])
            res["paired"][f"s{s}_vs_{m}"] = dict(diff=float(U[f"CMC-VAE_s{s}"].mean() - U[m].mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_boot=boot_p(d))
    ppi[setname] = res; log("ppi", setname, {k: round(v, 3) for k, v in res["auroc"].items()})
OUT["fig2f_surface"] = ppi

# ------------------------------------------------------------------ seed reproducibility and seed-stable edges (Supp Note 12; M9)
iuG = np.triu_indices(NG, 1)
def topk_idx(M, frac=0.01):
    a = np.abs(M[iuG]); k = int(round(frac * a.size)); return set(np.argpartition(a, -k)[-k:].tolist()), k
tops = {s: topk_idx(G[s]["rr"])[0] for s in RUNS}; K = topk_idx(G["42"]["rr"])[1]
jac = {f"{a}v{b}": len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for a, b in [("42", "1"), ("42", "2"), ("1", "2")]}
stable = tops["42"] & tops["1"] & tops["2"]
sp = {f"{a}v{b}": float(spearmanr(G[a]["rr"][iuG], G[b]["rr"][iuG]).correlation) for a, b in [("42", "1"), ("42", "2"), ("1", "2")]}
ctop, _ = topk_idx(BL["corr"][:NG, :NG])
stab = dict(k_top1pct=K, jaccard_top1pct=jac, spearman_all_edges=sp, n_stable=len(stable), frac_stable=len(stable) / K,
            jaccard_stable_vs_corr_top1pct=len(stable & ctop) / len(stable | ctop), frac_stable_in_corr_top1pct=len(stable & ctop) / len(stable))
# biological content of seed-stable edges vs same-size top-|corr| edges and vs all pairs
a_i, a_j = iuG[0], iuG[1]
IGTR = re.compile(r"^(IG[HKL][VDJ]|TR[ABDG][VDJ])")
isIGTR = np.array([bool(IGTR.match(g)) for g in genes])
refm = {}
for rn in ["STRING", "DoRothEA", "CollecTRI"]:
    full = np.zeros((NG, NG), bool); sub = ref[rn][np.ix_(ri, ri)].astype(bool); full[np.ix_(ci, ci)] = sub; refm[rn] = full | full.T
progm = {}
for name, d in pw.items():
    mem = np.array([gi[g] for g in d["members"] if g in gi]); inp = np.zeros(NG, bool); inp[mem] = True; progm[name] = np.outer(inp, inp)
corr_rank = np.argsort(-np.abs(BL["corr"][:NG, :NG][iuG]))
def content(sel):
    sel = np.array(sorted(sel)); I_, J_ = a_i[sel], a_j[sel]; o = {}
    o["n"] = int(len(sel)); o["frac_IGTR_edge"] = float(np.mean(isIGTR[I_] | isIGTR[J_]))
    for rn, Mk in refm.items():
        hit = Mk[I_, J_]; base = Mk[a_i, a_j].mean()
        tab = [[int(hit.sum()), int(len(hit) - hit.sum())], [int(Mk[a_i, a_j].sum() - hit.sum()), int(len(a_i) - len(hit) - (Mk[a_i, a_j].sum() - hit.sum()))]]
        o[rn] = dict(frac=float(hit.mean()), background=float(base), enrich=float(hit.mean() / base), fisher_p=float(fisher_exact(tab)[1]))
    for name, Mk in progm.items():
        hit = Mk[I_, J_]; o[f"prog::{name}"] = dict(n=int(hit.sum()), enrich=float(hit.mean() / Mk[a_i, a_j].mean()))
    return o
nst = len(stable)
stab["content_seed_stable_CMC"] = content(stable)
stab["content_top_corr_same_n"] = content(set(corr_rank[:nst].tolist()))
stab["content_top1pct_CMC_s42"] = content(tops["42"])
stab["content_stable_noIGTR"] = content({e for e in stable if not (isIGTR[a_i[e]] or isIGTR[a_j[e]])})
# top stable edges (gene pairs) for interpretation
st_sorted = sorted(stable, key=lambda e: -np.mean([abs(G[s]["rr"][a_i[e], a_j[e]]) for s in RUNS]))
stab["top_stable_edges"] = [(genes[a_i[e]], genes[a_j[e]]) for e in st_sorted[:60]]
stab["top_stable_edges_noIGTR"] = [(genes[a_i[e]], genes[a_j[e]]) for e in st_sorted if not (isIGTR[a_i[e]] or isIGTR[a_j[e]])][:60]
OUT["seed_stability"] = stab; log("seed", jac, "stable", nst)
json.dump(dict(stable_edges=[(genes[a_i[e]], genes[a_j[e]]) for e in st_sorted]), open(f"{BASE}/seed_stable_edges.json", "w"))

# ------------------------------------------------------------------ hubs (Supp Fig 1c) without IG/TR V(D)J genes; TFs among hubs
TFs = set()
for rn in ["DoRothEA", "CollecTRI"]:
    mk = ref[rn].astype(bool); TFs |= {rg[i] for i in np.where(mk.any(1))[0]}
TFs &= set(genes)
IGTR_ALL = re.compile(r"^(IG[HKL][VDJC]|IGH[GMADE]|IGLL|TR[ABDG][VDJC])")
hubs = {}
for label, excl in [("all_genes", np.zeros(NG, bool)), ("no_IGTR_VDJ", isIGTR), ("no_IGTR_incl_constant", np.array([bool(IGTR_ALL.match(g)) for g in genes]))]:
    keep = ~excl; o = {}
    for s in RUNS:
        A_ = np.abs(G[s]["rr"])[np.ix_(keep, keep)]; strength = A_.sum(1); kg = np.array(genes)[keep]; order = np.argsort(-strength)
        rown = np.linalg.norm(MOD[s]["Dr"], axis=1)[keep]
        o[f"s{s}"] = dict(top20=[(kg[i], float(strength[i]), kg[i] in TFs) for i in order[:20]], n_TF_top14=int(sum(kg[i] in TFs for i in order[:14])),
                          n_TF_top50=int(sum(kg[i] in TFs for i in order[:50])), TF_top50=[kg[i] for i in order[:50] if kg[i] in TFs],
                          r_strength_rownorm=float(pearsonr(strength, rown)[0]), spearman_strength_rownorm=float(spearmanr(strength, rown).correlation),
                          rank={kg[i]: int(r) for r, i in enumerate(order[:200])})
    # seed-consistent hubs: genes in the top 50 of all three seeds
    sets = [set(list(o[f"s{s}"]["rank"].keys())[:50]) for s in RUNS]
    o["top50_all_seeds"] = sorted(set.intersection(*sets), key=lambda g: o["s42"]["rank"][g])
    o["TF_in_top50_all_seeds"] = [g for g in o["top50_all_seeds"] if g in TFs]
    # correlation-network hubs for comparison
    Cc = np.abs(BL["corr"][:NG, :NG])[np.ix_(keep, keep)]; stc = Cc.sum(1); kg = np.array(genes)[keep]; oc = np.argsort(-stc)
    o["corr_top20"] = [(kg[i], float(stc[i]), kg[i] in TFs) for i in oc[:20]]; o["corr_n_TF_top50"] = int(sum(kg[i] in TFs for i in oc[:50]))
    o["n_TF_in_vocab"] = int(sum(g in TFs for g in kg)); o["n_genes"] = int(keep.sum())
    for s in RUNS: o[f"s{s}"].pop("rank")
    hubs[label] = o
OUT["hubs"] = hubs; OUT["n_IGTR_VDJ_genes"] = int(isIGTR.sum()); OUT["IGTR_VDJ_genes"] = [g for g in genes if IGTR.match(g)]
log("hubs no IGTR s42", [h[0] for h in hubs["no_IGTR_VDJ"]["s42"]["top20"][:14]], "TF top50", hubs["no_IGTR_VDJ"]["s42"]["n_TF_top50"])

json.dump(OUT, open(f"{BASE}/fig2_baselines.json", "w"), indent=1, default=float)
log("DONE")
