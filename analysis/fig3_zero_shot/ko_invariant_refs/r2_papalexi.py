#!/usr/bin/env python
# ROUND2 copy of code/reviewfix/rf_papalexi.py: op_controls_r2.py (R2_CACHE) and Papalexi cache dir from R2_PAP_CACHE.
"""Item 1 (POST HOC, not pre-registered): re-score the untouched Papalexi THP-1 ECCITE-seq screen with the PRIMARY
operator (signed clamp, clamp_zero, analysis/fig3_zero_shot/operator/op_controls.py::predict_op) under three DE definitions:
  cell_raw : the pre-registered cell-level DE (MW P<0.05 & |LFC|>=0.10; harness cache of analysis/fig3_zero_shot/papalexi_untouched/rc_untouched.py)
  cell_bh  : same cell-level test, BH within knockout (q<0.05 & |LFC|>=0.10)
  pb       : guide x replicate pseudobulks (>=10 cells; 4 guides x 3 replicates per target, 9 NT guides x 3 replicates as
             controls), limma-trend-like moderated t with replicate as stratum (data_prep/pseudobulk_de/pblimma.py, the condition-matched
             DE used for Figs 3-5), BH within knockout, q<0.05 & |LFC|>=0.10 (RNA) / 0.05 (protein).
Predictors: CMC-VAE clamp (this seed), the pre-registered edge cut (reference: must reproduce the frozen result on
cell_raw), and the same baselines as the pre-registered analysis (training-corpus correlation, totalVI, DeepSEM,
control-cell expression / variance, development-screen copy) plus the empirical-covariance conditioning comparator.
Joint knockout x control-cell bootstrap (999 replicates, replicate b uses default_rng(100003+b), identical across seeds so
replicates can be averaged over seeds) for the pb definition: control cells resampled within control pseudobulks, limma
re-run, fixed predictions re-scored, knockouts resampled.
usage: python -I rf_papalexi.py RUN_DIR SEEDTAG OUT_DIR [NBOOT]
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, runpy, pickle, hashlib, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
B = f"{_CMC_RESULTS}/paper"; REPO = f"{_CMC_ROOT}"
RUN, SEEDTAG, OUTD = sys.argv[1], sys.argv[2], sys.argv[3]; NBOOT = int(sys.argv[4]) if len(sys.argv) > 4 else 999
os.makedirs(OUTD, exist_ok=True)
sys.path.insert(0, f"{_CMC_ROOT}/analysis/fig3_zero_shot/stats"); sys.path.insert(0, f"{_CMC_ROOT}/data_prep/pseudobulk_de"); sys.path.insert(0, f"{_CMC_ROOT}/analysis/fig3_zero_shot/ko_invariant_refs")
from pblimma import limma_ko, bh
from rf_lib import EmpCov
os.environ["OP_DE"] = "orig"; os.environ["OP_PARTS"] = "none"
sys.argv = ["op_controls.py", RUN, f"{OUTD}/opmeta_pap_{SEEDTAG}.json"]
G = runpy.run_path(f"{_CMC_ROOT}/analysis/fig3_zero_shot/ko_invariant_refs/op_controls_r2.py", run_name="__opc__")
torch = G["torch"]; DEVICE = G["DEVICE"]; P = G["P"]
G["model"].scm.load_state_dict(G["SCM0"]); A0 = G["assemble"]().detach()
LFCR, LFCP = G["LFCR"], G["LFCP"]

# ---------------------------------------------------------------- Papalexi harness cache (align, cell-level DE, control eps)
ck = P.resolve_ckpt(RUN); h = hashlib.md5(open(ck, "rb").read()).hexdigest()[:12]
pk = f"{os.environ.get('R2_PAP_CACHE', B + '/analyses/untouched/cache')}/rc_papalexi_{os.path.basename(RUN)}_{h}_{LFCR}_{LFCP}.pkl"
cache = pickle.load(open(pk, "rb")); al, de_cell, eps = cache["align"], cache["de"], cache["eps"]
log("cache", os.path.basename(pk), "cell-level DE units", len(de_cell))
common = list(al["common_genes"]); rci = np.asarray(al["rna_eval_perturb_indices"]); pci = np.asarray(al["prot_eval_perturb_indices"])
PD = f"{B}/analyses/untouched/data/papalexi_proc"
ra = ad.read_h5ad(f"{PD}/papalexi_rna.h5ad"); pa = ad.read_h5ad(f"{PD}/papalexi_prot.h5ad")
assert list(ra.obs_names) == list(pa.obs_names)
R = ra.X[:, rci]; R = (R.toarray() if sp.issparse(R) else np.asarray(R)).astype(np.float32)
PX = pa.X; PX = (PX.toarray() if sp.issparse(PX) else np.asarray(PX)); PRT = PX[:, pci].astype(np.float32)
obs = ra.obs; tgt = obs["gene_target"].astype(str).values; guide = obs["guide_ID"].astype(str).values; rep = obs["replicate"].astype(str).values

# ---------------------------------------------------------------- pseudobulk DE (guide x replicate)
MINC, MIN_UNIT = 10, 30
PB = []
for (g, gd, st), idx in pd.Series(np.arange(len(tgt))).groupby([tgt, guide, rep]).apply(np.array).items():
    if len(idx) < MINC: continue
    PB.append(dict(gene=("control" if g == "NT" else g), guide=gd, stratum=st, n=len(idx), idx=idx, y_r=R[idx].mean(0), y_p=PRT[idx].mean(0)))
ct = [b for b in PB if b["gene"] == "control"]
CT = dict(Y_r=np.stack([b["y_r"] for b in ct]), Y_p=np.stack([b["y_p"] for b in ct]), strata=np.array([b["stratum"] for b in ct]),
          cells_r=np.concatenate([R[b["idx"]] for b in ct]), cells_p=np.concatenate([PRT[b["idx"]] for b in ct]),
          cell_pb=np.concatenate([np.full(b["n"], k) for k, b in enumerate(ct)]))
KO = {}
for g in sorted(set(b["gene"] for b in PB) - {"control"}):
    kb = [b for b in PB if b["gene"] == g]
    if sum(b["n"] for b in kb) < MIN_UNIT or len(kb) < 2: continue
    KO[g] = dict(Y_r=np.stack([b["y_r"] for b in kb]), Y_p=np.stack([b["y_p"] for b in kb]), strata=np.array([b["stratum"] for b in kb]),
                 n_pb=len(kb), n_cells=int(sum(b["n"] for b in kb)))
log("pseudobulks", len(PB), "control pb", len(ct), "KO units", len(KO))
def pb_de(Yc_r, Yc_p, sc):
    out = {}
    for g, k in KO.items():
        br, pr, _ = limma_ko(k["Y_r"], k["strata"], Yc_r, sc); bp, pp_, _ = limma_ko(k["Y_p"], k["strata"], Yc_p, sc)
        br = np.nan_to_num(br); bp = np.nan_to_num(bp); qr, qp = bh(pr), bh(pp_)
        out[g] = dict(lfc_rna=br, pvals_rna=pr, sig_rna=(qr < 0.05) & (np.abs(br) >= LFCR), lfc_prot=bp, pvals_prot=pp_,
                      sig_prot=(qp < 0.05) & (np.abs(bp) >= LFCP), n_cells=k["n_cells"])
    return out
DE = {"cell_raw": de_cell,
      "cell_bh": {g: dict(d, sig_rna=(bh(d["pvals_rna"]) < 0.05) & (np.abs(np.asarray(d["lfc_rna"], float)) >= LFCR),
                          sig_prot=((bh(d["pvals_prot"]) < 0.05) & (np.abs(np.asarray(d["lfc_prot"], float)) >= LFCP)) if len(d["pvals_prot"]) else np.zeros(0, bool))
                  for g, d in de_cell.items()},
      "pb": pb_de(CT["Y_r"], CT["Y_p"], CT["strata"])}

def mkctx(de):
    G["RAW"]["pap"] = (al, de, eps); c = G["build_ctx"]("pap")
    tg = set(c["evaluable"]); u = np.zeros(len(c["common"]), bool)
    for g in c["evaluable"]: u |= c["INFO"][g]["sig"]
    u &= ~np.isin(c["common"], list(tg)); c["S"] = np.where(u)[0]
    c["S_tr"] = torch.as_tensor(c["reti"][c["S"]], dtype=torch.long, device=DEVICE)
    cg = np.zeros(len(c["preti"]), bool)
    for g in tg: cg |= c["INFO"][g]["cog_mask"]
    c["Sp"] = np.where(~cg)[0]
    return c

# ---------------------------------------------------------------- predictions for every vocab unit (any definition)
VOC = sorted(g for g in set(de_cell) | set(KO) if g in al["train_gene_idx"] and g in al["train_gene_to_idx_common"])
c0 = mkctx(de_cell); reti, preti = c0["reti"], c0["preti"]
gi = torch.as_tensor([al["train_gene_idx"][g] for g in VOC], device=DEVICE)
ebar = torch.as_tensor(eps, dtype=torch.float32, device=DEVICE)[:4000].mean(0)
PRED = {}
with torch.no_grad():
    for nm, op in [("cmc_clamp", "clamp"), ("cmc_edgecut", "edgecut")]:
        dr, dp, delta = G["predict_op"](A0, ebar, gi, op, "zero")
        drn, dpn = dr.cpu().numpy().astype(float), dp.cpu().numpy().astype(float)
        PRED[nm] = {("pap", g): (drn[k][reti], dpn[k][preti]) for k, g in enumerate(VOC)}
        if op == "clamp": DELTA = delta.cpu().numpy().astype(float); ONT = drn[np.arange(len(VOC)), gi.cpu().numpy()]
log("on-target change < 0 (clamp):", float((ONT < 0).mean()), "units", len(VOC))
# baselines: same constructions as analysis/fig3_zero_shot/papalexi_untouched/rc_untouched.py::baseline_preds (knockdown sign)
tv = np.load(f"{_CMC_RESULTS}/baselines/totalvi_raw.npz", allow_pickle=True)
Grr_tv = np.asarray(tv["G_rr_tv"], float); Grp_tv = np.asarray(tv["G_rp_tv"], float); np.fill_diagonal(Grr_tv, 0.0)
tvg = {g: i for i, g in enumerate(np.asarray(tv["genes"]).astype(str))}
dsz = np.load(f"{_CMC_RESULTS}/baselines/deepsem_W_809k_clamp09.npz", allow_pickle=True)
Wd = np.asarray(dsz["W"], float); np.fill_diagonal(Wd, 0.0); dsg = {g: i for i, g in enumerate(np.asarray(dsz["genes"]).astype(str))}
Minv = np.linalg.inv(np.eye(Wd.shape[0]) - Wd.T)
ctr, cgp = al.get("corr_train"), al.get("gene_prot_corr"); g2i = al["train_gene_to_idx_common"]; n_p = len(preti)
tvcol = np.array([tvg.get(x, -1) for x in common]); dscol = np.array([dsg.get(x, -1) for x in common])
ctl = tgt == "NT"; mr, vr_ = R[ctl].mean(0), R[ctl].var(0); mpq, vpq = PRT[ctl].mean(0), PRT[ctl].var(0)
for k_ in ["corr", "totalvi", "deepsem", "zs_ctrl_expr", "zs_ctrl_var", "dev_copy", "empcov"]: PRED[k_] = {}
for g in VOC:
    PRED["corr"][("pap", g)] = (-np.asarray(ctr[g2i[g], :], float), -np.asarray(cgp[g2i[g], :], float) if (cgp is not None and n_p) else np.zeros(n_p))
    if g in tvg:
        vr = np.array([Grr_tv[tvg[g], j] if j >= 0 else 0.0 for j in tvcol]); PRED["totalvi"][("pap", g)] = (-vr, -Grp_tv[tvg[g], preti] if n_p else np.zeros(0))
    if g in dsg:
        vr = np.array([Minv[j, dsg[g]] if j >= 0 else 0.0 for j in dscol]); PRED["deepsem"][("pap", g)] = (-vr, np.full(n_p, np.nan))
    PRED["zs_ctrl_expr"][("pap", g)] = (-mr.astype(float), -mpq.astype(float)); PRED["zs_ctrl_var"][("pap", g)] = (-vr_.astype(float), -vpq.astype(float))
    for d in ["frangieh", "iccite", "arce"]:
        if g in G["RAW"][d][1]:
            ald, dd = G["RAW"][d][0], G["RAW"][d][1][g]; m = dict(zip(ald["common_genes"], np.asarray(dd["lfc_rna"], float)))
            mp = dict(zip(ald["prot_eval_names"], np.asarray(dd["lfc_prot"], float))) if len(dd["lfc_prot"]) else {}
            PRED["dev_copy"][("pap", g)] = (np.array([m.get(x, 0.0) for x in common]), np.array([mp.get(x, 0.0) for x in al["prot_eval_names"]])); break
EC = EmpCov()
assert EC.genes[:5] == list(al["train_gene_names"][:5]) if "train_gene_names" in al else True
er, ep = EC.predict(gi.cpu().numpy(), DELTA, reti, preti, al["train_rna_std"], al["train_prot_std"])
for k, g in enumerate(VOC): PRED["empcov"][("pap", g)] = (er[k], ep[k])
log("empcov: LW shrinkage", EC.shrink, "| corpus-sd vs align train_rna_std max rel diff",
    float(np.max(np.abs(EC.sd[:EC.G] - np.asarray(al["train_rna_std"], float)) / np.clip(np.asarray(al["train_rna_std"], float), 1e-6, None))))
NAMES = list(PRED)

MET = ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rr_auroc_meanresp", "rr_pe_meanresp", "rp_auroc", "rp_pe"]
def score_all(c):
    units = [g for g in c["evaluable"] if ("pap", g) in PRED["cmc_clamp"]]
    T = {}
    for nm in NAMES:
        u = [g for g in units if ("pap", g) in PRED[nm]]
        if len(u) < 3: continue
        rows = G["score"]({"pap": c}, {"pap": u}, PRED[nm], G["ref_means"]({"pap": c}, {"pap": u}, PRED[nm]))
        for r in rows: T[(nm, r["gene"])] = {m: r.get(m, np.nan) for m in MET}
    return T, units
RES = dict(seed=SEEDTAG, vocab_units=VOC, n_pb=len(PB), n_ctrl_pb=len(ct), pb_units={g: dict(n_pb=k["n_pb"], n_cells=k["n_cells"]) for g, k in KO.items()},
           on_target_neg_frac=float((ONT < 0).mean()), delta=dict(zip(VOC, DELTA.tolist())), empcov_shrink=EC.shrink, modes={})
for mode, de in DE.items():
    c = mkctx(de); T, units = score_all(c)
    nde = {g: int((c["INFO"][g]["sig"] & c["INFO"][g]["keep"]).sum()) for g in c["evaluable"]}
    RES["modes"][mode] = dict(units=units, n_de=nde, per_ko={f"{nm}|{g}": v for (nm, g), v in T.items()})
    log(mode, "units", len(units), "median #DE", np.median(list(nde.values())) if nde else None,
        {nm: round(float(np.nanmean([T[(nm, g)]["rr_pe_cent"] for g in units if (nm, g) in T])), 3) for nm in ["cmc_clamp", "cmc_edgecut", "corr", "empcov"]})
json.dump(RES, open(f"{OUTD}/papalexi_{SEEDTAG}.json", "w"), default=lambda o: None if (isinstance(o, float) and o != o) else (o.item() if hasattr(o, "item") else str(o)))

# ---------------------------------------------------------------- joint bootstrap, pb definition
cell_ids = [np.where(CT["cell_pb"] == k)[0] for k in range(len(ct))]
BOOT = []
for b in range(1, NBOOT + 1):
    rng = np.random.default_rng(100003 + b)
    rows, cols, vals = [], [], []
    for k, idx in enumerate(cell_ids):
        w = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx))); nz = w > 0
        rows.append(np.full(nz.sum(), k)); cols.append(idx[nz]); vals.append(w[nz] / len(idx))
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(ct), len(CT["cell_pb"])))
    de_b = pb_de(np.asarray(W @ CT["cells_r"]), np.asarray(W @ CT["cells_p"]), CT["strata"])
    c = mkctx(de_b); T, units = score_all(c)
    samp = rng.choice(len(units), len(units), replace=True) if units else []
    rec = dict(b=b, n_eval=len(units))
    for nm in NAMES:
        for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval"]:
            v = np.array([T.get((nm, units[i]), {}).get(m, np.nan) for i in samp], float)
            if np.isfinite(v).any(): rec[f"mean|{nm}|{m}"] = float(np.nanmean(v))
            if nm != "cmc_clamp":
                a = np.array([T.get(("cmc_clamp", units[i]), {}).get(m, np.nan) for i in samp], float); d = a - v
                if np.isfinite(d).any(): rec[f"diff|{nm}|{m}"] = float(np.nanmean(d))
    BOOT.append(rec)
    if b % 100 == 0: log("boot", b)
json.dump(BOOT, open(f"{OUTD}/papalexi_boot_{SEEDTAG}.json", "w"))
log("DONE")
