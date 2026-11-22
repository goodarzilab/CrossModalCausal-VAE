#!/usr/bin/env python
"""Reviewer M10: alternative DE ground truths for the Replogle 2022 essential-gene Perturb-seq screens (Jurkat, K562).

usage: python -I de_perturbseq.py LINE

Reads (read-only) results/replogle/data/{LINE}_X.npy (log1p(CP10k), 3,000 features), {LINE}_obs.csv, {LINE}_de.npz.
Writes analyses/review2/de_robust/de/{LINE}_de_variants.npz with, per KO of {LINE}_de.npz (target in features, >=30 cells):
  orig   : cell-level two-sided Mann-Whitney raw P<0.05 & |LFC|>=0.10, LFC = mean(KO cells) - mean(all NT cells)
           (exactly the published definition; P values taken from {LINE}_de.npz)
  bh     : same Mann-Whitney P, Benjamini-Hochberg across the 3,000 features within each KO, q<0.05 & |LFC|>=0.10
  pb     : pseudobulk limma-trend-like moderated t (code/stats/pblimma.limma_ko, the code used for the CITE-seq
           development screens).  Gem groups (10x lanes) are sorted numerically and cut into NBLK=8 contiguous blocks
           with ~equal cell numbers; pseudobulk = mean log-normalised expression of a KO's cells within a block
           (>=MINC=5 cells); control pseudobulk = all NT cells of the block.  Design = KO indicator + block dummies, so
           the KO coefficient is a block-matched LFC and the residual variance is the KO x block (replicate)
           variability.  Two-sided moderated-t P, BH across features within KO; q<0.05 & |LFC_pb|>=0.10.
           KOs need >=3 pseudobulks.
  pbgem  : sensitivity: pseudobulk per individual gem group (>=3 cells), otherwise as pb.
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
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, f"{_CMC_ROOT}/analysis/fig3_zero_shot/stats")
from pblimma import limma_ko, bh

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
LINE = sys.argv[1]
D = f"{_CMC_RESULTS}/replogle/data"
OUT = f"{_CMC_RESULTS}/paper/analyses/review2/de_robust/de/{LINE}_de_variants.npz"
LFCR, PTHR, QTHR, NBLK = 0.10, 0.05, 0.05, 8
FEATS = json.load(open(f"{D}/features.json"))["features"]; G = len(FEATS); FA = np.array(FEATS)

de = np.load(f"{D}/{LINE}_de.npz")
genes = de["genes"].astype(str); lfc = de["lfc"].astype(np.float64); pv = de["pvals"].astype(np.float64); K = len(genes)
log(LINE, "KOs", K)
sig_orig = (pv < PTHR) & (np.abs(lfc) >= LFCR)
q_mw = np.stack([bh(p) for p in pv]); sig_bh = (q_mw < QTHR) & (np.abs(lfc) >= LFCR)

obs = pd.read_csv(f"{D}/{LINE}_obs.csv")
log("loading X ...")
X = np.load(f"{D}/{LINE}_X.npy").astype(np.float32)
gem = obs.gem_group.astype(str).values
gnum = np.array(sorted(set(gem), key=lambda s: int(s) if s.isdigit() else s))
cnt = pd.Series(gem).value_counts().reindex(gnum).values
cum = np.cumsum(cnt) / cnt.sum(); blk_of = {g: min(NBLK - 1, int(c_ * NBLK - 1e-9)) for g, c_ in zip(gnum, cum)}
block = np.array([blk_of[g] for g in gem])
log("blocks (n cells):", np.bincount(block).tolist(), "gem groups per block:", np.bincount([blk_of[g] for g in gnum]).tolist())

is_nt = obs.is_nt.values; gene_col = obs.gene.astype(str).values

def run_pb(strata, minc, tag):
    st_vals = np.unique(strata)
    Yc = np.stack([X[is_nt & (strata == s)].mean(0) for s in st_vals]).astype(np.float64); sc_ = st_vals
    LF = np.full((K, G), np.nan); P = np.ones((K, G)); Q = np.ones((K, G)); NPB = np.zeros(K, int); DF = np.zeros(K, int)
    for k, g in enumerate(genes):
        idx = np.where(gene_col == g)[0]; s_ = strata[idx]
        Yk, sk = [], []
        for s in np.unique(s_):
            ii = idx[s_ == s]
            if len(ii) >= minc: Yk.append(X[ii].mean(0)); sk.append(s)
        NPB[k] = len(Yk)
        if len(Yk) < 3: continue
        b, p, info = limma_ko(np.stack(Yk).astype(np.float64), np.array(sk), Yc, sc_)
        if not np.isfinite(b).any(): continue
        LF[k], P[k], Q[k], DF[k] = b, p, bh(p), info["df"]
        if k % 200 == 0: log(f"  {tag} {k}/{K} {g}: n_pb {len(Yk)} df {info['df']} d0 {info.get('d0_median')}")
    sig = (Q < QTHR) & (np.abs(np.nan_to_num(LF)) >= LFCR)
    return LF, P, Q, sig, NPB, DF

lf_pb, p_pb, q_pb, sig_pb, npb, df_pb = run_pb(block, 5, "pb")
lf_pg, p_pg, q_pg, sig_pg, npg, df_pg = run_pb(np.array([int(x) for x in gem]), 3, "pbgem")

keep = FA[None, :] != genes[:, None]
def nde(sig): return (sig & keep).sum(1)
summ = {}
for nm, s_ in [("orig", sig_orig), ("bh", sig_bh), ("pb", sig_pb), ("pbgem", sig_pg)]:
    n = nde(s_); ev = n >= 3
    summ[nm] = dict(n_evaluable=int(ev.sum()), median_de_evaluable=float(np.median(n[ev])), median_de_all=float(np.median(n)),
                    iqr_evaluable=[float(np.percentile(n[ev], 25)), float(np.percentile(n[ev], 75))])
ok = np.isfinite(lf_pb).all(1)
summ["corr_lfc_cell_vs_pb_median"] = float(np.median([np.corrcoef(lfc[k], lf_pb[k])[0, 1] for k in np.where(ok)[0]]))
summ["n_pb_median"] = float(np.median(npb)); summ["df_pb_median"] = float(np.median(df_pb[ok]))
summ["n_pbgem_median"] = float(np.median(npg))
summ["jaccard_orig_bh_median"] = float(np.median([(sig_orig[k] & sig_bh[k]).sum() / max(1, (sig_orig[k] | sig_bh[k]).sum()) for k in range(K)]))
summ["frac_pb_in_orig_median"] = float(np.nanmedian([(sig_pb[k] & sig_orig[k]).sum() / sig_pb[k].sum() if sig_pb[k].sum() else np.nan for k in range(K)]))
log(json.dumps(summ, indent=1))
np.savez_compressed(OUT, genes=genes, n_cells=de["n_cells"], lfc=lfc.astype(np.float32), p_mw=pv.astype(np.float32), q_mw=q_mw.astype(np.float32),
                    sig_orig=sig_orig, sig_bh=sig_bh, lfc_pb=lf_pb.astype(np.float32), p_pb=p_pb.astype(np.float32), q_pb=q_pb.astype(np.float32),
                    sig_pb=sig_pb, n_pb=npb, df_pb=df_pb, lfc_pbgem=lf_pg.astype(np.float32), q_pbgem=q_pg.astype(np.float32), sig_pbgem=sig_pg,
                    n_pbgem=npg, summary=json.dumps(summ))
log("WROTE", OUT)
