#!/usr/bin/env python
"""Tasks 1 & 2 (+ Task-3 point values) on the point-estimate DE. Writes ../point_results.json and per-KO tables."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import sys, os, json, pickle, glob
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from lib import SL, TAGS, SETS
NB = 2000
C = lib.realise(None)
RES = {"n_evaluable": {t: len(C[t].evaluable) for t in TAGS}}

def ko_boot(v, nb=NB, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; rng = np.random.default_rng(seed)
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(nb)])
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

def resid_scores(score, cov):
    r1 = rankdata(score); r2 = rankdata(cov); X = np.c_[np.ones_like(r2), r2]
    beta = np.linalg.lstsq(X, r1, rcond=None)[0]; return r1 - X @ beta

def cross_matrix(units, sig, keep, common, pred, transform=None):
    """M[g,h] = AUROC(sig_h on keep_g&keep_h, |pred_g|). diagonal = headline (keep_g)."""
    K = len(units); M = np.full((K, K), np.nan)
    for a, g in enumerate(units):
        sc_full = np.abs(pred[g])
        for b_, h in enumerate(units):
            k = keep[g] & keep[h] if a != b_ else keep[g]
            sc = sc_full[k] if transform is None else transform(g, k)
            M[a, b_] = SL.auroc(sig[h][k], sc)
    return M

def summarise_cross(M, units, seed=0):
    K = len(units); d = np.diag(M); off = M.copy(); np.fill_diagonal(off, np.nan)
    mis = np.nanmean(off, 1); gap = d - mis
    # column view: for g's DE set (column g), percentile of pred_g among all predictions
    colpct = []
    for j in range(K):
        col = M[:, j]; f = np.isfinite(col); f[j] = False
        if f.sum() == 0 or not np.isfinite(M[j, j]): colpct.append(np.nan); continue
        o = col[f]; colpct.append(float(((M[j, j] > o).sum() + 0.5 * (M[j, j] == o).sum()) / len(o)))
    rowpct = []
    for i in range(K):
        r = off[i]; f = np.isfinite(r)
        rowpct.append(float(((d[i] > r[f]).sum() + 0.5 * (d[i] == r[f]).sum()) / f.sum()) if f.any() else np.nan)
    gf = gap[np.isfinite(gap)]
    out = dict(n_ko=int(np.isfinite(gap).sum()), n_pairs=int(np.isfinite(off).sum()),
               mean_matched=float(np.nanmean(d)), mean_mismatched=float(np.nanmean(mis)), gap=float(gf.mean()),
               gap_ci95=ko_boot(gap, seed=seed), frac_gap_pos=float((gf > 0).mean()),
               p_wilcoxon=float(wilcoxon(gf).pvalue) if len(gf) >= 5 and np.any(gf != 0) else None,
               specific_fraction=float(gf.mean() / (np.nanmean(d) - 0.5)) if np.nanmean(d) != 0.5 else None,
               mean_row_percentile=float(np.nanmean(rowpct)), mean_col_percentile=float(np.nanmean(colpct)),
               col_percentile_ci95=ko_boot(colpct, seed=seed + 1))
    return out, gap, mis

def paired_gap_boot(ga, gb, nb=NB, seed=0):
    ga, gb = np.asarray(ga), np.asarray(gb); f = np.isfinite(ga) & np.isfinite(gb); d = ga[f] - gb[f]
    rng = np.random.default_rng(seed); b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nb)])
    return dict(diff=float(d.mean()), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                p_boot=float(max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1 / (nb + 1))))

def ctrl_sd(t):
    X = lib.PB[t]["ct"]["cells_r"]; return np.asarray(X, float).std(0)

# ------------------------------------------------------------------ development screens
cross = {}; ko_rows = []; resid = {}
for t in TAGS:
    c = C[t]; units = c.evaluable
    sig = {g: c.sig[c.gi[g]] for g in units}; keep = {g: c.keep[c.gi[g]] for g in units}
    sdc = ctrl_sd(t); sdt = np.asarray(lib.META[t]["sd_r"], float)
    for nm in ["cmc_s42", "cmc_s1", "cmc_s2", "corr"]:
        pred = {g: SETS[nm][t][0][g][0] for g in units}
        M = cross_matrix(units, sig, keep, c.common, pred)
        s, gap, mis = summarise_cross(M, units); cross[f"{nm}|{t}"] = dict(s, _gap=gap, _mis=mis, _diag=np.diag(M))
        # residualisation on control-cell s.d. (primary) and training s.d. (secondary)
        rho_c = [spearmanr(np.abs(pred[g][keep[g]]), sdc[keep[g]]).correlation for g in units]
        rho_t = [spearmanr(np.abs(pred[g][keep[g]]), sdt[keep[g]]).correlation for g in units]
        au_raw = np.diag(M)
        au_res = np.array([SL.auroc(sig[g][keep[g]], resid_scores(np.abs(pred[g][keep[g]]), sdc[keep[g]])) for g in units])
        au_res_t = np.array([SL.auroc(sig[g][keep[g]], resid_scores(np.abs(pred[g][keep[g]]), sdt[keep[g]])) for g in units])
        au_sd = np.array([SL.auroc(sig[g][keep[g]], sdc[keep[g]]) for g in units])
        Mr = cross_matrix(units, sig, keep, c.common, pred, transform=lambda g, k: resid_scores(np.abs(pred[g][k]), sdc[k]))
        sr, gap_r, _ = summarise_cross(Mr, units, seed=5)
        resid[f"{nm}|{t}"] = dict(
            spearman_ctrl_sd_median=float(np.nanmedian(rho_c)), spearman_ctrl_sd_iqr=[float(np.nanpercentile(rho_c, 25)), float(np.nanpercentile(rho_c, 75))],
            spearman_train_sd_median=float(np.nanmedian(rho_t)),
            auroc_raw=float(np.nanmean(au_raw)), auroc_resid_ctrl_sd=float(np.nanmean(au_res)), auroc_resid_ctrl_sd_ci95=ko_boot(au_res, seed=2),
            resid_minus_raw=paired_gap_boot(au_res, au_raw, seed=3), auroc_resid_train_sd=float(np.nanmean(au_res_t)),
            auroc_resid_train_sd_ci95=ko_boot(au_res_t, seed=4), auroc_ctrl_sd_alone=float(np.nanmean(au_sd)),
            cross_ko_on_residualised=sr)
        for i, g in enumerate(units):
            ko_rows.append(dict(screen=t, method=nm, ko=g, auroc_matched=au_raw[i], auroc_mismatched_mean=mis[i], gap=gap[i],
                                spearman_ctrl_sd=rho_c[i], auroc_resid_ctrl_sd=au_res[i], auroc_resid_train_sd=au_res_t[i], auroc_ctrl_sd_alone=au_sd[i],
                                gap_resid=gap_r[i]))
    for s in ["s42", "s1", "s2"]:
        cross[f"cmc_{s}|{t}"]["vs_corr"] = paired_gap_boot(cross[f"cmc_{s}|{t}"]["_gap"], cross[f"corr|{t}"]["_gap"], seed=7)
    print(t, {k: round(v["gap"], 4) for k, v in cross.items() if k.endswith(t)}, flush=True)

# ------------------------------------------------------------------ Papalexi (untouched screen; cell-level DE as pre-registered)
UNT = f"{_CMC_RESULTS}/paper/analyses/untouched"
pap_src = pd.read_csv(f"{UNT}/source_data_papalexi_zeroshot_per_unit.csv")
for s, run, md5 in [("s42", "sp_dec_gl1000", "3a5d257c9323"), ("s1", "sp_dec_gl1000_s1", "7ff5ef2f5880"), ("s2", "sp_dec_gl1000_s2", "69407bfd5193")]:
    z = np.load(f"{UNT}/rc_{run}_preds.npz", allow_pickle=True)
    cf = [f"{UNT}/cache/rc_papalexi_{run}_{md5}_0.1_0.05.pkl"]; assert os.path.exists(cf[0]), cf
    b = pickle.load(open(cf[0], "rb")); al, de = b["align"], b["de"]; common = np.asarray(al["common_genes"])
    genes = sorted({k.split("|")[2] for k in z.files if k.startswith("cmc_obs|papalexi|")})
    sig = {g: np.asarray(de[g]["sig_rna"], bool) for g in genes}; keep = {g: common != g for g in genes}
    units = [g for g in genes if (sig[g] & keep[g]).sum() >= 3]
    sdc = np.sqrt(np.maximum(-z[f"zs_ctrl_var|papalexi|{units[0]}|r"].astype(float), 0))
    for nm, key in [(f"cmc_{s}", "cmc_obs"), ("corr", "corr")]:
        if nm == "corr" and s != "s42": continue
        pred = {g: z[f"{key}|papalexi|{g}|r"].astype(float) for g in units}
        M = cross_matrix(units, sig, keep, common, pred)
        srcm = "CMC-VAE " + run if key == "cmc_obs" else "corr"
        ref = pap_src[pap_src.method == srcm].set_index("gene")["rr_auroc"]
        repro = float(max(abs(M[i, i] - ref[g]) for i, g in enumerate(units)))
        sm, gap, mis = summarise_cross(M, units)
        rho_c = [spearmanr(np.abs(pred[g][keep[g]]), sdc[keep[g]]).correlation for g in units]
        au_res = np.array([SL.auroc(sig[g][keep[g]], resid_scores(np.abs(pred[g][keep[g]]), sdc[keep[g]])) for g in units])
        au_sd = np.array([SL.auroc(sig[g][keep[g]], sdc[keep[g]]) for g in units])
        cross[f"{nm}|papalexi"] = dict(sm, _gap=gap, _mis=mis, _diag=np.diag(M), repro_maxdiff_vs_source=repro, n_units=len(units))
        resid[f"{nm}|papalexi"] = dict(spearman_ctrl_sd_median=float(np.nanmedian(rho_c)),
                                       spearman_ctrl_sd_iqr=[float(np.nanpercentile(rho_c, 25)), float(np.nanpercentile(rho_c, 75))],
                                       auroc_raw=float(np.nanmean(np.diag(M))), auroc_resid_ctrl_sd=float(np.nanmean(au_res)),
                                       auroc_resid_ctrl_sd_ci95=ko_boot(au_res, seed=2), resid_minus_raw=paired_gap_boot(au_res, np.diag(M), seed=3),
                                       auroc_ctrl_sd_alone=float(np.nanmean(au_sd)))
        for i, g in enumerate(units):
            ko_rows.append(dict(screen="papalexi", method=nm, ko=g, auroc_matched=M[i, i], auroc_mismatched_mean=mis[i], gap=gap[i],
                                spearman_ctrl_sd=rho_c[i], auroc_resid_ctrl_sd=au_res[i], auroc_ctrl_sd_alone=au_sd[i]))
        print("papalexi", nm, "repro", repro, "gap", round(sm["gap"], 4), flush=True)
    cross[f"cmc_{s}|papalexi"]["vs_corr"] = paired_gap_boot(cross[f"cmc_{s}|papalexi"]["_gap"], cross["corr|papalexi"]["_gap"], seed=7)

# ------------------------------------------------------------------ Task-3 point per-KO AUROC (all sets)
t3 = []
for nm, d in SETS.items():
    for t in TAGS:
        ev = [g for g in C[t].evaluable if g in d[t][0]]
        for g, v in lib.rr_auroc_set(C[t], d[t], ev).items(): t3.append(dict(set=nm, screen=t, ko=g, rr_auroc=v))
pd.DataFrame(t3).to_csv(f"{lib.OUTD}/per_ko_rr_auroc_3seeds.csv", index=False)
pd.DataFrame(ko_rows).to_csv(f"{lib.OUTD}/per_ko_crossko_resid.csv", index=False)
for k in cross:
    for kk in ["_gap", "_mis", "_diag"]: cross[k].pop(kk, None)
RES["cross_ko"] = cross; RES["residualisation"] = resid
json.dump(RES, open(f"{lib.OUTD}/point_results.json", "w"), indent=1, default=float)
print("done")
