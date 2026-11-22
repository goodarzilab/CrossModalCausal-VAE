"""Standalone re-implementation of analysis/common/reviewer_controls.py::score() (the metrics used in Figs 3-5), operating on
dense arrays in the evaluation-feature order, so that DE ground truth can be re-derived inside a bootstrap and all
saved predictions re-scored.  Validated against the harness per-knockout output (see validate_scorer.py)."""
import numpy as np
from scipy.stats import rankdata

LFCR, LFCP = 0.10, 0.05


def auroc(lab, sc):
    lab = np.asarray(lab, bool); n1 = lab.sum(); n0 = len(lab) - n1
    if n1 < 3 or n0 < 1 or not np.isfinite(sc).all():
        return np.nan
    r = rankdata(sc)
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pe(p, o, m, kmin=5):
    m = np.asarray(m, bool)
    if m.sum() < kmin:
        return np.nan
    p = np.asarray(p, float)[m]; o = np.asarray(o, float)[m]
    if not (np.isfinite(p).all() and np.isfinite(o).all()) or p.std() == 0 or o.std() == 0:
        return np.nan
    return float(np.corrcoef(p, o)[0, 1])


def _z(v):
    v = v - v.mean(); n = np.linalg.norm(v); return v / n if n > 0 else v


class Cohort:
    """Observed responses for one cohort under one DE realisation.
    lfc [K,F], q/p [K,F] (sig computed outside), lfcp [K,Pn], sigp [K,Pn]; genes list; common feature names."""

    def __init__(self, genes, common, lfc, sig, lfcp, sigp, sd_r, sd_p, cog_mask, train_genes):
        self.genes = list(genes); self.gi = {g: i for i, g in enumerate(self.genes)}
        self.common = np.asarray(common); self.lfc = lfc; self.sig = sig; self.lfcp = lfcp; self.sigp = sigp
        self.sd_r = sd_r; self.sd_p = sd_p
        self.keep = np.stack([self.common != g for g in self.genes]) if len(self.genes) else np.zeros((0, len(common)), bool)
        self.evaluable = [g for g in self.genes if (self.sig[self.gi[g]] & self.keep[self.gi[g]]).sum() >= 3]
        self.has_p = {g: (sigp.shape[1] > 0 and sigp[self.gi[g]].sum() >= 3) for g in self.genes}
        ev = set(self.evaluable)
        u = np.zeros(len(common), bool)
        for g in train_genes:
            if g in ev: u |= self.sig[self.gi[g]]
        u &= ~np.isin(self.common, list(ev)); self.S = np.where(u)[0]
        cg = np.zeros(sigp.shape[1], bool)
        for g in ev: cg |= cog_mask.get(g, np.zeros(sigp.shape[1], bool))
        self.Sp = np.where(~cg)[0]


def score_set(C, preds, eval_genes, ref_genes, metrics=None):
    """preds: dict gene -> (rna[F], prot[Pn]).  Returns dict gene -> {metric: value}.  Mirrors harness score()."""
    eval_genes = [g for g in eval_genes if g in preds and g in C.gi]
    ref_genes = [g for g in ref_genes if g in preds and g in C.gi]
    out = {g: {} for g in eval_genes}
    if len(eval_genes) < 3 or not ref_genes:
        return out
    ri = [C.gi[g] for g in ref_genes]
    o_r = np.where(C.keep[ri], C.lfc[ri], np.nan)
    p_r = np.where(C.keep[ri], np.stack([preds[g][0] for g in ref_genes]), np.nan)
    o_p = C.lfcp[ri] if C.lfcp.shape[1] else None
    p_p = np.stack([preds[g][1] for g in ref_genes]) if C.lfcp.shape[1] else None
    pos = {g: j for j, g in enumerate(ref_genes)}
    sum_or, cnt_or = np.nansum(o_r, 0), np.isfinite(o_r).sum(0)
    sum_pr, cnt_pr = np.nansum(p_r, 0), np.isfinite(p_r).sum(0)
    def loo(sumv, cntv, arr, g):
        if g in pos:
            j = pos[g]; v = arr[j]; f = np.isfinite(v)
            s = sumv - np.where(f, v, 0); c = cntv - f
        else:
            s, c = sumv, cntv
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / np.maximum(c, 1), np.nan)
    if o_p is not None:
        sum_op, sum_pp = o_p.sum(0), p_p.sum(0); n_ref = len(ref_genes)
        def loop(sumv, arr, g):
            if g in pos: return (sumv - arr[pos[g]]) / max(n_ref - 1, 1)
            return sumv / n_ref
    MU = {}
    for g in eval_genes:
        MU[g] = (loo(sum_or, cnt_or, o_r, g), loo(sum_pr, cnt_pr, p_r, g))
    # retrieval (rr)
    S = C.S
    if len(S) >= 3:
        Pm = np.stack([_z((preds[g][0] - MU[g][1])[S]) for g in eval_genes])
        Om = np.stack([_z(((C.lfc[C.gi[g]] - MU[g][0]) / C.sd_r)[S]) for g in eval_genes])
        Cm = np.nan_to_num(Pm @ Om.T); d = np.diag(Cm)
        gt = (d[:, None] > Cm).sum(1); eq = (d[:, None] == Cm).sum(1) - 1; n = len(eval_genes) - 1
        for i, g in enumerate(eval_genes): out[g]["rr_retrieval"] = float((gt[i] + 0.5 * eq[i]) / n)
    gp = [g for g in eval_genes if C.has_p[g]]
    if o_p is not None and len(gp) >= 3 and len(C.Sp) >= 3:
        Sp = C.Sp
        MP = {g: (loop(sum_op, o_p, g), loop(sum_pp, p_p, g)) for g in gp}
        Pp = np.stack([_z((preds[g][1] - MP[g][1])[Sp]) for g in gp])
        Op = np.stack([_z(((C.lfcp[C.gi[g]] - MP[g][0]) / C.sd_p)[Sp]) for g in gp])
        if np.isfinite(Pp).all():
            Cp = np.nan_to_num(Pp @ Op.T); d = np.diag(Cp)
            gt = (d[:, None] > Cp).sum(1); eq = (d[:, None] == Cp).sum(1) - 1; n = len(gp) - 1
            for i, g in enumerate(gp): out[g]["rp_retrieval"] = float((gt[i] + 0.5 * eq[i]) / n)
    for g in eval_genes:
        i = C.gi[g]; k = C.keep[i]; sig = C.sig[i]; m = sig & k; lfc = C.lfc[i]; pr, pp = preds[g]
        mu_o, mu_p = MU[g]; r = out[g]; sd = C.sd_r
        r["n_de_rna"] = int(m.sum())
        r["rr_auroc"] = auroc(sig[k], np.abs(pr[k]))
        r["rr_pe"] = pe(pr, lfc, m)
        r["rr_pe_cent"] = pe(pr * sd - mu_p * sd, lfc - mu_o, m)
        r["rr_pe_meanresp"] = pe(mu_o, lfc, m)
        r["rr_auroc_meanresp"] = auroc(sig[k], np.abs(np.nan_to_num(mu_o[k])))
        if C.has_p[g] and o_p is not None:
            mp = C.sigp[i]; lp = C.lfcp[i]; mo, mq = MP[g] if g in gp and len(gp) >= 3 else (loop(sum_op, o_p, g), loop(sum_pp, p_p, g))
            r["n_de_prot"] = int(mp.sum())
            r["rp_auroc"] = auroc(mp, np.abs(pp))
            r["rp_pe"] = pe(pp, lp, mp)
            r["rp_pe_cent"] = pe((pp - mq) * C.sd_p, lp - mo, mp)
            r["rp_pe_meanresp"] = pe(mo, lp, mp)
            r["rp_auroc_meanresp"] = auroc(mp, np.abs(np.nan_to_num(mo)))
            r["rp_auroc_ge5"] = r["rp_auroc"] if mp.sum() >= 5 else np.nan
    return out


def pooled_protein(C, preds, genes, centre_ref=None):
    """Pooled per-feature analysis across knockouts: one correlation / AUROC over all (knockout, protein) pairs.
    genes: units with protein read-out.  Returns pooled raw pearDE (DE pairs), pooled centred pearDE (mean response
    of the other units removed from both pred and obs), pooled AUROC (DE vs non-DE pairs ranked by |pred|)."""
    genes = [g for g in genes if g in preds and g in C.gi and C.lfcp.shape[1]]
    if len(genes) < 3:
        return dict(n_units=len(genes))
    I = [C.gi[g] for g in genes]
    O = C.lfcp[I]; M = C.sigp[I]; Pp = np.stack([preds[g][1] for g in genes]).astype(float)
    if not np.isfinite(Pp).all():
        return dict(n_units=len(genes))
    K = len(genes)
    Oc = O - (O.sum(0, keepdims=True) - O) / (K - 1); Pc = Pp - (Pp.sum(0, keepdims=True) - Pp) / (K - 1)
    def cor(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 4 and a.std() > 0 and b.std() > 0 else np.nan
    return dict(n_units=K, n_pairs_de=int(M.sum()), n_pairs=int(M.size),
                pooled_pe=cor(Pp[M], O[M]), pooled_pe_cent=cor(Pc[M] * C.sd_p[None, :].repeat(K, 0)[M], Oc[M]),
                pooled_auroc=auroc(M.ravel(), np.abs(Pp).ravel()) if M.sum() >= 3 else np.nan,
                pooled_auroc_meanresp=auroc(M.ravel(), np.abs((O.sum(0, keepdims=True) - O) / (K - 1)).ravel()) if M.sum() >= 3 else np.nan)
