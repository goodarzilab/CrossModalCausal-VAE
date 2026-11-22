"""Pseudobulk limma-trend-like differential expression (numpy re-implementation).

Per knockout g: rows = pseudobulks (guide x stratum) of g plus the control pseudobulks (non-targeting guides x stratum)
from the SAME strata (condition / donor / HTO state) in which g has pseudobulks.  Design = [KO indicator, stratum
dummies] -> the KO coefficient is a condition-matched log-fold change (difference of mean log-normalized / CLR
expression between perturbed and control pseudobulks, stratum effects removed).  Residual variances are moderated
across features with Smyth's empirical-Bayes prior (fitFDist), estimated separately in 10 bins of average expression
(a binned version of limma-trend).  Two-sided moderated t P values; BH across features within each knockout.
"""
import numpy as np
from scipy import special, stats


def trigamma_inverse(x):
    """limma::trigammaInverse (Newton iteration)."""
    x = np.asarray(x, float)
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y = y + dif
        if np.all(-dif / y < 1e-8):
            break
    return y


def fit_fdist(s2, df):
    """Smyth (2004) moment estimate of the scaled-F prior (d0, s0^2) from residual variances s2 with df."""
    s2 = np.asarray(s2, float)
    ok = np.isfinite(s2) & (s2 > 1e-12)
    if ok.sum() < 3:
        return np.inf, float(np.nanmedian(s2[ok])) if ok.any() else 1e-6
    z = np.log(s2[ok])
    e = z - special.digamma(df / 2.0) + np.log(df / 2.0)
    emean = e.mean()
    evar = e.var(ddof=1) - special.polygamma(1, df / 2.0)
    if evar > 0:
        d0 = 2 * float(trigamma_inverse(evar))
        s0 = float(np.exp(emean + special.digamma(d0 / 2.0) - np.log(d0 / 2.0)))
    else:
        d0, s0 = np.inf, float(np.exp(emean))
    return d0, s0


def bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0:
        return p
    o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(n); q[o] = np.clip(r, 0, 1); return q


def limma_ko(Y_ko, s_ko, Y_ct, s_ct, nbins=10):
    """Y_ko [n1,F] KO pseudobulk means, s_ko [n1] stratum labels; Y_ct [n0,F] control pseudobulks, s_ct [n0].
    Returns beta [F], p [F], info dict.  Controls restricted to strata present among the KO pseudobulks."""
    strata = sorted(set(s_ko.tolist()))
    keep_c = np.isin(s_ct, strata)
    Y = np.vstack([Y_ko, Y_ct[keep_c]]); s = np.concatenate([s_ko, s_ct[keep_c]])
    n = Y.shape[0]
    X = np.zeros((n, 1 + len(strata)))
    X[:len(Y_ko), 0] = 1.0
    for j, st in enumerate(strata):
        X[s == st, 1 + j] = 1.0
    # drop stratum columns with no control rows (would be collinear with the KO column)
    good = [0] + [1 + j for j, st in enumerate(strata) if (s[len(Y_ko):] == st).any()]
    X = X[:, good]
    p_ = X.shape[1]; df = n - p_
    info = dict(n_ko_pb=int(len(Y_ko)), n_ct_pb=int(keep_c.sum()), n_strata=len(strata), df=int(df))
    if df < 1 or np.linalg.matrix_rank(X) < p_:
        F = Y.shape[1]
        return np.full(F, np.nan), np.ones(F), info
    XtXi = np.linalg.inv(X.T @ X)
    B = XtXi @ X.T @ Y
    R = Y - X @ B
    s2 = (R ** 2).sum(0) / df
    beta = B[0]; c = XtXi[0, 0]
    ave = Y.mean(0)
    s2p = np.empty_like(s2); dft = np.empty_like(s2)
    # binned limma-trend prior
    qs = np.quantile(ave, np.linspace(0, 1, nbins + 1)); qs[-1] += 1e-9
    b = np.clip(np.searchsorted(qs, ave, side="right") - 1, 0, nbins - 1)
    d0s = []
    for k in range(nbins):
        m = b == k
        if not m.any():
            continue
        d0, s0 = fit_fdist(s2[m], df); d0s.append(d0)
        if np.isinf(d0):
            s2p[m] = s0; dft[m] = 1e6
        else:
            s2p[m] = (d0 * s0 + df * s2[m]) / (d0 + df); dft[m] = d0 + df
    se = np.sqrt(np.clip(s2p * c, 1e-300, None))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dft)
    p[~np.isfinite(p)] = 1.0
    p[(np.abs(beta) < 1e-12)] = 1.0
    info["d0_median"] = float(np.median(d0s)) if d0s else float("nan")
    return beta, p, info
