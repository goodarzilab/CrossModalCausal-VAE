"""Shared helpers for the reviewfix workstream."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import numpy as np
EMPCOV = f"{_CMC_SCRATCH}/nov_reviewfix/out/empcov_lw.npz"

class EmpCov:
    """Gaussian conditioning on gene g under the Ledoit-Wolf training-corpus covariance (raw log-norm / CLR scale):
    E[dx | dx_g = d] = Sigma[:, g] / Sigma[g, g] * d.  The knockdown d is the clamp's delta_g (z units) converted to log
    units with the corpus sd of g; the response is returned on the model's z scale (divided by the corpus sd of each
    feature), i.e. exactly the scale on which clamp predictions are scored."""
    def __init__(self, path=EMPCOV, kind="raw"):
        z = np.load(path, allow_pickle=True)
        S = z[f"S_{kind}"].astype(np.float64); s = float(z[f"shrink_{kind}"]); mu = float(z[f"mu_{kind}"])
        self.S = (1 - s) * S; self.S[np.diag_indices_from(self.S)] += s * mu
        self.shrink, self.kind = s, kind
        self.sd = np.clip(z["sd"].astype(np.float64), 1e-8, None)
        self.genes = list(z["genes"].astype(str)); self.prots = list(z["prots"].astype(str)); self.G = len(self.genes)
    def predict(self, gidx, delta_z, reti, preti, train_rna_std, train_prot_std):
        """gidx [K] model gene indices; delta_z [K]; reti/preti model indices of the scored RNA / protein features."""
        trs = np.clip(np.asarray(train_rna_std, float), 1e-8, None); tps = np.clip(np.asarray(train_prot_std, float), 1e-8, None)
        out_r, out_p = [], []
        for g, d in zip(np.asarray(gidx), np.asarray(delta_z, float)):
            col = self.S[:, g] / self.S[g, g]
            if self.kind == "raw":
                d_log = d * trs[g]
                out_r.append(col[:self.G][reti] * d_log / trs[reti]); out_p.append(col[self.G:][preti] * d_log / tps[preti] if len(preti) else np.zeros(0))
            else:   # z-scored covariance: conditioning directly in z units
                out_r.append(col[:self.G][reti] * d); out_p.append(col[self.G:][preti] * d if len(preti) else np.zeros(0))
        return np.array(out_r), np.array(out_p)
