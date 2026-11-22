"""Task 9 prep: projected RNA maps |G_rr| = |D_r A_rr D_r^T| (off-diagonal) for a set of model dirs, saved as float32 npz.
usage: python -I task9_prep.py OUT.npz label=RUNDIR [label=RUNDIR ...]"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import sys, json, numpy as np, torch
R = f"{_CMC_ROOT}"; sys.path.insert(0, f"{R}/src"); sys.path.insert(0, f"{_CMC_ROOT}/analysis/common")
from model import CrossModalCausalLDVAE
import perturb_validation_analysis as P
JS = np.load(f"{_CMC_RESULTS}/paper/analyses/baselines/train_stats/joint_stats.npz", allow_pickle=True)
genes = list(map(str, JS["genes"])); NG, NP = len(genes), len(JS["prots"])
out = {"genes": np.array(genes)}
for a in sys.argv[2:]:
    lab, rd = a.split("=", 1); cfg = json.load(open(f"{rd}/config.json"))
    m = P.load_ctx_model(CrossModalCausalLDVAE, rd, NG, NP, cfg); m.eval()
    with torch.no_grad():
        A_rr = m.scm.get_block_adjacencies(tau=0.5)[0].double().numpy(); Dr = m._dec_r_w().detach().double().numpy()
    G = Dr @ A_rr @ Dr.T; np.fill_diagonal(G, 0.0); out[lab] = np.abs(G).astype(np.float32)
    print(lab, rd, "max", float(out[lab].max()), flush=True)
np.savez(sys.argv[1], **out)
