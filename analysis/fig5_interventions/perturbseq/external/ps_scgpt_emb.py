"""scGPT (whole-human checkpoint, Cui et al. Nat Methods 2024) gene embeddings for the 'foundation-model embedding +
ridge' Perturb-seq baseline (the embedding variant of Ahlmann-Eltze et al. 2025). Two embeddings per line:
  scgpt_static : pretrained gene-token embedding table (encoder.embedding.weight, 512-d).
  scgpt_ctx    : contextual embedding -- the 12-layer scGPT encoder applied to NCELL control (non-targeting) cells of the
                 line (all 3,000 features as tokens, per-cell 51-bin value binning as in pretraining, no fine-tuning);
                 a gene's embedding = mean over cells of its output token. (No perturbation fine-tuning of scGPT.)
Features absent from the scGPT vocabulary get no embedding (those KOs are excluded from the embedding baselines;
coverage reported). Output: analyses/benchmark/ext/scgpt_emb_<line>.npz (names, <n>|genes, <n>|X).
usage: python ps_scgpt_emb.py LINE
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, time
import numpy as np, pandas as pd, torch
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
LINE = sys.argv[1]; NCELL = int(os.environ.get("SCGPT_NCELL", "1000"))
REPO = f"{_CMC_ROOT}"; D = f"{_CMC_RESULTS}/replogle/data"; B = f"{_CMC_RESULTS}/paper/analyses/benchmark"
CK = f"{_CMC_SCRATCH}/nov_benchmark_scratch/scgpt_dl/scGPT_human"
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.preprocess import binning
FEATS = json.load(open(f"{D}/features.json"))["features"]
vocab = json.load(open(f"{CK}/vocab.json")); args = json.load(open(f"{CK}/args.json"))
inv = [f for f in FEATS if f in vocab]; fi = np.array([FEATS.index(f) for f in inv]); tok = torch.tensor([vocab[f] for f in inv])
log(f"features in scGPT vocab: {len(inv)}/{len(FEATS)}")
sd = torch.load(f"{CK}/best_model.pt", map_location="cpu")
static = sd["encoder.embedding.weight"][tok].numpy().astype(np.float32)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerModel(len(vocab), args["embsize"], args["nheads"], args["d_hid"], args["nlayers"], vocab=vocab, dropout=0.0,
                         pad_token=args["pad_token"], pad_value=args["pad_value"], do_mvc=True, input_emb_style="continuous",
                         n_input_bins=args["n_bins"], cell_emb_style="cls", use_fast_transformer=False)
model = load_pretrained(model, sd, verbose=False); model.to(dev).eval()
obs = pd.read_csv(f"{D}/{LINE}_obs.csv"); nt = np.sort(np.random.RandomState(0).choice(np.where(obs.is_nt.values)[0], NCELL, replace=False))
X = np.load(f"{D}/{LINE}_X.npy", mmap_mode="r")[nt][:, fi].astype(np.float32)
acc = torch.zeros(len(inv), args["embsize"], dtype=torch.float64); n = 0
with torch.no_grad():
    for s in range(0, len(X), 8):
        xb = np.stack([binning(r, args["n_bins"]) for r in X[s:s + 8]]).astype(np.float32)
        src = tok.unsqueeze(0).repeat(len(xb), 1).to(dev); val = torch.from_numpy(xb).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out = model._encode(src, val, src_key_padding_mask=torch.zeros_like(src, dtype=torch.bool))
        acc += out.float().sum(0).double().cpu(); n += len(xb)
ctx = (acc / n).numpy().astype(np.float32)
os.makedirs(f"{B}/ext", exist_ok=True); o = f"{B}/ext/scgpt_emb_{LINE}.npz"
np.savez_compressed(o, names=np.array(["scgpt_static", "scgpt_ctx"]), **{"scgpt_static|genes": np.array(inv), "scgpt_static|X": static,
                    "scgpt_ctx|genes": np.array(inv), "scgpt_ctx|X": ctx})
log("DONE", o, static.shape, ctx.shape, "n cells", n)
