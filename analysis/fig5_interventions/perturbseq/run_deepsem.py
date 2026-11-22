"""Train DeepSEM (Shu et al. 2021, Nat Comput Sci) non-cell-type GRN on our combined_v4 RNA subset,
and SAVE THE FULL gene x gene SEM adjacency W (adj_A) as npz, for the matched-mutilation baseline.
DeepSEM's SEM uses (I - W^T)^-1 exactly like our latent SCM, so knockout propagation is identical
but in gene space. Small ensemble (avg signed W over seeds). RNA-only -> RR only.
nov-2026 revision copy (paper_nov_revisions/code/perturbseq): adds --line/--cells to train on EXACTLY the CMC-VAE
pretraining cells read from results/replogle/data/{line}_X.npy (cells=obs: NT + training-perturbation cells, all of
them, not a 20k subsample; cells=ctrl: non-targeting cells only). Standardisation (own cells' mean/sd) unchanged."""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, time, argparse, numpy as np, scanpy as sc, torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, f"{_CMC_DEEPSEM}")
from src.Model import VAE_EAD

ap = argparse.ArgumentParser()
ap.add_argument("--data_file", default="")
ap.add_argument("--line", default=""); ap.add_argument("--cells", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--n_epochs", type=int, default=40)
ap.add_argument("--n_seeds", type=int, default=3)
ap.add_argument("--seed_list", type=str, default="")  # e.g. "1" -> train only these seeds (overrides n_seeds)
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--n_hidden", type=int, default=128)
ap.add_argument("--alpha", type=float, default=100.0)   # L1 on W
ap.add_argument("--beta", type=float, default=1.0)      # KL
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--K1", type=int, default=1); ap.add_argument("--K2", type=int, default=2)
ap.add_argument("--spectral_clamp", type=float, default=0.0)  # if>0 renormalize adj_A each epoch to rho<=clamp
ap.add_argument("--save_vae", action="store_true")  # also save full VAE state_dict (for VAE-native fine-tuning)
opt = ap.parse_args()
Tensor = torch.cuda.FloatTensor

def load():
    if opt.line:
        import json, pandas as pd
        D = f"{_CMC_RESULTS}/replogle/data"; genes = json.load(open(f"{D}/features.json"))["features"]
        o = pd.read_csv(f"{D}/{opt.line}_obs.csv"); sel = o.is_train.values if opt.cells == "obs" else o.is_nt.values
        X = np.load(f"{D}/{opt.line}_X.npy", mmap_mode="r")[np.where(sel)[0]].astype(np.float32)
    else:
        d = sc.read_h5ad(opt.data_file); genes = list(d.var_names)
        X = d.X.toarray() if hasattr(d.X, "toarray") else np.asarray(d.X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return torch.FloatTensor(X), genes

def init_A(n):
    A = np.ones([n, n]) / (n - 1) + (np.random.rand(n * n) * 2e-4).reshape([n, n])
    np.fill_diagonal(A, 0.0); return A

def train_once(feat, n_gene, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dl = DataLoader(TensorDataset(feat, torch.LongTensor(range(len(feat))), torch.FloatTensor((feat != 0).float())),
                    batch_size=opt.batch_size, shuffle=True, num_workers=1)
    vae = VAE_EAD(init_A(n_gene), 1, opt.n_hidden, 1).float().cuda()
    o1 = optim.RMSprop(vae.parameters(), lr=opt.lr); o2 = optim.RMSprop([vae.adj_A], lr=opt.lr * 0.2)
    sched = torch.optim.lr_scheduler.StepLR(o1, step_size=1, gamma=0.95)
    vae.train()
    for ep in range(opt.n_epochs + 1):
        opt_adj = (ep % (opt.K1 + opt.K2)) >= opt.K1
        vae.adj_A.requires_grad = opt_adj
        losses = []
        for inputs, _, _ in dl:
            o1.zero_grad(); inputs = Variable(inputs.type(Tensor))
            temp = max(0.95 ** ep, 0.5)
            loss, lrec, lg, lc, dec, y, h = vae(inputs, dropout_mask=None, temperature=temp, opt=opt)
            loss = loss + opt.alpha * torch.mean(torch.abs(vae.adj_A))
            loss.backward(); losses.append(loss.item())
            (o2 if opt_adj else o1).step()
        sched.step()
        if opt.spectral_clamp > 0:
            with torch.no_grad():
                Wd = vae.adj_A.detach().cpu().numpy().copy(); import numpy as _np; _np.fill_diagonal(Wd,0.0)
                _r = float(_np.max(_np.abs(_np.linalg.eigvals(Wd))))
                if _r > opt.spectral_clamp: vae.adj_A.data.mul_(opt.spectral_clamp/_r)
        if ep % 10 == 0: print(f"  seed{seed} ep{ep} loss {np.mean(losses):.3f} {time.strftime('%T')}", flush=True)
    if opt.save_vae:
        import os as _os; _os.makedirs(_os.path.dirname(opt.out), exist_ok=True)
        torch.save({'state_dict': vae.state_dict(), 'genes': genes, 'seed': seed}, opt.out.replace('.npz', f'_vae_seed{seed}.pt'))
    return vae.adj_A.detach().cpu().numpy()

if __name__ == "__main__":
    feat, genes = load(); n_gene = feat.shape[1]
    print(f"DeepSEM: {feat.shape} cells x genes, {opt.n_seeds} seeds, {opt.n_epochs} epochs", flush=True)
    SEEDS = [int(x) for x in opt.seed_list.split(",")] if opt.seed_list else list(range(opt.n_seeds))
    Ws = [train_once(feat, n_gene, s) for s in SEEDS]
    W_mean = np.mean(Ws, 0)
    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    np.savez_compressed(opt.out, W=W_mean, W_seeds=np.stack(Ws), genes=np.array(genes, dtype=object))
    print(f"WROTE {opt.out}  W shape {W_mean.shape}", flush=True)
