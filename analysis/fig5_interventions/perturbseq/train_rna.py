"""Train the RNA-only CMC-VAE on the observational-training cells of one Replogle cell line.
Schedules copied from src/train.py ExperimentConfig.get_params with the sp_dec_gl1000 config:
120 epochs, warmup 20 (alpha=0), alpha ramp 30, tau 1.0 -> 0.5 after epoch 30, beta 1e-4 -> 2.0 linearly post-warmup,
free bits 1.0, AdamW lr 1e-3, grad clip 1, ReduceLROnPlateau(10, 0.5) on val rec, batch 256, 80/10/10 cell split.
Usage: python train_rna.py LINE SEED CELLS [EPOCHS]
  CELLS = obs  : non-targeting + training-perturbation cells (the original Supp Note 16 pretraining; z-scored with
                 results/replogle/data/{line}_stats.npz)
  CELLS = ctrl : non-targeting control cells ONLY (control-only pretraining, review M6/C5; z-scored with NT-only stats)
(nov-2026 revision copy of eval/replogle/train_rna.py; only the cell set / stats / output dir differ)
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
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rna_model import RnaCMC, loss_fn

LINE, SEED, CELLS = sys.argv[1], int(sys.argv[2]), sys.argv[3]; EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 120
assert CELLS in ("obs", "ctrl")
D = f"{_CMC_RESULTS}/replogle/data"
AN = f"{_CMC_RESULTS}/paper/analyses/perturbseq"
OUT = f"{AN}/models/{LINE}_{CELLS}{'' if EPOCHS == 120 else f'_ep{EPOCHS}'}_s{SEED}"; os.makedirs(OUT, exist_ok=True)
STATS = f"{D}/{LINE}_stats.npz" if CELLS == "obs" else f"{AN}/data/{LINE}_ntstats.npz"
CFG = dict(line=LINE, seed=SEED, cells=CELLS, stats=STATS, epochs=EPOCHS, latent=192, hidden=512, batch_size=256, lr=1e-3, grad_clip=1.0,
           warmup_epochs=20, alpha_ramp_epochs=30, tau_switch_epoch=30, tau_high=1.0, tau_low=0.5,
           beta_init=1e-4, beta_final=2.0, free_bits=1.0, rec_weight=100.0, lambda_stable=20.0, stable_rho_bound=0.9,
           lambda_ortho=5.0, lambda_gl=1000 * 0.05, n_groups=16, feature_dropout=0.1)
json.dump(CFG, open(f"{OUT}/config.json", "w"), indent=1)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)
if DEV.type == "cpu": torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

obs = pd.read_csv(f"{D}/{LINE}_obs.csv"); st = np.load(STATS)
sel = obs.is_train.values if CELLS == "obs" else obs.is_nt.values
X = np.load(f"{D}/{LINE}_X.npy", mmap_mode="r")[np.where(sel)[0]]
X = ((X - st["mean"]) / (st["std"] + 1e-6)).astype(np.float32)        # same z-scoring as src/train.py
Xt = torch.from_numpy(X); n = len(Xt); print(f"{LINE} s{SEED} [{CELLS}]: {n} training cells x {X.shape[1]} genes on {DEV}", flush=True)
itr, itmp = train_test_split(np.arange(n), test_size=0.2, random_state=SEED); iva, _ = train_test_split(itmp, test_size=0.5, random_state=SEED)
tl = DataLoader(TensorDataset(Xt[itr]), batch_size=CFG["batch_size"], shuffle=True, drop_last=True)
vl = DataLoader(TensorDataset(Xt[iva]), batch_size=1024)
model = RnaCMC(X.shape[1], CFG["latent"], CFG["hidden"], CFG["feature_dropout"]).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

def params(ep):
    if ep < CFG["warmup_epochs"]: alpha, beta = 0.0, CFG["beta_init"]
    else:
        alpha = min(1.0, (ep - CFG["warmup_epochs"]) / CFG["alpha_ramp_epochs"])
        prog = (ep - CFG["warmup_epochs"]) / max(EPOCHS - CFG["warmup_epochs"], 1)
        beta = CFG["beta_init"] + (CFG["beta_final"] - CFG["beta_init"]) * prog
    tau = CFG["tau_low"] if ep > CFG["tau_switch_epoch"] else CFG["tau_high"]
    p = {k: CFG[k] for k in ["rec_weight", "free_bits", "lambda_stable", "stable_rho_bound", "lambda_ortho", "lambda_gl", "n_groups"]}
    p["beta"] = beta; return p, alpha, tau

hist = []; T0 = time.time()
for ep in range(EPOCHS):
    p, alpha, tau = params(ep); model.train(); tr = []
    for (xb,) in tl:
        xb = xb.to(DEV); opt.zero_grad(); out = model(xb, tau, alpha); loss, m = loss_fn(out, xb, model, p)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"]); opt.step(); tr.append(m)
    model.eval(); va = []
    with torch.no_grad():
        for (xb,) in vl:
            xb = xb.to(DEV); out = model(xb, tau, alpha); va.append(loss_fn(out, xb, model, p)[1])
    tm = {k: float(np.mean([d[k] for d in tr])) for k in tr[0]}; vm = {k: float(np.mean([d[k] for d in va])) for k in va[0]}
    sch.step(vm["rec"]); hist.append(dict(epoch=ep, alpha=alpha, tau=tau, beta=p["beta"], train=tm, val=vm))
    if ep % 5 == 0 or ep == EPOCHS - 1:
        with torch.no_grad(): rho = float(torch.linalg.eigvals(model.scm.get_A().cpu()).abs().max())
        print(f"ep {ep:3d} a={alpha:.2f} b={p['beta']:.3f} | val rec {vm['rec']:.3f} (mse {vm['rec']/100:.4f}) kl {vm['kl']:.3f} "
              f"sigma {vm['rho']:.3f} eig-rho {rho:.3f} | {time.time()-T0:.0f}s", flush=True)
torch.save(model.state_dict(), f"{OUT}/final_model.pt"); json.dump(hist, open(f"{OUT}/history.json", "w"))
print("saved", OUT, flush=True)
