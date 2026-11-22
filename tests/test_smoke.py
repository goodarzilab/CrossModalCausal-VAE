"""Smoke test: build CMC-VAE from the released configs, load the released weights, run a forward pass.

Checks, for each released seed (42, 1, 2):
  * the model in ``src/model.py`` can be constructed from ``weights/<run>/config.json``;
  * ``final_model.pt`` loads with every parameter/buffer matched (strict load, except the
    training-only ``context_of_batch`` buffer);
  * a forward pass on random standardized inputs returns reconstructions of the right shape;
  * the latent adjacency blocks have shapes A_rr (128x128), W_rp (128x64), W_pr (64x128), A_pp (64x64);
  * the projected feature-space graph G = D A D^T has blocks gene-gene (2692x2692),
    gene-protein (2692x144), protein-gene (144x2692) and protein-protein (144x144).

Run from the repository root:  python -m pytest -q tests
"""
import json
import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model import CrossModalCausalLDVAE  # noqa: E402

N_GENES, N_PROTEINS = 2692, 144
RUNS = ["sp_dec_gl1000", "sp_dec_gl1000_s1", "sp_dec_gl1000_s2"]


def build_model(cfg):
    """Mirror the constructor call in src/train.py::run_experiment for the released configuration."""
    return CrossModalCausalLDVAE(
        dim_rna=N_GENES,
        dim_prot=N_PROTEINS,
        num_batches=1,  # no parameter depends on the number of training batches
        latent_dim_rna=cfg["latent_dim_rna"],
        latent_dim_prot=cfg["latent_dim_prot"],
        recon_likelihood=cfg.get("recon_likelihood", "mse"),
        gate_mode=cfg.get("gate_mode", "gumbel"),
        cross_modal=cfg.get("cross_modal", True),
        multiscale_A=cfg.get("multiscale_A", False),
        global_rank=cfg.get("global_rank", 8),
        global_scale=cfg.get("global_scale", 0.5),
        coupled_noise=cfg.get("coupled_noise", False),
        coupled_rank=cfg.get("coupled_rank", 8),
        rna_feature_dropout=cfg.get("rna_feature_dropout", 0.1),
        prot_feature_dropout=cfg.get("prot_feature_dropout", 0.0),
        scm_nonlinear=cfg.get("scm_nonlinear", False),
        phi=cfg.get("phi", "tanh"),
        n_fp_iters=cfg.get("n_fp_iters", 20),
        scm_init_scale=cfg.get("scm_init_scale", 1.0),
        eps_bottleneck=cfg.get("eps_bottleneck", False),
    )


@pytest.mark.parametrize("run", RUNS)
def test_load_and_forward(run):
    wdir = os.path.join(ROOT, "weights", run)
    cfg = json.load(open(os.path.join(wdir, "config.json")))
    assert cfg["latent_dim_rna"] == 128 and cfg["latent_dim_prot"] == 64

    model = build_model(cfg)
    sd = torch.load(os.path.join(wdir, "final_model.pt"), map_location="cpu")
    sd.pop("context_of_batch", None)  # training-only buffer
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert set(missing) <= {"context_of_batch"}, f"missing keys: {missing}"
    model.eval()

    torch.manual_seed(0)
    n = 16
    x_r, x_p = torch.randn(n, N_GENES), torch.randn(n, N_PROTEINS)
    with torch.no_grad():
        out = model(x_r, x_p, torch.zeros(n, dtype=torch.long), current_tau=cfg.get("tau_low", 0.5))
    rec_r, rec_p = out["rec"]
    assert rec_r.shape == (n, N_GENES) and rec_p.shape == (n, N_PROTEINS)
    assert torch.isfinite(rec_r).all() and torch.isfinite(rec_p).all()

    # latent adjacency blocks (source x target)
    A_rr, A_pp, W_rp, W_pr = model.scm.get_block_adjacencies()
    assert tuple(A_rr.shape) == (128, 128)
    assert tuple(W_rp.shape) == (128, 64)
    assert tuple(W_pr.shape) == (64, 128)
    assert tuple(A_pp.shape) == (64, 64)
    assert tuple(out["adj_full"].shape) == (192, 192)

    # projected feature-space graph G = D A D^T
    grn, rna_to_prot, prot_to_rna, ppi = model.get_projected_graphs()
    assert tuple(grn.shape) == (N_GENES, N_GENES)
    assert tuple(rna_to_prot.shape) == (N_GENES, N_PROTEINS)
    assert tuple(prot_to_rna.shape) == (N_PROTEINS, N_GENES)
    assert tuple(ppi.shape) == (N_PROTEINS, N_PROTEINS)
    assert torch.isfinite(grn).all()
