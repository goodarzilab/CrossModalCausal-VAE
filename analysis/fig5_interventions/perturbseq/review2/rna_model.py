"""RNA-only CMC-VAE: trimmed copy of src/model.py CrossModalCausalLDVAE (RNA block only).

Kept verbatim in design: ResidualEncoder (512 hidden) -> eps (latent exogenous noise) -> linear latent SCM
z = eps (I - A)^-1 with A = GumbelSigmoid(logits) * (1-I) * W (gated, signed, no self loops; closed-form solve with
1e-6 jitter) -> linear decoder (no bias, + per-gene bias). Losses as src/train.py loss_fn with the sp_dec_gl1000
config: 100*MSE, beta-ramped KL with free bits, 20*relu(sigma_max(A)-0.9) spectral penalty, 5*ortho, group lasso
(16 latent groups) on the decoder with weight 1000*0.05, no DAG/prior terms (dag_w=prior_w=0 in that config;
scm_sparse_target=None there too -> no SCM L1). Protein block / cross-modal blocks removed.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class ResidualEncoder(nn.Module):   # == src/model.py ResidualEncoder
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.LeakyReLU())
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_lv = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        h = self.net(x); return self.fc_mu(h), self.fc_lv(h)


class RnaScm(nn.Module):
    """RNA block of BlockStructuredScmLayer (gate_mode='gumbel', linear)."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.logits_rr = nn.Parameter(torch.randn(d, d) * 0.05)
        self.weights_rr = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.weights_rr)
        self.register_buffer("mask_rr", 1.0 - torch.eye(d))

    def _gate(self, logits, tau):
        if self.training:
            g = -torch.empty_like(logits).exponential_().log()
            return torch.sigmoid((logits + g) / tau)
        return torch.sigmoid(logits)

    def get_A(self, tau=0.5):
        return self._gate(self.logits_rr, tau) * self.mask_rr * self.weights_rr

    def _scm_solve(self, A, eps):
        I = torch.eye(self.d, device=eps.device)
        return torch.linalg.solve(I - A.T + 1e-6 * I, eps.T).T

    def forward(self, eps, tau=1.0, alpha=1.0):
        A = self.get_A(tau); z = self._scm_solve(A, eps)
        return (1 - alpha) * eps + alpha * z, A


class RnaCMC(nn.Module):
    def __init__(self, n_genes, latent=192, hidden=512, feature_dropout=0.1):
        super().__init__()
        self.latent_r = latent; self.feature_dropout = feature_dropout
        self.enc_rna = ResidualEncoder(n_genes, hidden, latent)
        self.scm = RnaScm(latent)
        self.dec_rna = nn.Linear(latent, n_genes, bias=False)
        self.bias_rna = nn.Parameter(torch.zeros(n_genes))

    def forward(self, x, tau=1.0, alpha=1.0):
        if self.training and self.feature_dropout > 0: x = x * (torch.rand_like(x) > self.feature_dropout)
        mu, lv = self.enc_rna(x)
        eps = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        z, A = self.scm(eps, tau, alpha)
        return dict(rec=self.dec_rna(z) + self.bias_rna, mu=mu, lv=lv, A=A)


def group_lasso(W, n_groups=16):
    G, L = W.shape; k = max(1, L // n_groups)
    return W[:, :k * (L // k)].reshape(G, L // k, k).norm(dim=2).sum() / G


def ortho(W):
    g = W.T @ W; return torch.norm(g - torch.eye(g.shape[0], device=W.device), p="fro") ** 2 / g.numel()


def loss_fn(out, x, model, p):
    rec = F.mse_loss(out["rec"], x) * p["rec_weight"]
    mu, lv = out["mu"], out["lv"]
    kl = (-0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp(), dim=0)).clamp_min(p["free_bits"]).mean()
    rho = torch.linalg.svdvals(out["A"])[0]
    stable = torch.relu(rho - p["stable_rho_bound"])
    W = model.dec_rna.weight
    tot = rec + p["beta"] * kl + p["lambda_stable"] * stable + p["lambda_ortho"] * ortho(W) + p["lambda_gl"] * group_lasso(W, p["n_groups"])
    return tot, dict(rec=rec.item(), kl=kl.item(), rho=rho.item(), total=tot.item())
