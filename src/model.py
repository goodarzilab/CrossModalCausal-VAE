import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations

# ==========================================
# 1. BLOCK-STRUCTURED SCM LAYER
# ==========================================
class BlockStructuredScmLayer(nn.Module):
    def __init__(self, dim_rna, dim_prot, tau_init=1.0, gate_mode="gumbel", cross_modal=True,
                 multiscale=False, global_rank=8, global_scale=0.5,
                 n_context=1, context_cross_modal=False,
                 scm_nonlinear=False, phi="tanh", n_fp_iters=20, scm_init_scale=1.0,
                 nonlin_gain=1.0, fp_damping=0.0, scm_mechanism="elementwise", mlp_hidden=8):
        super().__init__()
        self.dim_r = dim_rna
        self.dim_p = dim_prot
        self.dim_total = dim_rna + dim_prot
        self.tau = tau_init
        # Engaged-nonlinearity knobs: pre-activation gain so phi operates in its CURVED regime (tanh with
        # tiny |A^T z| is ~identity -> effectively linear/unidentifiable), and damped (Mann) fixed-point
        # iteration z<-(1-d)z + d*(phi(gain*z@A)+eps) for convergence when gain*rho(A) approaches 1.
        self.nonlin_gain = nonlin_gain
        self.fp_damping = fp_damping
        # scm_mechanism: "elementwise" z=phi(g*z@A)+eps (default) | "mlp" per-node MLP over A-gated parents
        # (NOTEARS-MLP/GraN-DAG: enables parent INTERACTIONS, not just a linear combo) | "locscale"
        # heteroscedastic additive-noise z=phi(g*z@A)+softplus(a*z@A+b)*eps (Immer 2023, fits scRNA mean-var).
        self.scm_mechanism = scm_mechanism
        self.mlp_hidden = mlp_hidden
        if scm_nonlinear and scm_mechanism == "mlp":
            H = mlp_hidden; d = self.dim_total
            self.mlp_W1 = nn.Parameter(torch.randn(d, d, H) * 0.01)   # W1[i,j,h]: parent i -> node j hidden h
            self.mlp_b1 = nn.Parameter(torch.zeros(d, H))
            self.mlp_W2 = nn.Parameter(torch.randn(d, H) * (1.0 / max(H, 1)))
            self.mlp_b2 = nn.Parameter(torch.zeros(d))
        if scm_nonlinear and scm_mechanism == "locscale":
            self.ls_a = nn.Parameter(torch.zeros(self.dim_total))
            self.ls_b = nn.Parameter(torch.full((self.dim_total,), -1.0))  # softplus(-1)~0.31 baseline scale
        # Nonlinear structural equations: z = phi(A^T z) + eps, solved by K-step fixed-point (Picard)
        # iteration instead of the closed-form (I-A^T)^-1 eps. Linear is the default (phi=identity,
        # closed-form). phi=tanh has Lipschitz 1, so with spectral radius(A)<1 the map is a contraction
        # (unique fixed point, geometric convergence). At the origin phi'(0)=1 => the local/structural
        # projected GRN G = D_r A_rr D_r^T is EXACTLY the linear one; phi only saturates finite shifts.
        self.scm_nonlinear = scm_nonlinear
        self.phi = phi
        self.n_fp_iters = n_fp_iters
        # gate_mode: "gumbel" (Gumbel-Sigmoid) or "hardconcrete" (Louizos 2018 L0 gates)
        self.gate_mode = gate_mode
        # cross_modal=False zeros the RNA<->Protein blocks (W_rp, W_pr) -> intra-modal-only ablation
        self.cross_modal = cross_modal
        # multiscale: A_intra = A_local (sparse gated) + global_scale * low-rank dense global block.
        # DAGMA acyclicity is applied to A_local ONLY (the global block carries pathway-level effects).
        self.multiscale = multiscale
        self.global_scale = global_scale
        if multiscale:
            self.U_rr = nn.Parameter(torch.randn(dim_rna, global_rank) * 0.01)
            self.V_rr = nn.Parameter(torch.randn(dim_rna, global_rank) * 0.01)
            self.U_pp = nn.Parameter(torch.randn(dim_prot, global_rank) * 0.01)
            self.V_pp = nn.Parameter(torch.randn(dim_prot, global_rank) * 0.01)
        self.hc_beta = 0.5      # hard-concrete temperature
        self.hc_gamma = -0.1    # stretch lower bound (<0)
        self.hc_zeta = 1.1      # stretch upper bound (>1)
        
        # self.logits_rr = nn.Parameter(torch.randn(dim_rna, dim_rna) * 0.1 + 0.0) 
        # self.logits_pp = nn.Parameter(torch.randn(dim_prot, dim_prot) * 0.1 + 0.0)
        # self.weights_rr = nn.Parameter(torch.randn(dim_rna, dim_rna) * 0.05) # Was 0.01
        # self.weights_pp = nn.Parameter(torch.randn(dim_prot, dim_prot) * 0.05)

        # self.logits_rp = nn.Parameter(torch.randn(dim_rna, dim_prot) * 0.1 + 0.0)
        # self.logits_pr = nn.Parameter(torch.randn(dim_prot, dim_rna) * 0.1 + 0.0)
        # self.weights_rp = nn.Parameter(torch.randn(dim_rna, dim_prot) * 0.05)
        # self.weights_pr = nn.Parameter(torch.randn(dim_prot, dim_rna) * 0.05)
        
        # self.logits_rr= nn.Parameter(torch.zeros(dim_rna, dim_rna))
        # nn.init.xavier_uniform_(self.logits_rr)
        
        # self.logits_rp= nn.Parameter(torch.zeros(dim_rna, dim_prot))
        # nn.init.xavier_uniform_(self.logits_rp)
        
        # self.logits_pr= nn.Parameter(torch.zeros(dim_prot, dim_rna))
        # nn.init.xavier_uniform_(self.logits_pr)
        
        # self.logits_pp= nn.Parameter(torch.zeros(dim_prot, dim_prot))
        # nn.init.xavier_uniform_(self.logits_pp)
        
        # # Weights for Kinetics (Sign/Magnitude of edge)
        # self.weights_rr = nn.Parameter(torch.zeros(dim_rna, dim_rna))
        # nn.init.xavier_uniform_(self.weights_rr)
        
        # self.weights_rp = nn.Parameter(torch.zeros(dim_rna, dim_prot))
        # nn.init.xavier_uniform_(self.weights_rp)
        
        # self.weights_pr = nn.Parameter(torch.zeros(dim_prot, dim_rna))
        # nn.init.xavier_uniform_(self.weights_pr)
        
        # self.weights_pp = nn.Parameter(torch.zeros(dim_prot, dim_prot))
        # nn.init.xavier_uniform_(self.weights_pp)
        
        # self.logits_rr = nn.Parameter(torch.randn(dim_rna, dim_rna) * 0.05 + 0.0)
        # self.logits_pp = nn.Parameter(torch.randn(dim_prot, dim_prot) * 0.05 + 0.0)
        # self.weights_rr = nn.Parameter(torch.randn(dim_rna, dim_rna) * 0.01)
        # self.weights_pp = nn.Parameter(torch.randn(dim_prot, dim_prot) * 0.01)

        # self.logits_rp = nn.Parameter(torch.randn(dim_rna, dim_prot) * 0.05 + 0.0)
        # self.logits_pr = nn.Parameter(torch.randn(dim_prot, dim_rna) * 0.05 + 0.0)
        # self.weights_rp = nn.Parameter(torch.randn(dim_rna, dim_prot) * 0.01)
        # self.weights_pr = nn.Parameter(torch.randn(dim_prot, dim_rna) * 0.01)
        
        self.logits_rr = nn.Parameter(torch.randn(dim_rna, dim_rna) * 0.05)
        self.logits_pp = nn.Parameter(torch.randn(dim_prot, dim_prot) * 0.05)
        
        # --- WEIGHTS (Near-Zero Magnitude) ---
        # This makes the SCM act like an Identity Matrix at Epoch 0.
        # self.weights_rr = nn.Parameter(torch.randn(dim_rna, dim_rna) * 1e-4)
        # self.weights_pp = nn.Parameter(torch.randn(dim_prot, dim_prot) * 1e-4)
        
        self.weights_rr = nn.Parameter(torch.empty(dim_rna, dim_rna))
        nn.init.xavier_uniform_(self.weights_rr)
        self.weights_pp = nn.Parameter(torch.empty(dim_prot, dim_prot))
        nn.init.xavier_uniform_(self.weights_pp)

        # Do the same for inter-modal
        self.logits_rp = nn.Parameter(torch.randn(dim_rna, dim_prot) * 0.05)
        self.logits_pr = nn.Parameter(torch.randn(dim_prot, dim_rna) * 0.05)
        # self.weights_rp = nn.Parameter(torch.randn(dim_rna, dim_prot) * 1e-4)
        # self.weights_pr = nn.Parameter(torch.randn(dim_prot, dim_rna) * 1e-4)
        
        self.weights_rp = nn.Parameter(torch.empty(dim_rna, dim_prot))
        nn.init.xavier_uniform_(self.weights_rp)
        self.weights_pr = nn.Parameter(torch.empty(dim_prot, dim_rna))
        nn.init.xavier_uniform_(self.weights_pr)

        # gate_mode="none": no Gumbel/L0 gate -> A = mask * weights (raw SIGNED adjacency, +/- values,
        # never passed through sigmoid/softplus). Small randn*0.01 init so the SCM starts near identity;
        # sparsity then comes purely from the L1 penalty (set lambda_sparse_intra/inter + sum reduction).
        if gate_mode == "none":
            with torch.no_grad():
                for w in (self.weights_rr, self.weights_pp, self.weights_rp, self.weights_pr):
                    w.normal_(0.0, 0.01)

        # scm_init_scale<1 shrinks the initial edge weights so the SCM starts with a small spectral
        # radius / small pre-activations A^T z -> for the nonlinear SCM this keeps phi' near 1 (out of
        # tanh saturation) at init so A can propagate perturbations from the start.
        if scm_init_scale != 1.0:
            with torch.no_grad():
                for w in (self.weights_rr, self.weights_pp, self.weights_rp, self.weights_pr):
                    w.mul_(scm_init_scale)

        # --- per-context cross-modal coupling: each context gets its own W_rp/W_pr (gates+weights).
        # Intra-modal A_rr/A_pp stay SHARED. Same init across contexts (repeat) -> diverge in training.
        self.n_context = int(n_context)
        self.context_cross_modal = bool(context_cross_modal) and self.n_context > 1
        if self.context_cross_modal:
            self.logits_rp = nn.Parameter(self.logits_rp.detach().unsqueeze(0).repeat(self.n_context, 1, 1).clone())
            self.weights_rp = nn.Parameter(self.weights_rp.detach().unsqueeze(0).repeat(self.n_context, 1, 1).clone())
            self.logits_pr = nn.Parameter(self.logits_pr.detach().unsqueeze(0).repeat(self.n_context, 1, 1).clone())
            self.weights_pr = nn.Parameter(self.weights_pr.detach().unsqueeze(0).repeat(self.n_context, 1, 1).clone())

        # self.register_buffer('logits_pr', torch.zeros(dim_prot, dim_rna))
        # self.register_buffer('weights_pr', torch.zeros(dim_prot, dim_rna))
        
        # Masks to prevent self-loops in intra-modal blocks
        self.register_buffer('mask_rr', 1.0 - torch.eye(dim_rna))
        self.register_buffer('mask_pp', 1.0 - torch.eye(dim_prot))

    def _gumbel_sigmoid(self, logits, tau):
        """Helper to sample differentiable binary masks."""
        if self.training:
            # Sample Gumbel noise
            gumbels = -torch.empty_like(logits).exponential_().log()
            # (Logits + Noise) / Temp
            gumbels = (logits + gumbels) / tau
            # Sigmoid to get continuous relaxation of {0, 1}
            probs = torch.sigmoid(gumbels)
        else:
            # For inference, just use the logistic sigmoid probability
            probs = torch.sigmoid(logits)
        return probs

    def _hardconcrete(self, logits):
        """Louizos et al. 2018 hard-concrete gate (stretched-sigmoid, can hit exact 0/1)."""
        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + logits) / self.hc_beta)
        else:
            s = torch.sigmoid(logits)
        s_stretched = s * (self.hc_zeta - self.hc_gamma) + self.hc_gamma
        return torch.clamp(s_stretched, 0.0, 1.0)

    def _gate(self, logits, tau):
        if self.gate_mode == "none":
            return torch.ones_like(logits)   # no gating: A = mask * weights (raw signed adjacency)
        return self._hardconcrete(logits) if self.gate_mode == "hardconcrete" else self._gumbel_sigmoid(logits, tau)

    def _cross_blocks(self, t, ctx=0):
        """Inter-modal blocks for a given context (indexes the per-context params if enabled)."""
        if getattr(self, 'context_cross_modal', False):
            lrp, wrp = self.logits_rp[ctx], self.weights_rp[ctx]
            lpr, wpr = self.logits_pr[ctx], self.weights_pr[ctx]
        else:
            lrp, wrp, lpr, wpr = self.logits_rp, self.weights_rp, self.logits_pr, self.weights_pr
        W_rp = self._gate(lrp, t) * wrp
        W_pr = self._gate(lpr, t) * wpr
        if not self.cross_modal:   # ablate RNA<->Protein coupling (intra-modal-only)
            W_rp = torch.zeros_like(W_rp)
            W_pr = torch.zeros_like(W_pr)
        return W_rp, W_pr

    def get_block_adjacencies(self, tau=None, ctx=0):
        t = tau if tau is not None else self.tau

        # --- Intra-modal (Mask Diagonals): P(Edge) * Mask(No Self Loop) * Weight ---
        prob_rr = self._gate(self.logits_rr, t)
        A_rr = prob_rr * self.mask_rr * self.weights_rr
        if getattr(self, 'zero_A_rr', False):   # review2 M4a: A_rr fixed to 0 (no gradient to its gates/weights)
            A_rr = torch.zeros_like(A_rr).detach()
        prob_pp = self._gate(self.logits_pp, t)
        A_pp = prob_pp * self.mask_pp * self.weights_pp

        # multi-scale: keep the LOCAL (sparse) intra blocks for DAGMA, add a low-rank dense GLOBAL term
        self._A_local_rr, self._A_local_pp = A_rr, A_pp
        if self.multiscale:
            A_rr = A_rr + self.global_scale * ((self.U_rr @ self.V_rr.T) * self.mask_rr)
            A_pp = A_pp + self.global_scale * ((self.U_pp @ self.V_pp.T) * self.mask_pp)

        # --- Inter-modal (per-context if enabled; ctx ignored when shared) ---
        W_rp, W_pr = self._cross_blocks(t, ctx)
        return A_rr, A_pp, W_rp, W_pr

    def get_gate_probs(self):
        """Deterministic edge open-probabilities sigmoid(logits), used to penalize the
        *gates* (graph structure) rather than the edge *weights* (L0/L1-on-gates, à la
        DCDI / Louizos et al. 2018). No Gumbel noise / temperature so the penalty is stable.
        Intra-modal diagonals are masked out so self-loops are never penalized.
        For hard-concrete gates this is the expected-L0 P(gate>0) (Louizos 2018)."""
        def open_prob(logits):
            if self.gate_mode == "hardconcrete":
                import math
                return torch.sigmoid(logits - self.hc_beta * math.log(-self.hc_gamma / self.hc_zeta))
            return torch.sigmoid(logits)
        p_rr = open_prob(self.logits_rr) * self.mask_rr
        if getattr(self, 'zero_A_rr', False):
            p_rr = torch.zeros_like(p_rr).detach()
        p_pp = open_prob(self.logits_pp) * self.mask_pp
        p_rp = open_prob(self.logits_rp)
        p_pr = open_prob(self.logits_pr)
        return p_rr, p_pp, p_rp, p_pr

    def _phi(self, u):
        if self.phi == "tanh":    return torch.tanh(u)
        if self.phi == "relu":    return torch.relu(u)
        # Asymmetric, curvature-at-origin activations: engage nonlinearity even for small |A^T z| and
        # break the sign-flip symmetry that odd phi (tanh) preserves. All are 1-Lipschitz (contraction
        # holds with gain*rho(A)<1). Centered so phi(0)=0 -> origin Jacobian = gain*A (projection exact).
        if self.phi == "elu":      return F.elu(u)                       # elu(0)=0, elu'(0)=1, curved for u<0
        if self.phi == "softplus": return F.softplus(u) - 0.6931471805599453
        if self.phi == "gelu":     return F.gelu(u)
        if self.phi == "sine":     return torch.sin(u)             # nonlinear-ICA style; 1-Lipschitz, highly nonlinear
        if self.phi in ("none", "identity"): return u
        raise ValueError(f"unknown phi {self.phi}")

    def _scm_solve(self, A_full, eps):
        """Solve the structural equations for z given exogenous eps and the [d,d] latent adjacency
        A_full (A_full[j,i] = weight of edge j->i). Batch convention: eps,z are [B,d] row vectors, so
        (A^T z)_row = z @ A_full. Linear: z=(I-A^T)^{-1}eps (closed form). Nonlinear: z=phi(z@A_full)+eps
        by fixed-point iteration (z0=eps; converges since phi is 1-Lipschitz and rho(A)<1)."""
        if not self.scm_nonlinear:
            I = torch.eye(self.dim_total, device=eps.device)
            return torch.linalg.solve(I - A_full.T + 1e-6 * I, eps.T).T
        g = self.nonlin_gain; d = self.fp_damping; mech = getattr(self, "scm_mechanism", "elementwise")
        def step(z):
            if mech == "mlp":
                # per-node MLP over A-gated parents: M[b,i,j]=A[i,j]*z_i -> hidden_j=phi(sum_i M W1) -> f_j
                M = z.unsqueeze(2) * A_full.unsqueeze(0)                       # [B, i(parent), j(node)]
                hid = self._phi(torch.einsum('bij,ijh->bjh', M, self.mlp_W1) + self.mlp_b1)
                f = torch.einsum('bjh,jh->bj', hid, self.mlp_W2) + self.mlp_b2
                return g * f + eps
            if mech == "locscale":
                pre = z @ A_full
                scale = F.softplus(self.ls_a * pre + self.ls_b).clamp(0.05, 3.0)  # bound noise scale -> avoid blow-up
                return self._phi(g * pre) + scale * eps
            return self._phi(g * (z @ A_full)) + eps                          # elementwise (default)
        z = eps
        for _ in range(self.n_fp_iters):
            zt = step(z)
            z = (1 - d) * z + d * zt if d > 0 else zt          # damped (Mann) iteration for stability
        self._last_fp_resid = (z - step(z)).abs().mean().detach()             # convergence probe
        return z

    def forward(self, eps, tau=None, alpha=1.0, ctx_id=None):
        """SCM: z = (I - A^T)^{-1} eps (linear) or z = phi(A^T z)+eps (nonlinear fixed-point).
        If context_cross_modal, the inter-modal block is selected
        per-cell from ctx_id (a [B] long tensor); cells of each context are solved with that context's
        A_full. Returns A_full for context 0 (rho/stability proxy) and stacked cross-modal blocks for
        the inter-sparsity penalty so every context's W stays regularized."""
        t = tau if tau is not None else self.tau
        A_rr, A_pp, W_rp0, W_pr0 = self.get_block_adjacencies(t, ctx=0)
        I = torch.eye(self.dim_total, device=eps.device); jitter = 1e-6 * I

        def assemble(wrp, wpr):
            return torch.cat([torch.cat([A_rr, wrp], dim=1), torch.cat([wpr, A_pp], dim=1)], dim=0)

        A_full = assemble(W_rp0, W_pr0)
        if not getattr(self, 'context_cross_modal', False) or ctx_id is None:
            z_scm = self._scm_solve(A_full, eps)
            blocks = (A_rr, A_pp, W_rp0, W_pr0)
        else:
            z_scm = torch.zeros_like(eps)
            Wrp_list, Wpr_list = [W_rp0], [W_pr0]
            m0 = (ctx_id == 0)
            if m0.any():
                z_scm[m0] = self._scm_solve(A_full, eps[m0])
            for c in range(1, self.n_context):
                W_rp_c, W_pr_c = self._cross_blocks(t, c)
                Wrp_list.append(W_rp_c); Wpr_list.append(W_pr_c)
                m = (ctx_id == c)
                if m.any():
                    A_c = assemble(W_rp_c, W_pr_c)
                    z_scm[m] = self._scm_solve(A_c, eps[m])
            blocks = (A_rr, A_pp, torch.stack(Wrp_list), torch.stack(Wpr_list))

        z = (1.0 - alpha) * eps + alpha * z_scm
        return z, blocks, A_full

# ==========================================
# 2. ENCODER
# ==========================================
class ResidualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_lv = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_lv(h)

# ==========================================
# 3. BLOCK CROSS-MODAL CAUSAL LDVAE
# ==========================================
class CrossModalCausalLDVAE(nn.Module):
    def __init__(self, dim_rna, dim_prot, num_batches, latent_dim_rna=128, latent_dim_prot=64,
                 recon_likelihood="mse", gate_mode="gumbel", cross_modal=True,
                 multiscale_A=False, global_rank=8, global_scale=0.5,
                 coupled_noise=False, coupled_rank=8,
                 context_adapt=None, n_context=1, context_cross_modal=False,
                 rna_feature_dropout=0.1, prot_feature_dropout=0.0,
                 modality_dropout_p=0.0, modality_dropout_mode="both", modality_dropout_null="zero",
                 scm_nonlinear=False, phi="tanh", n_fp_iters=20, scm_init_scale=1.0,
                 nonlin_gain=1.0, fp_damping=0.0, scm_mechanism="elementwise", mlp_hidden=8,
                 dec_mask_rna=None, dec_mask_lambda=0.1, dec_mask_freeze=False,
                 eps_bottleneck=False, eps_gate_init=2.0, eps_bottleneck_target="both",
                 eps_prior="gaussian", eps_df=3.0,
                 cond_prior_on=None, cond_prior_n_classes=1, cond_prior_target="both"):
        super().__init__()
        self.dim_rna = dim_rna
        self.dim_prot = dim_prot
        self.latent_r = latent_dim_rna
        self.latent_p = latent_dim_prot
        self.recon_likelihood = recon_likelihood
        # Exogenous-prior family for eps. "gaussian" = standard VAE (closed-form KL; BYTE-IDENTICAL to
        # the original model — all non-gaussian code is guarded behind eps_prior != "gaussian"). "laplace"
        # / "studentt" = reparameterized non-Gaussian posterior q(eps|x) with a matching non-Gaussian
        # prior p(eps); KL(q||p) is then estimated by a single-sample MC estimate E_q[log q - log p]
        # (see reparameterize / _mc_kl_vec below and the KL block in train.loss_fn).
        self.eps_prior = eps_prior
        self.eps_df = float(eps_df)   # degrees-of-freedom for the Student-t prior (ignored otherwise)
        self.coupled_noise = coupled_noise
        # context_adapt: None | "latent" (per-context affine on eps = #6/#8 CPA cell-state) |
        #                "decoder" (per-context per-gene FiLM on the readout = #10).
        # context resolved per-cell from batch via context_of_batch (training) or self.eval_context (eval).
        self.context_adapt = context_adapt
        self.context_cross_modal = context_cross_modal
        self.n_context = n_context
        self.eval_context = 0
        self.register_buffer("context_of_batch", torch.zeros(0, dtype=torch.long))
        # --- modality / feature dropout (cross-modal prediction training) ---
        # feature dropout: randomly zero input features per modality (denoising).
        # modality dropout: replace a whole modality's eps with a learned NULL before the SCM, so the
        # cross-modal block must fill that modality's latent from the other -> forces RNA->Protein
        # (and Protein->RNA) prediction. mode in {"both","protein","rna"}.
        self.rna_feature_dropout = rna_feature_dropout
        self.prot_feature_dropout = prot_feature_dropout
        self.modality_dropout_p = modality_dropout_p
        self.modality_dropout_mode = modality_dropout_mode
        # null type: "zero" (FIX — bias holds the mean, cross-modal block must do per-cell work) vs
        # "learned" (a learnable null token, which empirically absorbs the marginal mean -> shortcut).
        self.modality_dropout_null = modality_dropout_null
        self.null_r = nn.Parameter(torch.zeros(latent_dim_rna))
        self.null_p = nn.Parameter(torch.zeros(latent_dim_prot))
        if context_adapt == "latent":
            L = latent_dim_rna + latent_dim_prot
            self.ctx_shift = nn.Parameter(torch.zeros(n_context, L))
            self.ctx_logscale = nn.Parameter(torch.zeros(n_context, L))
        elif context_adapt == "decoder":
            self.ctx_dec_logscale_r = nn.Parameter(torch.zeros(n_context, dim_rna))
            self.ctx_dec_bias_r = nn.Parameter(torch.zeros(n_context, dim_rna))
            self.ctx_dec_logscale_p = nn.Parameter(torch.zeros(n_context, dim_prot))
            self.ctx_dec_bias_p = nn.Parameter(torch.zeros(n_context, dim_prot))

        self.enc_rna = ResidualEncoder(dim_rna, 512, latent_dim_rna)
        self.enc_prot = ResidualEncoder(dim_prot, 128, latent_dim_prot)

        self.scm = BlockStructuredScmLayer(latent_dim_rna, latent_dim_prot, gate_mode=gate_mode,
                                           cross_modal=cross_modal, multiscale=multiscale_A,
                                           global_rank=global_rank, global_scale=global_scale,
                                           n_context=n_context, context_cross_modal=context_cross_modal,
                                           scm_nonlinear=scm_nonlinear, phi=phi, n_fp_iters=n_fp_iters,
                                           scm_init_scale=scm_init_scale,
                                           nonlin_gain=nonlin_gain, fp_damping=fp_damping,
                                           scm_mechanism=scm_mechanism, mlp_hidden=mlp_hidden)
        # coupled-noise: shared low-rank factor injecting cross-modal correlation into (eps_r, eps_p)
        if coupled_noise:
            self.coupling_r = nn.Parameter(torch.randn(latent_dim_rna, coupled_rank) * 0.01)
            self.coupling_p = nn.Parameter(torch.randn(latent_dim_prot, coupled_rank) * 0.01)
            self.coupled_rank = coupled_rank
        
        self.dec_rna = nn.Linear(latent_dim_rna, dim_rna, bias=False)
        self.dec_prot = nn.Linear(latent_dim_prot, dim_prot, bias=False)

        # Pathway-structured RNA decoder (SENA-style): effective decoder weight = W (x) M, where M is the
        # gene->pathway membership mask (1 in-set, dec_mask_lambda out-of-set). Forces gene-gene structure
        # OUT of the free decoder and INTO the latent SCM A_rr. latent_dim_rna must == n_pathways.
        self.dec_mask_freeze = dec_mask_freeze
        if dec_mask_rna is not None:
            mem = torch.as_tensor(dec_mask_rna).float()
            assert mem.shape == (dim_rna, latent_dim_rna), \
                f"dec_mask_rna {tuple(mem.shape)} != (dim_rna,latent_dim_rna)={(dim_rna, latent_dim_rna)}"
            self.register_buffer("dec_r_mask", mem + dec_mask_lambda * (1.0 - mem))
            if dec_mask_freeze:
                with torch.no_grad():
                    self.dec_rna.weight.copy_(self.dec_r_mask)
                self.dec_rna.weight.requires_grad_(False)
        else:
            self.dec_r_mask = None
        
        # self.dec_rna = parametrizations.orthogonal(nn.Linear(latent_dim_rna, dim_rna, bias=False))
        # self.dec_prot = parametrizations.orthogonal(nn.Linear(latent_dim_prot, dim_prot, bias=False))
        
        self.bias_rna = nn.Parameter(torch.zeros(dim_rna))
        self.bias_prot = nn.Parameter(torch.zeros(dim_prot))

        # --- CONDITIONAL exogenous prior p(eps|u)=N(mu(u), diag(exp(lv(u)))) (iVAE / CausalVAE
        #     auxiliary-variable prior, Khemakhem 2020) --------------------------------------------
        # OFF (cond_prior_on is None) => NO parameters created (so torch RNG / param-init order is
        # UNCHANGED -> byte-identical to the baseline) and forward emits cond_prior=None so loss_fn
        # uses the exact N(0,I) closed-form KL. When ON, a per-class embedding maps the one-hot
        # auxiliary u (celltype/donor/disease) -> (prior_mu, prior_logvar) over the joint RNA+protein
        # latent. Zero-init => at step 0 the prior == N(0,I); it then moves to the per-class aggregated
        # posterior, giving Khemakhem's identifiability signal. Only the KL uses it (the eps drawn for
        # reconstruction stays the posterior sample q(eps|x)).
        self.cond_prior_on = cond_prior_on
        self.cond_prior_target = cond_prior_target   # "both" (joint eps) | "rna" | "prot": which exogenous the aux prior conditions
        if cond_prior_on is not None:
            L = latent_dim_rna + latent_dim_prot
            self.cond_prior_mu = nn.Embedding(int(cond_prior_n_classes), L)
            self.cond_prior_lv = nn.Embedding(int(cond_prior_n_classes), L)
            nn.init.zeros_(self.cond_prior_mu.weight)
            nn.init.zeros_(self.cond_prior_lv.weight)

        # --- Exogenous ε-bottleneck (global L0 gate on eps) ---------------------------------------
        # Breaks the eps->z bijection that makes A non-load-bearing: a GLOBAL (shared-across-cells)
        # per-dim hard-concrete gate multiplies eps_joint. Dims whose gate closes get NO exogenous
        # input, so their latent can be produced ONLY by A propagating from the surviving "root" dims
        # -> A becomes load-bearing by construction. Global (not per-cell) so the closed dims are truly
        # endogenous (a per-cell gate would leak: the union of supports still spans the space).
        # eps_gate_init>0 starts gates OPEN (open-prob~0.88 at 2.0) so training begins ~unbottlenecked
        # and the ramped L0 penalty (train.py) closes dims gradually. Louizos et al. 2018 hard-concrete.
        self.eps_bottleneck = bool(eps_bottleneck)
        self.eps_bottleneck_target = eps_bottleneck_target   # "both" | "rna" | "prot"
        if self.eps_bottleneck:
            L = latent_dim_rna + latent_dim_prot
            self.eps_gate_logits = nn.Parameter(torch.full((L,), float(eps_gate_init)))
            self._eps_hc_beta, self._eps_hc_gamma, self._eps_hc_zeta = 0.5, -0.1, 1.1
            # which eps dims are gateable (target modality); the rest stay always-open (not penalized).
            # Gating ONLY eps_r targets making A_rr load-bearing without disrupting the (already
            # load-bearing) cross-modal protein path.
            active = torch.ones(L)
            if eps_bottleneck_target == "rna":  active[latent_dim_rna:] = 0.0
            elif eps_bottleneck_target == "prot": active[:latent_dim_rna] = 0.0
            self.register_buffer("eps_gate_active", active.bool())

        # Negative-binomial dispersions (per feature) — only for recon_likelihood="nb".
        # Decoder output is treated as LDVAE-style softmax rate logits; mu = library * softmax(Dz+b).
        if recon_likelihood == "nb":
            self.log_theta_rna = nn.Parameter(torch.zeros(dim_rna))
            self.log_theta_prot = nn.Parameter(torch.zeros(dim_prot))
        
        # self.batch_bias_rna = nn.Embedding(num_batches, dim_rna)
        # self.batch_bias_prot = nn.Embedding(num_batches, dim_prot)

        # Initialize to 0
        # nn.init.zeros_(self.batch_bias_rna.weight)
        # nn.init.zeros_(self.batch_bias_prot.weight)

    def reparameterize(self, mu, lv):
        std = torch.exp(0.5 * lv)
        # GAUSSIAN (default): byte-identical to the original model — this branch is unchanged and the
        # non-gaussian branches below never execute unless eps_prior is explicitly set.
        if getattr(self, "eps_prior", "gaussian") == "gaussian":
            return mu + torch.randn_like(std) * std
        # NON-GAUSSIAN posterior q(eps|x). We reinterpret the encoder's log-variance head lv as 2*log(scale)
        # so `std = exp(0.5*lv)` becomes the distribution's SCALE parameter (analogous role to the Gaussian
        # std). rsample() is the reparameterized (pathwise-differentiable) draw: Laplace uses the inverse-CDF
        # transform of a uniform, Student-t uses the Normal/Gamma reparameterization.
        if self.eps_prior == "laplace":
            return torch.distributions.Laplace(mu, std).rsample()
        if self.eps_prior == "studentt":
            return torch.distributions.StudentT(self.eps_df, mu, std).rsample()
        raise ValueError(f"unknown eps_prior {self.eps_prior}")

    def _mc_kl_vec(self, mu, lv, sample):
        """Per-latent-dim single-sample MC estimate of KL(q(eps|x) || p(eps)) for a NON-GAUSSIAN prior:
        returns a length-L vector E_batch[log q(eps) - log p(eps)] (mean over the batch, per dim), so the
        caller can apply the same per-dim free-bits floor + mean-over-dims reduction as the closed-form
        Gaussian KL. q = chosen family with loc=mu, scale=exp(0.5*lv); p = the same family standardized
        (loc 0, scale 1). Only called when eps_prior != 'gaussian'."""
        scale = torch.exp(0.5 * lv)
        zero, one = torch.zeros_like(mu), torch.ones_like(mu)
        if self.eps_prior == "laplace":
            q = torch.distributions.Laplace(mu, scale)
            p = torch.distributions.Laplace(zero, one)
        elif self.eps_prior == "studentt":
            df = torch.full_like(mu, self.eps_df)
            q = torch.distributions.StudentT(df, mu, scale)
            p = torch.distributions.StudentT(df, zero, one)
        else:
            raise ValueError(f"unknown eps_prior {self.eps_prior}")
        return (q.log_prob(sample) - p.log_prob(sample)).mean(dim=0)   # [L]

    def _eps_gate(self):
        """Global per-dim hard-concrete L0 gate on the exogenous eps. Returns (gate, open_prob):
        `gate` (sampled in train, deterministic in eval) multiplies eps_joint; `open_prob` (the
        deterministic expected-L0 P(gate>0)) feeds the L0 penalty. Closed dims -> eps=0 -> that latent
        is producible only via A (load-bearing)."""
        import math
        lg = self.eps_gate_logits
        if self.training:
            u = torch.rand_like(lg).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + lg) / self._eps_hc_beta)
        else:
            s = torch.sigmoid(lg)
        gate = torch.clamp(s * (self._eps_hc_zeta - self._eps_hc_gamma) + self._eps_hc_gamma, 0.0, 1.0)
        open_prob = torch.sigmoid(lg - self._eps_hc_beta * math.log(-self._eps_hc_gamma / self._eps_hc_zeta))
        # restrict to target modality: non-gateable dims stay always-open (gate=1) and are not penalized
        act = self.eps_gate_active
        gate = torch.where(act, gate, torch.ones_like(gate))
        open_prob = open_prob * act.to(open_prob.dtype)
        return gate, open_prob

    def _ctx_vec(self, batch_idx, B, device):
        """Resolve a [B] context index: from batch via context_of_batch (training) else eval_context."""
        if self.context_adapt is None and not getattr(self, 'context_cross_modal', False):
            return None
        if self.context_of_batch is not None and self.context_of_batch.numel() > 0 and batch_idx is not None:
            return self.context_of_batch.to(device)[batch_idx]
        return torch.full((B,), int(self.eval_context), dtype=torch.long, device=device)

    def apply_ctx_eps(self, eps_joint, ctx):
        """Per-context latent affine on eps (#6 affine / #8 cell-state embedding). No-op unless 'latent'."""
        if self.context_adapt != "latent" or ctx is None:
            return eps_joint
        if not torch.is_tensor(ctx):
            ctx = torch.full((eps_joint.shape[0],), int(ctx), dtype=torch.long, device=eps_joint.device)
        return torch.exp(self.ctx_logscale[ctx]) * eps_joint + self.ctx_shift[ctx]

    def apply_ctx_dec(self, rec_r, rec_p, ctx):
        """Per-context per-gene FiLM on the decoder output (#10). No-op unless 'decoder'."""
        if self.context_adapt != "decoder" or ctx is None:
            return rec_r, rec_p
        if not torch.is_tensor(ctx):
            ctx = torch.full((rec_r.shape[0],), int(ctx), dtype=torch.long, device=rec_r.device)
        rec_r = torch.exp(self.ctx_dec_logscale_r[ctx]) * rec_r + self.ctx_dec_bias_r[ctx]
        rec_p = torch.exp(self.ctx_dec_logscale_p[ctx]) * rec_p + self.ctx_dec_bias_p[ctx]
        return rec_r, rec_p

    def forward(self, x_r, x_p, batch_idx, current_tau=1.0, warmup=False, alpha=1.0, force_drop=None,
                compute_reversal=False, u=None):
        # 1. Encode (per-modality feature dropout)
        if self.training:
            rd, pd = getattr(self, 'rna_feature_dropout', 0.1), getattr(self, 'prot_feature_dropout', 0.0)
            if rd > 0: x_r = x_r * (torch.rand_like(x_r) > rd)
            if pd > 0: x_p = x_p * (torch.rand_like(x_p) > pd)

        mu_r, lv_r = self.enc_rna(x_r)
        mu_p, lv_p = self.enc_prot(x_p)

        eps_r = self.reparameterize(mu_r, lv_r)
        eps_p = self.reparameterize(mu_p, lv_p)

        # NON-GAUSSIAN prior: single-sample MC estimate of KL(q(eps|x)||p(eps)) per latent dim, computed
        # here from the FRESH posterior draws (before any modality-dropout / coupled-noise edits to eps).
        # None for the gaussian default -> train.loss_fn uses the exact closed-form Gaussian KL (no-op / byte-identical).
        kl_mc = None
        if getattr(self, "eps_prior", "gaussian") != "gaussian":
            kl_mc = (self._mc_kl_vec(mu_r, lv_r, eps_r), self._mc_kl_vec(mu_p, lv_p, eps_p))

        # CONDITIONAL exogenous prior p(eps|u): look up the per-class (prior_mu, prior_logvar) for this
        # batch's auxiliary u and hand them to loss_fn (closed-form Gaussian-Gaussian KL). None unless
        # cond_prior_on is set AND u is supplied -> loss_fn falls back to the exact N(0,I) KL (no-op).
        cond_prior = None
        if getattr(self, "cond_prior_on", None) is not None and u is not None:
            pri_mu = self.cond_prior_mu(u); pri_lv = self.cond_prior_lv(u)          # [B, L_r+L_p]
            pmu_r, pmu_p = torch.split(pri_mu, [self.latent_r, self.latent_p], dim=1)
            plv_r, plv_p = torch.split(pri_lv, [self.latent_r, self.latent_p], dim=1)
            # target-scoping: conditioning only one modality's exogenous zeros the other's prior params
            # (its KL reverts to N(0,I)); those embedding columns then receive no gradient -> stay N(0,I).
            tgt = getattr(self, "cond_prior_target", "both")
            if tgt == "rna":
                pmu_p = torch.zeros_like(pmu_p); plv_p = torch.zeros_like(plv_p)
            elif tgt == "prot":
                pmu_r = torch.zeros_like(pmu_r); plv_r = torch.zeros_like(plv_r)
            cond_prior = (pmu_r, plv_r, pmu_p, plv_p)

        # modality dropout: replace a whole modality's eps with the null vector -> the cross-modal
        # block must reconstruct it from the other modality (trains RNA->Protein / Protein->RNA).
        # null="zero": no learnable token (bias carries the mean), so the model can't shortcut by
        # absorbing the marginal into a learned null.
        _zero = getattr(self, 'modality_dropout_null', 'zero') == 'zero'
        nullp = torch.zeros_like(self.null_p) if _zero else self.null_p
        nullr = torch.zeros_like(self.null_r) if _zero else self.null_r
        if self.training and getattr(self, 'modality_dropout_p', 0.0) > 0:
            B = eps_r.shape[0]; u = torch.rand(B, device=eps_r.device); p = self.modality_dropout_p
            mode = self.modality_dropout_mode
            drop_p = (u < p) if mode == "protein" else (u < p / 2) if mode == "both" else None
            drop_r = (u < p) if mode == "rna" else ((u >= p / 2) & (u < p)) if mode == "both" else None
            if drop_p is not None: eps_p = torch.where(drop_p.unsqueeze(1), nullp, eps_p)
            if drop_r is not None: eps_r = torch.where(drop_r.unsqueeze(1), nullr, eps_r)
        # force_drop: deterministic single-modality drop for eval-time cross-modal imputation
        if force_drop == "protein": eps_p = nullp.expand_as(eps_p)
        elif force_drop == "rna":   eps_r = nullr.expand_as(eps_r)

        if self.coupled_noise:   # add shared low-rank factor -> correlated cross-modal eps
            u = torch.randn(eps_r.shape[0], self.coupled_rank, device=eps_r.device)
            eps_r = eps_r + u @ self.coupling_r.T
            eps_p = eps_p + u @ self.coupling_p.T
        eps_joint = torch.cat([eps_r, eps_p], dim=1)
        # exogenous bottleneck: gate eps so closed dims get no exogenous input (must be filled via A)
        eps_gate_openprob = None
        if getattr(self, 'eps_bottleneck', False):
            _gate, eps_gate_openprob = self._eps_gate()
            eps_joint = eps_joint * _gate.unsqueeze(0)
        ctx = self._ctx_vec(batch_idx, eps_joint.shape[0], eps_joint.device)   # context-adapt
        eps_joint = self.apply_ctx_eps(eps_joint, ctx)
        
        # 2. Block Causal Mechanism (ctx selects the per-context cross-modal block if enabled)
        z_joint, blocks, A_full = self.scm(eps_joint, tau=current_tau, alpha=alpha, ctx_id=ctx)
        
        # 3. Decode
        z_r, z_p = torch.split(z_joint, [self.latent_r, self.latent_p], dim=1)
        rec_r = z_r @ self._dec_r_w().T + self.bias_rna
        rec_p = self.dec_prot(z_p) + self.bias_prot
        rec_r, rec_p = self.apply_ctx_dec(rec_r, rec_p, ctx)   # context-adapt (decoder FiLM)

        out = {
            'rec': (rec_r, rec_p),
            'kl': ((mu_r, lv_r), (mu_p, lv_p)),
            'kl_mc': kl_mc,   # None (gaussian, closed-form KL) | (klvec_r, klvec_p) per-dim MC KL (non-gaussian)
            'cond_prior': cond_prior,  # None (unconditional N(0,I)) | (pmu_r,plv_r,pmu_p,plv_p) per-cell iVAE prior
            'blocks': blocks,
            'adj_full': A_full,
            'gates': self.scm.get_gate_probs(),
            'eps': (eps_r, eps_p),
            'eps_gate_openprob': eps_gate_openprob,   # global eps L0-gate open-probs (bottleneck), else None
            'blocks_dag': (self.scm._A_local_rr, self.scm._A_local_pp),  # local intra blocks for DAGMA
            'nb_theta': ((self.log_theta_rna.exp(), self.log_theta_prot.exp())
                         if self.recon_likelihood == "nb" else None),
            'dec_w': (self._dec_r_w(), self.dec_prot.weight)
        }
        # Reversal-discrepancy diagnostic (OFF by default): re-solve the SCM with ONLY the intra-modal
        # diagonal blocks transposed, using the SAME encoded eps + decoders, so the training loss can
        # penalize the reversed intra-graph reconstructing no-worse than the forward. No-op when off.
        if compute_reversal:
            lr = self.latent_r
            A_rr = A_full[:lr, :lr]; A_pp = A_full[lr:, lr:]
            W_rp = A_full[:lr, lr:]; W_pr = A_full[lr:, :lr]
            out['rec_rev'] = self.reversed_intra_recon(eps_joint, A_rr, A_pp, W_rp, W_pr, alpha, ctx)
        return out

    def reversed_intra_recon(self, eps_joint, A_rr, A_pp, W_rp, W_pr, alpha, ctx=None):
        """Reversal-discrepancy diagnostic. Re-solve the structural equations with ONLY the two
        intra-modal DIAGONAL blocks transposed (A_rr -> A_rr^T, A_pp -> A_pp^T); the cross-modal
        W_rp / W_pr blocks are left UNCHANGED. Uses the SAME encoded eps (eps_joint) that produced the
        forward z, and the model's own scm_solve + decoders, so it matches the forward path exactly.
        RNA latents occupy the first `latent_r` dims, protein the next `latent_p`. Returns (rec_r_rev,
        rec_p_rev). Used by the reversal-discrepancy causality loss to pressure A_rr/A_pp asymmetry."""
        A_full_rev = torch.cat([
            torch.cat([A_rr.transpose(-1, -2), W_rp], dim=1),
            torch.cat([W_pr, A_pp.transpose(-1, -2)], dim=1)], dim=0)
        z_rev_scm = self.scm._scm_solve(A_full_rev, eps_joint)
        z_rev = (1.0 - alpha) * eps_joint + alpha * z_rev_scm
        z_r_rev, z_p_rev = torch.split(z_rev, [self.latent_r, self.latent_p], dim=1)
        rec_r_rev = z_r_rev @ self._dec_r_w().T + self.bias_rna
        rec_p_rev = self.dec_prot(z_p_rev) + self.bias_prot
        rec_r_rev, rec_p_rev = self.apply_ctx_dec(rec_r_rev, rec_p_rev, ctx)
        return rec_r_rev, rec_p_rev

    def _dec_r_w(self):
        """Effective RNA decoder weight: W (x) mask (masked-learnable) / W (frozen or unmasked)."""
        w = self.dec_rna.weight
        if getattr(self, "dec_r_mask", None) is not None and not self.dec_mask_freeze:
            return w * self.dec_r_mask
        return w

    def get_projected_graphs(self):
        """Returns feature-space graphs (Genes x Genes, Prot x Prot, etc)"""
        with torch.no_grad():
            W_r = self._dec_r_w()
            W_p = self.dec_prot.weight
            
            # Get hard/clean adjacencies
            A_rr, A_pp, C_rp, C_pr = self.scm.get_block_adjacencies()
            
            # Project
            grn = W_r @ A_rr @ W_r.T
            ppi = W_p @ A_pp @ W_p.T
            translation = W_r @ C_rp @ W_p.T  # Effect of Prot latent on RNA genes
            transcription = W_p @ C_pr @ W_r.T # Effect of RNA latent on Proteins
            
            return grn, translation, transcription, ppi