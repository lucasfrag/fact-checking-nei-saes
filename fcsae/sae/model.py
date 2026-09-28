"""TopK Sparse Autoencoder [Gao et al., 2024; k-sparse AE, Makhzani & Frey, 2013].

Escolha de arquitetura: TopK dá esparsidade EXATA (k features ativas por token), sem
tunar coeficiente de L1, e treina de forma estável — mais robusto para uma primeira
rodada que a JumpReLU. Inclui a perda auxiliar AuxK (reconstrói o resíduo com as
features mortas) para conter features mortas, o principal modo de falha de SAEs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def normalize_input(x):
    """[Gao et al., 2024, §2.1] Tira a média sobre d_model e normaliza cada token à norma
    unitária antes do SAE (e antes de medir o erro de reconstrução). Torna a importância dos
    tokens uniforme, em vez de ponderada pela norma (§B). Retorna (x_norm, média, norma) p/
    permitir voltar ao espaço bruto: x = x_norm * norma + média."""
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    nrm = xc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return xc / nrm, mu, nrm


@torch.no_grad()
def geometric_median(X, iters: int = 200, tol: float = 1e-7):
    """Mediana geométrica por Weiszfeld — init do b_pre [Gao §A.1; Bricken et al., 2023]."""
    m = X.mean(0)
    for _ in range(iters):
        w = 1.0 / (X - m).norm(dim=1).clamp_min(1e-8)
        m_new = (w[:, None] * X).sum(0) / w.sum()
        if (m_new - m).norm() <= tol * m.norm().clamp_min(1e-8):
            return m_new
        m = m_new
    return m


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, width: int, k: int, k_aux: int = 512,
                 dead_steps: int = 1000):
        super().__init__()
        self.d_model, self.width, self.k = d_model, width, k
        self.k_aux = min(k_aux, width)
        self.dead_steps = dead_steps
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(d_model, width))
        self.b_enc = nn.Parameter(torch.zeros(width))
        self.W_dec = nn.Parameter(torch.empty(width, d_model))
        nn.init.kaiming_uniform_(self.W_dec)
        self._unit_norm_decoder()
        self.W_enc.data = self.W_dec.data.t().clone()   # init amarrado
        # nº de passos desde que cada feature disparou pela última vez
        self.register_buffer("steps_since_fire", torch.zeros(width, dtype=torch.long))

    @torch.no_grad()
    def _unit_norm_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def set_b_dec(self, mean_vec):
        self.b_dec.data = mean_vec.to(self.b_dec.dtype)

    def encode_pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc      # [B, width]

    def _topk(self, pre, k):
        vals, idx = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, torch.relu(vals))
        return z

    def forward(self, x):
        pre = self.encode_pre(x)
        z = self._topk(pre, self.k)
        x_hat = z @ self.W_dec + self.b_dec
        return x_hat, z, pre

    def aux_loss(self, x, x_hat, pre, normalized: bool = False):
        """AuxK: usa as features MORTAS (top-k_aux entre as mortas) para reconstruir o
        resíduo — reanima features que pararam de disparar.
        normalized=True [Gao §A.6; código oficial `normalized_mse`]: MSE dividido pela
        variância do resíduo no lote (a escala do erro muda ao longo do treino); NaN -> 0 (§A.2)."""
        dead = self.steps_since_fire >= self.dead_steps
        if dead.sum() == 0:
            return x.new_tensor(0.0)
        resid = x - x_hat.detach()
        pre_dead = pre.masked_fill(~dead.unsqueeze(0), float("-inf"))
        k = min(self.k_aux, int(dead.sum()))
        z_aux = self._topk(pre_dead.nan_to_num(neginf=0.0), k)
        resid_hat = z_aux @ self.W_dec
        if normalized:
            den = (resid - resid.mean(0, keepdim=True)).pow(2).mean()
            return ((resid - resid_hat).pow(2).mean() / den).nan_to_num(0.0)
        return (resid - resid_hat).pow(2).sum(-1).mean()

    @torch.no_grad()
    def project_decoder_grad(self):
        """[Gao §A.3; Bricken et al., 2023] Remove do gradiente de cada direção do decoder a
        componente paralela a ela (interação Adam × normalização do decoder). Assume decoder
        com norma unitária."""
        g = self.W_dec.grad
        if g is not None:
            g -= (g * self.W_dec.data).sum(1, keepdim=True) * self.W_dec.data

    @torch.no_grad()
    def update_dead(self, z):
        fired = (z > 0).any(0)
        self.steps_since_fire += 1
        self.steps_since_fire[fired] = 0
