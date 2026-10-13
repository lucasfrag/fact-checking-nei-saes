"""Dados do gráfico "como a feature foi escolhida" do viewer: Cohen's d × cosseno com a direção de abstenção,
para TODAS as features que disparam na posição de decisão (pedido do Lucas, 05/10/2026).

  - Cohen's d: ativação da feature (0 quando não dispara) em NEI × SUP/REF, desvio padrão combinado (ddof=1),
    sobre todas as decisões — a mesma definição de `abstention.py::cohens_d`;
  - cosseno: linha do decoder (normalizada) × direção de abstenção centralizada (v − média(v)), como em
    `abstention.py` (o SAE opera em entradas centralizadas); v vem de `abstention_direction.safetensors`.

Roda na CPU por padrão (não disputa a GPU com o steering).
    python -m fcsae.analysis.selection_map --preset v3_llama --out <viewer>/llama-3.1-8b-instruct/layer-15
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file

from fcsae.config import preset
from fcsae.sae.model import TopKSAE, normalize_input
from fcsae.steering.factcheck_sae import abstention_direction


@torch.no_grad()
def run(args):
    cfg = preset(args.preset); dev = args.device
    torch.set_num_threads(max(1, os.cpu_count() - 4))
    t = load_file(os.path.join(cfg.paths()["activations"], "decisions.safetensors"))
    A, L = t["acts"], t["label"]
    keep = (L == 0) | (L == 1) | (L == 2); A, y = A[keep], (L[keep] == 2)
    w = load_file(os.path.join(cfg.paths()["sae"], "sae.safetensors"))
    sae = TopKSAE(A.shape[1], cfg.sae_width, cfg.sae_k, cfg.sae_k_aux, cfg.sae_dead_steps)
    sae.load_state_dict({**w, "steps_since_fire": sae.steps_since_fire}); sae = sae.to(dev)
    W = cfg.sae_width
    s1 = torch.zeros(W, dtype=torch.float64); q1 = torch.zeros_like(s1); s0 = torch.zeros_like(s1); q0 = torch.zeros_like(s1)
    act = torch.zeros(W, dtype=torch.long)
    for i in range(0, len(A), 4096):
        x = A[i:i + 4096].to(dev).float()
        xin = normalize_input(x)[0] if cfg.sae_input_norm else x
        _, z, _ = sae(xin)
        z = z.double().cpu(); m = y[i:i + 4096]
        s1 += z[m].sum(0); q1 += (z[m] ** 2).sum(0); s0 += z[~m].sum(0); q0 += (z[~m] ** 2).sum(0)
        act += (z > 0).sum(0)
        if (i // 4096) % 20 == 0:
            print(f"  {i}/{len(A)}", flush=True)
    n1, n0 = int(y.sum()), int((~y).sum())
    m1, m0 = s1 / n1, s0 / n0
    v1 = (q1 - n1 * m1 ** 2) / (n1 - 1); v0 = (q0 - n0 * m0 ** 2) / (n0 - 1)
    sp = torch.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))
    d = ((m1 - m0) / (sp + 1e-9)).numpy()
    v = abstention_direction(cfg).numpy(); vc = v - v.mean(); vc /= np.linalg.norm(vc)
    Wd = w["W_dec"].float().numpy(); cos = (Wd / np.linalg.norm(Wd, axis=1, keepdims=True)) @ vc
    alive = np.where(act.numpy() > 0)[0]
    out = {"preset": cfg.name, "n_decisions": int(len(y)), "definition": {
               "d": "Cohen's d of the feature activation (0 when inactive), NEI vs SUP/REF, pooled SD, all decisions",
               "cos": "cosine between the feature's decoder row and the mean-centred abstention direction"},
           "features": [[int(a), round(float(d[a]), 4), round(float(cos[a]), 4)] for a in alive]}
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "selection.json"); json.dump(out, open(p, "w"), separators=(",", ":"))
    top = sorted(alive, key=lambda a: -d[a])[:3]
    print("maiores d:", [(int(a), round(float(d[a]), 3), round(float(cos[a]), 3)) for a in top])
    print(f"FIM: {len(alive)} features -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    run(ap.parse_args())
