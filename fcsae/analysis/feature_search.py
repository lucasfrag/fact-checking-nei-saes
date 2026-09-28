"""Busca a feature mais SELETIVA para abstenção em todo o dicionário do SAE.

Cohen's d / cosseno medem um deslocamento MÉDIO (a feature ativa mais em NEI na média);
seletividade é diferente: entre os top-ativadores da feature, quantos são NEI? Uma
"feature de abstenção" limpa tem top-ativadores majoritariamente NEI.

Sem GPU (numpy sobre as ativações da decisão cacheadas).

    python -m fcsae.analysis.feature_search --preset v1_domain
"""
from __future__ import annotations

import argparse
from itertools import islice

import numpy as np
from safetensors.torch import load_file

from fcsae.config import preset
from fcsae.common.io import iter_shards, LABEL2ID
from fcsae.data.loaders import iter_examples


def run(preset_name, topk_act, min_active, show):
    cfg = preset(preset_name)
    sae = load_file(f"{cfg.paths()['sae']}/sae.safetensors")
    W_enc = sae["W_enc"].float().numpy(); b_enc = sae["b_enc"].float().numpy()
    b_dec = sae["b_dec"].float().numpy()

    H, L = [], []
    for _, t in iter_shards(cfg.paths()["activations"]):
        m = t["is_decision"].bool()
        H.append(t["acts"][m].float().numpy()); L.append(t["label"][m].numpy())
    H = np.concatenate(H); L = np.concatenate(L)
    is_nei = (L == LABEL2ID["NEI"]).astype(np.float32)
    n, nei_base = len(H), is_nei.mean()
    print(f"tokens de decisão: {n} | taxa NEI geral: {nei_base:.3f}", flush=True)

    print("encodando (relu pré-ativação) — pode levar ~1-2 min...", flush=True)
    Z = np.maximum((H - b_dec) @ W_enc + b_enc, 0.0)     # [n, width]

    n_active = (Z > 0).sum(0)                              # por feature
    # top-K ativadores por feature (índices), depois taxa de NEI entre eles
    K = topk_act
    top_idx = np.argpartition(Z, -K, axis=0)[-K:]         # [K, width]
    nei_rate = is_nei[top_idx].mean(0)                    # [width]
    # só features com ativação suficiente (evita features raras/mortas)
    valid = n_active >= min_active
    nei_rate_valid = np.where(valid, nei_rate, -1)

    order = np.argsort(nei_rate_valid)[::-1]
    print(f"\n===== features mais SELETIVAS p/ NEI (top-{K} ativadores; min_active={min_active}) =====", flush=True)
    print(f"{'feat':>7} | {'NEI-rate top':>12} | {'enriq.':>6} | {'n_active':>8}", flush=True)
    for f in order[:show]:
        print(f"{f:>7} | {nei_rate[f]:>12.3f} | {nei_rate[f]/nei_base:>5.1f}x | {n_active[f]:>8}", flush=True)

    # também o outro extremo (features seletivas p/ DECISÃO) como contraste
    order_dec = np.argsort(np.where(valid, nei_rate, 2))[:5]
    print(f"\nfeatures mais seletivas p/ DECISÃO (NEI-rate baixo):", flush=True)
    for f in order_dec:
        print(f"  feat {f}: NEI-rate={nei_rate[f]:.3f} | n_active={n_active[f]}", flush=True)

    # mostra os top claims da feature nº1 mais seletiva p/ NEI
    best = int(order[0])
    print(f"\n===== TOP claims da feature mais seletiva ({best}, NEI-rate={nei_rate[best]:.3f}) =====", flush=True)
    exs = list(islice(iter_examples(cfg.datasets, cfg.max_per_dataset, shuffle=True), n))
    act_best = Z[:, best]
    for idx in np.argsort(act_best)[::-1][:15]:
        e = exs[idx]
        print(f"[act={act_best[idx]:.2f} | {e.label} | {e.source}] {e.claim[:120]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v1_domain")
    ap.add_argument("--topk-act", type=int, default=50)
    ap.add_argument("--min-active", type=int, default=200)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()
    run(args.preset, args.topk_act, args.min_active, args.show)


if __name__ == "__main__":
    main()
