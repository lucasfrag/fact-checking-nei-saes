"""Validação semântica de uma feature: os claims que mais a ativam na posição da decisão.

Sem GPU: os tokens de decisão nos shards estão na MESMA ordem dos exemplos do
`iter_examples` (shuffle com seed fixo → determinístico). Coletamos as ativações da
decisão dos shards, encodamos na CPU (numpy) e recuperamos o texto re-rodando o
`iter_examples`. Uma checagem de alinhamento (rótulo do shard vs do exemplo) garante
que os índices batem.

    python -m fcsae.analysis.feature_examples --preset v1_domain --feat 22280
"""
from __future__ import annotations

import argparse
from itertools import islice

import numpy as np
from safetensors.torch import load_file

from fcsae.config import preset
from fcsae.common.io import load_manifest, iter_shards, LABEL2ID
from fcsae.data.loaders import iter_examples
from fcsae.data.prompt import build_verdict_prompt

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def run(preset_name, feat, topn):
    cfg = preset(preset_name)
    shard_dir = cfg.paths()["activations"]
    sae = load_file(f"{cfg.paths()['sae']}/sae.safetensors")
    W_enc = sae["W_enc"].float().numpy(); b_enc = sae["b_enc"].float().numpy()
    b_dec = sae["b_dec"].float().numpy()

    # 1. ativações da decisão, em ordem, + rótulo
    H, L = [], []
    for _, t in iter_shards(shard_dir):
        m = t["is_decision"].bool()
        H.append(t["acts"][m].float().numpy()); L.append(t["label"][m].numpy())
    H = np.concatenate(H); L = np.concatenate(L)
    n = len(H)
    print(f"tokens de decisão: {n}", flush=True)

    # 2. ativação da feature `feat` (encode TopK não precisa: a pré-ativação já ordena;
    #    mas aplicamos relu — a feature só "conta" se positiva)
    act = np.maximum((H - b_dec) @ W_enc[:, feat] + b_enc[feat], 0.0)

    # 3. recupera o texto dos exemplos na mesma ordem
    exs = list(islice(iter_examples(cfg.datasets, cfg.max_per_dataset, shuffle=True), n))
    # checagem de alinhamento: rótulos batem?
    match = np.mean([LABEL2ID[exs[i].label] == L[i] for i in range(min(n, len(exs), 2000))])
    print(f"alinhamento shard↔exemplo (rótulo bate em 2k amostras): {match:.3f}", flush=True)

    order = np.argsort(act)[::-1]
    print(f"\n===== TOP {topn} claims que mais ativam a feature {feat} =====", flush=True)
    labs_top = []
    for idx in order[:topn]:
        e = exs[idx]; labs_top.append(e.label)
        print(f"[act={act[idx]:.2f} | {e.label} | {e.source}] {e.claim[:130]}", flush=True)

    from collections import Counter
    top200 = [exs[i].label for i in order[:200]]
    print(f"\ndistribuição de rótulos no TOP-200 ativadores: {dict(Counter(top200))}", flush=True)
    all_lab = Counter(e.label for e in exs)
    print(f"distribuição de rótulos GERAL: {dict(all_lab)}", flush=True)
    # taxa de NEI no topo vs geral (o teste-chave: a feature seleciona abstenção?)
    nei_top = top200.count("NEI") / 200
    nei_all = all_lab["NEI"] / n
    print(f"\ntaxa de NEI no TOP-200 = {nei_top:.2f}  vs  geral = {nei_all:.2f}  "
          f"(enriquecimento {nei_top / max(nei_all,1e-9):.1f}x)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v1_domain")
    ap.add_argument("--feat", type=int, default=22280)
    ap.add_argument("--topn", type=int, default=25)
    args = ap.parse_args()
    run(args.preset, args.feat, args.topn)


if __name__ == "__main__":
    main()
