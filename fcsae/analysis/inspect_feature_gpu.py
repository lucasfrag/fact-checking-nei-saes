"""Inspeção semântica de uma feature com TEXTO, auto-alinhada (uma passada só).

Faz forward de uma amostra de claims, encoda a posição da decisão pelo SAE e guarda
os top-N claims que mais ativam a feature — texto, rótulo e ativação juntos, sem o
bug de re-join do iter_examples. Salva o top-list periodicamente (GPU do WSL2 instável).

    python -m fcsae.analysis.inspect_feature_gpu --preset v1_domain --feat 22280 --limit 6000
"""
from __future__ import annotations

import os
import sys
import json
import heapq
import argparse

import torch
import numpy as np
from safetensors.torch import load_file

from fcsae.config import preset
from fcsae.common.modeling import load_model, residual_at_layer
from fcsae.data.loaders import iter_examples
from fcsae.data.prompt import build_verdict_prompt, templated_input_ids


@torch.no_grad()
def run(preset_name, feat, limit, topn):
    cfg = preset(preset_name)
    sae = load_file(f"{cfg.paths()['sae']}/sae.safetensors")
    W = sae["W_enc"][:, feat].float().cuda(); be = float(sae["b_enc"][feat]); bd = sae["b_dec"].float().cuda()
    model, tok = load_model(cfg.checkpoint, cfg.quant, cfg.device)
    print(f"inspecionando feature {feat} em até {limit} claims", flush=True)

    heap = []            # (act, tie, {claim,label,act}) — min-heap de tamanho topn
    labs, acts, tie = [], [], 0
    out = f"{cfg.paths()['results']}/feature_{feat}_examples.json"
    os.makedirs(cfg.paths()["results"], exist_ok=True)
    for ex in iter_examples(cfg.datasets, cfg.max_per_dataset, shuffle=True):
        if len(labs) >= limit:
            break
        prompt = build_verdict_prompt(ex.claim, ex.evidence, cfg.max_evidence)
        ids, dec = templated_input_ids(tok, prompt, cfg.device, cfg.max_seq_len)
        if ids.shape[1] < 4:
            continue
        h = residual_at_layer(model, ids, cfg.layer)[-1].float()        # decisão
        a = float(torch.relu((h - bd) @ W + be))
        labs.append(ex.label); acts.append(a)
        rec = {"claim": ex.claim, "label": ex.label, "source": ex.source, "act": round(a, 3)}
        if len(heap) < topn:
            heapq.heappush(heap, (a, tie, rec))
        elif a > heap[0][0]:
            heapq.heapreplace(heap, (a, tie, rec))
        tie += 1
        if len(labs) % 500 == 0:
            top = sorted([r for _, _, r in heap], key=lambda r: -r["act"])
            json.dump({"feat": feat, "n_seen": len(labs), "top": top}, open(out, "w"),
                      ensure_ascii=False, indent=1)
            print(f"  {len(labs)} claims | act_max={max(acts):.2f}", flush=True)
        torch.cuda.empty_cache()

    labs = np.array(labs); acts = np.array(acts)
    order = acts.argsort()[::-1]
    top_lab = labs[order[:200]]
    nei_top = (top_lab == "NEI").mean(); nei_all = (labs == "NEI").mean()
    top = sorted([r for _, _, r in heap], key=lambda r: -r["act"])
    json.dump({"feat": feat, "n_seen": int(len(labs)), "nei_top200": float(nei_top),
               "nei_all": float(nei_all), "top": top}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n===== TOP {topn} claims (feature {feat}) =====", flush=True)
    for r in top[:topn]:
        print(f"[act={r['act']:.2f} | {r['label']} | {r['source']}] {r['claim'][:110]}", flush=True)
    print(f"\nNEI no top-200: {nei_top:.2f} vs geral {nei_all:.2f} "
          f"(enriq {nei_top/max(nei_all,1e-9):.1f}x)", flush=True)
    print(f"salvo em {out}", flush=True)
    sys.stdout.flush(); os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v1_domain")
    ap.add_argument("--feat", type=int, default=22280)
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--topn", type=int, default=30)
    args = ap.parse_args()
    run(args.preset, args.feat, args.limit, args.topn)


if __name__ == "__main__":
    main()
