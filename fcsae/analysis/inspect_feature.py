"""Validação semântica de uma feature sobre TODAS as decisões cacheadas (sem o LLM).

As ativações da posição da decisão já estão nos shards do harvest, na mesma ordem das
linhas de `examples.jsonl` (o harvest lê o arquivo por índice). Então:
  1. junta as decisões de todos os shards num cache (`decisions.safetensors`, ~5 GB) —
     análises seguintes não precisam reler ~1 TB;
  2. CONFERE o alinhamento texto↔ativação (rótulo e fonte de cada decisão = linha i);
  3. encoda pelo SAE de verdade (normalização da entrada + TopK, como no treino) e mede
     a ativação da feature — o v1 (`inspect_feature_gpu`) usava só relu(pré-ativação);
  4. reporta top-claims + enriquecimento de NEI no top-50/200/1000 + AUC, por fonte.

    python -m fcsae.analysis.inspect_feature --preset v3_llama --feat 19219
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from fcsae.config import preset
from fcsae.common.io import iter_shards, LABEL2ID
from fcsae.harvest.harvest import SOURCE2ID
from fcsae.sae.model import TopKSAE, normalize_input
from fcsae.analysis.abstention import auc


def decision_cache(cfg):
    path = os.path.join(cfg.paths()["activations"], "decisions.safetensors")
    if os.path.exists(path):
        t = load_file(path)
        return t["acts"], t["label"], t["source"]
    A, L, S = [], [], []
    for i, (_, t) in enumerate(iter_shards(cfg.paths()["activations"])):
        m = t["is_decision"].bool()
        A.append(t["acts"][m].clone()); L.append(t["label"][m].clone()); S.append(t["source"][m].clone())
        if i % 20 == 0:
            print(f"  shard {i} | decisões até agora: {sum(len(x) for x in L)}", flush=True)
    A, L, S = torch.cat(A), torch.cat(L), torch.cat(S)
    tmp = path + ".tmp"
    save_file({"acts": A.contiguous(), "label": L.contiguous(), "source": S.contiguous()}, tmp)
    os.replace(tmp, path)
    return A, L, S


@torch.no_grad()
def run(preset_name, feat, topn):
    cfg = preset(preset_name)
    print("carregando decisões...", flush=True)
    A, L, S = decision_cache(cfg)
    exs = [json.loads(l) for l in open(os.path.join(cfg.paths()["activations"], "examples.jsonl"), encoding="utf-8")]
    n = len(L)
    assert n == len(exs), f"nº de decisões ({n}) != nº de exemplos ({len(exs)})"
    lab_ok = np.mean([LABEL2ID[e["label"]] == int(l) for e, l in zip(exs, L.tolist())])
    src_ok = np.mean([SOURCE2ID[e["source"]] == int(s) for e, s in zip(exs, S.tolist())])
    print(f"alinhamento texto↔ativação: rótulo {lab_ok:.4%} | fonte {src_ok:.4%} (n={n})", flush=True)
    assert lab_ok == 1.0 and src_ok == 1.0, "texto e ativações desalinhados — não confiar no top-list"

    dev = cfg.device if torch.cuda.is_available() else "cpu"
    w = load_file(os.path.join(cfg.paths()["sae"], "sae.safetensors"))
    sae = TopKSAE(A.shape[1], cfg.sae_width, cfg.sae_k, cfg.sae_k_aux, cfg.sae_dead_steps)
    sae.load_state_dict({**w, "steps_since_fire": sae.steps_since_fire}); sae = sae.to(dev)
    act = torch.empty(n); pre_f = torch.empty(n)
    for i in range(0, n, 8192):
        x = A[i:i + 8192].to(dev).float()
        x = normalize_input(x)[0] if cfg.sae_input_norm else x
        _, z, pre = sae(x)
        act[i:i + 8192] = z[:, feat].cpu(); pre_f[i:i + 8192] = pre[:, feat].cpu()
    act, pre_f, lab, src = act.numpy(), pre_f.numpy(), L.numpy(), S.numpy()

    nei = lab == LABEL2ID["NEI"]; base = nei.mean()
    order = np.argsort(-act, kind="stable")
    res = {"feat": feat, "preset": cfg.name, "n_decisions": int(n), "nei_base": float(base),
           "n_active": int((act > 0).sum()), "frac_active": float((act > 0).mean()),
           "nei_rate_when_active": float(nei[act > 0].mean()) if (act > 0).any() else None,
           "mean_act_nei": float(act[nei].mean()), "mean_act_dec": float(act[~nei].mean()),
           "auc_feature_topk": auc(act[nei], act[~nei]),
           "auc_feature_preact": auc(pre_f[nei], pre_f[~nei])}
    for K in (50, 200, 1000):
        r = float(nei[order[:K]].mean()); res[f"nei_top{K}"] = r; res[f"enrich_top{K}"] = r / base
    res["by_source"] = {}
    for sname, sid in SOURCE2ID.items():
        m = src == sid
        if m.sum() == 0:
            continue
        o = np.argsort(-act[m], kind="stable"); nm = nei[m]
        res["by_source"][sname] = {"n": int(m.sum()), "nei_base": float(nm.mean()),
                                   "nei_top200": float(nm[o[:200]].mean()),
                                   "auc_feature_topk": auc(act[m][nm], act[m][~nm])}
    res["top"] = [{"rank": r + 1, "idx": int(i), "act": round(float(act[i]), 4), "label": exs[i]["label"],
                   "source": exs[i]["source"], "claim": exs[i]["claim"],
                   "evidence": [e[:300] for e in exs[i]["evidence"][:2]]} for r, i in enumerate(order[:topn])]

    out = os.path.join(cfg.paths()["results"], f"feature_{feat}_examples.json")
    os.makedirs(cfg.paths()["results"], exist_ok=True)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("top",)}, indent=1), flush=True)
    print(f"\n===== TOP {topn} decisões (feature {feat}) =====", flush=True)
    for r in res["top"]:
        print(f"[#{r['rank']} act={r['act']:.3f} | {r['label']} | {r['source']}] {r['claim'][:120]}", flush=True)
    print(f"salvo em {out}", flush=True)
    sys.stdout.flush(); os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v3_llama")
    ap.add_argument("--feat", type=int, required=True)
    ap.add_argument("--topn", type=int, default=40)
    args = ap.parse_args()
    run(args.preset, args.feat, args.topn)


if __name__ == "__main__":
    main()
