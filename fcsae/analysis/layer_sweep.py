# -*- coding: utf-8 -*-
"""Varredura de camadas: onde a abstenção (NEI vs SUP/REF) vira uma direção linear?

Para cada camada L, reconstrói a direção diff-of-means (média NEI − média SUP/REF) na
posição da decisão (último token do prompt) num split de treino, e mede o AUC held-out.
Também registra a norma média do resíduo por camada (motiva a normalização β do steering).
Não gera texto — é um único forward por prompt com output_hidden_states.

    python -m fcsae.analysis.layer_sweep --preset v1_deepseek --n-per-class 1200
"""
import argparse, json, os, random
import numpy as np, torch
from sklearn.metrics import roc_auc_score

from fcsae.config import preset
from fcsae.common.modeling import load_model, n_layers
from fcsae.data.prompt import build_verdict_prompt, templated_input_ids


def _examples_path(cfg):
    p = os.path.join(cfg.paths()["activations"], "examples.jsonl")
    if os.path.exists(p):
        return p
    # v1_domain não tem examples.jsonl próprio; os exemplos (claim/evidence/label) são os
    # mesmos dados (VitaminC+FEVER) entre modelos — reusa de outra pasta.
    for alt in ("v1_deepseek", "v1_qwen3_l19", "v1_qwen3"):
        ap = os.path.join(preset(alt).paths()["activations"], "examples.jsonl")
        if os.path.exists(ap):
            return ap
    raise FileNotFoundError("nenhum examples.jsonl encontrado")


def run(preset_name, n_per_class=1200, seed=0):
    cfg = preset(preset_name)
    exs = [json.loads(l) for l in open(_examples_path(cfg), encoding="utf-8")]
    nei = [e for e in exs if e["label"] == "NEI"]
    dec = [e for e in exs if e["label"] in ("SUP", "REF")]
    random.seed(seed); random.shuffle(nei); random.shuffle(dec)
    k = min(n_per_class, len(nei), len(dec))
    sample = [(e, 1) for e in nei[:k]] + [(e, 0) for e in dec[:k]]
    random.shuffle(sample)
    print(f"[{preset_name}] {k} NEI + {k} SUP/REF = {2*k} prompts", flush=True)

    model, tok = load_model(cfg.checkpoint, cfg.quant, "cuda")
    nL = n_layers(model)
    per_layer = [[] for _ in range(nL)]          # vetor último-token por camada
    labels = []
    for i, (ex, y) in enumerate(sample):
        ids, _ = templated_input_ids(tok, build_verdict_prompt(
            ex["claim"], ex["evidence"], cfg.max_evidence), "cuda", cfg.max_seq_len)
        if ids.shape[1] < 4:
            continue
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states                    # tupla len nL+1 (0=embeddings)
        for L in range(nL):
            per_layer[L].append(hs[L + 1][0, -1, :].float().cpu())
        labels.append(y)
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(sample)}", flush=True)
        if i % 64 == 0:
            torch.cuda.empty_cache()

    y = np.array(labels)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y)); cut = len(y) // 2
    tr, te = idx[:cut], idx[cut:]
    aucs, norms = [], []
    for L in range(nL):
        H = torch.stack(per_layer[L]).numpy()     # [N, d]
        d = H[tr][y[tr] == 1].mean(0) - H[tr][y[tr] == 0].mean(0)
        s = H[te] @ d
        aucs.append(float(roc_auc_score(y[te], s)))
        norms.append(float(np.linalg.norm(H, axis=1).mean()))

    best = int(np.argmax(aucs))
    out = {"preset": preset_name, "n_layers": nL, "n_per_class": k,
           "layers": list(range(nL)), "auc_by_layer": aucs, "resid_norm_by_layer": norms,
           "best_layer": best, "best_auc": aucs[best],
           "layer_used": cfg.layer, "auc_at_used": aucs[cfg.layer]}
    od = cfg.paths()["results"]; os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "layer_sweep.json"), "w"), indent=1)
    print(f"[{preset_name}] melhor camada={best} (AUC {aucs[best]:.3f}) | "
          f"usada={cfg.layer} (AUC {aucs[cfg.layer]:.3f})", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--n-per-class", type=int, default=1200)
    a = ap.parse_args()
    run(a.preset, a.n_per_class)
