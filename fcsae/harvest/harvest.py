"""Harvest resumível e determinístico.

Duas correções importantes:
1. O `iter_examples` com shuffle em streaming NÃO é reprodutível entre processos, o que
   quebraria o resume e o alinhamento texto↔ativação. Solução: MATERIALIZAR a ordem uma
   vez em `examples.jsonl` (e `general.jsonl`); o harvest lê por índice → ordem fixa,
   resume exato, e o texto da decisão i é a linha i de examples.jsonl.
2. RESUMÍVEL (a GPU do WSL2 cai sob carga): grava shard + `progress.json` a cada
   `shard_tokens`; ao reiniciar, pula os prompts/passagens já commitados. `manifest.json`
   só no fim = marcador de conclusão.

    python -m fcsae.harvest.harvest --preset v1_domain
    python -m fcsae.harvest.harvest --preset v2_mixed
"""
from __future__ import annotations

import os
import sys
import json
import glob
import time
import argparse
from itertools import islice

import torch
from safetensors.torch import save_file

from fcsae.config import preset, Config
from fcsae.common.modeling import load_model, n_layers, residual_at_layer
from fcsae.common.io import LABEL2ID
from fcsae.data.loaders import iter_examples, iter_general
from fcsae.data.prompt import build_verdict_prompt, templated_input_ids

SOURCE2ID = {"vitaminc": 0, "fever": 1, "averitec": 2, "scifact": 3, "general": 9}


def _atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def ensure_example_files(cfg, out_dir):
    """Materializa a ordem dos exemplos UMA vez (determinístico e resumível)."""
    ex_path = os.path.join(out_dir, "examples.jsonl")
    if not os.path.exists(ex_path):
        print("materializando examples.jsonl (ordem fixa)...", flush=True)
        lines = []
        for ex in iter_examples(cfg.datasets, cfg.max_per_dataset, shuffle=True):
            lines.append(json.dumps({"claim": ex.claim, "evidence": ex.evidence,
                                     "label": ex.label, "source": ex.source}, ensure_ascii=False))
        _atomic(ex_path, "\n".join(lines) + "\n")
        print(f"  {len(lines)} exemplos de domínio", flush=True)
    if cfg.mix_general:
        g_path = os.path.join(out_dir, "general.jsonl")
        if not os.path.exists(g_path):
            print("materializando general.jsonl...", flush=True)
            lines = [json.dumps({"text": t}, ensure_ascii=False)
                     for t in islice(iter_general(cfg.general_dataset), 40_000)]
            _atomic(g_path, "\n".join(lines) + "\n")
            print(f"  {len(lines)} passagens gerais", flush=True)
    return ex_path


# Tipo de gravação das ativações. float16 (padrão) basta para Llama/Qwen3/DeepSeek; nos Gemma uma dimensão do
# resíduo vale ~32.768 em todo token ("massive activations", Sun et al. 2024) e estoura o teto do float16 (65.504)
# → inf → treino nan. bfloat16 = formato em que o próprio modelo calcula (compute dtype do nf4): sem estouro.
STORE_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}

class Sharder:
    def __init__(self, out_dir, shard_tokens, prog):
        self.out_dir, self.shard_tokens, self.prog = out_dir, shard_tokens, prog
        self.acts, self.dec, self.lab, self.src, self.buf = [], [], [], [], 0

    def add(self, h, is_dec, label, source):
        self.acts.append(h); self.dec.append(is_dec.to(torch.uint8))
        self.lab.append(torch.full((h.shape[0],), LABEL2ID[label], dtype=torch.uint8))
        self.src.append(torch.full((h.shape[0],), SOURCE2ID.get(source, 8), dtype=torch.uint8))
        self.buf += h.shape[0]
        if source == "general":
            self.prog["gen_passages"] += 1; self.prog["gen_tokens"] += h.shape[0]
        else:
            self.prog["dom_prompts"] += 1
        if self.buf >= self.shard_tokens:
            self.flush()

    def flush(self):
        if self.buf == 0:
            return
        k = self.prog["n_shards"]
        save_file({"acts": torch.cat(self.acts), "is_decision": torch.cat(self.dec),
                   "label": torch.cat(self.lab), "source": torch.cat(self.src)},
                  os.path.join(self.out_dir, f"shard_{k:05d}.safetensors"))
        self.prog["n_shards"] = k + 1
        _atomic(os.path.join(self.out_dir, "progress.json"), json.dumps(self.prog))
        self.acts, self.dec, self.lab, self.src, self.buf = [], [], [], [], 0


@torch.no_grad()
def run(cfg: Config):
    out_dir = cfg.paths()["activations"]
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        print(f"manifest.json já existe — harvest concluído ({out_dir}).", flush=True)
        sys.stdout.flush(); os._exit(0)

    dev = cfg.device
    ex_path = ensure_example_files(cfg, out_dir)
    model, tok = load_model(cfg.checkpoint, cfg.quant, dev)
    nL = n_layers(model)
    d_model = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size

    prog_path = os.path.join(out_dir, "progress.json")
    if os.path.exists(prog_path):
        prog = json.load(open(prog_path))
        print(f"RESUME: dom={prog['dom_prompts']} gen_pass={prog['gen_passages']} "
              f"shards={prog['n_shards']}", flush=True)
    else:
        prog = {"n_shards": 0, "dom_prompts": 0, "domain_complete": False,
                "gen_passages": 0, "gen_tokens": 0, "d_model": int(d_model), "layer": cfg.layer}
    print(f"modelo {cfg.model_tag} ({nL} cam, d={d_model}) camada={cfg.layer} "
          f"quant={cfg.quant} mix_general={cfg.mix_general}", flush=True)

    sh = Sharder(out_dir, cfg.shard_tokens, prog)
    t0 = time.time()

    # ---- domínio: lê examples.jsonl por índice, pula os já commitados ----
    if not prog["domain_complete"]:
        with open(ex_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < prog["dom_prompts"]:
                    continue
                ex = json.loads(line)
                ids, _ = templated_input_ids(tok, build_verdict_prompt(
                    ex["claim"], ex["evidence"], cfg.max_evidence), dev, cfg.max_seq_len, cfg.empty_think)
                if ids.shape[1] < 4:
                    # mantém alinhamento com a linha: grava um token dummy? Não — em vez disso
                    # nunca acontece (prompt de veredito sempre > 4 tokens). Conta como feito.
                    prog["dom_prompts"] += 1
                    continue
                h = residual_at_layer(model, ids, cfg.layer)[1:].to(STORE_DTYPE[cfg.dtype_store]).cpu()
                is_dec = torch.zeros(h.shape[0], dtype=torch.bool); is_dec[-1] = True
                sh.add(h, is_dec, ex["label"], ex["source"])
                if prog["dom_prompts"] % 500 == 0:
                    print(f"  domínio {prog['dom_prompts']} | "
                          f"{(prog['dom_prompts'])/(time.time()-t0+1e-9):.1f}/s shards={prog['n_shards']}",
                          flush=True)
                if dev == "cuda":
                    torch.cuda.empty_cache()
        sh.flush()
        prog["domain_complete"] = True
        _atomic(prog_path, json.dumps(prog))
        print(f"domínio COMPLETO: {prog['dom_prompts']} prompts", flush=True)

    # ---- geral (v2): lê general.jsonl por índice ----
    if cfg.mix_general:
        dom_tokens = sum(_shard_tokens(p) for p in glob.glob(os.path.join(out_dir, "shard_*.safetensors"))) \
            - prog["gen_tokens"]
        target = int(dom_tokens * cfg.general_fraction / (1 - cfg.general_fraction))
        print(f"geral: alvo≈{target} tokens (já {prog['gen_tokens']})", flush=True)
        with open(os.path.join(out_dir, "general.jsonl"), encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < prog["gen_passages"]:
                    continue
                if prog["gen_tokens"] >= target:
                    break
                ids = tok(json.loads(line)["text"], add_special_tokens=True, return_tensors="pt",
                          truncation=True, max_length=cfg.max_seq_len).input_ids.to(dev)
                if ids.shape[1] < 8:
                    prog["gen_passages"] += 1
                    continue
                h = residual_at_layer(model, ids, cfg.layer)[1:].to(STORE_DTYPE[cfg.dtype_store]).cpu()
                sh.add(h, torch.zeros(h.shape[0], dtype=torch.bool), "NEI", "general")
                if prog["gen_tokens"] % 200_000 < cfg.max_seq_len:
                    print(f"  geral ~{prog['gen_tokens']}/{target}", flush=True)
                if dev == "cuda":
                    torch.cuda.empty_cache()
        sh.flush()

    # ---- manifest (conclusão) ----
    n_tokens = sum(_shard_tokens(p) for p in glob.glob(os.path.join(out_dir, "shard_*.safetensors")))
    _atomic(manifest_path, json.dumps({
        "model_tag": cfg.model_tag, "checkpoint": cfg.checkpoint, "layer": cfg.layer,
        "d_model": int(d_model), "n_shards": prog["n_shards"], "n_tokens": int(n_tokens),
        "dom_prompts": prog["dom_prompts"], "gen_tokens": prog["gen_tokens"],
        "sources": SOURCE2ID, "label2id": LABEL2ID, "config": cfg.to_dict()}))
    print(f"\nCONCLUÍDO: {prog['dom_prompts']} prompts + {prog['gen_tokens']} tokens gerais "
          f"| {n_tokens} tokens | {prog['n_shards']} shards -> {out_dir}", flush=True)
    sys.stdout.flush(); os._exit(0)


def _shard_tokens(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return f.get_slice("is_decision").get_shape()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v1_domain")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--max-per-dataset", type=int, default=None)
    ap.add_argument("--quant", default=None)
    args = ap.parse_args()
    cfg = preset(args.preset)
    if args.layer is not None: cfg.layer = args.layer
    if args.max_per_dataset is not None: cfg.max_per_dataset = args.max_per_dataset
    if args.quant is not None: cfg.quant = args.quant
    run(cfg)


if __name__ == "__main__":
    main()
