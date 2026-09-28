"""Carregamento do modelo e captura do residual stream.

Lições herdadas dos experimentos anteriores (mantidas aqui de propósito):
- 4-bit (nf4) é rápido (~5GB) e é a precisão que produziu os vereditos; carregar
  passando SÓ `quantization_config` + `device_map` (passar `dtype`/`low_cpu_mem_usage`
  junto quebra a conversão de pesos no transformers 5.14).
- Modelos multimodais (ex.: Gemma) não são reconhecidos por AutoModelForCausalLM ->
  fallback para AutoModelForImageTextToText.
- `logits_to_keep=1`: computa logits só da última posição. Os hidden states ficam
  idênticos, mas evita materializar logits de todas as posições sobre o vocabulário
  (que estoura a VRAM em modelos de vocab grande).
- Para modelos RMSNorm (Llama/Gemma), `hidden_states[L+1]` == `blocks.L.hook_resid_post`.
"""
from __future__ import annotations

import torch


def load_model(checkpoint: str, quant: str = "4bit", device: str = "cuda"):
    from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                              AutoTokenizer, BitsAndBytesConfig)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    kw = {}
    if device == "cuda" and quant in ("4bit", "8bit"):
        from fcsae.common.gpu_env import ensure_bitsandbytes_cuda_libs
        ensure_bitsandbytes_cuda_libs()
        if quant == "4bit":
            cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            cfg = BitsAndBytesConfig(load_in_8bit=True)
        kw = dict(quantization_config=cfg, device_map={"": 0})
    else:
        kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **kw)
    except (ValueError, KeyError):
        model = AutoModelForImageTextToText.from_pretrained(checkpoint, **kw)
    if not (device == "cuda" and quant in ("4bit", "8bit")):
        model = model.to(device)
    return model.eval(), tok


def n_layers(model) -> int:
    c = model.config
    return getattr(c, "num_hidden_layers", None) or c.text_config.num_hidden_layers


@torch.no_grad()
def residual_at_layer(model, input_ids, layer: int):
    """Retorna o residual stream (resid_post) da camada `layer` para todas as posições:
    tensor [seq_len, d_model]. `input_ids`: [1, seq_len] no device do modelo."""
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False,
                logits_to_keep=1)
    return out.hidden_states[layer + 1][0]     # [seq_len, d]
