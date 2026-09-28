"""Exporta nosso TopK SAE para o formato do `sae_lens` (cfg.json + sae_weights.safetensors).

No formato do sae_lens, o SAE fica carregável por `SAE.load_from_disk(...)` e pronto
para subir no Hugging Face Hub e submeter à Neuronpedia. Construímos um `TopKSAE` do
próprio sae_lens com nossos pesos e usamos o `save_model` dele — assim o formato fica
garantidamente compatível com a versão instalada (sem cfg.json escrito à mão).

Normalização da entrada (v3, Gao et al. 2024 §2.1). Nosso SAE opera em x_n = (x − média)/‖x − média‖.
O sae_lens oferece `normalize_activations="layer_norm"`: x_ln = (x − média)/(std + eps), com
std = ‖x − média‖/√(d−1)  →  x_ln ≈ c·x_n, c = √(d−1). Exportamos os pesos reescalados
(W_enc/c, b_dec·c, W_dec·c, b_enc igual): a pré-ativação e as features ativas ficam idênticas,
e a reconstrução no espaço bruto também (a menos do eps=1e-5 do sae_lens). O v1 (sem
normalização) é exportado como antes (`normalize_activations="none"`).

Roda num ambiente que tenha o sae_lens (usamos um venv separado, para não mexer no ambiente
dos experimentos):
    PYTHONPATH=. ~/venvs/saelens/bin/python -m fcsae.export.to_saelens --preset v3_llama \
        --out "/mnt/d/Projetos WSL/fact-checking-abstention-saes/llama-3.1-8b-instruct/layer-15" --feat 19219
"""
from __future__ import annotations

import os
import sys
import json
import math
import argparse

import torch

from fcsae.common.gpu_env import ensure_bitsandbytes_cuda_libs
try:
    ensure_bitsandbytes_cuda_libs()   # silencia o erro de lib do bnb ao importar sae_lens
except ModuleNotFoundError:           # ambiente só-CPU do sae_lens (sem os wheels nvidia)
    pass

from safetensors import safe_open
from safetensors.torch import load_file
from fcsae.config import preset, MODELS


MODEL_CARD_V1 = """---
license: other
base_model: {base}
tags: [sparse-autoencoder, interpretability, fact-checking, sae-lens]
library_name: sae_lens
---

# {name} — TopK Sparse Autoencoder (domínio: fact-checking)

SAE de domínio treinado sobre ativações do **{base}** (residual stream,
`{hook}`), para o estudo do mecanismo de **abstenção** em verificação de fatos.
Parte de uma dissertação de mestrado (PPGCC/PUCRS).

## Especificação
- Arquitetura: **TopK** (Gao et al., 2024), esparsidade L0 exata = **k={k}**
- `d_in` = {d_in} · `d_sae` = {d_sae}
- Camada / hook: `{hook}`
- Dados de treino: {datasets} (evidência-ouro, renderizada no template de veredito)
- Tokens de treino: ~{tokens}
- Métricas: FVU(held-out) = {fvu} · FVU(decisão) = {fvu_dec} · L0 = {l0} · features mortas = {dead}

## Uso
```python
from sae_lens import SAE
sae = SAE.load_from_disk("caminho/deste/repo")
```

## Licenças / atribuição
- **Modelo base:** {base} — sujeito à licença do modelo base (ex.: Llama 3.1
  Community License). Este repositório contém apenas os pesos do SAE, não os do
  modelo base.
- **Datasets de treino:** respeitar as licenças originais ({datasets}).

## Referência
Gao et al. (2024), *Scaling and evaluating sparse autoencoders*.
"""

MODEL_CARD_V3 = """---
license: other
base_model: {base}
tags: [sparse-autoencoder, interpretability, mechanistic-interpretability, fact-checking, abstention, sae-lens]
library_name: sae_lens
---

# TopK Sparse Autoencoder — {base_short}, layer {layer} (fact-checking domain)

A domain SAE trained on residual-stream activations of **{base}** (`{hook}`) while the model reads
fact-checking verdict prompts, to study the model's **abstention** ("NOT ENOUGH EVIDENCE") mechanism.

## Architecture and training (following Gao et al., 2024)
| | |
|---|---|
| architecture | **TopK** SAE, exact L0 = **k = {k}**; `d_in` = {d_in}, `d_sae` = {d_sae} |
| auxiliary loss | AuxK, α = 1/32, k_aux = {k_aux}; dead-latent threshold ≈ 10M tokens |
| optimization | Adam (lr {lr}, ε = 6.25e-10), gradient projection parallel to decoder rows, geometric-median b_pre init, weight EMA 0.999 (Gao et al., App. A) |
| input normalization | mean-subtracted and unit-normed over `d_model` (Gao et al., §2.1) — exported as `normalize_activations="layer_norm"` with weights rescaled by √(d_in−1). sae_lens adds ε = 1e-5 to the std, which changes one of the 32 active latents on ~0.1% of tokens (near-ties); reconstruction FVU is identical to 5 decimals (see `export_verification.json`) |
| batch | 2,048 tokens (Gao et al. use 131,072; their §A.4 reports L(N) is not strongly batch-dependent) |
| training data | {n_prompts:,} claims from VitaminC + FEVER (gold evidence) rendered in a verdict prompt → {n_tokens:,} token activations |
| base-model precision | activations collected from the model loaded in **4-bit NF4** (bitsandbytes) |

## Metrics
| | |
|---|---|
| dead latents (end of training) | {dead} / {d_sae} |
| FVU held-out (normalized space) | {fvu:.3f} |
| FVU at the decision position (raw space, variance across decisions) | {fvu_dec_raw} |
| L0 | {l0:.1f} |

## Abstention feature
{feat_block}

## Usage
```python
from sae_lens import SAE
sae = SAE.load_from_disk("path/to/this/folder")    # or SAE.from_pretrained(...) once on the Hub
# hook: {hook}  (output of transformer block {layer})
feats = sae.encode(resid)          # resid: [..., {d_in}] raw residual stream
```

## Licenses / attribution
- **Base model:** {base} — subject to the base model's license. This repository contains only the SAE weights.
- **Training data:** VitaminC ([Schuster et al., 2021](https://arxiv.org/abs/2103.08541)) and FEVER ([Thorne et al., 2018](https://arxiv.org/abs/1803.05355)) — see their licenses.

## References
- Gao, L., et al. (2024). *Scaling and Evaluating Sparse Autoencoders.* [arXiv:2406.04093](https://arxiv.org/abs/2406.04093)
- Ferrando, J., et al. (2025). *Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models.* ICLR 2025. [arXiv:2411.14257](https://arxiv.org/abs/2411.14257)
"""


def _norm_scale(d_in):
    return math.sqrt(d_in - 1)


def run(preset_name: str, out_dir: str | None, verify: bool, feat: int | None, n_verify: int):
    from sae_lens.saes.topk_sae import TopKSAE, TopKSAEConfig
    from sae_lens.saes.sae import SAEMetadata
    from sae_lens import SAE

    cfg = preset(preset_name)
    sae_dir = cfg.paths()["sae"]
    weights = {k: v.float() for k, v in load_file(os.path.join(sae_dir, "sae.safetensors")).items()}
    metrics = json.load(open(os.path.join(sae_dir, "metrics.json")))
    base = MODELS[cfg.model_tag]
    hook = f"blocks.{cfg.layer}.hook_resid_post"
    d_in = weights["b_dec"].shape[0]
    d_sae = weights["b_enc"].shape[0]
    out_dir = out_dir or os.path.join(sae_dir, "saelens")

    if cfg.sae_input_norm:
        c = _norm_scale(d_in)
        export_w = {"W_enc": weights["W_enc"] / c, "b_enc": weights["b_enc"],
                    "W_dec": weights["W_dec"] * c, "b_dec": weights["b_dec"] * c}
        norm = "layer_norm"
    else:
        c, export_w, norm = 1.0, {k: weights[k] for k in ("W_enc", "b_enc", "W_dec", "b_dec")}, "none"

    meta = SAEMetadata(
        model_name=base, hook_name=hook, hook_layer=cfg.layer,
        prepend_bos=True, dataset_path=",".join(cfg.datasets),
        context_size=cfg.max_seq_len,
        description=f"TopK domain SAE (fact-checking) for abstention study; layer {cfg.layer}; "
                    f"activations from the 4-bit NF4 base model",
    )
    sae_cfg = TopKSAEConfig(
        d_in=d_in, d_sae=d_sae, k=cfg.sae_k, dtype="float32", device="cpu",
        apply_b_dec_to_input=True, normalize_activations=norm,
        rescale_acts_by_decoder_norm=False, metadata=meta,
    )
    sae = TopKSAE(sae_cfg)
    sae.load_state_dict(export_w)
    os.makedirs(out_dir, exist_ok=True)
    sae.save_model(out_dir)
    print(f"SAE exportado (formato sae_lens, normalize_activations={norm}, c={c:.4f}) -> {out_dir}", flush=True)

    # ---- verificação: nosso SAE × sae_lens recarregado do disco
    report = {}
    if verify:
        sae2 = SAE.load_from_disk(out_dir, device="cpu")
        dec_path = os.path.join(cfg.paths()["activations"], "decisions.safetensors")
        if os.path.exists(dec_path):
            with safe_open(dec_path, "pt") as f:
                x = f.get_slice("acts")[:n_verify].float()
            src = f"{n_verify} decisões reais (cache do harvest)"
        else:
            x = torch.randn(n_verify, d_in) * 8
            src = f"{n_verify} vetores aleatórios (cache de decisões não encontrado)"
        with torch.no_grad():
            mu = x.mean(-1, keepdim=True); xc = x - mu; nrm = xc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            xin = xc / nrm if cfg.sae_input_norm else x
            pre = (xin - weights["b_dec"]) @ weights["W_enc"] + weights["b_enc"]
            vals, idx = pre.topk(cfg.sae_k, dim=-1)
            z = torch.zeros_like(pre).scatter_(-1, idx, torch.relu(vals))
            ours = z @ weights["W_dec"] + weights["b_dec"]
            ours = ours * nrm + mu if cfg.sae_input_norm else ours
            z2 = sae2.encode(x)
            theirs = sae2.decode(z2)
        same_set = ((z > 0) == (z2 > 0)).all(dim=-1).float().mean().item()
        rel = ((ours - theirs).norm(dim=-1) / x.norm(dim=-1)).max().item()
        fvu = lambda a: (((x - a) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()).item()
        report = {"source": src, "same_active_set": same_set, "max_rel_recon_diff": rel,
                  "fvu_ours": fvu(ours), "fvu_saelens": fvu(theirs),
                  "max_abs_feature_diff": (z - z2).abs().max().item()}
        if feat is not None:
            report["feat_corr"] = torch.corrcoef(torch.stack([z[:, feat], z2[:, feat]]))[0, 1].item()
            report["feat_max_abs_diff"] = (z[:, feat] - z2[:, feat]).abs().max().item()
        # o sae_lens soma eps=1e-5 ao desvio padrão na normalização "layer_norm": em ~0,1% dos tokens,
        # onde a 32ª e a 33ª pré-ativação estão praticamente empatadas, isso troca uma feature. Testamos
        # que essa é a ÚNICA fonte de diferença: reproduzindo o caminho do sae_lens sem o eps, o
        # conjunto de features ativas tem de ser idêntico em 100% dos tokens.
        if cfg.sae_input_norm:
            with torch.no_grad():
                xln0 = xc / xc.std(dim=-1, keepdim=True)
                pre0 = (xln0 - export_w["b_dec"]) @ export_w["W_enc"] + export_w["b_enc"]
                z0 = torch.zeros_like(pre0).scatter_(-1, pre0.topk(cfg.sae_k, dim=-1).indices, 1.0)
            report["same_active_set_without_saelens_eps"] = ((z > 0) == (z0 > 0)).all(dim=-1).float().mean().item()
            ok = report["same_active_set_without_saelens_eps"] == 1.0 and abs(report["fvu_ours"] - report["fvu_saelens"]) < 1e-4
        else:
            ok = same_set > 0.999 and rel < 1e-3
        print(f"VERIFICAÇÃO ({src}): mesmo conjunto de features ativas em {same_set:.4%} | "
              f"máx. dif. relativa da reconstrução = {rel:.2e} | FVU nosso {report['fvu_ours']:.5f} × "
              f"sae_lens {report['fvu_saelens']:.5f}" + (f" | feature {feat}: corr {report['feat_corr']:.6f}, "
              f"máx |dif| {report['feat_max_abs_diff']:.2e}" if feat is not None else "") +
              f"  → {'OK' if ok else 'DIVERGE — revisar'}"
              + (f" (sem o eps do sae_lens: {report['same_active_set_without_saelens_eps']:.4%})" if cfg.sae_input_norm else ""), flush=True)
        report["ok"] = ok
        json.dump(report, open(os.path.join(out_dir, "export_verification.json"), "w"), indent=1)

    # ---- model card
    if cfg.sae_input_norm:
        man = json.load(open(os.path.join(cfg.paths()["activations"], "manifest.json")))
        abst = os.path.join(cfg.paths()["results"], "abstention.json")
        fex = os.path.join(cfg.paths()["results"], f"feature_{feat}_examples.json") if feat is not None else ""
        fvu_dec_raw = f"{json.load(open(abst))['fvu_decision_domainSAE']:.3f}" if os.path.exists(abst) else "n/a"
        if feat is not None and os.path.exists(fex):
            fe = json.load(open(fex))
            feat_block = (f"Latent **{feat}** is the candidate abstention feature: it is the latent most aligned with the "
                          f"diff-of-means abstention direction and the one with the largest Cohen's d between NEI and "
                          f"SUP/REF decisions. Among the 200 claims that activate it most, **{100*fe['nei_top200']:.1f}%** "
                          f"are NEI (base rate {100*fe['nei_base']:.1f}%; {fe['enrich_top200']:.1f}×); AUC(NEI vs. "
                          f"SUP/REF) = {fe['auc_feature_topk']:.3f}. Its **magnitude** (not on/off) encodes abstention: it is "
                          f"active on {100*fe['frac_active']:.0f}% of decision positions.")
        else:
            feat_block = "See the accompanying repository."
        card = MODEL_CARD_V3.format(
            base=base, base_short=base.split("/")[-1], layer=cfg.layer, hook=hook, k=cfg.sae_k,
            k_aux=cfg.sae_k_aux, d_in=d_in, d_sae=d_sae, lr=cfg.sae_lr, n_prompts=man["dom_prompts"],
            n_tokens=man["n_tokens"], dead=metrics.get("dead_features", "?"),
            fvu=metrics.get("fvu_heldout", float("nan")), fvu_dec_raw=fvu_dec_raw,
            l0=metrics.get("l0", float("nan")), feat_block=feat_block)
        open(os.path.join(out_dir, "README.md"), "w").write(card)
        json.dump({"preset": cfg.name, "config": cfg.to_dict(), "train_metrics": metrics,
                   "normalization_scale_c": c}, open(os.path.join(out_dir, "fcsae_training_config.json"), "w"), indent=1)
        print("README.md (model card) + fcsae_training_config.json gerados", flush=True)
    else:
        card = MODEL_CARD_V1.format(
            name=cfg.name, base=base, hook=hook, k=cfg.sae_k, d_in=d_in, d_sae=d_sae,
            datasets=", ".join(cfg.datasets), tokens=metrics.get("steps", "?"),
            fvu=round(metrics.get("fvu_heldout", float("nan")), 3),
            fvu_dec=round(metrics.get("fvu_decision", float("nan")), 3),
            l0=round(metrics.get("l0", float("nan")), 1),
            dead=metrics.get("dead_features", "?"))
        open(os.path.join(out_dir, "MODEL_CARD.md"), "w").write(card)
        print("MODEL_CARD.md gerado", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v1_domain")
    ap.add_argument("--out", default=None)
    ap.add_argument("--feat", type=int, default=None, help="feature de abstenção (p/ o model card e a verificação)")
    ap.add_argument("--n-verify", type=int, default=20000)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    run(args.preset, args.out, verify=not args.no_verify, feat=args.feat, n_verify=args.n_verify)


if __name__ == "__main__":
    main()
