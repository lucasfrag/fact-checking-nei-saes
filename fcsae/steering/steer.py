"""Steering causal: somar α·(direção de abstenção) no resíduo da camada-alvo durante a
geração e medir se o veredito muda. +α deve INDUZIR abstenção; −α deve SUPRIMIR.

Prova causalidade (a direção *causa* a abstenção), além da correlação já mostrada.
Direção = diferença de médias (abstém − decisivo) nas ativações de treino do SAE (v1),
na camada-alvo — a mesma que separa abstenção (à la Arditi et al., 2024).

    python -m fcsae.steering.steer --sae v1_domain --layer 15 --n 200
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import torch
import numpy as np

from fcsae.config import preset
from fcsae.common.modeling import load_model
from fcsae.common.io import iter_shards, LABEL2ID
from fcsae.data.prompt import build_verdict_prompt, templated_input_ids
from fcsae.analysis.generalize import TESTS

COEFF = [0.0]   # multiplicador global de steering (setado no loop)


def parse_label(txt):
    t = (txt or "").upper()
    # ordem importa: "NOT ENOUGH" antes de SUPPORT/REFUTE
    if "NOT ENOUGH" in t or "INSUFFICIENT" in t:
        return "NEI"
    if "REFUT" in t:
        return "REF"
    if "SUPPORT" in t:
        return "SUP"
    if "NOT ENOUGH" in t:
        return "NEI"
    return "?"


@torch.no_grad()
def run(sae_name, layer, n, alphas, no_evidence=False, max_new_tokens=15, skip_think=False,
        scale_by_residual=False):
    cfg = preset(sae_name); dev = cfg.device
    suffix = ("_beta" if scale_by_residual else "") + ("_noevidence" if no_evidence else "")
    # direção = diff-of-means (não-normalizada) na camada, das ativações de treino do v1
    Ht, Lt = [], []
    for _, t in iter_shards(cfg.paths()["activations"]):
        m = t["is_decision"].bool()
        Ht.append(t["acts"][m].float()); Lt.append(t["label"][m])
    Ht = torch.cat(Ht); Lt = torch.cat(Lt)
    ab = Lt == LABEL2ID["NEI"]; de = (Lt == LABEL2ID["SUP"]) | (Lt == LABEL2ID["REF"])
    dvec = (Ht[ab].mean(0) - Ht[de].mean(0))               # unidade "1 = um gap médio"
    R = Ht.norm(dim=1).mean()
    print(f"direção (camada {layer}): ||dvec||={dvec.norm():.2f} | ||resíduo médio||={R:.1f}", flush=True)
    if scale_by_residual:
        # α passa a ser FRAÇÃO da norma do resíduo (comparável entre modelos):
        # α=1 soma uma perturbação de magnitude = norma média do resíduo.
        dvec = dvec / (dvec.norm() + 1e-9) * R
        print(f"  [scale-by-residual] α agora = fração do resíduo (||add em α=1||={dvec.norm():.1f})", flush=True)

    out_dir = f"results/{cfg.name}"; os.makedirs(out_dir, exist_ok=True)
    preds_path = f"{out_dir}/steering_preds{suffix}.json"
    cached = json.load(open(preds_path)) if os.path.exists(preds_path) else {}
    if cached:
        print(f"RESUME: α já feitos = {list(cached)}", flush=True)

    model, tok = load_model(cfg.checkpoint, cfg.quant, dev)
    dv = dvec.to(dev)

    def hook(mod, inp, out):
        if COEFF[0] == 0.0:
            return out
        add = (COEFF[0] * dv).to(out[0].dtype if isinstance(out, tuple) else out.dtype)
        if isinstance(out, tuple):
            return (out[0] + add,) + out[1:]
        return out + add
    h = model.model.layers[layer].register_forward_hook(hook)

    # amostra de claims (AVeriTeC) — usamos a predição BASELINE (α=0) como referência
    data = TESTS["averitec"]()
    rng = np.random.default_rng(0); idx = rng.permutation(len(data))[:n]
    claims = [data[i] for i in idx]

    def gen(ids):
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        if "</think>" in txt:                      # reasoning model: parseia só a resposta
            txt = txt.split("</think>")[-1]
        return parse_label(txt)

    # pré-tokeniza
    # reasoning models (DeepSeek): injeta "</think>" p/ pular o raciocínio e ir ao veredito
    skip_ids = tok("</think>\n\n", add_special_tokens=False, return_tensors="pt").input_ids.to(dev) if skip_think else None
    prompts = []
    for claim, ev, gold in claims:
        ev_use = [] if no_evidence else ev            # linha "sem evidência" do 2x2
        ids, _ = templated_input_ids(tok, build_verdict_prompt(claim, ev_use, cfg.max_evidence), dev, cfg.max_seq_len)
        if skip_ids is not None:
            ids = torch.cat([ids, skip_ids], 1)
        prompts.append((ids, claim, gold))

    preds = {a: cached.get(str(a)) for a in alphas}
    for a in alphas:
        if preds[a] and len(preds[a]) == len(prompts):     # já feito (resume)
            print(f"α={a:+.1f} | (cache) taxa NEI={np.mean([p=='NEI' for p in preds[a]]):.2f}", flush=True)
            continue
        COEFF[0] = float(a)
        preds[a] = [gen(ids) for ids, claim, gold in prompts]
        torch.cuda.empty_cache()
        cached[str(a)] = preds[a]
        json.dump(cached, open(preds_path, "w"))            # checkpoint por α
        nei = np.mean([p == "NEI" for p in preds[a]])
        bad = np.mean([p == "?" for p in preds[a]])
        print(f"α={a:+.1f} | taxa NEI={nei:.2f} | ilegível={bad:.2f}", flush=True)
    h.remove()

    base = preds[0.0]
    base_dec = [i for i, p in enumerate(base) if p in ("SUP", "REF")]
    base_nei = [i for i, p in enumerate(base) if p == "NEI"]
    # usa o maior |α| LEGÍVEL (ilegível<20%) — evita medir num α que quebrou o modelo
    unread = {a: float(np.mean([p == "?" for p in preds[a]])) for a in alphas}
    clean_pos = [a for a in alphas if a > 0 and unread[a] < 0.2]
    clean_neg = [a for a in alphas if a < 0 and unread[a] < 0.2]
    amax = max(clean_pos) if clean_pos else max(alphas)
    amin = min(clean_neg) if clean_neg else min(alphas)
    induced = np.mean([preds[amax][i] == "NEI" for i in base_dec]) if base_dec else float("nan")
    suppressed = np.mean([preds[amin][i] in ("SUP", "REF") for i in base_nei]) if base_nei else float("nan")

    res = {"sae": sae_name, "layer": layer, "n": len(prompts), "alphas": list(alphas),
           "nei_rate": {str(a): float(np.mean([p == "NEI" for p in preds[a]])) for a in alphas},
           "unreadable_rate": unread,
           "n_base_decisive": len(base_dec), "n_base_nei": len(base_nei),
           "alpha_induce": amax, "alpha_suppress": amin,
           "induced_abstention": float(induced), "suppressed_abstention": float(suppressed),
           "dvec_norm": float(dvec.norm()), "residual_norm": float(R),
           "scale_by_residual": scale_by_residual,
           "preds": {str(a): preds[a] for a in alphas},
           "gold": [p[2] for p in prompts]}
    # alguns exemplos de flip decisivo->NEI ao adicionar a direção
    ex = []
    for i in base_dec:
        if preds[amax][i] == "NEI":
            ex.append({"claim": prompts[i][1][:120], "base": base[i], f"alpha_{amax}": "NEI"})
        if len(ex) >= 6:
            break
    res["exemplos_flip"] = ex
    res["no_evidence"] = no_evidence
    json.dump(res, open(f"{out_dir}/steering{suffix}.json", "w"), ensure_ascii=False, indent=1)
    if os.path.exists(preds_path):
        os.remove(preds_path)                              # limpa o cache ao concluir
    print(f"\ninduzir abstenção (base decisivo→NEI em α={amax}): {induced:.2f} (de {len(base_dec)})", flush=True)
    print(f"suprimir abstenção (base NEI→decisivo em α={amin}): {suppressed:.2f} (de {len(base_nei)})", flush=True)
    print(f"salvo em {out_dir}/steering{suffix}.json", flush=True)
    sys.stdout.flush(); os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default="v1_domain")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--alphas", default="-8,-4,-2,0,2,4,8")
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=15)
    ap.add_argument("--skip-think", action="store_true")
    ap.add_argument("--scale-by-residual", action="store_true")
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    run(args.sae, args.layer, args.n, alphas, no_evidence=args.no_evidence,
        max_new_tokens=args.max_new_tokens, skip_think=args.skip_think,
        scale_by_residual=args.scale_by_residual)


if __name__ == "__main__":
    main()
