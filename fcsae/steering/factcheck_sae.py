# -*- coding: utf-8 -*-
"""Steering com a FEATURE do SAE v3 no passo de veredito do fact-checking (AVeriTeC dev).

Desenho (ver notebooks/07-steering-sae-v3.ipynb para as justificativas e referências):
  - 3 condições de evidência: `none` (só o claim), `single` e `iterative` (a evidência que
    os pipelines Single-step / Iterative HF recuperaram) — "sem evidência" é a mesma para os
    dois métodos, pois na hora do veredito eles só diferem pela evidência;
  - steering x ← x + α·d_f no residual stream da camada do SAE, em todas as posições
    [Ferrando et al. 2025, Eq. 4 — mesma notação; Arditi et al. 2024, Eq. 3], com d_f = direção
    da feature no decoder (norma 1). A grade é dada em α/R (R = norma média do resíduo na posição
    da decisão), como Ferrando et al. descrevem o α deles (nota 5: "around two times the norm of
    the residual stream"). NOME INTERNO: o argumento/campo `beta` (e o `b<valor>` dos arquivos)
    é α/R; o vetor somado é α·d_f = beta·R·d_f;
  - grade de β (curva dose-resposta) + controle com features ALEATÓRIAS do mesmo SAE, mesmo
    β [Ferrando et al. 2025: 10 latentes aleatórios; Turner et al. 2023: vetores aleatórios].
  - O prompt, o system prompt, o chat template, a geração (greedy) e a normalização do rótulo
    são os do pipeline original (`LLMVerdict` / `HFLLM`), reconstruídos a partir do JSONL
    (mesma evidência e mesmos pares Q&A que o veredito viu). Com β=0 as predições devem
    coincidir com as do pipeline — isso é verificado (`--check`).

Resumível: cada célula (condição, direção, β) é gravada em results/<preset>/steer_sae/.

    python -m fcsae.steering.factcheck_sae --preset v3_llama --feat 19219 --check
    python -m fcsae.steering.factcheck_sae --preset v3_llama --feat 19219 --betas=-2,-1,0,1,2
    python -m fcsae.steering.factcheck_sae --preset v3_llama --feat 19219 --random 10 --betas=-1,1
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import torch
from safetensors.torch import load_file

from fcsae.config import preset, ROOT
from fcsae.common.modeling import load_model

OUT = os.path.join(ROOT, "old_method/corpora/averitec/outputs")
EV_DIR = {"single": "averitec_original_pipeline", "iterative": "averitec_iterative_pipeline"}
HF_TAG = {"llama3.1:8b": "llama3.1-8b-hf", "deepseek-r1:8b": "deepseek-r1-8b-hf", "qwen3:8b": "qwen3-8b-hf"}
MAX_NEW_TOKENS = 1536        # = HF_MAX_NEW_TOKENS do pipeline; a normalização lê o texto inteiro ("FALSE" em qualquer ponto -> REF)

# ---------------------------------------------------------------- pipeline original (cópia fiel)
# shared/llm/hf_llm.py::HFLLM._build_system_prompt (LANGUAGE=en)
SYSTEM_PROMPT = """
You are a fact-checking assistant.

Respond in English.

General rules:
- Be factual and precise
- Do not hallucinate
- Follow the instructions strictly
"""


def build_prompt(claim, claim_date, speaker, evidence, qa_pairs):
    """shared/modules/verdict/llm_verdict.py::LLMVerdict.build_prompt (idêntico)."""
    evidence_block = "\n\n".join(f"[{i+1}] {e['text']}" for i, e in enumerate(evidence[:8]))
    qa_block = "\n\n".join(f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}" for qa in qa_pairs[:5])
    return f"""
Task: Determine the final verdict of a claim based on evidence.

Claim:
{claim}

Claim date:
{claim_date}

Speaker:
{speaker}

---

Evidence:
{evidence_block}

---

Answers derived from evidence (may be incomplete or noisy):
{qa_block}

---

Instructions:

- You must determine whether the evidence SUPPORTS or REFUTES the claim.

- IMPORTANT:
  Evidence may refute a statement that is OPPOSITE to the claim.
  Always reason directly about the claim.

- Do NOT rely on keywords like "false", "fake", or "not true" alone.
  Interpret what is being negated.

- If evidence confirms the claim → SUPPORTED
- If evidence contradicts the claim → REFUTED
- If evidence is insufficient or unclear → NOT ENOUGH EVIDENCE
- If both strong support and refutation exist → CONFLICTING EVIDENCE/CHERRYPICKING

---

Output:
Return ONLY one label:

SUPPORTED
REFUTED
NOT ENOUGH EVIDENCE
CONFLICTING EVIDENCE/CHERRYPICKING
"""


def normalize(response):
    """LLMVerdict.normalize (idêntico): sem rótulo reconhecível -> NEI."""
    if not response:
        return "NEI"
    r = response.strip().upper()
    if "REFUTE" in r or "FALSE" in r:
        return "REF"
    if "SUPPORT" in r:
        return "SUP"
    if "CONFLICT" in r:
        return "CONF"
    if "NOT ENOUGH" in r or "INSUFFICIENT" in r:
        return "NEI"
    return "NEI"


# Rótulos como o modelo os ESCREVE: expressão completa, em MAIÚSCULAS (verificado nas células: as
# ocorrências em minúsculas são o modelo falando dos rótulos, p.ex. "whether the claim is supported,
# refuted, or…", e palavras soltas como "support or refute" não são veredito).
_LABEL_RE = __import__("re").compile(r"NOT ENOUGH EVIDENCE|CONFLICTING EVIDENCE|SUPPORTED|REFUTED")
_LABEL_MAP = {"NOT ENOUGH EVIDENCE": "NEI", "CONFLICTING EVIDENCE": "CONF", "SUPPORTED": "SUP", "REFUTED": "REF"}


def first_label(response):
    """Rótulo que o modelo PRODUZIU: o primeiro rótulo escrito como rótulo (maiúsculas, por extenso).
    Se a resposta começa com o rótulo, é esse; se raciocina antes, é o primeiro que aparece; sem
    rótulo -> NEI (padrão do pipeline). Corrige o defeito de `normalize` (pipeline original), que
    procura "REFUTE"/"FALSE" em qualquer ponto ANTES de "NOT ENOUGH" (achado em 28/09/2026)."""
    m = _LABEL_RE.search(response or "")
    return _LABEL_MAP[m.group(0)] if m else "NEI"


def readable(response):
    """Saída contém algum dos 4 rótulos ESCRITO COMO RÓTULO (maiúsculas, por extenso)."""
    return bool(_LABEL_RE.search(response or ""))


# Recusa por FALTA DE INFORMAÇÃO/EVIDÊNCIA escrita por extenso, sem usar o rótulo — detectada por
# correspondência de frases, como a "knowledge refusal" de Ferrando et al. (2025, §5: "we define a
# set of common knowledge refusal completions and detect if any of these occur with string
# matching"). Lista CONGELADA em 27/09/2026, definida a partir das células sem evidência (β=0 e
# β=+0,25) e mantida fixa para todas as células. Recusa de SEGURANÇA ("illegal or harmful") não
# conta — Ferrando et al. distinguem a recusa por conhecimento da recusa "due to safety concerns".
REFUSAL_PATTERNS = [
    r"\bunable to (find|verify|locate|confirm|determine|process|access|provide)",
    r"\b(couldn't|could not|cannot|can't|can not) (find|verify|locate|confirm|determine|access)",
    r"\b(i )?(do not|don't) have (any |enough |sufficient |the )?(access|evidence|information|data|details)",
    r"\bno (provided )?(evidence|information|data) (is |was |has been )?(provided|available|given|found|to)",
    r"\bnot (able|possible) to (verify|determine|confirm|find)",
    r"\b(there is|there's) (no|not enough|insufficient) (provided )?(evidence|information)",
]
_REFUSAL = __import__("re").compile("|".join(REFUSAL_PATTERNS), __import__("re").I)


def refusal(response):
    """Abstenção por falta de informação escrita por extenso (sem rótulo)."""
    return bool(_REFUSAL.search(response or ""))


def coherent(response):
    """Critério de qualidade do steering (a partir de 27/09): rótulo OU recusa por falta de
    informação. O que sobra (texto sem conclusão, recusa de segurança) conta como 'quebrou'."""
    return readable(response) or refusal(response)


def strip_thinking(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


GOLD = {"SUPPORTED": "SUP", "REFUTED": "REF", "NOT ENOUGH EVIDENCE": "NEI", "CONFLICTING EVIDENCE/CHERRYPICKING": "CONF"}


def load_claims(model_tag):
    """Claims + a evidência/Q&A que o veredito do pipeline viu (result_writer grava em todo
    step a MESMA lista final de evidências; os Q&A são um por step)."""
    tag = HF_TAG[model_tag]
    runs = {}
    for cond, d in EV_DIR.items():
        rows = {}
        for line in open(f"{OUT}/{d}/{tag}/averitec_dev.jsonl", encoding="utf-8"):
            r = json.loads(line); rows[r["claim_id"]] = r
        runs[cond] = rows
    ids = sorted(runs["single"])
    claims = []
    for cid in ids:
        base = runs["single"][cid]
        c = {"id": cid, "gold": GOLD[base["gold_label"].strip().upper()], "claim": base["claim"],
             "date": base.get("claim_date"), "speaker": base.get("speaker")}
        for cond in EV_DIR:
            r = runs[cond][cid]; st = r["pipeline"]["steps"]
            c[cond] = {"evidence": (st[0].get("evidence") or []) if st else [],
                       "qa": [{"question": s.get("question"), "answer": s.get("answer")} for s in st],
                       "pipeline_pred": {"SUPPORTED": "SUP", "REFUTED": "REF", "NOT ENOUGH EVIDENCE": "NEI",
                                         "CONFLICTING EVIDENCE/CHERRYPICKING": "CONF"}.get(r["prediction"].strip().upper(), "?")}
        claims.append(c)
    return claims


def prompt_for(c, cond):
    if cond == "none":      # mesmo template, sem evidência e sem Q&A
        return build_prompt(c["claim"], c["date"], c["speaker"], [], [])
    e = c[cond]
    return build_prompt(c["claim"], c["date"], c["speaker"], e["evidence"], e["qa"])


# ---------------------------------------------------------------- steering
STEER = {"vec": None}


def make_hook():
    def hook(mod, inp, out):
        v = STEER["vec"]
        if v is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + v.to(out[0].dtype),) + out[1:]
        return out + v.to(out.dtype)
    return hook


def directions(cfg, feat, n_random, seed=0):
    w = load_file(os.path.join(cfg.paths()["sae"], "sae.safetensors"))
    W = w["W_dec"].float()
    W = W / W.norm(dim=1, keepdim=True)
    dirs = {f"feat{feat}": W[feat]}
    if n_random:
        rng = np.random.default_rng(seed)
        pool = [i for i in range(W.shape[0]) if i != feat]
        for j in rng.choice(pool, size=n_random, replace=False):
            dirs[f"rand{int(j)}"] = W[int(j)]
    return dirs


def residual_norm(cfg):
    """R = norma média do resíduo (bruto) na posição da decisão, camada do SAE."""
    cache = os.path.join(cfg.paths()["results"], "residual_norm_decision.json")
    if os.path.exists(cache):
        return json.load(open(cache))["R"]
    t = load_file(os.path.join(cfg.paths()["activations"], "decisions.safetensors"))
    R = float(t["acts"].float().norm(dim=1).mean())
    os.makedirs(cfg.paths()["results"], exist_ok=True)
    json.dump({"R": R, "n": int(t["acts"].shape[0])}, open(cache, "w"))
    return R


@torch.no_grad()
def run(args):
    cfg = preset(args.preset); dev = cfg.device
    claims = load_claims(cfg.model_tag)
    # metades fixas do AVeriTeC dev: validação = claim_id par, teste = claim_id ímpar
    # [seleção do coeficiente em validação: Arditi et al. 2024; Ferrando et al. 2025]
    if args.split == "val":
        claims = [c for c in claims if c["id"] % 2 == 0]
    elif args.split == "test":
        claims = [c for c in claims if c["id"] % 2 == 1]
    if args.n:
        claims = claims[:args.n]
    tag = "" if (args.split == "all" and not args.n) else f"__{args.split}{args.n or len(claims)}"
    R = residual_norm(cfg)
    dirs = directions(cfg, args.feat, args.random)
    if args.random:                      # só as aleatórias (a feature roda sem --random)
        dirs = {k: v for k, v in dirs.items() if k.startswith("rand")}
    betas = [float(b) for b in args.betas.split(",")]
    conds = args.conds.split(",")
    out_dir = os.path.join(ROOT, "results", cfg.name, "steer_sae"); os.makedirs(out_dir, exist_ok=True)
    print(f"[{cfg.name}] camada {cfg.layer} | R={R:.3f} | claims={len(claims)} | direções={list(dirs)[:4]}{'...' if len(dirs) > 4 else ''} "
          f"| β={betas} | condições={conds}", flush=True)

    model, tok = load_model(cfg.checkpoint, cfg.quant, dev)
    model.model.layers[cfg.layer].register_forward_hook(make_hook())

    def generate(prompt):
        inputs = tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": prompt}],
                                         add_generation_prompt=True, return_tensors="pt",
                                         return_dict=True, enable_thinking=False).to(dev)
        o = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        return strip_thinking(tok.decode(o[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))

    cells = [(c, d, b) for c in conds for d in dirs for b in betas if not (b == 0.0 and d != list(dirs)[0])]
    for cond, dname, b in cells:
        dkey = "none" if b == 0.0 else dname          # β=0 é o mesmo baseline p/ qualquer direção
        path = os.path.join(out_dir, f"{cond}__{dkey}__b{b:+.2f}{tag}.json")
        if os.path.exists(path):
            continue
        STEER["vec"] = None if b == 0.0 else (b * R * dirs[dname]).to(dev)
        part = path + ".partial"                       # retomada dentro da célula (GPU instável)
        raw = json.load(open(part)) if os.path.exists(part) else []
        if raw:
            print(f"  RESUME {cond} {dkey} β={b:+.2f}: {len(raw)}/{len(claims)}", flush=True)
        for i, c in enumerate(claims[len(raw):], start=len(raw)):
            raw.append(generate(prompt_for(c, cond)))
            if (i + 1) % 25 == 0:
                tmp = part + ".tmp"; json.dump(raw, open(tmp, "w"), ensure_ascii=False); os.replace(tmp, part)
            if (i + 1) % 100 == 0:
                print(f"  {cond} {dkey} β={b:+.2f}: {i+1}/{len(claims)}", flush=True)
            torch.cuda.empty_cache() if i % 50 == 0 else None
        preds = [normalize(r) for r in raw]
        res = {"cond": cond, "direction": dkey, "beta": b, "R": R, "layer": cfg.layer,
               "max_new_tokens": args.max_new_tokens, "ids": [c["id"] for c in claims],
               "gold": [c["gold"] for c in claims], "pred": preds, "raw": raw,
               "readable": [readable(r) for r in raw]}
        if cond != "none":
            res["pipeline_pred"] = [c[cond]["pipeline_pred"] for c in claims]
        tmp = path + ".tmp"; json.dump(res, open(tmp, "w"), ensure_ascii=False); os.replace(tmp, path)
        if os.path.exists(part):
            os.remove(part)
        acc = np.mean([p == g for p, g in zip(preds, res["gold"])]); nei = np.mean([p == "NEI" for p in preds])
        msg = f"CÉLULA {cond} {dkey} β={b:+.2f} | acc={acc:.3f} | NEI={nei:.3f} | legível={np.mean(res['readable']):.3f}"
        if cond != "none" and b == 0.0:
            agree = np.mean([p == q for p, q in zip(preds, res["pipeline_pred"])])
            msg += f" | concordância c/ pipeline={agree:.3f}"
        print(msg, flush=True)
    STEER["vec"] = None
    print("FIM", flush=True)
    sys.stdout.flush(); os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v3_llama")
    ap.add_argument("--feat", type=int, required=True)
    ap.add_argument("--betas", default="0")
    ap.add_argument("--conds", default="none,single,iterative")
    ap.add_argument("--random", type=int, default=0, help="nº de features aleatórias (controle)")
    ap.add_argument("--n", type=int, default=0, help="usa só os n primeiros claims (após o split)")
    ap.add_argument("--split", default="all", choices=["all", "val", "test"],
                    help="val = claim_id par, test = claim_id ímpar")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--check", action="store_true", help="só β=0 nas condições single/iterative")
    args = ap.parse_args()
    if args.check:
        args.betas, args.conds = "0", "single,iterative"
    run(args)


if __name__ == "__main__":
    main()
