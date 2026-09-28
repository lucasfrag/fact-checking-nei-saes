# -*- coding: utf-8 -*-
"""Estágio 2: steering aplicado ao fact-checking (AVeriTeC dev, 500 claims).

Estudo fatorial: 3 condições de EVIDÊNCIA (none / single-step / iterative) × grade de β
(β<0 suprime a abstenção, β>0 induz). Para cada célula, mede acurácia, F1 por classe
(SUP/REF/NEI/CONF) e taxa de NEI. A evidência single/iterative é a MESMA recuperada pelos
pipelines do trabalho anterior (campo pipeline.steps[].evidence).

    python -m fcsae.analysis.factcheck_steer --model llama3.1-8b-hf --evidence single \
        --betas="-1,-0.5,-0.25,0,0.25,0.5,1"
"""
import argparse, json, os
import numpy as np, torch

from fcsae.config import preset
from fcsae.common.modeling import load_model
from fcsae.common.io import iter_shards, LABEL2ID

MODEL_PRESET = {
    "llama3.1-8b-hf":    ("v1_domain",    15),
    "deepseek-r1-8b-hf": ("v1_deepseek",  15),
    "qwen3-8b-hf":       ("v1_qwen3_l19", 19),
}
EV_DIR = {"single": "averitec_original_pipeline", "iterative": "averitec_iterative_pipeline"}
OUT = "old_method/corpora/averitec/outputs"
LABELS = ["SUP", "REF", "NEI", "CONF"]
COEFF = [0.0]


def norm_gold(s):
    s = (s or "").strip().lower()
    if "not enough" in s: return "NEI"
    if "support" in s:    return "SUP"
    if "refut" in s:      return "REF"
    if "conflict" in s or "cherry" in s: return "CONF"
    return "?"


def parse_verdict(txt):
    t = (txt or "").upper()
    if "CONFLICT" in t or "CHERRY" in t: return "CONF"
    if "NOT ENOUGH" in t or "INSUFFICIENT" in t: return "NEI"
    if "REFUT" in t: return "REF"
    if "SUPPORT" in t: return "SUP"
    return "?"


def _steps(row):
    st = row.get("pipeline", {}).get("steps", [])
    if isinstance(st, str):
        try: st = json.loads(st.replace("'", '"'))
        except Exception: st = []
    return st or []


def build_prompt(row, evmode):
    """evmode: 'none' (sem evidência), 'single' ou 'iterative' (passagens recuperadas)."""
    if evmode == "none":
        ev_block, qa = "(no evidence provided — decide from the claim alone)", "(none)"
    else:
        steps = _steps(row)
        seen, passages = set(), []
        for s in steps:
            for e in (s.get("evidence") or []):
                t = ((e.get("text") if isinstance(e, dict) else str(e)) or "").strip()[:500]
                if t and t not in seen:
                    seen.add(t); passages.append(t)
        ev_block = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(passages[:8])) or "(no evidence retrieved)"
        qa = "\n\n".join(f"Q: {s.get('question','')}\nA: {s.get('answer','')}" for s in steps[:5]) or "(none)"
    return f"""Task: Determine the final verdict of a claim based on evidence.

Claim:
{row.get('claim','')}

Claim date:
{row.get('claim_date','')}

Speaker:
{row.get('speaker','')}

---

Evidence:
{ev_block}

---

Answers derived from evidence (may be incomplete or noisy):
{qa}

---

Instructions:
- Determine whether the evidence SUPPORTS or REFUTES the claim. Reason directly about the claim.
- If evidence confirms the claim -> SUPPORTED
- If evidence contradicts the claim -> REFUTED
- If evidence is insufficient or unclear -> NOT ENOUGH EVIDENCE
- If both strong support and refutation exist -> CONFLICTING EVIDENCE/CHERRYPICKING

Output: return ONLY one label:
SUPPORTED
REFUTED
NOT ENOUGH EVIDENCE
CONFLICTING EVIDENCE/CHERRYPICKING"""


SYS = ("You are a fact-checking assistant. Respond in English. Be factual and precise; "
       "do not hallucinate; follow the instructions strictly.")


def _direction(cfg, dev):
    """Direção de abstenção (norma = R). Cache em results/<preset>/abstention_dir.json —
    os shards podem ser removidos após computada uma vez."""
    cache = os.path.join(cfg.paths()["results"], "abstention_dir.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        return torch.tensor(d["dvec"], dtype=torch.float32).to(dev), float(d["R"])
    Ht, Lt = [], []
    for _, t in iter_shards(cfg.paths()["activations"]):
        m = t["is_decision"].bool()
        Ht.append(t["acts"][m].float()); Lt.append(t["label"][m])
    Ht = torch.cat(Ht); Lt = torch.cat(Lt)
    ab = Lt == LABEL2ID["NEI"]; de = (Lt == LABEL2ID["SUP"]) | (Lt == LABEL2ID["REF"])
    dvec = Ht[ab].mean(0) - Ht[de].mean(0)
    R = float(Ht.norm(dim=1).mean())
    dvec = dvec / (dvec.norm() + 1e-9) * R
    os.makedirs(cfg.paths()["results"], exist_ok=True)
    json.dump({"dvec": dvec.tolist(), "R": R}, open(cache, "w"))
    return dvec.to(dev), R


def _metrics(preds, gold):
    acc = float(np.mean([p == g for p, g in zip(preds, gold)]))
    per = {}
    for c in LABELS:
        tp = sum(p == c and g == c for p, g in zip(preds, gold))
        fp = sum(p == c and g != c for p, g in zip(preds, gold))
        fn = sum(p != c and g == c for p, g in zip(preds, gold))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
                  "precision": round(prec, 4), "recall": round(rec, 4),
                  "support": sum(g == c for g in gold)}
    macro_f1 = float(np.mean([per[c]["f1"] for c in LABELS]))
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "nei_rate": round(float(np.mean([p == "NEI" for p in preds])), 4),
            "unparsed": sum(p == "?" for p in preds), "per_class": per}


@torch.no_grad()
def run(model_tag, betas, evidence, n=0, max_new_tokens=10):
    pname, layer = MODEL_PRESET[model_tag]
    cfg = preset(pname); dev = cfg.device
    src = EV_DIR.get(evidence, EV_DIR["single"])          # 'none' usa as claims do single
    rows = [json.loads(l) for l in open(f"{OUT}/{src}/{model_tag}/averitec_dev.jsonl", encoding="utf-8")]
    if n: rows = rows[:n]
    gold = [norm_gold(r["gold_label"]) for r in rows]
    print(f"[{model_tag}] evidência={evidence} | {len(rows)} claims | camada={layer}", flush=True)

    dv, R = _direction(cfg, dev)
    print(f"direção camada {layer}: ||add em β=1||={dv.norm():.1f}", flush=True)
    model, tok = load_model(cfg.checkpoint, cfg.quant, dev)

    def hook(mod, inp, out):
        if COEFF[0] == 0.0: return out
        add = (COEFF[0] * dv).to(out[0].dtype if isinstance(out, tuple) else out.dtype)
        return (out[0] + add,) + out[1:] if isinstance(out, tuple) else out + add
    model.model.layers[layer].register_forward_hook(hook)

    # reasoning models (DeepSeek-R1): injeta "</think>" p/ pular o raciocínio e ir ao veredito
    skip_think = "deepseek" in model_tag
    skip_ids = (tok("</think>\n\n", add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
                if skip_think else None)

    def templ(prompt):
        try:
            text = tok.apply_chat_template([{"role": "system", "content": SYS},
                {"role": "user", "content": prompt}], add_generation_prompt=True,
                tokenize=False, enable_thinking=False)   # Qwen3: veredito direto
        except TypeError:
            text = tok.apply_chat_template([{"role": "system", "content": SYS},
                {"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt",
                  truncation=True, max_length=1536).input_ids.to(dev)
        if skip_ids is not None:
            ids = torch.cat([ids, skip_ids], dim=1)
        return ids

    ids_all = [templ(build_prompt(r, evidence)) for r in rows]
    out_dir = f"results/{pname}"; os.makedirs(out_dir, exist_ok=True)
    cache_path = f"{out_dir}/factcheck_preds_{evidence}.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    results = {}
    for b in betas:
        key = f"{b:+.2f}"
        if key in cache and len(cache[key]) == len(rows):
            preds = cache[key]
        else:
            COEFF[0] = float(b); preds = []
            for i, ids in enumerate(ids_all):
                o = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
                txt = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
                if "</think>" in txt: txt = txt.split("</think>")[-1]
                preds.append(parse_verdict(txt))
                if (i + 1) % 100 == 0: print(f"  β={key} {i+1}/{len(ids_all)}", flush=True)
            cache[key] = preds; json.dump(cache, open(cache_path, "w"))
        m = _metrics(preds, gold); results[key] = m
        print(f"β={key} | acc={m['accuracy']:.3f} | macroF1={m['macro_f1']:.3f} | "
              f"NEI={m['nei_rate']:.3f} | F1(S/R/N/C)="
              + "/".join(f"{m['per_class'][c]['f1']:.2f}" for c in LABELS), flush=True)

    base = {"model": model_tag, "preset": pname, "layer": layer, "evidence": evidence,
            "n": len(rows), "betas": [f"{b:+.2f}" for b in betas],
            "gold_dist": {c: gold.count(c) for c in LABELS}, "by_beta": results}
    json.dump(base, open(f"{out_dir}/factcheck_{evidence}.json", "w"), indent=1)
    print(f"salvo: {out_dir}/factcheck_{evidence}.json", flush=True)
    return base


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_PRESET))
    ap.add_argument("--evidence", default="single", choices=["none", "single", "iterative"])
    ap.add_argument("--betas", default="-1,-0.5,-0.25,0,0.25,0.5,1")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=10)
    a = ap.parse_args()
    run(a.model, [float(x) for x in a.betas.split(",")], a.evidence, a.n, a.max_new_tokens)
