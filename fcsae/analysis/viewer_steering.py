"""Dados do painel de steering do SAE viewer (um JSON por feature steerada).

Lê as células prontas de `results/<preset>/steer_sae/` (grade completa, 500 claims; ignora `.partial`,
`__test100` e o controle aleatório) e gera `<out>/steering/<feat>.json` com:
  - curvas dose-resposta por condição (métricas da §3.1 do nb03; rótulo = `verdict_label`, regra v2 de 15/10/2026,
    qualidade = `coherent`), incluindo α = 0 (baseline de cada condição), e a % de cada rótulo
    respondido em cada dose (`dist`; `gold_dist` = distribuição do gabarito);
  - exemplos: claims em que a dose muda o veredito, com a resposta (truncada) em cada α/R;
  - `<feat>_all_<cond>.json`: TODOS os claims da condição, com veredito e resposta (truncada) em cada α/R.

    python -m fcsae.analysis.viewer_steering --preset v3_llama --feat 19219 --out <viewer>/llama-3.1-8b-instruct/layer-15
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from fcsae.config import preset
from fcsae.steering.factcheck_sae import coherent, verdict_label, load_claims

CLS = ["SUP", "REF", "NEI", "CONF"]
DIST = CLS + ["NOLABEL"]          # regra v2: resposta sem veredito nem abstenção por extenso
COND_LABEL = {"none": "No evidence", "single": "Single-step", "iterative": "Iterative"}
MAX_CHARS = 700          # resposta exibida (o resto é cortado com "…")
N_EX = 6                 # exemplos por tipo e condição


def f1(pred, gold, c):
    tp = sum(p == c and g == c for p, g in zip(pred, gold)); fp = sum(p == c and g != c for p, g in zip(pred, gold))
    fn = sum(p != c and g == c for p, g in zip(pred, gold))
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def metrics(pred, gold, ok, capped):
    F = {c: f1(pred, gold, c) for c in CLS}
    nn = [(p, g) for p, g in zip(pred, gold) if p != "NEI"]
    return {"acc": float(np.mean([p == g for p, g in zip(pred, gold)])),
            "macro_f1": float(np.mean([F["SUP"], F["REF"], F["NEI"]])),
            "f1_nei": F["NEI"], "f1_conf": F["CONF"],
            "nei_rate": float(np.mean([p == "NEI" for p in pred])),
            "err_non_nei": float(np.mean([p != g for p, g in nn])) if nn else None,
            "coherent": float(np.mean(ok)),
            "hit_cap": float(np.mean(capped)),   # resposta chegou ao teto de tokens (loop)
            "dist": {c: float(np.mean([p == c for p in pred])) for c in DIST}}   # % de cada rótulo respondido


def cut(t):
    t = (t or "").strip()
    return t if len(t) <= MAX_CHARS else t[:MAX_CHARS].rstrip() + " …"


def run(args):
    cfg = preset(args.preset)
    d = os.path.join("results", args.preset, "steer_sae")
    cells = {}
    for p in sorted(glob.glob(f"{d}/*.json")):
        m = re.match(r"(\w+?)__(\w+?)__b([+-][\d.]+)\.json$", os.path.basename(p))
        if not m or m.group(2) not in (f"feat{args.feat}", "none"):
            continue
        r = json.load(open(p, encoding="utf-8"))
        if len(r["ids"]) < 500:
            continue
        cells[(r["cond"], round(float(r["beta"]), 2))] = r
    claims = {c["id"]: c for c in load_claims(cfg.model_tag)}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.checkpoint)
    cap = lambda t, n: len(tok(t, add_special_tokens=False).input_ids) >= n - 6   # margem p/ tokens especiais

    curves, examples, all_rows = {}, {}, {}
    # régua da dose: R_c (norma centrada) para todos os modelos (decisão do Lucas, 12/10/2026). Células rodadas com a
    # norma bruta (Llama, Qwen3) são convertidas exatamente: α/R_c = (α/R) × R / R_c.
    norms = json.load(open(os.path.join(cfg.paths()["results"], "residual_norm_decision.json")))
    rkind = next(iter(cells.values())).get("R_kind", "raw")
    conv = 1.0 if rkind == "centered" else norms["R"] / norms["R_centered"]
    D = lambda b: round(b * conv, 2)
    for cond in COND_LABEL:
        betas = sorted(b for (c, b) in cells if c == cond)
        if 0.0 not in betas or len(betas) < 2:          # condição sem steering rodado (só o baseline)
            continue
        base = cells[(cond, 0.0)]; ids = base["ids"]; gold = base["gold"]
        preds = {b: [verdict_label(x) for x in cells[(cond, b)]["raw"]] for b in betas}   # regra v2 (15/10/2026)
        curves[cond] = [{"alpha_over_R": D(b), **metrics(preds[b], gold, [coherent(x) for x in cells[(cond, b)]["raw"]],
                                                      [cap(x, cells[(cond, b)]["max_new_tokens"]) for x in cells[(cond, b)]["raw"]])}
                        for b in betas]
        # exemplos: dose que tira a abstenção e acerta / dose que cria abstenção onde o modelo acertava
        neg = [b for b in betas if b < 0]; pos = [b for b in betas if b > 0]
        kinds = []
        if neg:
            bn = max(neg)  # a menor dose negativa que já muda o veredito é a mais legível
            for b in sorted(neg, reverse=True):
                if sum(preds[0.0][i] == "NEI" and preds[b][i] == gold[i] != "NEI" for i in range(len(ids))) >= N_EX:
                    bn = b; break
            kinds.append(("decrease: abstention removed, verdict now correct", bn,
                          [i for i in range(len(ids)) if preds[0.0][i] == "NEI" and preds[bn][i] == gold[i] != "NEI"]))
            kinds.append(("decrease: correct abstention lost", bn,
                          [i for i in range(len(ids)) if gold[i] == "NEI" == preds[0.0][i] and preds[bn][i] != "NEI"]))
        if pos:
            bp = min(pos)
            for b in sorted(pos):
                if sum(preds[0.0][i] == gold[i] != "NEI" and preds[b][i] == "NEI" for i in range(len(ids))) >= N_EX:
                    bp = b; break
            kinds.append(("increase: correct verdict turned into abstention", bp,
                          [i for i in range(len(ids)) if preds[0.0][i] == gold[i] != "NEI" and preds[bp][i] == "NEI"]))
        rng = np.random.default_rng(0)
        ex = []
        for kind, bsel, idx in kinds:
            pick = sorted(rng.choice(idx, size=min(N_EX, len(idx)), replace=False).tolist()) if idx else []
            for i in pick:
                c = claims[ids[i]]
                ev = [] if cond == "none" else [{"text": (e.get("text") or "")[:300] + ("…" if len(e.get("text") or "") > 300 else ""),
                                                "url": e.get("url")} for e in c[cond]["evidence"][:3]]
                ex.append({"kind": kind, "alpha_over_R_selected": D(bsel), "claim_id": ids[i], "claim": c["claim"],
                           "gold": gold[i], "evidence": ev,
                           "responses": [{"alpha_over_R": D(b), "pred": preds[b][i], "text": cut(cells[(cond, b)]["raw"][i])}
                                         for b in betas]})
        examples[cond] = ex
        # todas as instâncias (página "browse all claims" do viewer): um arquivo por condição
        rows = []
        for i, cid in enumerate(ids):
            c = claims[cid]
            ev = [] if cond == "none" else [{"text": (e.get("text") or "")[:300] + ("…" if len(e.get("text") or "") > 300 else ""),
                                            "url": e.get("url")} for e in c[cond]["evidence"][:3]]
            rows.append({"claim_id": cid, "claim": c["claim"], "gold": gold[i], "evidence": ev,
                         "pred": [preds[b][i] for b in betas], "text": [cut(cells[(cond, b)]["raw"][i]) for b in betas]})
        all_rows[cond] = {"betas": [D(b) for b in betas], "rows": rows}
    R = norms["R_centered"]
    out = {"feature": args.feat, "preset": cfg.name, "model": cfg.checkpoint, "layer": cfg.layer, "R": R,
           "R_raw": norms["R"], "dose_unit": "alpha/R_c", "run_with": rkind,
           "n_claims": 500, "dataset": "AVeriTeC dev (first 500 claims)",
           "conditions": {k: v for k, v in COND_LABEL.items() if k in curves},
           "formula": "x <- x + alpha * d_feat at every position of layer L, prompt and generated tokens (Ferrando et al., 2025, Eq. 4); "
                      "alpha is reported as alpha/R, R = mean residual norm at the decision position",
           "label_rule": "first label written by the model (uppercase, tolerating misspellings; or on the first line); otherwise an abstention written in words counts as NEI; otherwise NOLABEL (no verdict)",
           "gold_dist": {c: float(np.mean([g == c for g in gold])) for c in CLS},
           "curves": curves, "examples": examples}
    os.makedirs(os.path.join(args.out, "steering"), exist_ok=True)
    p = os.path.join(args.out, "steering", f"{args.feat}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    for cond, d_all in all_rows.items():
        pa = os.path.join(args.out, "steering", f"{args.feat}_all_{cond}.json")
        json.dump({"feature": args.feat, "cond": cond, "condition": COND_LABEL[cond], **d_all},
                  open(pa, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
        print(f"  todas as instâncias ({cond}): {pa} ({os.path.getsize(pa)/1e6:.1f} MB)")
    for cond, cv in curves.items():
        print(f"{cond:10s} α/R={[c['alpha_over_R'] for c in cv]}  exemplos={len(examples[cond])}")
    print(f"FIM: {p} ({os.path.getsize(p)/1e3:.0f} kB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v3_llama")
    ap.add_argument("--feat", type=int, default=19219)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
