"""Teste de GENERALIZAÇÃO multi-dataset: aplica a feature/direção de abstenção
(treinada em VitaminC/FEVER) a datasets-alvo nunca vistos pelo SAE, e mede se separam
abstém (NEI) vs decisivo (SUP/REF).

    python -m fcsae.analysis.generalize --sae v1_domain --feat 22280 --test climate_fever
    python -m fcsae.analysis.generalize --sae v1_domain --feat 22280 --test scifact
    python -m fcsae.analysis.generalize --sae v1_domain --feat 22280 --test healthver
    python -m fcsae.analysis.generalize --sae v1_domain --feat 22280 --test averitec
"""
from __future__ import annotations

import os
import sys
import csv
import json
import glob
import argparse
from itertools import islice

import torch
import numpy as np
from safetensors.torch import load_file

from fcsae.config import preset
from fcsae.common.modeling import load_model, residual_at_layer
from fcsae.common.io import iter_shards, LABEL2ID
from fcsae.data.prompt import build_verdict_prompt, templated_input_ids

AVERITEC_DIR = "old_method/corpora/averitec/data"
HEALTHVER_DIR = "data_raw/healthver"


# ------------------------------------------------------------------ loaders de teste
def _averitec():
    out = []
    for s in ("train", "dev"):
        for x in json.load(open(f"{AVERITEC_DIR}/{s}.json", encoding="utf-8")):
            lab = x.get("label") or ""
            if "Conflict" in lab or "Cherry" in lab:
                continue
            y = "REF" if "Refut" in lab else "SUP" if "Support" in lab else "NEI"
            ev = [f"{q.get('question','')} {a['answer']}".strip()
                  for q in x.get("questions", []) for a in q.get("answers", []) if a.get("answer")]
            out.append((x.get("claim", ""), ev, y))
    return out


def _climate_fever():
    from datasets import load_dataset
    m = {0: "SUP", 1: "REF", 2: "NEI"}          # 3=DISPUTED -> descarta
    out = []
    for x in load_dataset("tdiggelm/climate_fever", split="test"):
        if x["claim_label"] not in m:
            continue
        ev = [e["evidence"] for e in x["evidences"] if e.get("evidence")]
        out.append((x["claim"], ev, m[x["claim_label"]]))
    return out


def _scifact():
    from datasets import load_dataset
    corpus = {int(d["doc_id"]): " ".join(d["abstract"]) for d in load_dataset("allenai/scifact", "corpus", split="train")}
    out = []
    for x in load_dataset("allenai/scifact", "claims", split="train"):
        el = x.get("evidence_label") or ""
        y = "SUP" if el == "SUPPORT" else "REF" if el == "CONTRADICT" else "NEI"
        docs = x.get("cited_doc_ids") or []
        ev = [corpus[int(d)] for d in docs if int(d) in corpus]
        if not ev:
            continue
        out.append((x["claim"], ev, y))
    return out


def _healthver(max_n=6000):
    m = {"Supports": "SUP", "Refutes": "REF", "Neutral": "NEI"}
    rows = []
    for f in sorted(glob.glob(f"{HEALTHVER_DIR}/*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r.get("label") in m and r.get("claim"):
                rows.append((r["claim"], [r.get("evidence", "")], m[r["label"]]))
    np.random.default_rng(0).shuffle(rows)
    return rows[:max_n]


TESTS = {"averitec": _averitec, "climate_fever": _climate_fever,
         "scifact": _scifact, "healthver": _healthver}


def auc(pos, neg):
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def run(sae_name, feat, test_name):
    cfg = preset(sae_name); dev = cfg.device
    sae = load_file(f"{cfg.paths()['sae']}/sae.safetensors")
    W_enc = sae["W_enc"].float().numpy(); b_enc = sae["b_enc"].float().numpy(); b_dec = sae["b_dec"].float().numpy()

    # direção + referência in-domain (das ativações de treino do SAE)
    Ht, Lt = [], []
    for _, t in iter_shards(cfg.paths()["activations"]):
        m = t["is_decision"].bool()
        Ht.append(t["acts"][m].float().numpy()); Lt.append(t["label"][m].numpy())
    Ht = np.concatenate(Ht); Lt = np.concatenate(Lt)
    ab_t = Lt == LABEL2ID["NEI"]; de_t = (Lt == LABEL2ID["SUP"]) | (Lt == LABEL2ID["REF"])
    vdir = Ht[ab_t].mean(0) - Ht[de_t].mean(0); vdir /= np.linalg.norm(vdir) + 1e-9
    act_t = np.maximum((Ht - b_dec) @ W_enc[:, feat] + b_enc[feat], 0.0)
    auc_feat_in = auc(act_t[ab_t], act_t[de_t]); auc_dir_in = auc((Ht @ vdir)[ab_t], (Ht @ vdir)[de_t])

    data = TESTS[test_name]()
    model, tok = load_model(cfg.checkpoint, cfg.quant, dev)
    print(f"[{test_name}] {len(data)} claims | forward camada {cfg.layer}...", flush=True)
    H, L = [], []
    for i, (claim, ev, lab) in enumerate(data):
        ids, _ = templated_input_ids(tok, build_verdict_prompt(claim, ev, cfg.max_evidence), dev, cfg.max_seq_len)
        if ids.shape[1] < 4:
            continue
        H.append(residual_at_layer(model, ids, cfg.layer)[-1].to(torch.float16).cpu().numpy())
        L.append(lab)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(data)}", flush=True)
        torch.cuda.empty_cache()
    H = np.stack(H); L = np.array(L)
    ab = L == "NEI"; de = (L == "SUP") | (L == "REF")

    act = np.maximum((H - b_dec) @ W_enc[:, feat] + b_enc[feat], 0.0)
    proj = H @ vdir
    auc_feat = auc(act[ab], act[de]); auc_dir = auc(proj[ab], proj[de])
    order = np.argsort(act)[::-1]
    k = min(200, int(ab.sum()) * 2 or 200)
    nei_top = float(ab[order[:k]].mean()); nei_all = float(ab.mean())

    res = {"sae": sae_name, "feat": feat, "test": test_name, "n": int(len(L)),
           "n_nei": int(ab.sum()), "n_dec": int(de.sum()),
           "auc_feat_indomain": auc_feat_in, "auc_feat": auc_feat,
           "auc_dir_indomain": auc_dir_in, "auc_dir": auc_dir,
           "topk": k, "nei_top": nei_top, "nei_base": nei_all}
    out = f"results/{cfg.name}"; os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/generalize_{test_name}.json", "w"), indent=1)
    print(f"\n[{test_name}] NEI={int(ab.sum())} dec={int(de.sum())} | "
          f"AUC feature {auc_feat_in:.3f}->{auc_feat:.3f} | AUC direção {auc_dir_in:.3f}->{auc_dir:.3f} | "
          f"NEI top{k} {nei_top:.2f} vs base {nei_all:.2f} ({nei_top/max(nei_all,1e-9):.1f}x)", flush=True)
    print(f"salvo em {out}/generalize_{test_name}.json", flush=True)
    sys.stdout.flush(); os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default="v1_domain")
    ap.add_argument("--feat", type=int, default=22280)
    ap.add_argument("--test", required=True, choices=list(TESTS))
    args = ap.parse_args()
    run(args.sae, args.feat, args.test)


if __name__ == "__main__":
    main()
