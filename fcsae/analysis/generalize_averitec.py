"""Teste de GENERALIZAÇÃO: a feature/direção de abstenção (treinada em VitaminC/FEVER)
transfere para o AVeriTeC — dataset-alvo nunca visto pelo SAE?

Aplica, SEM re-treinar, o SAE do v1 aos claims do AVeriTeC (train+dev, evidência-ouro
= QA pairs), na posição da decisão, e mede se a feature 22280 e a direção diff-of-means
separam abstém (NEI) vs decisivo (SUP/REF) no AVeriTeC.

    python -m fcsae.analysis.generalize_averitec --sae v1_domain --feat 22280
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import torch
import numpy as np
from safetensors.torch import load_file

from fcsae.config import preset
from fcsae.common.modeling import load_model, residual_at_layer
from fcsae.common.io import iter_shards, LABEL2ID
from fcsae.data.prompt import build_verdict_prompt, templated_input_ids

AVERITEC = "old_method/corpora/averitec/data"


def auc(pos, neg):
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_averitec(splits=("train", "dev")):
    """Retorna [(claim, [evidências], label SUP/REF/NEI)], descartando CONFLICTING."""
    out = []
    for s in splits:
        for x in json.load(open(f"{AVERITEC}/{s}.json", encoding="utf-8")):
            lab = (x.get("label") or "")
            if "Conflict" in lab or "Cherry" in lab:
                continue
            y = "REF" if "Refut" in lab else "SUP" if "Support" in lab else "NEI"
            ev = []
            for q in x.get("questions", []):
                for a in q.get("answers", []):
                    if a.get("answer"):
                        ev.append(f"{q.get('question','')} {a['answer']}".strip())
            out.append((x.get("claim", ""), ev, y))
    return out


@torch.no_grad()
def run(sae_name, feat):
    cfg = preset(sae_name)
    dev = cfg.device
    sae = load_file(f"{cfg.paths()['sae']}/sae.safetensors")
    W_enc = sae["W_enc"].float().numpy(); b_enc = sae["b_enc"].float().numpy()
    b_dec = sae["b_dec"].float().numpy()

    # direção de abstenção do SAE de treino (diff-of-means nas ativações de treino)
    Ht, Lt = [], []
    for _, t in iter_shards(cfg.paths()["activations"]):
        m = t["is_decision"].bool()
        Ht.append(t["acts"][m].float().numpy()); Lt.append(t["label"][m].numpy())
    Ht = np.concatenate(Ht); Lt = np.concatenate(Lt)
    ab_t = Lt == LABEL2ID["NEI"]; de_t = (Lt == LABEL2ID["SUP"]) | (Lt == LABEL2ID["REF"])
    vdir = Ht[ab_t].mean(0) - Ht[de_t].mean(0); vdir /= np.linalg.norm(vdir) + 1e-9
    # referência in-domain: AUC da feature e da direção no próprio treino
    act_t = np.maximum((Ht - b_dec) @ W_enc[:, feat] + b_enc[feat], 0.0)
    auc_feat_in = auc(act_t[ab_t], act_t[de_t]); auc_dir_in = auc((Ht @ vdir)[ab_t], (Ht @ vdir)[de_t])
    print(f"in-domain ({sae_name}): AUC feature {feat}={auc_feat_in:.3f} | AUC direção={auc_dir_in:.3f}", flush=True)

    # AVeriTeC: forward + resíduo na decisão
    data = load_averitec()
    model, tok = load_model(cfg.checkpoint, cfg.quant, dev)
    print(f"AVeriTeC: {len(data)} claims (forward na camada {cfg.layer})...", flush=True)
    H, L = [], []
    for i, (claim, ev, lab) in enumerate(data):
        ids, _ = templated_input_ids(tok, build_verdict_prompt(claim, ev, cfg.max_evidence),
                                     dev, cfg.max_seq_len)
        if ids.shape[1] < 4:
            continue
        H.append(residual_at_layer(model, ids, cfg.layer)[-1].to(torch.float16).cpu().numpy())
        L.append(lab)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(data)}", flush=True)
        torch.cuda.empty_cache()
    H = np.stack(H); L = np.array(L)
    ab = L == "NEI"; de = (L == "SUP") | (L == "REF")
    n_nei, n_dec = int(ab.sum()), int(de.sum())

    # aplica feature e direção do SAE de treino ao AVeriTeC
    act = np.maximum((H - b_dec) @ W_enc[:, feat] + b_enc[feat], 0.0)
    proj = H @ vdir
    auc_feat = auc(act[ab], act[de]); auc_dir = auc(proj[ab], proj[de])
    order = np.argsort(act)[::-1]
    nei_top = float(ab[order[:200]].mean()); nei_all = float(ab.mean())

    print(f"\n===== GENERALIZAÇÃO no AVeriTeC (SAE treinado em VitaminC/FEVER) =====", flush=True)
    print(f"claims: {len(L)} | NEI={n_nei} decisivo={n_dec}", flush=True)
    print(f"AUC feature {feat}: in-domain {auc_feat_in:.3f} -> AVeriTeC {auc_feat:.3f}", flush=True)
    print(f"AUC direção:        in-domain {auc_dir_in:.3f} -> AVeriTeC {auc_dir:.3f}", flush=True)
    print(f"NEI no top-200 da feature: {nei_top:.2f} vs base {nei_all:.2f} "
          f"(enriq {nei_top/max(nei_all,1e-9):.1f}x)", flush=True)

    res = {"sae": sae_name, "feat": feat, "n": int(len(L)), "n_nei": n_nei, "n_dec": n_dec,
           "auc_feat_indomain": auc_feat_in, "auc_feat_averitec": auc_feat,
           "auc_dir_indomain": auc_dir_in, "auc_dir_averitec": auc_dir,
           "nei_top200": nei_top, "nei_base": nei_all,
           "nei_mean": float(act[ab].mean()), "dec_mean": float(act[de].mean())}
    out = f"results/{cfg.name}"; os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/generalize_averitec.json", "w"), indent=1)
    print(f"salvo em {out}/generalize_averitec.json", flush=True)
    sys.stdout.flush(); os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default="v1_domain")
    ap.add_argument("--feat", type=int, default=22280)
    args = ap.parse_args()
    run(args.sae, args.feat)


if __name__ == "__main__":
    main()
