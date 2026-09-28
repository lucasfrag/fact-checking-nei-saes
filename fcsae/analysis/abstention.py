"""Análise-payoff: com o SAE de DOMÍNIO treinado, a abstenção agora se localiza?

Reúne as ativações da POSIÇÃO DA DECISÃO (marcadas no harvest) com seus rótulos, e:
  1. reconstrói a direção de abstenção (diff-of-means NEI vs SUP/REF) e sua separação
     held-out (AUC) — o sinal continua lá;
  2. mede o FVU do SAE de domínio NA DECISÃO (onde os SAEs oficiais falhavam);
  3. decompõe a direção na base do decoder (cosseno) — ela vira uma feature?
  4. Cohen's d por feature na decisão — há feature de abstenção limpa?

Compara diretamente com o achado do método antigo (SAE oficial: max cos ≤ 0.29).

    python -m fcsae.analysis.abstention --preset v1_domain
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import torch
from safetensors.torch import load_file

from fcsae.config import preset, Config
from fcsae.common.io import load_manifest, iter_shards


def auc(pos, neg):
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(x); r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    return (a.mean(0) - b.mean(0)) / (sp + 1e-9)


def run(cfg: Config):
    dev = cfg.device if torch.cuda.is_available() else "cpu"
    shard_dir = cfg.paths()["activations"]
    sae_dir = cfg.paths()["sae"]
    man = load_manifest(shard_dir)

    # 1. coleta ativações da posição da decisão + rótulo (0=SUP,1=REF,2=NEI)
    Hs, Ls = [], []
    for _, t in iter_shards(shard_dir):
        dec = t["is_decision"].bool()
        Hs.append(t["acts"][dec].float()); Ls.append(t["label"][dec])
    H = torch.cat(Hs).numpy(); lab = torch.cat(Ls).numpy()
    ab = lab == 2                                   # NEI
    de = (lab == 0) | (lab == 1)                    # SUP|REF
    H, ab = H[ab | de], ab[ab | de]
    print(f"posições de decisão: {len(ab)} | abstém={int(ab.sum())} decisivo={int((~ab).sum())}", flush=True)

    # split held-out (par/ímpar por índice)
    idx = np.arange(len(ab)); tr = idx % 2 == 0; te = ~tr
    v = H[tr & ab].mean(0) - H[tr & ~ab].mean(0); v /= np.linalg.norm(v) + 1e-9
    proj = H[te] @ v
    auc_dir = auc(proj[ab[te]], proj[~ab[te]])
    print(f"direção de abstenção (held-out): AUC={auc_dir:.3f}", flush=True)

    # 2-4. carrega o SAE de domínio
    sd = load_file(os.path.join(sae_dir, "sae.safetensors"))
    W_dec = sd["W_dec"].float().numpy()             # [width, d]
    b_dec = sd["b_dec"].float().numpy()
    W_enc = sd["W_enc"].float().numpy(); b_enc = sd["b_enc"].float().numpy()

    # decomposição da direção no decoder (barato, não precisa do encode). Com input_norm
    # [Gao §2.1] o SAE vive no espaço centrado em d_model: compara com a direção centrada.
    Wn = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-9)
    vc = v - v.mean() if cfg.sae_input_norm else v
    cos = Wn @ (vc / (np.linalg.norm(vc) + 1e-9))
    top_cos = np.argsort(cos)[::-1][:15]

    # FVU na decisão + Cohen's d por feature — BATCHADO (o encode [N x width] inteiro
    # estoura a RAM p/ N grande; acumulamos estatísticas em blocos).
    k, width = cfg.sae_k, W_enc.shape[1]
    mean = H.mean(0)
    num = den = 0.0
    numn = denn = 0.0; sn = np.zeros(H.shape[1]); nn_ = 0     # FVU no espaço normalizado
    if cfg.sae_input_norm:                                    # média do espaço normalizado (1ª passada)
        for i in range(0, len(H), 8192):
            xc = H[i:i + 8192] - H[i:i + 8192].mean(1, keepdims=True)
            sn += (xc / (np.linalg.norm(xc, axis=1, keepdims=True) + 1e-8)).sum(0); nn_ += len(xc)
        mean_n = sn / nn_
    sa = np.zeros(width); sqa = np.zeros(width); na = 0
    sde = np.zeros(width); sqd = np.zeros(width); nd = 0
    for i in range(0, len(H), 8192):
        x = H[i:i + 8192]
        if cfg.sae_input_norm:          # [Gao §2.1]: SAE opera no espaço normalizado
            mu = x.mean(1, keepdims=True); nrm = np.linalg.norm(x - mu, axis=1, keepdims=True) + 1e-8
            xin = (x - mu) / nrm
        else:
            xin = x
        pre = (xin - b_dec) @ W_enc + b_enc
        idxk = np.argpartition(pre, -k, axis=1)[:, -k:]
        z = np.zeros_like(pre)
        rows = np.arange(x.shape[0])[:, None]
        z[rows, idxk] = np.clip(pre[rows, idxk], 0, None)
        xhat_in = z @ W_dec + b_dec
        if cfg.sae_input_norm:          # volta ao espaço bruto (comparável ao v1)
            numn += ((xin - xhat_in) ** 2).sum(); denn += ((xin - mean_n) ** 2).sum()
            xhat = xhat_in * nrm + mu
        else:
            xhat = xhat_in
        num += ((x - xhat) ** 2).sum(); den += ((x - mean) ** 2).sum()
        m = ab[i:i + 8192]
        sa += z[m].sum(0); sqa += (z[m] ** 2).sum(0); na += int(m.sum())
        sde += z[~m].sum(0); sqd += (z[~m] ** 2).sum(0); nd += int((~m).sum())
    fvu_dec = float(num / (den + 1e-9))
    mean_a, mean_d = sa / na, sde / nd
    var_a = (sqa - na * mean_a ** 2) / (na - 1)
    var_d = (sqd - nd * mean_d ** 2) / (nd - 1)
    sp = np.sqrt(((na - 1) * var_a + (nd - 1) * var_d) / (na + nd - 2))
    dcoh = (mean_a - mean_d) / (sp + 1e-9)
    top_d = np.argsort(np.abs(dcoh))[::-1][:15]

    res = {
        "n_decision": int(len(ab)), "abstem": int(ab.sum()), "decisivo": int((~ab).sum()),
        "auc_direction_heldout": auc_dir,
        "fvu_decision_domainSAE": fvu_dec,     # espaço bruto (comparável ao v1)
        "fvu_decision_normalized": (float(numn / (denn + 1e-9)) if cfg.sae_input_norm else None),
        "max_decoder_cos": float(cos.max()),
        "top_decoder_cos": [{"feat": int(i), "cos": float(cos[i])} for i in top_cos],
        "max_abs_cohens_d_feature": float(np.abs(dcoh).max()),
        "top_feats_cohens_d": [{"feat": int(i), "d": float(dcoh[i])} for i in top_d],
    }
    out = cfg.paths()["results"]; os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "abstention.json"), "w"), indent=1)
    print(f"FVU(decisão) do SAE de domínio = {fvu_dec:.3f}  "
          f"(método antigo: SAE oficial reconstrói mal a decisão)", flush=True)
    print(f"max cos(direção, feature) = {cos.max():.3f}  "
          f"(método antigo: ≤ 0.29 → distribuída)", flush=True)
    print(f"max |Cohen's d| por feature na decisão = {np.abs(dcoh).max():.2f}", flush=True)
    print(f"salvo em {out}", flush=True)
    sys.stdout.flush()
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v1_domain")
    args = ap.parse_args()
    run(preset(args.preset))


if __name__ == "__main__":
    main()
