"""Quão "espalhada" está a abstenção entre as features do SAE? (pergunta do Lucas, 01/10/2026)

Duas medidas, nas posições de decisão (metade par = treino, ímpar = teste, como em `abstention.py`):
  1. **Separação com k features:** regressão logística NEI × SUP/REF usando só as k features de maior
     |Cohen's d| (k = 1, 2, 3, 5, 10, 20, 50, 100) e usando todas as 32.768 (teto); AUC no teste.
     Responde: "quantas features são precisas para separar tão bem quanto o SAE inteiro?"
  2. **Cobertura da direção de abstenção:** a direção diff-of-means (NEI − decidido, no espaço normalizado
     em que o SAE opera) projetada no subespaço das k linhas do decoder mais alinhadas com ela; reporta
     ‖proj‖/‖v‖ (1 = a direção está inteira nessas k features).

    python -m fcsae.analysis.spread --preset v3_deepseek
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fcsae.config import preset
from fcsae.analysis.inspect_feature import decision_cache
from fcsae.sae.model import TopKSAE, normalize_input

KS = [1, 2, 3, 5, 10, 20, 50, 100]


@torch.no_grad()
def run(args):
    cfg = preset(args.preset); dev = "cuda" if torch.cuda.is_available() else "cpu"
    A, L, _ = decision_cache(cfg)
    keep = (L == 0) | (L == 1) | (L == 2)
    A, y = A[keep], (L[keep] == 2).numpy().astype(np.int8)
    n = len(y); tr = np.arange(n) % 2 == 0; te = ~tr
    w = load_file(os.path.join(cfg.paths()["sae"], "sae.safetensors"))
    sae = TopKSAE(A.shape[1], cfg.sae_width, cfg.sae_k, cfg.sae_k_aux, cfg.sae_dead_steps)
    sae.load_state_dict({**w, "steps_since_fire": sae.steps_since_fire}); sae = sae.to(dev)
    W = cfg.sae_width
    rows, cols, vals = [], [], []
    vsum = {0: torch.zeros(A.shape[1], device=dev), 1: torch.zeros(A.shape[1], device=dev)}
    for i in range(0, n, 8192):
        x = A[i:i + 8192].to(dev).float()
        xin = normalize_input(x)[0] if cfg.sae_input_norm else x
        _, z, _ = sae(xin)
        v, ix = z.topk(cfg.sae_k, dim=-1)
        m = v > 0
        r = torch.arange(i, i + len(x), device=dev)[:, None].expand_as(ix)
        rows.append(r[m].cpu()); cols.append(ix[m].cpu()); vals.append(v[m].cpu())
        yy = torch.as_tensor(y[i:i + 8192], device=dev).bool(); tt = torch.as_tensor(tr[i:i + 8192], device=dev)
        vsum[1] += xin[yy & tt].sum(0); vsum[0] += xin[~yy & tt].sum(0)
    Z = sparse.csr_matrix((torch.cat(vals).numpy(), (torch.cat(rows).numpy(), torch.cat(cols).numpy())), shape=(n, W))

    # Cohen's d por feature (treino; inativa = 0)
    Ztr, ytr = Z[tr], y[tr]
    def stats(M):
        mu = np.asarray(M.mean(0)).ravel(); sq = np.asarray(M.multiply(M).mean(0)).ravel()
        return mu, np.maximum(sq - mu ** 2, 0)
    m1, v1 = stats(Ztr[ytr == 1]); m0, v0 = stats(Ztr[ytr == 0])
    d = (m1 - m0) / (np.sqrt((v1 + v0) / 2) + 1e-9)
    order = np.argsort(-np.abs(d))

    res = {"preset": cfg.name, "n": int(n), "top_features_by_abs_d": [[int(f), float(d[f])] for f in order[:20]], "auc_by_k": {}}
    for k in KS:
        f = order[:k]
        clf = LogisticRegression(max_iter=2000).fit(Ztr[:, f], ytr)
        res["auc_by_k"][str(k)] = float(roc_auc_score(y[te], clf.decision_function(Z[te][:, f])))
        print(f"k={k:4d} features (maior |d|): AUC teste = {res['auc_by_k'][str(k)]:.3f}", flush=True)
    clf = LogisticRegression(max_iter=3000, C=0.1).fit(Ztr, ytr)
    res["auc_all_features"] = float(roc_auc_score(y[te], clf.decision_function(Z[te])))
    print(f"todas as {W} features (teto): AUC teste = {res['auc_all_features']:.3f}", flush=True)

    # cobertura da direção de abstenção pelas k linhas do decoder mais alinhadas
    vdir = (vsum[1] / max(1, int((ytr == 1).sum())) - vsum[0] / max(1, int((ytr == 0).sum()))).cpu().numpy()
    vdir /= np.linalg.norm(vdir)
    Wd = w["W_dec"].float().numpy(); Wd = Wd / np.linalg.norm(Wd, axis=1, keepdims=True)
    cos = Wd @ vdir; oc = np.argsort(-np.abs(cos))
    res["direction_coverage_by_k"] = {}
    for k in KS:
        B = Wd[oc[:k]].T                                     # [d, k]
        coef, *_ = np.linalg.lstsq(B, vdir, rcond=None)
        res["direction_coverage_by_k"][str(k)] = float(np.linalg.norm(B @ coef))
        print(f"k={k:4d} linhas do decoder: cobrem {res['direction_coverage_by_k'][str(k)]:.3f} da direção de abstenção", flush=True)
    out = os.path.join(cfg.paths()["results"], "spread.json")
    json.dump(res, open(out, "w"), indent=1); print("salvo em", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--preset", required=True); run(ap.parse_args())
