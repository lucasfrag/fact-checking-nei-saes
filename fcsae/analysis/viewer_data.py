"""Gera os dados do SAE viewer (adaptado do sae-viewer da OpenAI, MIT) para TODAS as features.

Usa só o SAE e o cache das ativações da decisão (`decisions.safetensors`) — não precisa do LLM.
Para cada feature, um JSON no formato do sae-viewer (`density`, `mean_act`, `mean_act_squared`,
`hist`, `top`, `random`), acrescido de campos do nosso estudo:
  - por exemplo: `label` (gold SUP/REF/NEI), `source`, `evidence` (trecho);
  - por feature: `fc` (fração ativa, taxa de NEI quando ativa, AUC NEI×SUP/REF aproximada por
    histograma, % NEI no top-10/top-50, taxa de NEI por decil entre as ativas), `hist_by_label`,
    `neighbors` (cosseno no decoder), `logits` (tokens promovidos/suprimidos: logit lens da direção
    do decoder pela norma final + unembedding do modelo base).
Os exemplos mostram o CLAIM tokenizado + um token final "▶ verdict" com a ativação da feature na
posição da decisão (a única posição cacheada). Heatmaps do prompt inteiro: etapa separada (LLM).

    python -m fcsae.analysis.viewer_data --preset v3_llama --out <dir>
Saída: <dir>/atoms/<atom//1000>/<atom>.json + <dir>/index.json
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from fcsae.config import preset
from fcsae.sae.model import TopKSAE, normalize_input

K_TOP, K_RAND, N_BINS, CHUNK = 20, 5, 40, 1024   # bloco pequeno: a GPU é compartilhada com o steering
LAB = {0: "SUP", 1: "REF", 2: "NEI"}
SRC = {0: "vitaminc", 1: "fever"}


def load_unembed(checkpoint, dev):
    """lm_head e norma final do modelo base (bf16, direto dos safetensors do cache do HF)."""
    from huggingface_hub import hf_hub_download
    idx = json.load(open(hf_hub_download(checkpoint, "model.safetensors.index.json")))["weight_map"]
    def get(name):
        with safe_open(hf_hub_download(checkpoint, idx[name]), "pt") as f:
            return f.get_tensor(name)
    dt = torch.float16 if dev == "cuda" else torch.float32          # CPU: matmul em float32
    # nomes variam: Llama/Qwen3 = "lm_head.weight" + "model.norm.weight"; Gemma3 (multimodal, embeddings amarradas)
    # = "language_model.model.embed_tokens.weight" + "language_model.model.norm.weight"
    first = lambda names: next(n for n in names if n in idx)
    U = get(first(["lm_head.weight", "language_model.lm_head.weight", "model.embed_tokens.weight",
                   "language_model.model.embed_tokens.weight", "model.language_model.embed_tokens.weight"]))
    g = get(first(["model.norm.weight", "language_model.model.norm.weight", "model.language_model.norm.weight"])).float()
    if "gemma" in checkpoint.lower():   # RMSNorm do Gemma escala por (1 + peso), não pelo peso
        g = 1.0 + g
    return U.to(dev, dt), g.to(dev)


@torch.no_grad()
def run(args):
    cfg = preset(args.preset); dev = args.device
    if dev == "cpu":
        torch.set_num_threads(os.cpu_count() or 8)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.checkpoint)
    # ler do disco do Linux: mapear (mmap) o arquivo direto do HD do Windows (/mnt/d, DrvFs) pode travar o
    # processo no kernel (visto em 28/09/2026) — por isso a opção --decisions com uma cópia local
    t = load_file(args.decisions or os.path.join(cfg.paths()["activations"], "decisions.safetensors"))
    A, L, S = t["acts"], t["label"].long(), t["source"].long()
    N = len(L)
    exs = [json.loads(l) for l in open(os.path.join(cfg.paths()["activations"], "examples.jsonl"), encoding="utf-8")]
    assert len(exs) == N
    w = load_file(os.path.join(cfg.paths()["sae"], "sae.safetensors"))
    sae = TopKSAE(A.shape[1], cfg.sae_width, cfg.sae_k); sae.load_state_dict({**w, "steps_since_fire": sae.steps_since_fire})
    sae = sae.to(dev).eval(); W = cfg.sae_width
    Ld = L.to(dev); nei = (Ld == 2)

    def enc(i):
        x = A[i:i + CHUNK].to(dev).float()
        x = normalize_input(x)[0] if cfg.sae_input_norm else x
        return sae(x)[1]                                   # z [B, W]

    ck = os.path.join(args.out, "_checkpoint.pt"); os.makedirs(args.out, exist_ok=True)
    state = torch.load(ck) if os.path.exists(ck) else {}
    if state:
        print(f"RESUME: etapas prontas = {sorted(state)}", flush=True)
    # ---- passada 1: momentos, contagens, máximos, top-K e aleatórios (reservatório por chave aleatória)
    if "passes" not in state:
        cnt = torch.zeros(W, device=dev); s1 = torch.zeros(W, device=dev); s2 = torch.zeros(W, device=dev)
        cnt_nei = torch.zeros(W, device=dev); mx = torch.zeros(W, device=dev)
        top_v = torch.full((K_TOP, W), -1.0, device=dev); top_i = torch.zeros((K_TOP, W), dtype=torch.long, device=dev)
        rnd_k = torch.full((K_RAND, W), -1.0, device=dev); rnd_i = torch.zeros((K_RAND, W), dtype=torch.long, device=dev)
        rnd_v = torch.zeros((K_RAND, W), device=dev)
        g = torch.Generator(device=dev).manual_seed(0)
        for i in range(0, N, CHUNK):
            z = enc(i); B = z.shape[0]; act = z > 0; ids = torch.arange(i, i + B, device=dev)
            cnt += act.sum(0); s1 += z.sum(0); s2 += (z * z).sum(0); cnt_nei += (act & nei[i:i + B, None]).sum(0)
            mx = torch.maximum(mx, z.max(0).values)
            v = torch.cat([top_v, z]); ix = torch.cat([top_i, ids[:, None].expand(B, W)])
            top_v, sel = v.topk(K_TOP, dim=0); top_i = ix.gather(0, sel)
            rk = torch.where(act, torch.rand(z.shape, generator=g, device=dev), torch.full_like(z, -1.0))
            v = torch.cat([rnd_k, rk]); ix = torch.cat([rnd_i, ids[:, None].expand(B, W)])
            rnd_k, sel = v.topk(K_RAND, dim=0); rnd_i = ix.gather(0, sel); rnd_v = torch.cat([rnd_v, z]).gather(0, sel)
            del z
        # ---- passada 2: histogramas por rótulo (bins em [0, max] de cada feature) -> AUC aproximada e decis
        hist = torch.zeros((3, W, N_BINS), device=dev)
        scale = (N_BINS / mx.clamp_min(1e-8))
        for i in range(0, N, CHUNK):
            z = enc(i); B = z.shape[0]
            b = (z * scale).long().clamp_(0, N_BINS - 1)
            for lab in range(3):
                m = (Ld[i:i + B] == lab)
                if m.any():
                    zz, bb = z[m], b[m]; a = zz > 0
                    flat = (torch.arange(W, device=dev)[None, :].expand_as(bb) * N_BINS + bb)[a]
                    hist[lab].view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
            del z
        state["passes"] = {k: v.cpu() for k, v in dict(cnt=cnt, s1=s1, s2=s2, cnt_nei=cnt_nei, mx=mx, top_v=top_v, top_i=top_i, rnd_k=rnd_k, rnd_i=rnd_i, rnd_v=rnd_v, hist=hist).items()}
        torch.save(state, ck + ".tmp"); os.replace(ck + ".tmp", ck); print("  checkpoint: passadas", flush=True)
    P = state["passes"]; cnt, s1, s2, cnt_nei, mx, top_v, top_i, rnd_k, rnd_i, rnd_v, hist = [P[k] for k in ("cnt", "s1", "s2", "cnt_nei", "mx", "top_v", "top_i", "rnd_k", "rnd_i", "rnd_v", "hist")]

    if "extras" not in state:
        # ---- vizinhos no decoder (cosseno) e logit lens
        Wd = w["W_dec"].to(dev).float(); Wn = Wd / Wd.norm(dim=1, keepdim=True)
        nb_v = torch.zeros((W, 10), device=dev); nb_i = torch.zeros((W, 10), dtype=torch.long, device=dev)
        for i in range(0, W, 2048):
            c = Wn[i:i + 2048] @ Wn.T; c[torch.arange(c.shape[0]), torch.arange(i, i + c.shape[0])] = -2
            nb_v[i:i + 2048], nb_i[i:i + 2048] = c.topk(10, dim=1)
        U, g_norm = load_unembed(cfg.checkpoint, dev)
        up_i = torch.zeros((W, 10), dtype=torch.long, device=dev); up_v = torch.zeros((W, 10), device=dev)
        dn_i = torch.zeros((W, 10), dtype=torch.long, device=dev); dn_v = torch.zeros((W, 10), device=dev)
        for i in range(0, W, 512):
            lg = ((Wn[i:i + 512] * g_norm).to(U.dtype) @ U.T).float()
            up_v[i:i + 512], up_i[i:i + 512] = lg.topk(10, dim=1); dn_v[i:i + 512], dn_i[i:i + 512] = (-lg).topk(10, dim=1)
        del U

        state["extras"] = {k: v.cpu() for k, v in dict(nb_v=nb_v, nb_i=nb_i, up_v=up_v, up_i=up_i, dn_v=dn_v, dn_i=dn_i).items()}
        torch.save(state, ck + ".tmp"); os.replace(ck + ".tmp", ck); print("  checkpoint: vizinhos+logits", flush=True)
    E = state["extras"]; nb_v, nb_i, up_v, up_i, dn_v, dn_i = [E[k] for k in ("nb_v", "nb_i", "up_v", "up_i", "dn_v", "dn_i")]

    # ---- para CPU
    cnt, s1, s2, cnt_nei, mx = [x.cpu().numpy() for x in (cnt, s1, s2, cnt_nei, mx)]
    top_v, top_i, rnd_k, rnd_i, rnd_v = [x.cpu().numpy() for x in (top_v, top_i, rnd_k, rnd_i, rnd_v)]
    hist = hist.cpu().numpy(); nb_v, nb_i = nb_v.cpu().numpy(), nb_i.cpu().numpy()
    up_v, up_i, dn_v, dn_i = up_v.cpu().numpy(), up_i.cpu().numpy(), dn_v.cpu().numpy(), dn_i.cpu().numpy()
    Lc, Sc = L.numpy(), S.numpy()
    n_nei = int((Lc == 2).sum()); n_dec = int((Lc != 2).sum())
    tok_cache = {}

    def seq(j, act_val):
        j = int(j)
        if j not in tok_cache:
            ids = tok(exs[j]["claim"], add_special_tokens=False).input_ids
            tok_cache[j] = tok.batch_decode([[x] for x in ids])
        toks = tok_cache[j] + [" ▶ verdict"]
        acts = [0.0] * (len(toks) - 1) + [round(float(act_val), 4)]
        ev = " | ".join(exs[j]["evidence"])[:300]
        return {"doc_id": j, "idx": len(toks) - 1, "act": round(float(act_val), 4), "acts": acts, "tokens": toks,
                "density": 0, "label": LAB[int(Lc[j])], "source": SRC.get(int(Sc[j]), "?"), "evidence": ev}

    def auc_from_hist(hn, hd, zn, zd):
        # Mann-Whitney com bins (empates dentro do bin contam 1/2); zeros = bin "-1"
        pn = np.concatenate([[zn], hn]); pd_ = np.concatenate([[zd], hd])
        if pn.sum() == 0 or pd_.sum() == 0: return None
        cd = np.cumsum(pd_) - pd_
        return float(((pn * (cd + 0.5 * pd_)).sum()) / (pn.sum() * pd_.sum()))

    out = args.out; os.makedirs(os.path.join(out, "atoms"), exist_ok=True)
    index = []
    for a in range(W):
        if cnt[a] == 0:
            info = {"density": 0.0, "mean_act": 0.0, "mean_act_squared": 0.0, "hist": {}, "top": [], "random": [], "dead": True}
        else:
            hn, hd = hist[2, a], hist[0, a] + hist[1, a]
            zn, zd = n_nei - hn.sum(), n_dec - hd.sum()
            au = auc_from_hist(hn, hd, zn, zd)
            order = [k for k in range(K_TOP) if top_v[k, a] > 0]
            top = [seq(top_i[k, a], top_v[k, a]) for k in order]
            rnd = [seq(rnd_i[k, a], rnd_v[k, a]) for k in range(K_RAND) if rnd_k[k, a] >= 0]
            labs_top = [t_["label"] for t_ in top]
            edges = (np.arange(N_BINS + 1) / N_BINS * mx[a]).round(5).tolist()
            info = {
                "density": float(cnt[a] / N), "mean_act": float(s1[a] / N), "mean_act_squared": float(s2[a] / N),
                "hist": {str(k): int(v) for k, v in enumerate(hn + hd) if v > 0}, "bin_width": float(mx[a] / N_BINS),
                "hist_by_label": {"edges": edges, "NEI": hn.astype(int).tolist(), "SUP/REF": hd.astype(int).tolist()},
                "top": top, "random": rnd,
                "fc": {"frac_active": float(cnt[a] / N), "nei_rate_when_active": float(cnt_nei[a] / cnt[a]),
                       "nei_base": n_nei / N, "auc_nei_vs_dec_approx": au,
                       "nei_top10": float(np.mean([l == "NEI" for l in labs_top[:10]])) if labs_top else None,
                       "nei_top20": float(np.mean([l == "NEI" for l in labs_top])) if labs_top else None},
                "neighbors": [{"atom": int(nb_i[a, k]), "cos": round(float(nb_v[a, k]), 4)} for k in range(10)],
                "logits": {"up": [[tok.decode([int(up_i[a, k])]), round(float(up_v[a, k]), 3)] for k in range(10)],
                           "down": [[tok.decode([int(dn_i[a, k])]), round(float(-dn_v[a, k]), 3)] for k in range(10)]},
            }
            index.append({"atom": a, "frac_active": round(info["fc"]["frac_active"], 5),
                          "nei_rate_when_active": round(info["fc"]["nei_rate_when_active"], 4),
                          "auc": None if au is None else round(au, 4), "nei_top20": info["fc"]["nei_top20"]})
        d = os.path.join(out, "atoms", str(a // 1000)); os.makedirs(d, exist_ok=True)
        json.dump(info, open(os.path.join(d, f"{a}.json"), "w"), ensure_ascii=False, separators=(",", ":"))
        if a % 4096 == 0:
            print(f"  feature {a}/{W}", flush=True)
    meta = {"model": cfg.checkpoint, "preset": cfg.name, "layer": cfg.layer, "num_features": W, "n_decisions": N,
            "nei_base": n_nei / N, "note": "stats at the decision position of verdict prompts (VitaminC+FEVER)"}
    json.dump({"meta": meta, "features": index}, open(os.path.join(out, "index.json"), "w"), separators=(",", ":"))
    print(f"FIM: {W} features -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v3_llama")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decisions", default=None, help="caminho alternativo (cópia local) do decisions.safetensors")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu: não disputa a GPU com outros jobs")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
