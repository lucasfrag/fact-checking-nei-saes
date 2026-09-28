"""Treino do TopK SAE sobre as ativações cacheadas, com AUTO-RESUME.

A GPU do WSL2 cai intermitentemente sob carga sustentada ("device not ready"). Para
completar mesmo assim, o treino:
  - usa orçamento de PASSOS (não de épocas), retomável;
  - salva checkpoint (pesos + otimizador + passo) a cada `ckpt_every` passos;
  - ao reiniciar, retoma do último checkpoint e continua até o orçamento;
  - só escreve `metrics.json` (marcador de conclusão) ao terminar de fato.
Um wrapper externo pode relançar em loop até `metrics.json` existir.

Otimização opcional do Gao et al. (2024), Apêndice A (flags `sae_*` no Config; desligadas
no v1, ligadas no v3): normalização da entrada (§2.1), perdas normalizadas (§A.6), Adam ε
pequeno (§A.3.1), projeção do gradiente do decoder (§A.3), b_pre = mediana geométrica
(§A.1) e EMA dos pesos (§A.5). Ordem do passo segue o código oficial
(openai/sparse_autoencoder): backward -> normaliza decoder -> projeta grad -> step.

    python -m fcsae.sae.train --preset v1_domain
"""
from __future__ import annotations

import os
import sys
import json
import glob
import math
import argparse

import torch
from safetensors.torch import save_file, load_file

from fcsae.config import preset, Config
from fcsae.common.io import load_manifest
from fcsae.sae.model import TopKSAE, normalize_input, geometric_median

CKPT_EVERY = 500
N_INIT_SAMPLE = 32_768      # amostra p/ b_pre e constante do MSE (mesmo tamanho do código do Gao)


def fvu(x, x_hat):
    num = (x - x_hat).pow(2).sum()
    den = (x - x.mean(0, keepdim=True)).pow(2).sum().clamp_min(1e-8)
    return (num / den).item()


def save_sae(sae, out_dir, metrics=None):
    os.makedirs(out_dir, exist_ok=True)
    save_file({"W_enc": sae.W_enc.data.contiguous().cpu(), "b_enc": sae.b_enc.data.contiguous().cpu(),
               "W_dec": sae.W_dec.data.contiguous().cpu(), "b_dec": sae.b_dec.data.contiguous().cpu()},
              os.path.join(out_dir, "sae.safetensors"))
    if metrics is not None:
        json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)


def run(cfg: Config):
    dev = cfg.device if torch.cuda.is_available() else "cpu"
    out = cfg.paths()["sae"]; os.makedirs(out, exist_ok=True)
    state_path = os.path.join(out, "train_state.pt")
    metrics_path = os.path.join(out, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"metrics.json já existe em {out} — treino concluído; nada a fazer.", flush=True)
        sys.stdout.flush(); os._exit(0)

    shard_dir = cfg.paths()["activations"]
    man = load_manifest(shard_dir)
    d = man["d_model"]
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.safetensors")))
    assert shards, f"nenhum shard em {shard_dir} — rode o harvest primeiro"
    eval_shard = shards[-1] if len(shards) > 1 else None
    train_shards = shards[:-1] if eval_shard else shards
    tokens_train = man["n_tokens"] * len(train_shards) / len(shards)
    total_steps = int(math.ceil(tokens_train / cfg.sae_batch) * cfg.sae_epochs)

    def prep(x):                    # espaço de entrada do SAE
        return normalize_input(x)[0] if cfg.sae_input_norm else x

    sae = TopKSAE(d, cfg.sae_width, cfg.sae_k, cfg.sae_k_aux, cfg.sae_dead_steps).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.sae_lr, eps=cfg.sae_adam_eps)
    ema = [torch.zeros_like(p) for p in sae.parameters()] if cfg.sae_ema > 0 else None
    ema_steps = 0
    mse_scale = 1.0

    gstep = 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev)
        sae.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); gstep = st["step"]
        mse_scale = st.get("mse_scale", 1.0)
        if ema is not None:
            ema = st["ema"]; ema_steps = st["ema_steps"]
        print(f"RESUME do passo {gstep}/{total_steps}", flush=True)
    elif not (cfg.sae_init_geomedian or cfg.sae_normalized_loss):
        sae.set_b_dec(load_file(shards[0])["acts"][:50_000].float().mean(0).to(dev))
    else:
        a0 = load_file(shards[0])["acts"]
        g = torch.Generator().manual_seed(0)
        sample = prep(a0[torch.randperm(a0.shape[0], generator=g)[:N_INIT_SAMPLE]].float()).to(dev)
        del a0
        sae.set_b_dec(geometric_median(sample) if cfg.sae_init_geomedian else sample.mean(0))
        if cfg.sae_normalized_loss:
            # §A.6: constante calculada UMA vez no início (código oficial: `mse_scale`)
            mse_scale = (1.0 / (sample - sample.mean(0, keepdim=True)).pow(2).mean()).item()
        del sample
    print(f"treino SAE | d={d} width={cfg.sae_width} k={cfg.sae_k} | shards={len(shards)} "
          f"| passos alvo={total_steps} | batch={cfg.sae_batch} | lr={cfg.sae_lr} | dev={dev} "
          f"| input_norm={cfg.sae_input_norm} norm_loss={cfg.sae_normalized_loss} "
          f"grad_proj={cfg.sae_grad_proj} ema={cfg.sae_ema} | mse_scale={mse_scale:.4g}", flush=True)

    def checkpoint():
        save_sae(sae, out)
        tmp = state_path + ".tmp"      # escrita atômica (crash no meio não corrompe o resume)
        torch.save({"model": sae.state_dict(), "opt": opt.state_dict(), "step": gstep,
                    "mse_scale": mse_scale, "ema": ema, "ema_steps": ema_steps}, tmp)
        os.replace(tmp, state_path)

    while gstep < total_steps:
        order = torch.randperm(len(train_shards)).tolist()
        for si in order:
            acts = load_file(train_shards[si])["acts"].to(dev)     # fp16; float32 por lote
            perm = torch.randperm(acts.shape[0], device=dev)
            for i in range(0, acts.shape[0], cfg.sae_batch):
                x = prep(acts[perm[i:i + cfg.sae_batch]].float())
                if x.shape[0] < 8:
                    continue
                x_hat, z, pre = sae(x)
                if cfg.sae_normalized_loss:
                    loss_rec = mse_scale * (x - x_hat).pow(2).mean()
                else:
                    loss_rec = (x - x_hat).pow(2).sum(-1).mean()
                loss = loss_rec + cfg.sae_aux_coeff * sae.aux_loss(
                    x, x_hat, pre, normalized=cfg.sae_normalized_loss)
                opt.zero_grad(); loss.backward()
                if cfg.sae_grad_proj:
                    sae._unit_norm_decoder(); sae.project_decoder_grad()
                opt.step()
                sae._unit_norm_decoder(); sae.update_dead(z)
                if ema is not None:
                    with torch.no_grad():
                        torch._foreach_mul_(ema, cfg.sae_ema)
                        torch._foreach_add_(ema, list(sae.parameters()), alpha=1 - cfg.sae_ema)
                    ema_steps += 1
                gstep += 1
                if gstep % 100 == 0:
                    dead = int((sae.steps_since_fire >= cfg.sae_dead_steps).sum())
                    print(f"  passo {gstep}/{total_steps} | FVU(train)={fvu(x, x_hat):.3f} "
                          f"| rec={loss_rec.item():.4g} | mortas={dead}/{cfg.sae_width}", flush=True)
                if gstep % CKPT_EVERY == 0:
                    checkpoint()
                if gstep >= total_steps:
                    break
            del acts
            torch.cuda.empty_cache() if dev == "cuda" else None
            if gstep >= total_steps:
                break

    # §A.5: o SAE final usa os pesos da EMA (com correção de viés, como no Adam)
    if ema is not None and ema_steps > 0:
        with torch.no_grad():
            corr = 1 - cfg.sae_ema ** ema_steps
            for p, e in zip(sae.parameters(), ema):
                p.copy_(e / corr)

    # avaliação held-out final — BATCHADA (o shard inteiro de uma vez gera
    # pré-ativações [N, width] gigantes que estouram a VRAM e derrubam a GPU).
    # Com input_norm, o FVU é medido no espaço normalizado (convenção do Gao, §2.1).
    metrics = {"steps": gstep}
    if eval_shard:
        ev = load_file(eval_shard)
        xe = prep(ev["acts"].float())            # em CPU
        dec = ev["is_decision"].bool()
        mean = xe.mean(0)                        # para o denominador do FVU
        num = den = numd = dend = l0sum = ntok = 0.0
        with torch.no_grad():
            for i in range(0, xe.shape[0], 8192):
                b = xe[i:i + 8192].to(dev)
                xh, z, _ = sae(b)
                num += (b - xh).pow(2).sum().item()
                den += (b - mean.to(dev)).pow(2).sum().item()
                l0sum += (z > 0).float().sum().item(); ntok += b.shape[0]
                dm = dec[i:i + 8192].to(dev)
                if dm.any():
                    numd += (b[dm] - xh[dm]).pow(2).sum().item()
                    dend += (b[dm] - mean.to(dev)).pow(2).sum().item()
                del b, xh, z
                torch.cuda.empty_cache() if dev == "cuda" else None
        metrics["fvu_heldout"] = num / max(den, 1e-8)
        metrics["fvu_decision"] = numd / max(dend, 1e-8)
        metrics["l0"] = l0sum / max(ntok, 1)
        metrics["fvu_space"] = "normalized" if cfg.sae_input_norm else "raw"
        print(f"HELD-OUT: FVU={metrics['fvu_heldout']:.3f} | "
              f"FVU(decisão)={metrics['fvu_decision']:.3f} | L0={metrics['l0']:.1f}", flush=True)
    metrics["dead_features"] = int((sae.steps_since_fire >= cfg.sae_dead_steps).sum())
    metrics["config"] = cfg.to_dict()
    save_sae(sae, out, metrics)                 # metrics.json = marcador de conclusão
    if os.path.exists(state_path):
        os.remove(state_path)
    print(f"CONCLUÍDO. SAE salvo em {out} | mortas={metrics['dead_features']}/{cfg.sae_width}", flush=True)
    sys.stdout.flush(); os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v1_domain")
    args = ap.parse_args()
    run(preset(args.preset))


if __name__ == "__main__":
    main()
