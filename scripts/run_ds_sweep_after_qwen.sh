#!/usr/bin/env bash
# Lucas (06/10/2026): depois do treino/análise do Qwen3, refazer a varredura de camadas do DeepSeek com
# 1.200 exemplos por classe (a do v1 tinha só 40) — a camada 15 explica a abstenção espalhada?
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
until grep -q "fim v3_qwen3" logs/plan_models.log; do sleep 120; done
echo "[$(date +%F\ %T)] início varredura de camadas DeepSeek (1.200/classe)" >> logs/plan_models.log
for t in 1 2 3 4 5; do
  /home/systemp2w/miniconda3/envs/fcsae/bin/python -m fcsae.analysis.layer_sweep --preset v3_deepseek --n-per-class 1200 >> logs/layer_sweep_v3_deepseek.log 2>&1 && break
  sleep 30
done
echo "[$(date +%F\ %T)] fim varredura DeepSeek" >> logs/plan_models.log
