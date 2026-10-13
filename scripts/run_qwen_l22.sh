#!/usr/bin/env bash
# Lucas (06/10/2026): (1) varredura de camadas do DeepSeek com 1.200/classe (recomeçar o DeepSeek do zero);
# (2) Qwen3 na camada 22 (harvest -> treino -> análise) + espalhamento.
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
PY=/home/systemp2w/miniconda3/envs/fcsae/bin/python
echo "[$(date +%F\ %T)] início varredura de camadas DeepSeek (1.200/classe)" >> logs/plan_models.log
for t in 1 2 3 4 5; do $PY -m fcsae.analysis.layer_sweep --preset v3_deepseek --n-per-class 1200 >> logs/layer_sweep_v3_deepseek.log 2>&1 && break; sleep 30; done
echo "[$(date +%F\ %T)] fim varredura DeepSeek; início v3_qwen3_l22" >> logs/plan_models.log
MAX_TRIES=1000 bash scripts/run_v3.sh v3_qwen3_l22 >> logs/v3_qwen3_l22.log 2>&1
for t in 1 2 3; do $PY -m fcsae.analysis.spread --preset v3_qwen3_l22 >> logs/spread_v3_qwen3_l22.log 2>&1 && break; sleep 30; done
echo "[$(date +%F\ %T)] fim v3_qwen3_l22" >> logs/plan_models.log
