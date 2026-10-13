#!/usr/bin/env bash
# Lucas (02/10/2026): terminado o steering do DeepSeek (feature 7531), seguir para o próximo modelo: pipeline v3 do
# Qwen3 (harvest -> treino -> análise) + análise de espalhamento. (A publicação no HF eu faço por fora, sem GPU.)
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
until grep -qE "^\[.*\] (CONCLUÍDO|desisti)" logs/steer_sae_v3_deepseek_feat7531.log 2>/dev/null; do sleep 120; done
echo "[$(date +%F\ %T)] steering DeepSeek terminou; início v3_qwen3" >> logs/plan_models.log
MAX_TRIES=1000 bash scripts/run_v3.sh v3_qwen3 >> logs/v3_qwen3.log 2>&1
for t in 1 2 3; do /home/systemp2w/miniconda3/envs/fcsae/bin/python -m fcsae.analysis.spread --preset v3_qwen3 >> logs/spread_v3_qwen3.log 2>&1 && break; sleep 30; done
echo "[$(date +%F\ %T)] fim v3_qwen3" >> logs/plan_models.log
