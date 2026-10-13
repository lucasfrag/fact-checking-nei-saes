#!/usr/bin/env bash
# Lucas (05/10/2026): terminar só a última dose da 18658 (+1) e partir para o Qwen3. As doses restantes da 25349
# (±0,25, ±0,75, ±1) ficam PENDENTES para o final (ver LEIA-ME).
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
until grep -q "CÉLULA none feat18658 β=+1.00" logs/steer_sae_v3_deepseek_feat18658.log; do sleep 60; done
sleep 30   # o wrapper encerra sozinho após o FIM
echo "[$(date +%F\ %T)] 18658 concluída; DeepSeek pausado (25349 pendente); início v3_qwen3" >> logs/plan_models.log
MAX_TRIES=1000 bash scripts/run_v3.sh v3_qwen3 >> logs/v3_qwen3.log 2>&1
for t in 1 2 3; do /home/systemp2w/miniconda3/envs/fcsae/bin/python -m fcsae.analysis.spread --preset v3_qwen3 >> logs/spread_v3_qwen3.log 2>&1 && break; sleep 30; done
echo "[$(date +%F\ %T)] fim v3_qwen3" >> logs/plan_models.log
