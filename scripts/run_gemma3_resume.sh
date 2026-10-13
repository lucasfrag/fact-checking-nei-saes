#!/usr/bin/env bash
# Lucas: retomar o treino do Gemma3 (checkpoint) → análise → espalhamento.
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
echo "[$(date +%F\ %T)] v3_gemma3: treino RETOMADO do checkpoint" >> logs/plan_models.log
MAX_TRIES=1000 bash scripts/run_v3.sh v3_gemma3 >> logs/v3_gemma3.log 2>&1
for t in 1 2 3; do /home/systemp2w/miniconda3/envs/fcsae/bin/python -m fcsae.analysis.spread --preset v3_gemma3 >> logs/spread_v3_gemma3.log 2>&1 && break; sleep 30; done
echo "[$(date +%F\ %T)] fim ciclo v3_gemma3 (coleta → treino → análise → espalhamento)" >> logs/plan_models.log
