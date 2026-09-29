#!/usr/bin/env bash
# Gera os dados do SAE viewer com relançamento automático (checkpoint por etapa em <out>/_checkpoint.pt).
#   bash scripts/run_viewer_data.sh <preset> <out>
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"}
export HF_HUB_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
LOG="logs/viewer_data_$1.log"
for try in $(seq 1 30); do
  echo "[$(date +%F\ %T)] tentativa $try" >> "$LOG"
  /home/systemp2w/miniconda3/envs/fcsae/bin/python -m fcsae.analysis.viewer_data --preset "$1" --out "$2" ${3:+--device $3} ${4:+--decisions $4} >> "$LOG" 2>&1
  grep -q "^FIM:" "$LOG" && { echo "[$(date +%F\ %T)] CONCLUÍDO" >> "$LOG"; exit 0; }
  sleep 20
done
