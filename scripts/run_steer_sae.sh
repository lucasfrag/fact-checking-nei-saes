#!/usr/bin/env bash
# Roda fcsae.steering.factcheck_sae com relançamento automático até o "FIM" (a GPU do WSL2
# dá segfault intermitente; o script retoma de onde parou, com checkpoint a cada 25 claims).
#   bash scripts/run_steer_sae.sh <nome-do-log> <args do factcheck_sae...>
set -u
NAME=$1; shift
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"}
export HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"}
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
PY=${PY:-/home/systemp2w/miniconda3/envs/fcsae/bin/python}
cd "$(dirname "$0")/.."
LOG="logs/${NAME}.log"
for try in $(seq 1 200); do
  echo "[$(date +%F\ %T)] tentativa $try: $*" >> "$LOG"
  $PY -m fcsae.steering.factcheck_sae "$@" >> "$LOG" 2>&1
  tail -1 "$LOG" | grep -q "^FIM" && { echo "[$(date +%F\ %T)] CONCLUÍDO" >> "$LOG"; exit 0; }
  sleep 20
done
echo "[$(date +%F\ %T)] desisti após 200 tentativas" >> "$LOG"; exit 1
