#!/usr/bin/env bash
# Valida features candidatas (monta o cache de decisões na 1ª vez; depois é rápido).
#   bash scripts/run_inspect.sh <preset> <feat> [feat ...]
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
P=$1; shift
for f in "$@"; do
  for try in 1 2 3 4 5; do
    echo "[$(date +%F\ %T)] feat $f tentativa $try" >> "logs/inspect_$P.log"
    /home/systemp2w/miniconda3/envs/fcsae/bin/python -m fcsae.analysis.inspect_feature --preset "$P" --feat "$f" >> "logs/inspect_$P.log" 2>&1 && break
    sleep 20
  done
done
echo "[$(date +%F\ %T)] FIM" >> "logs/inspect_$P.log"
