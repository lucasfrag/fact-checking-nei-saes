#!/usr/bin/env bash
# Lucas (12/10/2026): Gemma3 após o ciclo — inspeção das candidatas (12903 = 1ª no cosseno e no d positivo;
# 2170 = 1ª no |d|, espelho; 13819 = 2ª no cosseno) + dados do viewer + mapa de seleção. Publicação à parte.
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
PY=/home/systemp2w/miniconda3/envs/fcsae/bin/python
P=v3_gemma3; OUT=$HOME/viewer_build/data/gemma-3-4b-it/layer-19
echo "[$(date +%F\ %T)] gemma3 após ciclo: inspect 12903 2170 13819" >> logs/plan_models.log
bash scripts/run_inspect.sh $P 12903 2170 13819
echo "[$(date +%F\ %T)] gemma3: viewer_data" >> logs/plan_models.log
mkdir -p "$OUT"; bash scripts/run_viewer_data.sh $P "$OUT"
echo "[$(date +%F\ %T)] gemma3: selection_map" >> logs/plan_models.log
for t in 1 2 3; do $PY -m fcsae.analysis.selection_map --preset $P --out "$OUT" >> logs/selection_map_$P.log 2>&1 && break; sleep 20; done
echo "[$(date +%F\ %T)] fim gemma3 após ciclo" >> logs/plan_models.log
