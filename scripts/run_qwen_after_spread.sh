#!/usr/bin/env bash
# Lucas (08/10/2026): ciclo do Qwen3 L22 após o spread — cache de decisões + exemplos da feature candidata
# (12482: 1ª no cosseno e no |d|), dados do viewer e mapa de seleção. Publicação no HF é feita à parte.
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
PY=/home/systemp2w/miniconda3/envs/fcsae/bin/python
P=v3_qwen3_l22; OUT=$HOME/viewer_build/data/qwen3-8b/layer-22
until grep -q "fim v3_qwen3_l22" logs/plan_models.log; do sleep 60; done
echo "[$(date +%F\ %T)] qwen após spread: inspect 12482" >> logs/plan_models.log
bash scripts/run_inspect.sh $P 12482
echo "[$(date +%F\ %T)] qwen: viewer_data" >> logs/plan_models.log
mkdir -p "$OUT"
bash scripts/run_viewer_data.sh $P "$OUT"
echo "[$(date +%F\ %T)] qwen: selection_map" >> logs/plan_models.log
for t in 1 2 3; do $PY -m fcsae.analysis.selection_map --preset $P --out "$OUT" >> logs/selection_map_$P.log 2>&1 && break; sleep 20; done
echo "[$(date +%F\ %T)] fim qwen após spread" >> logs/plan_models.log
