#!/usr/bin/env bash
# Replay do veredito (α = 0, sem SAE) dos modelos do pipeline HF, em fila, para reler o rótulo com
# `first_label` (defeito do `normalize`, nb01 §7). Cada modelo usa o relançamento automático do
# run_steer_sae.sh (checkpoint a cada 25 claims).  Saída: results/replay/<modelo>-hf/
#   bash scripts/run_replay_all.sh [tag ...]     (padrão: deepseek-r1:8b qwen3:8b gemma3:4b gemma4:e4b)
set -u
cd "$(dirname "$0")/.."
TAGS=${*:-"deepseek-r1:8b qwen3:8b gemma3:4b gemma4:e4b"}
for t in $TAGS; do
  bash scripts/run_steer_sae.sh "replay_${t//:/-}" --replay "$t"
done
