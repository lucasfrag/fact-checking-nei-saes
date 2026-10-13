#!/usr/bin/env bash
# Fila (Lucas, 01/10/2026): steering com a DIREÇÃO de abstenção (diff-of-means, sem SAE), grade completa,
# primeiro no DeepSeek e depois no Llama — comparação "feature do SAE × direção média" nos dois modelos.
set -u
cd "$(dirname "$0")/.."
B="--betas=0,-0.25,0.25,-0.5,0.5,-0.75,0.75,-1,1 --conds none,single,iterative --direction abstention"
bash scripts/run_steer_sae.sh steer_dirabst_v3_deepseek --preset v3_deepseek $B
bash scripts/run_steer_sae.sh steer_dirabst_v3_llama --preset v3_llama $B
echo "[$(date +%F\ %T)] fila dirabst concluída" >> logs/steer_dirabst_queue.log
