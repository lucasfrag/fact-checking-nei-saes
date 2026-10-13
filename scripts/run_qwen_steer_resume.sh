#!/usr/bin/env bash
# Lucas (10/10/2026, após reiniciar o PC): retomar o steering do Qwen3 12482 como combinado em 09/10 —
# terminar o Single (dose +1 retoma do checkpoint) e rodar o Iterative SÓ com −0,75…+0,75 (±1 pendente, LEIA-ME #9).
set -u
cd "$(dirname "$0")/.."
echo "[$(date +%F\ %T)] retomada steering Qwen3 12482 (single até +1; iterative moderado)" >> logs/plan_models.log
bash scripts/run_steer_sae.sh steer_sae_v3_qwen3_l22_feat12482 --preset v3_qwen3_l22 --feat 12482 --conds single \
  --betas=0,-0.25,0.25,-0.5,0.5,-0.75,0.75,-1,1
bash scripts/run_steer_sae.sh steer_sae_v3_qwen3_l22_feat12482 --preset v3_qwen3_l22 --feat 12482 --conds iterative \
  --betas=0,-0.25,0.25,-0.5,0.5,-0.75,0.75
echo "[$(date +%F\ %T)] fim steering Qwen3 12482 (iterative moderado)" >> logs/plan_models.log
