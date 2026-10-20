#!/usr/bin/env bash
# Lucas (14/10/2026): entre o Single e o Iterative do Gemma3, completar a pendência do Qwen3 (LEIA-ME #9):
# 12482, condição Iterative, doses α/R = ±1 na régua bruta (--norm raw, como o resto da grade do Qwen3).
set -u
cd "$(dirname "$0")/.."
G=logs/steer_sae_v3_gemma3_feat12903.log
until grep -aq "CÉLULA single feat12903 β=+6.00" $G; do sleep 60; done
for p in /proc/[0-9]*; do pid=${p#/proc/}; [ "$pid" = "$$" ] && continue; c=$(timeout 1 tr '\0' ' ' < $p/cmdline 2>/dev/null)
  case "$c" in "bash scripts/run_steer_sae.sh steer_sae_v3_gemma3"*|*"fcsae.steering.factcheck_sae --preset v3_gemma3"*) kill $pid 2>/dev/null;; esac; done
sleep 15
echo "[$(date +%F\ %T)] Gemma3 PAUSADO após o Single; início Qwen3 12482 iterative ±1 (régua bruta)" >> logs/plan_models.log
bash scripts/run_steer_sae.sh steer_sae_v3_qwen3_l22_feat12482 --preset v3_qwen3_l22 --feat 12482 --norm raw --conds iterative --betas=0,-1,1
echo "[$(date +%F\ %T)] fim Qwen3 12482 iterative ±1; retomando Gemma3 (iterative)" >> logs/plan_models.log
bash scripts/run_steer_sae.sh steer_sae_v3_gemma3_feat12903 --preset v3_gemma3 --feat 12903 --norm centered --conds iterative \
  --betas=0,-1.5,1.5,-3,3,-4.5,4.5,-6,6
echo "[$(date +%F\ %T)] fim steering Gemma3 12903 (grade completa)" >> logs/plan_models.log
