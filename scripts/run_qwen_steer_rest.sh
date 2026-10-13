#!/usr/bin/env bash
# Lucas (09/10/2026): Qwen3 12482 — terminar single (±1 incluídas) e rodar iterative SÓ com as doses moderadas
# (−0,75…+0,75). Iterative ±1 fica PENDENTE para o final (LEIA-ME).
set -u
cd "$(dirname "$0")/.."
L=logs/steer_sae_v3_qwen3_l22_feat12482.log
until grep -aq "CÉLULA single feat12482 β=+1.00" $L; do sleep 60; done
# mata o wrapper atual (ele seguiria para iterative com ±1) e relança só iterative moderado
for p in /proc/[0-9]*; do pid=${p#/proc/}; c=$(timeout 1 tr '\0' ' ' < $p/cmdline 2>/dev/null)
  case "$c" in *run_steer_sae.sh*feat12482*|*factcheck_sae*--feat\ 12482*) kill $pid 2>/dev/null;; esac; done
sleep 10
echo "[$(date +%F\ %T)] Qwen3 12482: single completo; iterative só −0,75…+0,75 (±1 pendente)" >> logs/plan_models.log
bash scripts/run_steer_sae.sh steer_sae_v3_qwen3_l22_feat12482 --preset v3_qwen3_l22 --feat 12482 --conds iterative \
  --betas=0,-0.25,0.25,-0.5,0.5,-0.75,0.75
echo "[$(date +%F\ %T)] fim steering Qwen3 12482 (iterative moderado)" >> logs/plan_models.log
