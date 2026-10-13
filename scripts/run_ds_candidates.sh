#!/usr/bin/env bash
# Lucas (02/10/2026): terminar a condição SEM EVIDÊNCIA da feature 7531 e repetir o mesmo experimento (sem evidência,
# 9 doses) com as outras candidatas do DeepSeek (top-5 por |Cohen's d|: 32642, 19789, 18658, 25349); depois, Qwen3.
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
L7=logs/steer_sae_v3_deepseek_feat7531.log
until grep -q "CÉLULA none feat7531 β=+1.00" $L7; do sleep 60; done
# a grade da 7531 seguiria para Single/Iterative: interrompe aqui (wrapper primeiro, para não relançar)
for p in $(ls /proc | grep -E "^[0-9]+$"); do
  a=$(timeout 1 tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$a" in "bash scripts/run_steer_sae.sh steer_sae_v3_deepseek_feat7531"*) kill $p;; esac
done
sleep 2
for p in $(ls /proc | grep -E "^[0-9]+$"); do
  a=$(timeout 1 tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$a" in *"factcheck_sae --preset v3_deepseek --feat 7531"*) kill $p;; esac
done
echo "[$(date +%F\ %T)] 7531: condição sem evidência completa; Single/Iterative NÃO rodados (decisão do Lucas)" >> $L7
B="--betas=0,-0.25,0.25,-0.5,0.5,-0.75,0.75,-1,1 --conds none"
for f in 32642 19789 18658 25349; do
  bash scripts/run_steer_sae.sh "steer_sae_v3_deepseek_feat$f" --preset v3_deepseek --feat $f $B
done
echo "[$(date +%F\ %T)] candidatas DeepSeek concluídas; início v3_qwen3" >> logs/plan_models.log
MAX_TRIES=1000 bash scripts/run_v3.sh v3_qwen3 >> logs/v3_qwen3.log 2>&1
for t in 1 2 3; do /home/systemp2w/miniconda3/envs/fcsae/bin/python -m fcsae.analysis.spread --preset v3_qwen3 >> logs/spread_v3_qwen3.log 2>&1 && break; sleep 30; done
echo "[$(date +%F\ %T)] fim v3_qwen3" >> logs/plan_models.log
