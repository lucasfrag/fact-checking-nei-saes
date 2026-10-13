#!/usr/bin/env bash
# Lucas (04/10/2026): triagem rápida das candidatas restantes do DeepSeek (18658, 25349) só em α/R = ±0,5,
# sem evidência; depois completar as doses restantes de todas (19789, 18658, 25349); depois Qwen3.
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
FULL="--betas=0,-0.25,0.25,-0.5,0.5,-0.75,0.75,-1,1 --conds none"
for f in 18658 25349; do
  bash scripts/run_steer_sae.sh "steer_sae_v3_deepseek_feat$f" --preset v3_deepseek --feat $f --betas=-0.5,0.5 --conds none
done
echo "[$(date +%F\ %T)] triagem ±0,5 concluída; completando as doses" >> logs/plan_models.log
for f in 19789 18658 25349; do            # células prontas são puladas (o script confere os arquivos)
  bash scripts/run_steer_sae.sh "steer_sae_v3_deepseek_feat$f" --preset v3_deepseek --feat $f $FULL
done
echo "[$(date +%F\ %T)] candidatas DeepSeek concluídas; início v3_qwen3" >> logs/plan_models.log
MAX_TRIES=1000 bash scripts/run_v3.sh v3_qwen3 >> logs/v3_qwen3.log 2>&1
for t in 1 2 3; do /home/systemp2w/miniconda3/envs/fcsae/bin/python -m fcsae.analysis.spread --preset v3_qwen3 >> logs/spread_v3_qwen3.log 2>&1 && break; sleep 30; done
echo "[$(date +%F\ %T)] fim v3_qwen3" >> logs/plan_models.log
