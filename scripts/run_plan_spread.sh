#!/usr/bin/env bash
# Plano (Lucas, 01/10/2026): a abstenção espalhada do DeepSeek se repete? (1) SAE n=8.192 nas mesmas ativações do
# DeepSeek (teste de feature splitting) + análise de espalhamento; (2) pipeline v3 do Qwen3 + espalhamento.
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
PY=/home/systemp2w/miniconda3/envs/fcsae/bin/python
for p in v3_deepseek_w8k v3_qwen3; do
  echo "[$(date +%F\ %T)] início $p" >> logs/plan_spread.log
  MAX_TRIES=1000 bash scripts/run_v3.sh "$p" >> "logs/$p.log" 2>&1
  for t in 1 2 3; do $PY -m fcsae.analysis.spread --preset "$p" >> "logs/spread_$p.log" 2>&1 && break; sleep 30; done
  echo "[$(date +%F\ %T)] fim $p" >> logs/plan_spread.log
done
