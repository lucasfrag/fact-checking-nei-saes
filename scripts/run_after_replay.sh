#!/usr/bin/env bash
# Fila da madrugada de 30/09/2026 (pedido do Lucas: "quando terminar [o replay] já pode pôr os SAEs pra treinar"):
# espera a fila de replay terminar e roda o pipeline v3 (harvest -> treino -> análise) do DeepSeek e depois do Qwen3.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "[r]un_replay_all.sh" > /dev/null; do sleep 60; done
echo "[$(date +%F\ %T)] replay terminou; iniciando SAEs" >> logs/after_replay.log
for p in v3_deepseek v3_qwen3; do
  echo "[$(date +%F\ %T)] início $p" >> logs/after_replay.log
  MAX_TRIES=1000 bash scripts/run_v3.sh "$p" >> "logs/$p.log" 2>&1
  echo "[$(date +%F\ %T)] fim $p (rc=$?)" >> logs/after_replay.log
done
