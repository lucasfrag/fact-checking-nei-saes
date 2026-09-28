#!/usr/bin/env bash
# Pipeline v3 (harvest -> treino -> análise) para um preset, com retomada automática.
# Cada fase roda até o marcador de conclusão existir (manifest.json / metrics.json /
# abstention.json). Watchdog: se o log não avança por STALL segundos (hang da GPU no
# WSL2), mata e relança — o harvest e o treino retomam do último checkpoint.
#
#   bash scripts/run_v3.sh v3_llama 2>&1 | tee -a logs/v3_llama.log
set -u
PRESET=${1:-v3_llama}
STALL=${STALL:-2400}
MAX_TRIES=${MAX_TRIES:-60}
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"}
export HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"}
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1     # segfault nativo -> stack trace Python no log
# env conda-forge py3.12 (o py3.13 do miniconda base dava segfault aleatório no próprio
# interpretador: 5/250 vs 0/250; mesmas versões de pacotes, ativações idênticas bit a bit)
PY=${PY:-/home/systemp2w/miniconda3/envs/fcsae/bin/python}
cd "$(dirname "$0")/.."
mkdir -p logs

ACT="$FCSAE_STORE/activations/$PRESET"
CKP="$FCSAE_STORE/checkpoints/$PRESET"
RES="$FCSAE_STORE/results/$PRESET"

run_phase() {   # nome, marcador, módulo, [stall em s]
  local name=$1 marker=$2 mod=$3 stall=${4:-$STALL} try=0
  while [ ! -f "$marker" ]; do
    try=$((try + 1))
    if [ $try -gt $MAX_TRIES ]; then echo "[$(date +%F\ %T)] $name: desisti após $MAX_TRIES tentativas"; exit 1; fi
    local log="logs/${PRESET}_${name}.log"
    echo "[$(date +%F\ %T)] $name: tentativa $try (log: $log)"
    touch "$log"    # o relógio do watchdog começa no lançamento, não na última escrita antiga
    $PY -m "$mod" --preset "$PRESET" >>"$log" 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      local age=$(( $(date +%s) - $(stat -c %Y "$log") ))
      if [ $age -gt $stall ]; then
        echo "[$(date +%F\ %T)] $name: log parado há ${age}s — matando $pid e relançando"
        kill -9 $pid 2>/dev/null; break
      fi
    done
    wait $pid 2>/dev/null; local rc=$?
    echo "[$(date +%F\ %T)] $name: processo saiu (rc=$rc)"
    [ -f "$marker" ] || sleep 30
  done
  echo "[$(date +%F\ %T)] $name: CONCLUÍDO"
}

echo "[$(date +%F\ %T)] preset=$PRESET store=$FCSAE_STORE"
run_phase harvest "$ACT/manifest.json"   fcsae.harvest.harvest
run_phase train   "$CKP/metrics.json"    fcsae.sae.train
# a análise lê TODAS as decisões (~1 TB no HD) antes do 1º print -> watchdog bem mais longo
run_phase analysis "$RES/abstention.json" fcsae.analysis.abstention 21600
echo "[$(date +%F\ %T)] PIPELINE $PRESET COMPLETO"
