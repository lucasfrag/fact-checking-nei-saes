#!/usr/bin/env bash
# Ciclo automático de um modelo (Lucas, 10/10/2026): varredura de camadas (1.200/classe) → camada pela MESMA regra de
# todos os modelos (maior AUC da direção de abstenção, 3 casas; empate → a primeira) → coleta → treino → análise
# (run_v3.sh) → espalhamento. Para em caso de erro. Publicação, exclusão das ativações e steering são feitos depois.
#   bash scripts/run_model_cycle.sh <preset> ["texto que precisa aparecer em logs/plan_models.log antes de começar"]
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
PY=/home/systemp2w/miniconda3/envs/fcsae/bin/python
P=$1; WAIT=${2:-}
[ -n "$WAIT" ] && until grep -q "$WAIT" logs/plan_models.log; do sleep 120; done
echo "[$(date +%F\ %T)] $P: início varredura de camadas (1.200/classe)" >> logs/plan_models.log
for t in 1 2 3 4 5; do $PY -m fcsae.analysis.layer_sweep --preset $P --n-per-class 1200 >> logs/layer_sweep_$P.log 2>&1 && break; sleep 30; done
L=$(P=$P $PY - <<'PY'
import json, os
d = json.load(open(os.path.join(os.environ["FCSAE_STORE"], "results", os.environ["P"], "layer_sweep.json")))
a = [round(x, 3) for x in d["auc_by_layer"]]
print(a.index(max(a)))          # maior AUC (3 casas); empate -> primeira camada
PY
)
[ -z "$L" ] && { echo "[$(date +%F\ %T)] ERRO: varredura $P sem resultado — parado" >> logs/plan_models.log; exit 1; }
sed -i -E "s/(return Config\(name=\"$P\", model_tag=\"[^\"]+\", layer=)[0-9]+,/\1$L,/" fcsae/config.py
[ "$($PY -c "from fcsae.config import preset; print(preset('$P').layer)")" = "$L" ] || { echo "[$(date +%F\ %T)] ERRO ao gravar camada $L em $P — parado" >> logs/plan_models.log; exit 1; }
echo "[$(date +%F\ %T)] $P: camada $L pela regra (maior AUC; empate → primeira); início coleta/treino/análise" >> logs/plan_models.log
MAX_TRIES=1000 bash scripts/run_v3.sh $P >> logs/$P.log 2>&1
for t in 1 2 3; do $PY -m fcsae.analysis.spread --preset $P >> logs/spread_$P.log 2>&1 && break; sleep 30; done
echo "[$(date +%F\ %T)] fim ciclo $P (coleta → treino → análise → espalhamento)" >> logs/plan_models.log
