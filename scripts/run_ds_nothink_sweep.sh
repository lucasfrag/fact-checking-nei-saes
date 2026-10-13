#!/usr/bin/env bash
# Lucas (09–10/10/2026): DeepSeek refeito do zero, SEM raciocínio (<think> vazio) e sem truncamento (max_seq_len 16.384).
# Camada pela MESMA regra do Llama e do Qwen3: maior AUC da direção de abstenção na varredura (1.200/classe);
# empate na 3ª casa decimal → a primeira camada. Depois: coleta → treino → análise (run_v3.sh) → espalhamento.
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"} PYTHONUNBUFFERED=1
PY=/home/systemp2w/miniconda3/envs/fcsae/bin/python
P=v3_deepseek_nothink
until grep -q "fim steering Qwen3 12482" logs/plan_models.log; do sleep 120; done
echo "[$(date +%F\ %T)] início varredura de camadas $P (1.200/classe)" >> logs/plan_models.log
for t in 1 2 3 4 5; do $PY -m fcsae.analysis.layer_sweep --preset $P --n-per-class 1200 >> logs/layer_sweep_$P.log 2>&1 && break; sleep 30; done
L=$($PY - <<'PY'
import json, os
d = json.load(open(os.path.join(os.environ["FCSAE_STORE"], "results", "v3_deepseek_nothink", "layer_sweep.json")))
a = [round(x, 3) for x in d["auc_by_layer"]]
print(a.index(max(a)))          # maior AUC (3 casas); empate -> primeira camada
PY
)
[ -z "$L" ] && { echo "[$(date +%F\ %T)] ERRO: varredura $P sem resultado — parado" >> logs/plan_models.log; exit 1; }
sed -i "s/return Config(name=\"v3_deepseek_nothink\", model_tag=\"deepseek-r1:8b\", layer=[0-9]*,/return Config(name=\"v3_deepseek_nothink\", model_tag=\"deepseek-r1:8b\", layer=$L,/" fcsae/config.py
grep -q "name=\"v3_deepseek_nothink\", model_tag=\"deepseek-r1:8b\", layer=$L," fcsae/config.py || { echo "[$(date +%F\ %T)] ERRO ao gravar camada $L — parado" >> logs/plan_models.log; exit 1; }
echo "[$(date +%F\ %T)] $P: camada $L pela regra (maior AUC; empate → primeira); início coleta/treino/análise" >> logs/plan_models.log
MAX_TRIES=1000 bash scripts/run_v3.sh $P >> logs/$P.log 2>&1
for t in 1 2 3; do $PY -m fcsae.analysis.spread --preset $P >> logs/spread_$P.log 2>&1 && break; sleep 30; done
echo "[$(date +%F\ %T)] fim $P (coleta → treino → análise → espalhamento)" >> logs/plan_models.log
