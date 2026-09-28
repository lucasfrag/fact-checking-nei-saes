#!/usr/bin/env bash
# Pipeline cross-model completo do Qwen3: harvest -> treino -> discovery ->
# generalização (4 domínios) -> steering 2x2. Cada etapa com retry (GPU WSL2 instável).
# Qwen3 usa thinking desligado (veredito direto), então o steering é rápido/limpo como o Llama.
cd /home/systemp2w/projects/mestrado
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# timeout por tentativa: mata processo TRAVADO (hang de GPU) e o retry retoma do checkpoint
PY="timeout 2400 /home/systemp2w/miniconda3/bin/python"
SP=/tmp/claude-1000/-home-systemp2w-projects-mestrado/29c5db7f-1119-4020-b3df-6729287e796e/scratchpad
mkdir -p "$SP"
ACT=activations/v1_qwen3/manifest.json
MET=checkpoints/v1_qwen3/metrics.json
DISC=results/v1_qwen3/abstention.json

echo "===== QWEN3 PIPELINE INÍCIO $(date +%H:%M:%S) ====="

# 1) harvest
for i in $(seq 1 15); do [ -f "$ACT" ] && break
  echo "== HARVEST qwen tent $i $(date +%H:%M:%S) =="
  $PY -m fcsae.harvest.harvest --preset v1_qwen3 2>&1 | tee -a "$SP/qwen_harvest.log"; sleep 4; done
[ -f "$ACT" ] || { echo "HARVEST qwen FALHOU"; exit 1; }

# 2) treino
for i in $(seq 1 25); do [ -f "$MET" ] && break
  echo "== TREINO qwen tent $i $(date +%H:%M:%S) =="
  $PY -m fcsae.sae.train --preset v1_qwen3 2>&1 | tee -a "$SP/qwen_train.log"; sleep 4; done
[ -f "$MET" ] || { echo "TREINO qwen FALHOU"; exit 1; }

# 3) discovery
for i in $(seq 1 6); do [ -f "$DISC" ] && break
  echo "== DISCOVERY qwen tent $i $(date +%H:%M:%S) =="
  $PY -m fcsae.analysis.abstention --preset v1_qwen3 2>&1 | tee "$SP/qwen_disc.log"; sleep 4; done
[ -f "$DISC" ] || { echo "DISCOVERY qwen FALHOU"; exit 1; }

FEAT=$($PY -c "import json;print(json.load(open('results/v1_qwen3/abstention.json'))['top_decoder_cos'][0]['feat'])")
echo "== feature candidata qwen3 = $FEAT =="

# 4) generalização (4 domínios)
for T in averitec climate_fever scifact healthver; do
  for i in 1 2 3 4; do [ -f "results/v1_qwen3/generalize_$T.json" ] && break
    echo "== GEN qwen $T tent $i $(date +%H:%M:%S) =="
    $PY -m fcsae.analysis.generalize --sae v1_qwen3 --feat "$FEAT" --test $T 2>&1 | tee "$SP/qwen_gen_${T}.log"; sleep 4; done
done

# 5) steering 2x2 em β (fração do resíduo, comparável entre modelos); veredito direto
BETAS="-1.5,-1,-0.5,-0.25,0,0.25,0.5,0.75,1"
for i in $(seq 1 15); do [ -f "results/v1_qwen3/steering_beta.json" ] && break
  echo "== STEER qwen (com ev) tent $i $(date +%H:%M:%S) =="
  $PY -m fcsae.steering.steer --sae v1_qwen3 --layer 15 --n 200 --alphas="$BETAS" --scale-by-residual --max-new-tokens 24 2>&1 | tee -a "$SP/qwen_steer.log"; sleep 4; done
for i in $(seq 1 15); do [ -f "results/v1_qwen3/steering_beta_noevidence.json" ] && break
  echo "== STEER qwen (sem ev) tent $i $(date +%H:%M:%S) =="
  $PY -m fcsae.steering.steer --sae v1_qwen3 --layer 15 --n 200 --alphas="$BETAS" --scale-by-residual --no-evidence --max-new-tokens 24 2>&1 | tee -a "$SP/qwen_steer_noev.log"; sleep 4; done

echo "===== QWEN3 PIPELINE FIM $(date +%H:%M:%S) ====="
