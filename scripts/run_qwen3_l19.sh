#!/usr/bin/env bash
# Qwen3 na camada 19 (cross-model mais justo). Mesmo pipeline, layer 19.
cd /home/systemp2w/projects/mestrado
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="timeout 2400 /home/systemp2w/miniconda3/bin/python"
SP=/tmp/claude-1000/-home-systemp2w-projects-mestrado/29c5db7f-1119-4020-b3df-6729287e796e/scratchpad
mkdir -p "$SP"
P=v1_qwen3_l19
ACT=activations/$P/manifest.json; MET=checkpoints/$P/metrics.json; DISC=results/$P/abstention.json
BETAS="-1.5,-1,-0.5,-0.25,0,0.25,0.5,0.75,1"
echo "##### QWEN3-L19 INÍCIO $(date +%H:%M:%S) #####"

for i in $(seq 1 15); do [ -f "$ACT" ] && break
  echo "== HARVEST $i $(date +%H:%M:%S) =="; $PY -m fcsae.harvest.harvest --preset $P 2>&1 | tee -a "$SP/q19_harvest.log"; sleep 4; done
[ -f "$ACT" ] || { echo "harvest falhou"; exit 1; }
for i in $(seq 1 25); do [ -f "$MET" ] && break
  echo "== TREINO $i $(date +%H:%M:%S) =="; $PY -m fcsae.sae.train --preset $P 2>&1 | tee -a "$SP/q19_train.log"; sleep 4; done
[ -f "$MET" ] || { echo "treino falhou"; exit 1; }
for i in $(seq 1 6); do [ -f "$DISC" ] && break
  echo "== DISCOVERY $i =="; $PY -m fcsae.analysis.abstention --preset $P 2>&1 | tee "$SP/q19_disc.log"; sleep 4; done
[ -f "$DISC" ] || { echo "discovery falhou"; exit 1; }
FEAT=$(/home/systemp2w/miniconda3/bin/python -c "import json;print(json.load(open('results/$P/abstention.json'))['top_decoder_cos'][0]['feat'])")
echo "== feature qwen3-l19 = $FEAT =="
for T in averitec climate_fever scifact healthver; do
  for i in 1 2 3 4; do [ -f "results/$P/generalize_$T.json" ] && break
    echo "== GEN $T $i =="; $PY -m fcsae.analysis.generalize --sae $P --feat "$FEAT" --test $T 2>&1 | tee "$SP/q19_gen_${T}.log"; sleep 4; done
done
for i in $(seq 1 15); do [ -f "results/$P/steering_beta.json" ] && break
  echo "== STEER (com ev) $i =="; $PY -m fcsae.steering.steer --sae $P --layer 19 --n 200 --alphas="$BETAS" --scale-by-residual --max-new-tokens 24 2>&1 | tee -a "$SP/q19_steer.log"; sleep 4; done
for i in $(seq 1 15); do [ -f "results/$P/steering_beta_noevidence.json" ] && break
  echo "== STEER (sem ev) $i =="; $PY -m fcsae.steering.steer --sae $P --layer 19 --n 200 --alphas="$BETAS" --scale-by-residual --no-evidence --max-new-tokens 24 2>&1 | tee -a "$SP/q19_steer_noev.log"; sleep 4; done
echo "##### QWEN3-L19 FIM $(date +%H:%M:%S) #####"
