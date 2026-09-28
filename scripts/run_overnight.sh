#!/usr/bin/env bash
# Trabalho noturno autônomo. Sequencial (uma GPU):
# 1) Llama steering em β (com/sem evidência) — comparável cross-model, rápido.
# 2) Qwen3 pipeline completo (harvest→treino→discovery→generalização→steering β).
# 3) DeepSeek steering em β, faixa corrigida (perturbação = fração do resíduo), reasoning.
cd /home/systemp2w/projects/mestrado
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# timeout 2400s por tentativa: se a GPU TRAVAR (hang, sem crashar), o processo é morto e
# o retry retoma do checkpoint (harvest/treino são resumíveis; steering tem cache por-α).
PY="timeout 2400 /home/systemp2w/miniconda3/bin/python"
SP=/tmp/claude-1000/-home-systemp2w-projects-mestrado/29c5db7f-1119-4020-b3df-6729287e796e/scratchpad
mkdir -p "$SP"
BETAS="-1.5,-1,-0.5,-0.25,0,0.25,0.5,0.75,1"

echo "########## OVERNIGHT INÍCIO $(date +%H:%M:%S) ##########"

# ---- 1) Llama steering em β (rápido, veredito direto) ----
for i in $(seq 1 12); do [ -f results/v1_domain/steering_beta.json ] && break
  echo "## LLAMA steer β (com ev) $i $(date +%H:%M:%S)"
  $PY -m fcsae.steering.steer --sae v1_domain --layer 15 --n 200 --alphas="$BETAS" --scale-by-residual 2>&1 | tee -a "$SP/llama_steer_beta.log"; sleep 3; done
for i in $(seq 1 12); do [ -f results/v1_domain/steering_beta_noevidence.json ] && break
  echo "## LLAMA steer β (sem ev) $i $(date +%H:%M:%S)"
  $PY -m fcsae.steering.steer --sae v1_domain --layer 15 --n 200 --alphas="$BETAS" --scale-by-residual --no-evidence 2>&1 | tee -a "$SP/llama_steer_beta_noev.log"; sleep 3; done

# ---- 2) Qwen3 pipeline completo ----
echo "## QWEN3 pipeline $(date +%H:%M:%S)"
bash scripts/run_qwen3_pipeline.sh 2>&1 | tee -a "$SP/qwen_pipeline.log" || echo "qwen pipeline retornou erro (segue)"

# ---- 3) DeepSeek steering em β (reasoning; n e β reduzidos p/ custo, faixa corrigida) ----
DSBETAS="-1,-0.5,-0.25,0,0.25,0.5,1"
for i in $(seq 1 15); do [ -f results/v1_deepseek/steering_beta.json ] && break
  echo "## DEEPSEEK steer β (com ev) $i $(date +%H:%M:%S)"
  $PY -m fcsae.steering.steer --sae v1_deepseek --layer 15 --n 30 --alphas="$DSBETAS" --scale-by-residual --max-new-tokens 512 2>&1 | tee -a "$SP/ds_steer_beta.log"; sleep 4; done

echo "########## OVERNIGHT FIM $(date +%H:%M:%S) ##########"
