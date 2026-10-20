#!/usr/bin/env bash
# Lucas (15/10/2026): "publicar conforme cada etapa for concluída". A cada célula de steering nova (Gemma3 12903 e
# Qwen3 12482), regenera os dados do painel de steering (fcsae.analysis.viewer_steering) e sobe para o HF.
# O viewer lê os dados direto do HF: não precisa reconstruir a página. Termina quando a grade do Gemma3 acabar.
set -u
cd "$(dirname "$0")/.."
export FCSAE_STORE=${FCSAE_STORE:-"/mnt/d/Projetos WSL/fcsae-store"} HF_HOME=${HF_HOME:-"/mnt/d/Projetos WSL/hf-cache"}
PY=/home/systemp2w/miniconda3/envs/fcsae/bin/python
declare -A LAST
publish() {  # $1 preset  $2 feat  $3 pasta do viewer
  local n; n=$(ls results/$1/steer_sae/*.json 2>/dev/null | grep -vc "__all\|__test" )
  [ "${LAST[$1]:-}" = "$n" ] && return
  HF_HUB_OFFLINE=1 timeout 1800 $PY -m fcsae.analysis.viewer_steering --preset $1 --feat $2 --out "$HOME/viewer_build/data/$3" >> logs/auto_publish_steering.log 2>&1 || return
  HF_HUB_DISABLE_XET=1 timeout 1800 ~/venvs/saelens/bin/python -c "
import glob, os
from huggingface_hub import HfApi, CommitOperationAdd
B=os.path.expanduser('~/viewer_build/data/'); fs=glob.glob(B+'$3/steering/$2*.json')
HfApi().create_commit('lucasfrag/fact-checking-nei-saes-data', [CommitOperationAdd(path_in_repo='viewer/'+os.path.relpath(f,B), path_or_fileobj=f) for f in fs], repo_type='dataset', commit_message='Steering $2: $n cells (auto)')
" >> logs/auto_publish_steering.log 2>&1 && { LAST[$1]=$n; echo "[$(date +%F\ %T)] publicado $1 feat $2: $n células" >> logs/auto_publish_steering.log; }
}
while true; do
  publish v3_gemma3 12903 gemma-3-4b-it/layer-19
  publish v3_qwen3_l22 12482 qwen3-8b/layer-22
  grep -q "fim steering Gemma3 12903 (grade completa)" logs/plan_models.log && { publish v3_gemma3 12903 gemma-3-4b-it/layer-19; echo "[$(date +%F\ %T)] FIM auto-publicação" >> logs/auto_publish_steering.log; break; }
  sleep 600
done
