# Fact-Checking Abstention SAEs

**Domain sparse autoencoders for finding, validating and steering the feature behind abstention
("NOT ENOUGH EVIDENCE") in fact-checking LLMs.**

We train **TopK sparse autoencoders** (following Gao et al., 2024) on the residual stream of
instruction-tuned LLMs *while they read fact-checking verdict prompts* (claim + evidence + instructions),
locate the feature that encodes abstention at the decision position, check what it responds to, and
**manipulate it** during the verdict (activation steering) to test whether it causally controls abstention
and whether suppressing it improves fact-checking on AVeriTeC.

| resource | link |
|---|---|
| 🤗 **SAE weights** (SAELens format, one folder per model/layer) | [`lucasfrag/fact-checking-abstention-saes`](https://huggingface.co/lucasfrag/fact-checking-abstention-saes) |
| 🤗 **Data** (ordered training claims + decision-position activations) | [`lucasfrag/fact-checking-abstention-saes-data`](https://huggingface.co/datasets/lucasfrag/fact-checking-abstention-saes-data) |
| 🔎 **SAE viewer** (browse every feature: top claims, label histograms, logit effects) | [`lucasfrag/fact-checking-abstention-saes-viewer`](https://huggingface.co/spaces/lucasfrag/fact-checking-abstention-saes-viewer) |
| Fact-checking pipeline (Single-step vs. Iterative question generation) | [`lucasfrag/iterative-question-generation`](https://github.com/lucasfrag/iterative-question-generation) |

Part of an MSc dissertation (PPGCC/PUCRS). Status: **work in progress** — the Llama-3.1-8B SAE is released; SAEs for the other models of the Single-step vs. Iterative study
(DeepSeek-R1-Distill-Llama-8B, being redone from scratch; Qwen3-8B, Gemma-3-4B-it, Gemma-4-E4B-it, Ministral-3-8B-Instruct) are being trained.

---

## Why domain SAEs

Off-the-shelf SAEs (Gemma Scope, Llama Scope, …) reconstruct the **verdict-decision position** (the last
prompt token, where the model commits to SUPPORTED / REFUTED / NOT ENOUGH EVIDENCE) poorly: it is out
of their training distribution. Training the SAE on the task distribution itself puts the decision
position in-distribution, so the abstention mechanism can be read off the SAE's features.

## Released SAEs

| base model | layer | d_sae | k | dead features | FVU at decision | abstention feature |
|---|---|---|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | 15 | 32,768 | 32 | 2 | 0.234 | **19219** |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | *tbd* | | | | | *being redone* |
| `Qwen/Qwen3-8B` | 22 | | | | | *in progress* |
| `google/gemma-3-4b-it` | *tbd* | | | | | *planned* |
| `google/gemma-4-E4B-it` | *tbd* | | | | | *planned* |
| `mistralai/Ministral-3-8B-Instruct-2512` | *tbd* | | | | | *planned* |

Feature 19219 (Llama): 93.5% of the 200 claims that activate it most are NEI (4.7× the base rate);
AUC (NEI vs. SUP/REF) = 0.783; its **magnitude** (not on/off) encodes abstention.
Browse it in the viewer: [feature 19219](https://lucasfrag-fact-checking-abstention-saes-viewer.static.hf.space/#/model/fact-checking/family/fc_topk_32k/model/llama-3.1-8b-instruct/feature/19219).

DeepSeek: the first SAE (layer 15) is being redone from scratch, starting from a new layer sweep.

```python
from huggingface_hub import snapshot_download
from sae_lens import SAE

path = snapshot_download("lucasfrag/fact-checking-abstention-saes", allow_patterns="llama-3.1-8b-instruct/layer-15/*")
sae = SAE.load_from_disk(f"{path}/llama-3.1-8b-instruct/layer-15")
feats = sae.encode(resid)      # raw residual stream at blocks.15.hook_resid_post, shape [..., 4096]
```

---

## Repository layout

```
fcsae/                    the package
  config.py               presets (v3_llama, v3_deepseek, v3_qwen3; Gemma presets to come) and the Gao et al. optimization flags
  data/                   VitaminC + FEVER loaders, verdict-prompt template
  harvest/harvest.py      collect residual-stream activations (resumable, deterministic order)
  sae/                    TopK SAE (AuxK, input normalization, EMA, …) and trainer (auto-resume)
  analysis/               abstention analysis, feature inspection, layer sweep, generalization
  steering/factcheck_sae.py  feature steering in the verdict step (faithful replay of the pipeline prompt)
  export/to_saelens.py    export to SAELens format, with verification
  analysis/viewer_data.py per-feature JSONs for the SAE viewer
viewer/                   SAE viewer (adapted from OpenAI's sae-viewer, MIT), deployed as a static HF Space
scripts/                  long-running wrappers with watchdog + auto-restart (run_v3.sh, run_steer_sae.sh)
notebooks/                analysis notebooks (NN-name.ipynb) and their builders (build_NN_name.py)
  v1-historico/           earlier SAE (v1) notebooks, kept as history
results/                  small JSON results read by the notebooks (incl. raw steering generations)
old_method/corpora/averitec/outputs/   Single-step / Iterative pipeline outputs (evidence + predictions)
```

Heavy artifacts are **not** versioned: activations (~1 TB per model), SAE checkpoints and caches live
outside git (`FCSAE_STORE`); the SAEs and the decision-position activations are on the Hugging Face Hub.

## Installation

Python **3.12** (conda-forge) and a CUDA GPU (models are loaded in 4-bit NF4). Versions used:

```bash
conda create -n fcsae --override-channels -c conda-forge python=3.12
conda activate fcsae
pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.10.0+cu130"
pip install "transformers==5.14.1" "bitsandbytes==0.49.2" "datasets==5.0.1" "safetensors==0.8.0" \
            "numpy==2.5.3" "scikit-learn==1.9.1" "accelerate==1.15.0" "tokenizers==0.22.2" \
            nbformat jupyter matplotlib pandas scipy
pip install -e .
```

The SAELens export/verification runs in a separate environment (`pip install sae-lens`, CPU is enough).

## Reproducing (Llama-3.1-8B, preset `v3_llama`)

```bash
export FCSAE_STORE=/path/to/large/disk      # activations, checkpoints and results are written here

# 1–3. harvest (≈1 TB, ~30 h) → train the SAE (~17 h) → abstention analysis, with watchdog/resume
bash scripts/run_v3.sh v3_llama

# 4. semantic validation of the abstention feature (uses the cached decision activations)
python -m fcsae.analysis.inspect_feature --preset v3_llama --feat 19219

# 5. steering in the verdict step: 3 evidence conditions × α grid (+ random-feature control)
bash scripts/run_steer_sae.sh steer --preset v3_llama --feat 19219 \
     --conds none,single,iterative --betas=0,-0.25,0.25,-0.5,0.5,-0.75,0.75,-1,1
bash scripts/run_steer_sae.sh steer_rand --preset v3_llama --feat 19219 --random 10 --split test --n 100 --betas=<chosen>

# 6. export to SAELens (+ verification on real activations)
PYTHONPATH=. python -m fcsae.export.to_saelens --preset v3_llama --out <dir> --feat 19219

# 7. notebooks
python notebooks/build_02_sae_v3.py && jupyter nbconvert --to notebook --execute --inplace notebooks/02-sae-v3.ipynb
```

The harvest is deterministic: the exact claim order is materialized in `examples.jsonl` (also on the Hub),
so regenerated activations are bit-identical given the same environment.

## Notes on methodology

- **SAE training** follows Gao et al. (2024): TopK (k=32), 32,768 features, AuxK, input normalization,
  normalized losses, small Adam ε, decoder-gradient projection, geometric-median init and weight EMA.
  The only deviation is the batch size (2,048 tokens), discussed in `notebooks/02-sae-v3.ipynb` §3.
- **Steering** follows Ferrando et al. (2025, Eq. 4): `x ← x + α·d_j` with the feature's decoder direction,
  α reported in multiples of the residual-stream norm; random-feature control as in their work.
- **Verdict parsing.** The original pipeline's label parser (`LLMVerdict.normalize`) checks for "REFUTE"/"FALSE"
  anywhere in the response before "NOT ENOUGH", so answers such as *"NOT ENOUGH EVIDENCE — the evidence does
  not support or refute the claim"* are recorded as REFUTED. Our steering analysis uses the label the model
  actually wrote (`first_label`); see `notebooks/03-steering-sae-v3.ipynb` §3.1 and
  `notebooks/01-single-vs-iterative-it.ipynb` §7.

## References

- Gao, L., et al. (2024). *Scaling and Evaluating Sparse Autoencoders.* [arXiv:2406.04093](https://arxiv.org/abs/2406.04093)
- Ferrando, J., et al. (2025). *Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models.* ICLR. [arXiv:2411.14257](https://arxiv.org/abs/2411.14257)
- Arditi, A., et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Schlichtkrull, M., et al. (2023). *AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web.* NeurIPS Datasets & Benchmarks. [arXiv:2305.13117](https://arxiv.org/abs/2305.13117)
- Schuster, T., et al. (2021). *Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence.* NAACL. [arXiv:2103.08541](https://arxiv.org/abs/2103.08541)
- Thorne, J., et al. (2018). *FEVER: a Large-scale Dataset for Fact Extraction and VERification.* NAACL. [arXiv:1803.05355](https://arxiv.org/abs/1803.05355)

## Citation

Citation information will be added with the dissertation / paper.
