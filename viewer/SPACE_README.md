---
title: Fact-Checking NEI SAE Viewer
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Explore SAE features of LLMs reading fact-checking verdicts
models:
  - lucasfrag/fact-checking-nei-saes
datasets:
  - lucasfrag/fact-checking-nei-saes-data
tags: [sparse-autoencoder, sae-lens, mechanistic-interpretability, fact-checking, nei, evidence-sufficiency]
---

# Fact-Checking NEI SAE Viewer

Interactive viewer for the features of **domain TopK sparse autoencoders** trained on the residual stream of
instruction-tuned LLMs while they read **fact-checking verdict prompts** (claim + evidence + instructions), built to
study **when the model judges the evidence is not enough** to decide (NEI, "NOT ENOUGH EVIDENCE"). For each of the 32,768 features: activation statistics at the verdict-decision
position, histogram split by gold label (NEI vs. SUP/REF), top and random activating claims with their gold labels,
logit effects and nearest features. Start with feature **19219** (Llama-3.1-8B-Instruct, layer 15), the NEI feature (confirmed by steering), feature **12482** (Qwen3-8B, layer 22) or feature **12903** (Gemma-3-4B, layer 19).

- SAEs: [lucasfrag/fact-checking-nei-saes](https://huggingface.co/lucasfrag/fact-checking-nei-saes)
- Data (incl. the per-feature JSONs this viewer reads): [lucasfrag/fact-checking-nei-saes-data](https://huggingface.co/datasets/lucasfrag/fact-checking-nei-saes-data)

Adapted from OpenAI's [sae-viewer](https://github.com/openai/sparse_autoencoder/tree/main/sae-viewer) (MIT License, © 2023 OpenAI).
