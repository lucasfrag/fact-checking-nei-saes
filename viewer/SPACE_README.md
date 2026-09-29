---
title: Fact-Checking Abstention SAE Viewer
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Explore SAE features of LLMs reading fact-checking verdicts
models:
  - lucasfrag/fact-checking-abstention-saes
datasets:
  - lucasfrag/fact-checking-abstention-saes-data
tags: [sparse-autoencoder, sae-lens, mechanistic-interpretability, fact-checking, abstention]
---

# Fact-Checking Abstention SAE Viewer

Interactive viewer for the features of **domain TopK sparse autoencoders** trained on the residual stream of
instruction-tuned LLMs while they read **fact-checking verdict prompts** (claim + evidence + instructions), built to
study **abstention** ("NOT ENOUGH EVIDENCE"). For each of the 32,768 features: activation statistics at the verdict-decision
position, histogram split by gold label (NEI vs. SUP/REF), top and random activating claims with their gold labels,
logit effects and nearest features. Start with feature **19219** (Llama-3.1-8B-Instruct, layer 15), the candidate abstention latent.

- SAEs: [lucasfrag/fact-checking-abstention-saes](https://huggingface.co/lucasfrag/fact-checking-abstention-saes)
- Data (incl. the per-feature JSONs this viewer reads): [lucasfrag/fact-checking-abstention-saes-data](https://huggingface.co/datasets/lucasfrag/fact-checking-abstention-saes-data)

Adapted from OpenAI's [sae-viewer](https://github.com/openai/sparse_autoencoder/tree/main/sae-viewer) (MIT License, © 2023 OpenAI).
