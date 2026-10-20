# Attribution

This viewer is adapted from OpenAI's **sae-viewer** (https://github.com/openai/sparse_autoencoder/tree/main/sae-viewer),
released under the MIT License (Copyright (c) 2023 OpenAI) — see `LICENSE-OPENAI`.

Changes: data loaded from the Hugging Face dataset `lucasfrag/fact-checking-nei-saes-data` (`viewer/`);
model/feature selector for the fact-checking domain SAEs; per-example gold label and source; fact-checking
statistics (NEI rate, AUC NEI vs. SUP/REF); histogram split by label; decoder-cosine neighbors and logit effects;
new welcome page listing NEI-related features (evidence not enough).
