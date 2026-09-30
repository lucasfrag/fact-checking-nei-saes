# Notebooks — índice (na ordem do trabalho)

Cada notebook `NN-nome.ipynb` é gerado pelo seu builder `build_NN_nome.py` (fonte única de verdade)
e depois executado:

    python notebooks/build_NN_nome.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/NN-nome.ipynb

## Sequência principal (SAE v3)

| # | notebook | o que tem | estado |
|---|---|---|---|
| 00 | `00-faq.ipynb` | perguntas & respostas do projeto | vivo |
| 00 | `00-datasets-fact-checking.md` | levantamento dos datasets (treino × teste) | — |
| 01 | `01-single-vs-iterative-it.ipynb` | **RQ1** com modelos Instruct via HuggingFace (tabelas do SA + F1 CONF, NEI, erro fora de NEI, custo) e comparação com o Ollama | pronto |
| 02 | `02-sae-v3.ipynb` | **SAE v3** (todos os dados + Gao et al. 2024): auditoria, features mortas, v1 × v3, feature 19219 | pronto |
| 03 | `03-steering-sae-v3.ipynb` | steering com a feature 19219: desenho + referências, fidelidade ao pipeline, curvas × β, β escolhido na validação, controle aleatório (resultados atualizam ao reexecutar) | em andamento |
| 04 | `04-generalizacao-v3.ipynb` | a feature 19219 em AVeriTeC, Climate-FEVER, SciFact, HealthVer | a fazer |
| 05 | `05-cross-model-v3.ipynb` | todos os modelos do estudo Single × Iterative (Llama, DeepSeek, Qwen3, Gemma3, Gemma4, Ministral3) no v3 | a fazer |

## Histórico — SAE v1 (`v1-historico/`)

A jornada que levou ao v3; resultados preservados (base da seção de metodologia).

| notebook | o que tem |
|---|---|
| `v1-historico/02-feature-discovery.ipynb` | SAE v1, feature 22280, ablation v2, generalização |
| `v1-historico/03-causal-steering.ipynb` | steering causal 2×2 com a direção v1 |
| `v1-historico/04-cross-model.ipynb` | Llama × DeepSeek × Qwen3 (v1), varredura de camadas |
| `v1-historico/05-steering-factcheck.ipynb` | estágio 2 v1 (atenção: prompt de veredito aproximado, não o do pipeline) |

Scripts de execução longa ficam em `scripts/` na raiz do projeto.
