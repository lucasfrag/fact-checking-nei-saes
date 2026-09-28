"""Configuração central: caminhos, modelos, e presets de experimento.

A configuração é um dataclass tipado; presets nomeados (`smoke`, `v1_domain`) são
selecionados por nome na linha de comando. Sem YAML para reduzir partes móveis.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

# ------------------------------------------------------------------ caminhos
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Onde ficam os artefatos pesados (gitignored). Padrão = drive rápido do projeto
# (I/O rápido, ideal para treino que relê shards). Para overflow/arquivo use o HD
# de 4 TB: `export FCSAE_STORE=/mnt/d/Mestrado/fcsae` (mais espaço, I/O mais lento).
STORE = os.environ.get("FCSAE_STORE", ROOT)
DATA_RAW = os.path.join(STORE, "data_raw")          # cache de datasets baixados
ACTIVATIONS = os.path.join(STORE, "activations")    # shards de ativações cacheadas
CHECKPOINTS = os.path.join(STORE, "checkpoints")    # SAEs treinados
RESULTS = os.path.join(STORE, "results")            # métricas/figuras de análise

for _d in (DATA_RAW, ACTIVATIONS, CHECKPOINTS, RESULTS):
    os.makedirs(_d, exist_ok=True)

# ------------------------------------------------------------------ modelos
# checkpoint HF por tag curta. O modelo alvo da v1 é o Llama-3.1-8B-Instruct
# (temos SAEs oficiais andyrdt em várias camadas, inclusive a 15, como baseline).
MODELS = {
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-r1:8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "qwen3:8b": "Qwen/Qwen3-8B",
}

# rótulos canônicos do fact-checking
LABELS = ("SUP", "REF", "NEI")


@dataclass
class Config:
    name: str = "v1_domain"

    # --- modelo / captura ---
    model_tag: str = "llama3.1:8b"
    layer: int = 15                       # camada do residual stream a capturar (meio da rede)
    quant: str = "4bit"                   # "4bit" | "8bit" | "bf16"
    device: str = "cuda"

    # --- dados ---
    datasets: tuple = ("vitaminc", "fever")   # loaders a usar
    mix_general: bool = False             # False = só-domínio (v1); True = domínio + texto geral
    general_dataset: str = "monology/pile-uncopyrighted"
    general_fraction: float = 0.25        # fração de tokens gerais na mistura (quando mix_general)
    max_per_dataset: int = 60_000         # teto de claims por dataset (0 = sem teto)
    max_evidence: int = 4                 # nº máx. de trechos de evidência no prompt
    max_seq_len: int = 512                # truncamento do prompt (tokens)

    # --- harvest ---
    shard_tokens: int = 500_000           # nº de token-ativações por shard em disco
    dtype_store: str = "float16"

    # --- SAE (TopK) ---
    sae_width: int = 32_768               # nº de features do dicionário
    sae_k: int = 32                       # features ativas por token (L0 exato)
    sae_k_aux: int = 512                  # features na perda auxiliar (anti-morte)
    sae_dead_steps: int = 1000            # passos sem disparo p/ considerar morta
    sae_lr: float = 4e-4
    sae_batch: int = 2048                  # menor = menos carga/pico por passo (GPU WSL2 instável)
    sae_epochs: int = 2                    # FVU já platô em ~1 época; 2 reduz exposição a crashes
    sae_aux_coeff: float = 1.0 / 32.0     # peso da perda auxiliar
    # --- otimização do Gao et al. (2024), Apêndice A. Padrões = comportamento do v1
    #     (desligado); os presets v3 ligam tudo. Referências por item:
    sae_input_norm: bool = False          # §2.1: tira a média em d_model e normaliza cada token à norma 1
    sae_normalized_loss: bool = False     # §A.6: MSE principal / constante (1x no início); AuxK / var. do resíduo
    sae_adam_eps: float = 1e-8            # §A.3.1: ε pequeno p/ o Adam seguir invariante à escala (6.25e-10)
    sae_grad_proj: bool = False           # §A.3 [Bricken 2023]: remove do grad a componente paralela ao decoder
    sae_init_geomedian: bool = False      # §A.1 [Bricken 2023]: b_pre = mediana geométrica de uma amostra
    sae_ema: float = 0.0                  # §A.5: média móvel exponencial dos pesos (0.999); 0 = desligada

    def paths(self) -> dict:
        base = self.name
        return {
            "activations": os.path.join(ACTIVATIONS, base),
            "sae": os.path.join(CHECKPOINTS, base),
            "results": os.path.join(RESULTS, base),
        }

    @property
    def checkpoint(self) -> str:
        return MODELS[self.model_tag]

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ presets
# Otimização do Gao et al. (2024), Apêndice A (e código oficial openai/sparse_autoencoder).
# lr 1e-4 = padrão do código deles p/ n=32.768 latentes; §3.1.2: lr máximo ∝ 1/√n; nota 16:
# "lowering learning rates helped with decreasing dead latents".
GAO_OPT = dict(sae_lr=1e-4, sae_input_norm=True, sae_normalized_loss=True,
               sae_adam_eps=6.25e-10, sae_grad_proj=True, sae_init_geomedian=True,
               sae_ema=0.999)


def preset(name: str) -> Config:
    if name == "smoke":
        # ponta-a-ponta rápido para validar o pipeline inteiro
        return Config(name="smoke", max_per_dataset=200, sae_width=4096,
                      shard_tokens=50_000, sae_epochs=1)
    if name == "smoke_mixed":
        return Config(name="smoke_mixed", max_per_dataset=150, sae_width=4096,
                      shard_tokens=50_000, sae_epochs=1, mix_general=True, general_fraction=0.25)
    # ── LEGADO (v1): base dos resultados atuais da dissertação. Mantidos até o v3
    #    substituí-los (re-rodar as análises no v3). Não usar em experimentos novos.
    if name == "v1_domain":
        return Config(name="v1_domain", mix_general=False)
    if name == "v1_deepseek":
        return Config(name="v1_deepseek", model_tag="deepseek-r1:8b",
                      max_per_dataset=25_000, mix_general=False)
    if name == "v1_qwen3_l19":
        return Config(name="v1_qwen3_l19", model_tag="qwen3:8b", layer=19,
                      max_per_dataset=25_000, mix_general=False)
    # ── v3: versão FINAL "following prior works" (Gao et al. 2024; He et al. 2024, Llama
    #    Scope). Config padrão (TopK k=32, width 32.768, AuxK α=1/32, k_aux=512) + duas
    #    correções que alinham à literatura: limiar de "morta" ~10M tokens (dead_steps≈5000,
    #    antes ~2M) e MAIS DADOS (max_per_dataset alto → menos features mortas).
    #    Mapeamento completo em CONFIG.md no repositório de publicação.
    #    max_per_dataset: teto por dataset (0 = todo VitaminC ~370k + FEVER ~145k). Ajuste ao disco.
    #    Otimização alinhada ao Apêndice A do Gao (ver GAO_OPT). Única divergência: batch 2048
    #    (Gao usa 131.072); o próprio §A.4 cita 2048/4096 como usual [Bricken 2023; Conerly
    #    2024; Rajamanoharan 2024] e mostra L(N) pouco sensível ao batch (Fig. 12).
    #    Nome do preset = modelo (v3_llama / v3_deepseek / v3_qwen3). O v3_llama se chamava
    #    "v3_domain" até 27/09/2026 (herança do v1_domain); pastas e logs foram renomeados.
    if name == "v3_llama":
        return Config(name="v3_llama", model_tag="llama3.1:8b", mix_general=False, layer=15,
                      max_per_dataset=0, sae_dead_steps=5000, **GAO_OPT)   # 0 = todo o dataset
    if name == "v3_deepseek":
        return Config(name="v3_deepseek", model_tag="deepseek-r1:8b", layer=15,
                      max_per_dataset=0, sae_dead_steps=5000, **GAO_OPT)
    if name == "v3_qwen3":
        return Config(name="v3_qwen3", model_tag="qwen3:8b", layer=19,
                      max_per_dataset=0, sae_dead_steps=5000, **GAO_OPT)
    raise KeyError(f"preset desconhecido: {name}")
