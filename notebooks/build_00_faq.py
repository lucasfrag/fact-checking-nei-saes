# -*- coding: utf-8 -*-
"""Gera notebooks/00-faq.ipynb — registro das perguntas do Lucas e as respostas, para
consulta rápida. Documento vivo (markdown); acrescentar novas perguntas no fim.
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell

cells = []
def md(t): cells.append(new_markdown_cell(t))

md(r"""# FAQ — perguntas & respostas do projeto

**Dissertação de mestrado (PPGCC/PUCRS).** Registro das dúvidas que foram surgindo sobre o
método (SAEs, modelos, camadas), com respostas curtas para consulta rápida. Documento vivo —
perguntas novas entram no fim.

**Índice**
1. [Como exatamente treinamos o Sparse Autoencoder?](#1)
2. ["Usamos o Llama" — é modelo pré-treinado ou o SAE é do zero?](#2)
3. [Quantos SAEs temos treinados?](#3)
4. [Quais modelos no Single-vs-Iterative e quais têm SAE?](#4)
5. [Qual a diferença das versões dos SAEs? Por que tantos no Llama e no Qwen?](#5)
6. [Como escolhemos a camada de cada modelo (15 no Llama, 22 no Qwen3; DeepSeek sendo refeito)?](#6)
7. [O que é o SAE v3 e o que significa "seguir o Gao et al. (2024)"?](#7)
8. [Por que as features mortas caíram de 33–48% para ~0%?](#8)
9. [Por que a AUC da direção caiu no v3 (0,839 → 0,796)?](#9)
10. [A feature 19219 "liga" só quando o modelo abstém?](#10)
11. [Como é feita a validação — como escolhemos o α do steering?](#11)
12. [O que significa "todas as linhas usam os mesmos 250 claims da metade de teste"?](#12)
13. [O que é CAA?](#13)
14. [Como decidimos se o steering "melhorou" ou "piorou"? O que é o "p (McNemar)"?](#14)
15. [Por que o critério de qualidade do steering mudou de "legível" para "coerente"?](#15)
16. [O que significa a fórmula do steering x ← x + α·d₁₉₂₁₉ — e de onde ela vem?](#16)
17. [Que defeito foi achado na regra de rótulo do pipeline, e o que ele muda?](#17)
18. [Por que a grade de steering foi cortada em |α/R| ≤ 1?](#18)
19. [No DeepSeek a abstenção ficou espalhada em várias features. A literatura trata disso? Como fazemos o steering?](#19)
20. [Steering com a feature × steering com a direção média: qual a diferença? O que fizemos antes é descartado?](#20)
21. [Quantas features o SAE deve ter (32.768)? Como justificar esse hiperparâmetro?](#21)
22. [No DeepSeek, quais features controlam a abstenção? O que a triagem de steering mostrou?](#22)
23. [Como o cosseno entre uma feature e a direção de abstenção é calculado? O que ele significa?](#23)
24. [O que é a "ativação" de uma feature (o eixo "how strongly the feature fires" do viewer)?](#24)
25. [Qual a diferença entre "feature" e "ativação"?](#25)
26. [Como a ativação é medida a partir da direção? Como sei se ela é forte ou fraca?](#26)
27. [O que é, em detalhe, a "régua" de uma feature (o vetor do encoder)?](#27)
28. [O que o steering da feature 12482 mostrou no Qwen3?](#28)
29. [O que deu errado no treino do SAE do Gemma3 (perda "nan") e como foi corrigido?](#29)
30. ["Ativação" do modelo × ativação de uma feature: de onde até onde vai cada número?](#30)
31. [Por que a dose do steering passou a usar a norma centrada ($R_c$) e não a norma média?](#31)
32. ["Resíduo" é o mesmo que *residual stream*?](#32)
33. [Por que a feature do Qwen3 (12482) gera REF/SUP no steering, e não NEI? Fizemos algo errado?](#33)
34. [No viewer, por que só os "top 20" e por que "not shown: 17.614 claims"? Esses dados são ignorados?](#34)
35. [Como a literatura escolhe a feature de um SAE para um comportamento? E o nosso método?](#35)
36. [De onde até onde vão o d de Cohen, o cosseno e a AUC?](#36)
37. [O que o steering da feature 12903 mostrou no Gemma3?](#37)

> **Atualização 27/09/2026:** o SAE principal agora é o **v3** (ver `02-sae-v3.ipynb`). As
> respostas 1, 3 e 5 foram atualizadas; as 7–10 são novas.
""")

md(r"""<a id="1"></a>
## 1. Como exatamente treinamos o Sparse Autoencoder?

**Ideia em uma frase.** O SAE aprende a **reescrever** cada vetor de ativação do modelo (o
*residual stream* na camada L, dim **4096**) como uma **soma esparsa** de poucas features de
um dicionário grande. Se cada feature captura um conceito, ganhamos direções interpretáveis.

**O que entra:** as ativações que o *harvest* cacheou (residual da camada-alvo em todas as
posições dos prompts de veredito VitaminC+FEVER), ~**9–11 milhões** de vetores de 4096 dims.

**Arquitetura — TopK SAE** (Gao et al., 2024), dicionário **8× overcomplete** (32.768 features):
```
pre = (x − b_dec) @ W_enc + b_enc      # pontua as 32.768 features
z   = TopK(pre, k=32)                   # mantém só as 32 maiores (ReLU), zera o resto
x̂  = z @ W_dec + b_dec                  # reconstrói somando as 32 features ativas
```
O **TopK** força **exatamente 32 features ativas por token** (esparsidade L0 exata) — sem
tunar coeficiente de L1, e treina estável (por isso preferimos a JumpReLU nesta 1ª rodada).

**Perda:** `||x − x̂||²  +  (1/32)·AuxK`. A **AuxK** pega as features **mortas** (sem disparar
há 1000 passos) e as usa pra reconstruir o resíduo — reanima features mortas, o principal
modo de falha de SAE.

**Hiperparâmetros (v1):** width 32.768 · k 32 · Adam lr 4e-4 · batch 2.048 · 2 épocas.
**Avaliação:** num shard *held-out* — FVU (erro de reconstrução), L0 (~32) e features mortas.

> **No v3** a arquitetura e a perda são as mesmas, mas a **otimização segue o Apêndice A do
> Gao et al. (2024)**: entrada normalizada (média 0, norma 1), MSE e AuxK normalizados, Adam
> ε = 6,25e-10, projeção do gradiente do decoder, b_pre = mediana geométrica, EMA dos pesos
> (0,999), lr 1e-4, limiar de morte ~10M tokens (5.000 passos). Batch mantido em 2.048.
> Dados: **todo** o VitaminC + FEVER (598.930 claims, 132M tokens). Ver pergunta 7.

> Detalhe de infra: o treino é **retomável por passos** (checkpoint a cada 500), porque a GPU
> no WSL2 cai sob carga. Código: `fcsae/sae/train.py` + `fcsae/sae/model.py`.
""")

md(r"""<a id="2"></a>
## 2. "Usamos o Llama" — é modelo pré-treinado ou o SAE é do zero?

São **dois modelos** com papéis diferentes:

**O Llama — pré-treinado e congelado.** Usamos o Llama-3.1-8B pronto **só como fonte de
ativações**: rodamos ele pra frente (inferência 4-bit, **sem gradiente**) e gravamos o
residual stream. **Nunca treinamos nem ajustamos o Llama** — os pesos dele não mudam.

**O SAE — treinado do zero.** É uma rede separada e pequena, com pesos **inicializados
aleatoriamente**, treinada por gradiente **do zero** só pra reconstruir aqueles vetores.
Não é um SAE que baixamos pronto.

```
Llama (congelado) ──produz──► ativações (arquivos) ──treina──► SAE (do zero)
   pré-treinado                   "os dados"            a única coisa que treinamos
```
Analogia: o **Llama é um aparelho de ressonância** (não mexemos nele); as **ativações são os
exames**; o **SAE é um modelo novo** que treinamos pra achar padrões nos exames. Como cada
modelo produz ativações diferentes, treinamos **um SAE por modelo**. (E foi por isso que, na
faxina de disco, pudemos apagar as ativações brutas e manter o SAE treinado.)
""")

md(r"""<a id="3"></a>
## 3. Quantos SAEs temos treinados?

**6 SAEs**, de **3 famílias de modelos**:

| preset | modelo-fonte | camada | dados | k | papel |
|---|---|---|---|---|---|
| `v1_domain` | Llama-3.1-8B | 15 | só domínio | 32 | **principal** |
| `v2_mixed` | Llama-3.1-8B | 15 | domínio + geral | 48 | ablation de dados |
| `v2_domain_only` | Llama-3.1-8B | 15 | só domínio | 48 | controle do ablation |
| `v1_deepseek` | DeepSeek-R1-8B | 15 | só domínio | 32 | cross-model |
| `v1_qwen3` | Qwen3-8B | 15 | só domínio | 32 | 1ª tentativa (substituída) |
| `v1_qwen3_l19` | Qwen3-8B | **19** | só domínio | 32 | cross-model (camada boa) |

Todos com dicionário **32.768** (8× overcomplete). **Features mortas: 22–44%** — a principal
limitação anotada (candidata a melhorar com dicionário menor).

**Atualização (27/09): versão final v3.** Os v2 e o `v1_qwen3` (camada 15) foram
**descartados**. A versão final é a **v3** (todos os dados + otimização do Gao):

| preset | modelo-fonte | camada | estado |
|---|---|---|---|
| `v3_llama` | Llama-3.1-8B | 15 | **pronto** — 2 features mortas (0,006%), feature candidata **19219** |
| `v3_deepseek` | DeepSeek-R1-8B | 15 | harvest pausado (44.499 / 598.930) |
| `v3_qwen3` | Qwen3-8B | 19 | a fazer |

Os `v1_*` continuam guardados como base dos resultados anteriores até o v3 substituí-los.
""")

md(r"""<a id="4"></a>
## 4. Quais modelos no Single-vs-Iterative e quais têm SAE?

**Trabalho Single-vs-Iterative — 7 famílias** (cada uma rodada em variantes `-hf` = HuggingFace
e sem sufixo = outro backend; os experimentos de fact-checking usam as `-hf`):

Llama-3.1-8B · DeepSeek-R1-8B · Qwen3-8B · Gemma3-4B · Gemma4-e4b · Mistral-7B · Ministral-3-8B

**SAEs — só 3 dessas famílias:** Llama, DeepSeek, Qwen3 (exatamente os do cross-model e do
fatorial de fact-checking).

**Próximo passo mapeado:** os outros **4 (Gemma3, Gemma4, Mistral, Ministral)** têm resultados
de Single-vs-Iterative mas **não têm SAE** — estendê-los exige rodar *harvest + treino de SAE*
(do zero) para cada um.
""")

md(r"""<a id="5"></a>
## 5. Qual a diferença das versões dos SAEs? Por que tantos no Llama e no Qwen?

**v1 vs v2 — diferença principal é de HIPERPARÂMETROS do SAE, não de dataset.** O que cada
preset (do Llama) seta:

| | `v1_domain` | `v2_domain_only` | `v2_mixed` |
|---|---|---|---|
| **dados** | só domínio | só domínio | domínio **+ 25% texto geral (Pile)** |
| **k (features ativas/token)** | 32 | 48 | 48 |
| **peso da perda auxiliar** | 1/32 | 1/16 | 1/16 |
| **limiar de "morta"** | 1000 passos | 400 passos | 400 passos |

**Por que o v2 mudou os hiperparâmetros?** O v1 teve **33% de features mortas** (alto). O v2
tentou reduzir: k 32→48, perda auxiliar 1/32→1/16, limiar de morte 1000→400.

**Duas comparações controladas embutidas:**
1. **`v1_domain` vs `v2_domain_only`** → mesmos **dados** (só domínio), **hiperparâmetros**
   diferentes → isola: *os ajustes anti-features-mortas ajudaram?*
2. **`v2_domain_only` vs `v2_mixed`** → mesmos **hiperparâmetros**, **dados** diferentes (mixed
   acrescenta texto geral) → isola: *misturar texto geral muda a localização da abstenção?*

Ou seja: **não é (só) o dataset** — o v1→v2 mudou os hiperparâmetros (contra features mortas);
o texto geral só entra no `v2_mixed`, e o `v2_domain_only` existe pra separar as duas coisas.

**E o v3?** Não é um ajuste nosso como o v2: volta aos valores do **Gao et al. (2024)** em
tudo (inclusive k=32 e AuxK 1/32, como o v1) e adiciona a otimização do Apêndice A dele, mais
todos os dados. Foi o que resolveu as features mortas (pergunta 8).

**E os 2 SAEs do Qwen3 — é outra coisa (correção de camada).** `v1_qwen3` foi na camada 15
(ficou fraco); `v1_qwen3_l19` foi refeito na camada 19 (onde o sinal é forte) → é o que usamos.
O DeepSeek tem só 1 SAE porque não precisou de nenhum dos dois ajustes.
""")

md(r"""<a id="6"></a>
## 6. Como escolhemos a camada de cada modelo (15 no Llama, 22 no Qwen3; DeepSeek sendo refeito)?

> **Atualização (06/10/2026) — leia primeiro.** Regra adotada para todos os modelos: **a camada onde a direção de
> abstenção é mais legível na varredura** (AUC máxima; em caso de empate, a primeira do platô).
> - **Qwen3:** trocado da camada **19** para a **22** (início do platô; AUC 0,867, empatada com a 26 — diferença de 0,0003,
>   dentro do ruído; a 19 ficava 0,03 abaixo). O SAE do Qwen3 está sendo treinado na 22 (preset `v3_qwen3_l22`).
> - **Qwen3, resultado (08/10/2026):** SAE na camada 22 pronto (120.092 passos; FVU held-out 0,065; 7 features mortas).
>   Direção de abstenção: AUC held-out **0,831**. A feature **12482** lidera os dois critérios (cosseno 0,655; d de Cohen
>   1,08), como a 19219 no Llama; 87,5% de NEI no top-200; AUC 0,782; 1 feature cobre 64% da direção (Llama: 70%).
>   Publicado no HF (SAE, dados, viewer). Steering da 12482 em andamento (mesma grade do Llama).
> - **DeepSeek:** a varredura antiga usou só **40 exemplos por classe** (descuido na sessão de 22/09; os outros modelos
>   usaram 1.200). **Refeita com 1.200 por classe (06/10/2026):** a melhor camada é a **15** (AUC **0,772**), com platô de
>   0,76–0,77 da 16 até a 30 — ou seja, a camada 15 estava certa, e o DeepSeek **não** é "fraco em toda a rede" (a
>   varredura de 40 exemplos subestimava a AUC: 0,672). A camada não explica a abstenção espalhada (pergunta 19).
>   Por decisão do Lucas, o SAE do DeepSeek será **refeito do zero** (camada e ordem **a definir pelo Lucas**); o antigo foi retirado do HF.
> - O texto abaixo é o histórico de como a escolha foi feita até aqui.

Em **duas etapas** (com uma ressalva honesta no fim).

**1) Pista inicial.** Quando o Qwen3 ficou fraco na camada 15, hipotetizei que o sinal dele se
formava mais fundo — a partir de uma varredura antiga que apontava **~19–21**. O "19" começou
como palpite informado.

**2) Confirmação sistemática — `fcsae/analysis/layer_sweep.py`.** Para **cada camada**:
1. pego exemplos balanceados NEI vs SUP/REF;
2. um forward por exemplo capturando o residual no **token da decisão** em **todas as camadas**
   (`output_hidden_states=True`);
3. numa metade (treino) reconstruo a **direção de abstenção** (diff-of-means) daquela camada;
4. na outra (teste) meço o **AUC** (quão bem a direção separa NEI de decisivo).

**AUC alto numa camada = a abstenção é linearmente legível ali.** Resultado do Qwen3 (36 cam.):

| camada | 14 | 16 | **18** | **20** | 22 | **26** | 30 |
|---|---|---|---|---|---|---|---|
| AUC | 0.60 | 0.62 | **0.74** | **0.85** | 0.87 | **0.87** | 0.86 |

~acaso até a 16, **salto brusco em 18→20**, e **platô de 0.85–0.87 entre 20 e 26**.

**Ressalva honesta.** A camada 19 (AUC **0.837**) está **na região forte** (no joelho da
subida), mas o **pico exato é a camada 26 (0.867)** — não a 19. Ou seja, a 19 não é
literalmente a melhor; é uma camada **boa, no platô**, fixada pela pista inicial e confirmada
pela varredura. A diferença pro pico é pequena; refazer o Qwen3 na 26 é um ajuste fino possível.

**E o Llama e o DeepSeek (camada 15)?** A mesma varredura (`results/<preset>/layer_sweep.json`):

| modelo | exemplos por classe | melhor camada | AUC máx. | camada usada | AUC na usada |
|---|---|---|---|---|---|
| Llama-3.1-8B | 1.200 | 15 | 0,835 | 15 | 0,835 |
| Qwen3-8B | 1.200 | 26 | 0,867 | 19 | 0,837 |
| DeepSeek-R1-Distill-Llama-8B | **40** | 15 | 0,672 | 15 | 0,672 |

- **Llama:** a AUC sobe entre as camadas 10 e 13 e faz platô a partir da 15 (0,81–0,83) — a 15 é o pico. Também é uma
  camada com SAEs públicos para o Llama-3.1-8B (Llama Scope, He et al., 2024), o que dá referência externa.
- **DeepSeek:** sobe na 13–15 e faz platô 0,65–0,67 entre 15 e 22; a 15 é a melhor da varredura. **Ressalva:** essa
  varredura usou só **40 exemplos por classe** (ruidosa); com os dados completos, a AUC da direção na camada 15 é
  **0,779**. Como a camada pode influenciar o quanto a abstenção se concentra numa feature do SAE (pergunta 19), a
  varredura do DeepSeek será **refeita com 1.200 por classe** (na fila depois do Qwen3, 06/10/2026;
  `results/v3_deepseek/layer_sweep.json`).
- **Por que a camada importa:** o mesmo comportamento aparece em profundidades diferentes em cada modelo (o Qwen3 só
  "acende" a partir da 18–20), então usar a mesma camada para todos subestimaria alguns. As normas do resíduo também
  crescem ~100× com a profundidade — por isso o steering usa α/R (pergunta 16).

> O gráfico **AUC × camada** dos 3 modelos (com a camada usada marcada) está no notebook
> **`v1-historico/04-cross-model.ipynb`**, §1 (metodologia).
""")

md(r"""<a id="7"></a>
## 7. O que é o SAE v3 e o que significa "seguir o Gao et al. (2024)"?

É a versão **final** do SAE de domínio, com duas mudanças em relação ao v1:
1. **Todos os dados:** VitaminC + FEVER inteiros (598.930 claims, 132M token-ativações; o v1
   tinha 50k claims, ~11M).
2. **Receita de treino do Gao et al. (2024)**, conferida linha a linha contra o paper e o código
   oficial (`openai/sparse_autoencoder`).

**O que já seguíamos desde o v1:** TopK (k=32), 32.768 features, AuxK (α=1/32, k_aux=512),
init do encoder como transposta do decoder, decoder de norma 1.

**O que faltava e entrou no v3** (cada item com a seção do paper): normalização da entrada
(§2.1), MSE e AuxK normalizados (§A.6), Adam ε pequeno (§A.3.1), projeção do gradiente do
decoder (§A.3), b_pre = mediana geométrica (§A.1), EMA dos pesos (§A.5), lr 1e-4 (código
oficial; §3.1.2 e nota 16).

**A única divergência é o batch** (2.048 em vez de 131.072), justificada pelo próprio §A.4:
*"prior work tends to use batch sizes like 2048 or 4096 tokens"* [Bricken et al. 2023; Conerly
et al. 2024; Rajamanoharan et al. 2024], e a perda na convergência *"does not depend strongly on
batch size"* (Fig. 12). Tabela completa no `02-sae-v3.ipynb`, §3.
""")

md(r"""<a id="8"></a>
## 8. Por que as features mortas caíram de 33–48% para ~0%?

**O que aconteceu.** O 1º treino v3 usava a otimização do v1 e, com o treino longo (128k
passos), matava features sem parar: 0,2% no passo 10k → **48,6% no 35,9k**. Com a otimização
do Gao, o mesmo treino terminou com **2 mortas (0,006%)**.

**Por que (o que o paper sugere).** Não isolamos qual item pesou mais (seria preciso uma
ablação por item), mas o paper aponta três suspeitos diretos:
- **§A.3.1:** quando o gradiente é pequeno, o Adam **deixa de ser invariante à escala da
  perda**; por isso ε tem de ser muito pequeno. As features mortas só recebem gradiente da
  AuxK — e, sem normalizar a AuxK e com ε = 1e-8, esse gradiente pode ser "engolido" pelo ε.
- **Nota 16:** *"lowering learning rates helped with decreasing dead latents"* (usávamos 4×
  o lr do código oficial).
- **§A.6:** AuxK normalizada pela escala do resíduo, que diminui ao longo do treino.

**Detalhe:** no passo 10k (o tamanho do v1 inteiro), o 1º treino v3 tinha só 0,2% de mortas —
não era "mais dados piora", era a otimização que não aguentava um treino longo.
""")

md(r"""<a id="9"></a>
## 9. Por que a AUC da direção caiu no v3 (0,839 → 0,796)?

Porque **os dados mudaram**, não o SAE. Essa AUC é da **direção diff-of-means calculada no
resíduo bruto** — nenhum SAE entra no cálculo. O v1 usava uma amostra de 50k claims; o v3 usa
os 598.930 do VitaminC + FEVER completos. Nesse conjunto maior, separar NEI de SUP/REF com uma
única direção linear é mais difícil.

As métricas que **dependem do SAE** melhoraram: FVU na decisão 0,294 → 0,234; cosseno máximo
0,673 → 0,716; features mortas 33% → 0,006%.
""")

md(r"""<a id="10"></a>
## 10. A feature 19219 "liga" só quando o modelo abstém?

**Não.** Ela está ativa em **97%** das posições de decisão. O que codifica a abstenção é a
**intensidade**: a média em NEI (0,126) é ~2× a média em SUP/REF (0,064), e a taxa de NEI
cresce de forma **monotônica** com a ativação — de ~5% no decil mais baixo para **~66%** no
mais alto.

Mesmo assim, ela é **muito seletiva no topo**: 93,5% dos 200 claims que mais a ativam são NEI
(4,7× a taxa geral), e esses claims são afirmações que não dá para verificar com a evidência
dada (trivialidades, juízos subjetivos, comparações sem base).

**Como descrevê-la no texto:** "uma feature cuja intensidade na posição da decisão cresce com
a insuficiência de evidência", e não "uma feature que acende quando o modelo abstém".

**Pendente:** medir quanto ela ativa **fora** da posição de decisão, para separar "feature da
decisão modulada pela abstenção" de "feature de posição".
""")

# 11
md(r"""<a id="11"></a>
## 11. Como é feita a validação — como escolhemos o α do steering?

**A pergunta.** O steering tem um "botão de volume", o **α** (quanto empurramos a feature 19219;
reportado também como **α/R**, em múltiplos da norma do resíduo — ver pergunta 16). Pouca força não faz nada; força demais "quebra" o modelo, que
passa a gerar texto sem rótulo. A validação escolhe **quanto empurrar**, sem espiar o resultado.

**Passo a passo:**
1. **Cada célula da grade roda nos 500 claims** do AVeriTeC dev (não há rodada separada para
   a validação — a divisão em metades é feita só na análise).
2. **Metade de validação** = os 250 claims de `claim_id` **par**. Nela medimos, para cada α, a
   **coerência**: fração de respostas que contêm um dos 4 rótulos **ou** uma recusa por falta de
   informação escrita por extenso (ver pergunta 15). Isso **não olha se o rótulo está certo** — só
   se o modelo ainda está funcionando.
3. **Escolha:** para cada condição (sem evidência / Single / Iterative) e cada sentido (diminuir /
   aumentar), o **maior |α| com coerência ≥ 80%** na validação — empurrar o mais forte possível
   sem quebrar o modelo. São 6 escolhas; cada uma pode dar um α diferente.
4. **Metade de teste** = os 250 claims de `claim_id` **ímpar**. É nela que reportamos acurácia,
   F1 etc. do α escolhido, contra o α = 0 nos mesmos claims.

**Exemplo (números inventados):** Single-step, diminuir NEI — coerência na validação:
α/R −0,25 → 99%, −0,5 → 95%, −0,75 → 88%, −1 → 71%, −1,5 → 30%. O maior que passa de 80% é
**α/R = −0,75** (α = −6,28); esse é o α reportado.

**Por que é justo:** a escolha não usa a acurácia nem o gabarito, e o resultado é medido em claims
que não participaram da escolha. Escolher o α pela melhor acurácia nos 500 e reportar essa mesma
acurácia daria um número inflado.

**Referências:** selecionar o vetor/coeficiente num conjunto de **validação** é o protocolo de
Arditi et al. (2024, §2.3) e Ferrando et al. (2025, nota 5). Limitar o α pela **qualidade do
texto** vem do CAA (Panickssery et al., 2024; ver pergunta 13).
**Escolha nossa, sem referência direta:** o limiar de **80%** — por isso o notebook mostra a
curva completa, para ver o que aconteceria com 70% ou 90%.

> Detalhes e resultados: `03-steering-sae-v3.ipynb`, §2.3 e §5.3.
""")

# 12
md(r"""<a id="12"></a>
## 12. O que significa "todas as linhas usam os mesmos 250 claims da metade de teste"?

São duas ideias:

**(a) O AVeriTeC dev foi dividido em duas metades fixas** (pelo número do claim):

| metade | claims | para que serve |
|---|---|---|
| validação | os 250 de `claim_id` par | só para **escolher o α** (pergunta 11) |
| teste | os 250 de `claim_id` ímpar | para **reportar o resultado** |

Separar evita escolher o α que "deu sorte" justamente nos claims em que o resultado é reportado —
seria como montar a prova depois de ver as respostas dos alunos.

**(b) Na tabela de comparação (§5.0 do notebook 03), toda linha é calculada nos mesmos 250
claims de teste** — pipeline Single, pipeline Iterative, sem steering, diminuir NEI, aumentar NEI.
Alguns claims são mais fáceis que outros; se cada linha usasse claims diferentes, parte da
diferença de acurácia viria dos claims e não do método. Com os mesmos claims, a diferença é do
método — e isso também permite o teste de McNemar (pergunta 14), que compara acerto a acerto.

**Custo:** 250 claims em vez de 500 → estimativas menos precisas e mais difícil dar
"significativo". Por isso as **curvas** (§5.1), a tabela por célula (§5.2) e as matrizes de confusão
(§5.2.1) usam os **500** — ali não há escolha de α, então não há o que "vazar". Pelo mesmo motivo, a
comparação Single × Iterative do `01-single-vs-iterative-it.ipynb` (§8.4) usa os 500: são dois métodos
fixos, nada é escolhido.

**Alternativa que usaria os 500 no teste (não aplicada):** validação cruzada em 2 partes — escolher o α na
metade par e testar na ímpar, depois o inverso, e juntar os dois resultados. Cada claim vira "teste" uma
vez, sem ser usado para escolher o próprio α. Não exige rodar nada novo.
""")

# 13
md(r"""<a id="13"></a>
## 13. O que é CAA?

**CAA = *Contrastive Activation Addition*** — um dos métodos de steering mais citados:

> **Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. M. (2024).**
> *Steering Llama 2 via Contrastive Activation Addition.* arXiv:2312.06681.

**O que fazem:** pegam pares de exemplos **contrastantes** (a mesma pergunta com uma resposta que
tem um comportamento — p.ex. bajuladora — e outra que não tem), calculam a **diferença média das
ativações** entre eles no *residual stream* (o "vetor de steering") e, na geração, **somam esse
vetor × um multiplicador** para aumentar o comportamento (ou subtraem para diminuir). É
praticamente o que fizemos no v1 com a direção diff-of-means; no v3 usamos a direção de uma
**feature do SAE**, seguindo Ferrando et al. (2025).

**Por que citamos o CAA:** pelo achado sobre o tamanho do multiplicador:
> *"After initially exploring a wider range of multipliers, we find that steering with larger
> multipliers results in a degradation in the quality of the open-ended text … Therefore, we
> choose to limit the multiplier range to strike a balance between effectively steering the
> model's behavior and maintaining the overall quality of the generated text."*

Isso sustenta **limitar o α pela qualidade da resposta** (o critério de coerência da
pergunta 11). O CAA não fixa *qual* limiar usar — os 80% são escolha nossa.
""")

# 14
md(r"""<a id="14"></a>
## 14. Como decidimos se o steering "melhorou" ou "piorou"? O que é o "p (McNemar)"?

Cada manipulação (diminuir ou aumentar a feature) é comparada com o **α = 0 da mesma condição**
— mesmo prompt, mesma evidência; só muda o steering —, nos mesmos 250 claims de teste.

- **Métrica principal:** acurácia (a do SA). Abaixo do veredito aparecem também taxa de NEI,
  F1 por classe e erro fora de NEI.
- **▲ MELHOROU / ▼ PIOROU** só se a diferença for **estatisticamente significativa** (p < 0,05);
  senão, **= sem diferença significativa**.
- **Teste de McNemar (exato):** compara dois classificadores **nos mesmos exemplos**. Conta só os
  claims em que **exatamente um** dos dois acerta (b = só o steering acerta; c = só o baseline
  acerta) e testa se b e c diferem mais do que o acaso explicaria.
- **Por que McNemar:** Dietterich (1998) comparou testes para esse caso e mostrou que o McNemar
  tem **baixo erro do tipo I** (raramente acusa diferença que não existe), enquanto outros
  testes comuns não devem ser usados.

**O que é o "p (McNemar)".** É a probabilidade de ver uma diferença **pelo menos tão grande** quanto a
observada **se os dois fossem, na verdade, igualmente bons** (e as diferenças fossem só sorte, claim a
claim). p pequeno = difícil explicar a diferença por acaso. O corte usual é **p < 0,05** (menos de 5% de
chance); p = 5e-9 significa 0,000000005.

**Exemplo com os nossos números** (Single-step, α = 0 × α/R = −0,5, 250 claims de teste):

| | claims |
|---|---|
| os dois acertam | 84 |
| os dois erram | 103 |
| **só o steering acerta** (b) | **56** |
| **só o baseline acerta** (c) | **7** |

Os 187 claims em que os dois concordam não dizem nada sobre quem é melhor e são ignorados. Sobram 63
claims em que eles discordam; se fossem igualmente bons, cada um "ganharia" ~metade (≈ 31 a 32). O
steering ganhou 56 de 63. A chance disso acontecer por sorte (uma moeda honesta dar 56 caras em 63
lançamentos, ou algo mais extremo) é **p = 1,4 × 10⁻¹⁰** → a melhora é real. É por isso que o teste é
"exato": o p é calculado diretamente pela distribuição binomial, sem aproximação.

**Referências:** McNemar, Q. (1947). *Psychometrika, 12*(2), 153–157 · Dietterich, T. G. (1998).
Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.
*Neural Computation, 10*(7), 1895–1923.

> Onde ver: `03-steering-sae-v3.ipynb`, §5.0; e `01-single-vs-iterative-it.ipynb`, §8.4 (Single × Iterative).
""")

# 15
md(r"""<a id="15"></a>
## 15. Por que o critério de qualidade do steering mudou de "legível" para "coerente"?

**O critério original ("com rótulo"):** uma resposta só contava como "o modelo ainda funciona" se
contivesse um dos 4 rótulos (SUPPORTED, REFUTED, NOT ENOUGH EVIDENCE, CONFLICTING).

**O que apareceu nos dados (27/09/2026).** Sem evidência e com a feature **aumentada**
(α/R = +0,25), 26% das respostas não tinham rótulo. Lendo essas respostas, **não eram o modelo
quebrando** — eram **abstenções escritas por extenso**:
- *"I'm unable to find any information about the claim…"* (79 casos)
- *"Unfortunately, I don't have any evidence to review… The task requires evidence…"*
- *"I'm unable to process the evidence…"*

Ou seja: aumentar a feature fazia o modelo abster **ainda mais**, só que às vezes sem usar o
rótulo. O critério antigo contava isso como "quebrou" e penalizava o sentido "aumentar"
justamente por produzir o comportamento que queremos induzir.

**O critério novo ("coerente"):** a resposta contém um dos 4 rótulos **ou** uma **recusa por
falta de informação**, detectada por uma lista de frases (*"unable to find/verify…"*, *"I don't
have any evidence/information…"*, *"there is no evidence…"* etc.).

**Referência:** é o procedimento de **Ferrando et al. (2025, §5)**, que detectam a recusa por
desconhecimento com correspondência de frases: *"we define a set of common knowledge refusal
completions and detect if any of these occur with string matching"*. Recusa de **segurança**
(*"I cannot provide information… on illegal or harmful activities"*) **não conta** — Ferrando et
al. também separam a recusa por conhecimento da recusa por segurança.

**Cuidados, já que o critério mudou depois de vermos dados:**
1. a lista de frases foi definida a partir das 2 primeiras células sem evidência e **congelada**
   para todas as células (`REFUSAL_PATTERNS` em `fcsae/steering/factcheck_sae.py`);
2. o notebook mostra **os dois critérios** lado a lado, e qual α cada um escolheria;
3. as **métricas não mudam** (a regra do pipeline já convertia essas respostas em NEI) — muda só
   qual α é considerado "sem quebrar o modelo".

> Onde ver: `03-steering-sae-v3.ipynb`, §2.3.1, §5.1 (curvas com as duas medidas) e §5.3.
""")

# 16
md(r"""<a id="16"></a>
## 16. O que significa a fórmula do steering x ← x + α·d₁₉₂₁₉ — e de onde ela vem?

$$x \leftarrow x + \alpha \cdot d_{19219}$$

**De onde vem:** é **exatamente a Eq. 4 de Ferrando et al. (2025, ICLR)** — $x_{new} \leftarrow x + \alpha\,d_j$
—, com a mesma notação (o $d_j$ deles é o nosso $d_{19219}$). Eles mostram que aumentar/diminuir a
ativação de uma feature do SAE equivale a somar a direção dela no decoder (*activation steering*,
Turner et al., 2023). Somar numa camada só, em todas as posições, segue Arditi et al. (2024, Eq. 3).

| símbolo | o que é | tipo | valor |
|---|---|---|---|
| $x$ | o **residual stream** na camada 15, num token | vetor de 4.096 números | muda a cada token |
| $\alpha$ | o **coeficiente de steering** — o "botão de volume" (**não é bias**); o mesmo α do Ferrando | um número | α < 0 diminui a feature; α > 0 aumenta; α = 0 = sem steering |
| $d_{19219}$ | a **direção da feature 19219** (linha do decoder, comprimento 1) | vetor de 4.096 números | fixo |
| $\leftarrow$ | "substitui $x$ por…" | — | — |
| $R$ *(para interpretar α)* | o **tamanho típico de $x$** (norma média do resíduo na decisão) | um número | 8,375 |
| $\alpha/R$ *(para interpretar α)* | α em **múltiplos do tamanho do resíduo** | um número | grade de −2 a +2 |

**Em 2 passos:** (1) monta o empurrão $\alpha \cdot d$ — um vetor na direção da feature (ou oposta, se
α < 0) com comprimento |α|; (2) **soma** ao residual stream. A única operação sobre $x$ é a soma.

**Por que reportamos α/R:** o valor absoluto de α depende da escala do resíduo de cada modelo (o do
Qwen3 é ~10× maior que o do Llama). α/R diz "quanto do tamanho do resíduo" — interpretável e
comparável. É como o próprio Ferrando descreve o α deles: *"around two times the norm of the
residual stream"* (nota 5), ou seja, α/R ≈ 2. A grade é definida em α/R e convertida por
α = (α/R) × 8,375.

**Exemplo, α/R = −0,25 (α = −2,09):** empurrão de comprimento 2,09 — cerca de **25% do tamanho de
$x$** — no sentido contrário à feature de abstenção → a abstenção cai. α/R = +0,25 (α = +2,09) →
mesmo tamanho, no sentido da feature → a abstenção sobe.

> **Nota de notação:** até 27/09/2026 escrevíamos $x \leftarrow x + \beta \cdot R \cdot d$ (β = α/R).
> Trocamos para a notação literal do Ferrando para não criar um símbolo novo. Nos nomes dos
> arquivos de resultado, o valor gravado continua sendo α/R (campo interno `beta`).
>
> Onde ver: `03-steering-sae-v3.ipynb`, §2.2.
""")

# 17
md(r"""<a id="17"></a>
## 17. Que defeito foi achado na regra de rótulo do pipeline, e o que ele muda?

**O defeito.** A função que extrai o rótulo da resposta (`LLMVerdict.normalize`, do pipeline original)
procura "REFUTE" ou "FALSE" **em qualquer ponto do texto antes** de procurar "NOT ENOUGH". Os modelos
respondem o rótulo e depois explicam; a explicação típica de uma abstenção é *"NOT ENOUGH EVIDENCE —
the evidence does not … support or **refute** the claim"* → contada como **REFUTED**.

**Tamanho (Llama, Single-step, 500 claims, estimado por replay que reproduz o pipeline em 99%):**
76 abstenções lidas como REF, 45 delas com gold REF (viram "acertos"). Acurácia **0,436 → ~0,36**;
taxa de NEI **0,42 → ~0,62** com a leitura correta.

**O que muda:**
- **RQ1 (`01-single-vs-iterative-it.ipynb`):** números absolutos enviesados a favor de REF; aviso no
  topo e estimativa corrigida na §7. A comparação Single × Iterative precisa ser revista.
- **SA / artigo do ENIAC (Ollama):** não dá para verificar — os JSONL não guardam o texto das
  respostas. Registrado como limitação.
- **Steering (`03-steering-sae-v3.ipynb`):** a métrica principal passou a ser o **rótulo que o modelo
  produziu, ignorando a explicação** — contando como rótulo só a expressão completa em MAIÚSCULAS
  (SUPPORTED, REFUTED, NOT ENOUGH EVIDENCE, CONFLICTING EVIDENCE), que é como o modelo escreve o veredito
  (em minúsculas ele está só falando dos rótulos): (1) se a resposta começa com um rótulo, usa esse rótulo — quase
  sempre o caso com evidência; (2) se não começa (sem evidência o modelo às vezes raciocina antes), usa
  o primeiro rótulo que aparece — **ignorando enumerações das opções** como *"I cannot determine whether
  the claim is SUPPORTED, REFUTED, NOT ENOUGH EVIDENCE, or CONFLICTING…"*, que são o modelo citando o formato,
  não decidindo (achado no viewer em 28/09; afeta no máximo 8/500 respostas por célula, nb03 §3.1.1);
  (3) sem rótulo nenhum → NEI (p.ex. recusas por extenso). A regra do
  pipeline fica ao lado. Com ela, o efeito do steering fica mais claro
  (p.ex. aumentar a feature com evidência passa a aumentar a abstenção, como esperado).

**Nada foi re-rodado:** as células de steering guardam o texto completo das respostas; basta
reclassificar.
""")

# 18
md(r"""<a id="18"></a>
## 18. Por que a grade de steering foi cortada em |α/R| ≤ 1?

A grade planejada ia até α/R = ±2. Com os dados sem evidência vimos que **em |α/R| = 1 o modelo já
degenera**: com α/R = +1, 91% das respostas entram em *loop* (*"I have no information about this
claim…"* repetido) até o limite de 1.536 tokens; com α/R = −1, 30%. Doses maiores só degenerariam
mais, nenhum critério de qualidade as escolheria, e cada célula levaria ~7 h.

**Referência:** é o procedimento do CAA (Panickssery et al., 2024): explorar multiplicadores, observar
a degradação do texto e **limitar a faixa** (pergunta 13).

> Onde ver: `03-steering-sae-v3.ipynb`, §2.3.
""")

# 19
md(r"""<a id="19"></a>
## 19. No DeepSeek a abstenção ficou espalhada em várias features. A literatura trata disso? Como fazemos o steering?

> **Atenção (06/10/2026):** os resultados do DeepSeek nesta pergunta são do **SAE antigo (camada 15)**, que foi descartado e
> será refeito do zero a partir de uma nova varredura de camadas (pergunta 6). Valem como histórico do método, não como resultado final.

**O que vimos (01/10/2026, `fcsae/analysis/spread.py`, resultados em `results/<preset>/spread.json` no store).**
Separação NEI × decidido (AUC no teste) usando só as k features de maior |Cohen's d|, e fração da direção
de abstenção (diff-of-means) coberta pelas k linhas do decoder mais alinhadas:

| k features | AUC Llama | AUC DeepSeek | cobertura Llama | cobertura DeepSeek |
|---|---|---|---|---|
| 1 | **0,784** | 0,689 | **70%** | 45% |
| 5 | 0,808 | **0,780** | 82% | 73% |
| 10 | 0,815 | 0,799 | 93% | 85% |
| todas (teto) | 0,850 | 0,819 | — | — |

No Llama **uma** feature (19219) já chega perto do teto; no DeepSeek são precisas **~5** (7531, 32642 e 18658,
que sobem com NEI; 19789 e 25349, que sobem quando dá para decidir).

**O que a literatura diz** (verificado nas fontes em 01/10/2026; conferir as citações internas no PDF antes de
usar no texto):
- **É esperado, não anomalia.** *Feature splitting*: um conceito se divide em várias features à medida que o
  dicionário cresce (Bricken et al., 2023; Templeton et al., 2024 — *"we suspect that there are many features
  representing different kinds of errors"*). *Feature absorption*: uma feature aparentemente pura deixa de
  disparar em parte dos casos porque outras "absorvem" esse pedaço (Chanin et al., 2024, arXiv:2409.14507).
- **Cuidado ao comparar modelos.** SAEs treinados no mesmo modelo e nos mesmos dados, mudando só a semente,
  aprendem conjuntos de features diferentes (Paulo & Belrose, 2025, arXiv:2501.16615); se uma feature isolada
  captura um comportamento depende também dos dados do SAE (Kissane et al., 2024, LessWrong — não revisado
  por pares). Então **não** dá para afirmar "o DeepSeek representa a abstenção de forma mais distribuída";
  o correto é "no nosso SAE ela aparece espalhada" → limitação.
- **Steering com uma feature só** funciona quando há uma feature forte (O'Brien et al., 2024,
  arXiv:2411.11296; Ferrando et al., 2025, que notam efeito mais fraco em outro modelo; Durmus et al., 2024).
- **Várias features combinadas** têm precedente, mas não idêntico ao nosso caso: o **FGAA** (Soo et al., 2025,
  arXiv:2501.09929) parte da diferença média das ativações do SAE entre exemplos com e sem o comportamento,
  **descarta features densas** (ativas em > 1% dos tokens — a 7531 está ativa em 99% das decisões), fica com as
  n₁ mais positivas (na prática **descarta as negativas**, n₂ = 0, porque pioravam o resultado) e passa por um
  "effect approximator" treinado (do SAE-TS, Chalnev et al., 2024, arXiv:2411.02193) antes de somar ao resíduo.
  Supera o steering com uma feature e o CAA no Gemma-2. O'Brien et al. (apêndice) também combinam duas features.
  Uma combinação simples das nossas 5 features seria um método **inspirado** nesses, não o FGAA em si.
  Arad et al. (2025, arXiv:2505.20063): escolher as features pelo **efeito na saída**, não só pela detecção.
- **A direção média (diff-of-means / CAA), sem SAE, é a comparação mais forte**: com uma feature de SAE por
  conceito, o steering é pior que a direção média (AxBench, Wu et al., 2025, arXiv:2501.17148: *"SAEs are not
  competitive"*; Kantamneni et al., 2025, arXiv:2502.16681); Arditi et al. (2024) controlam a recusa com uma
  direção média em 13 modelos.

**Decisão (Lucas, 01/10/2026):** comparar, **nos dois modelos**, o steering com a **direção de abstenção
(diff-of-means)** — assim a comparação entre modelos usa o mesmo método e responde "a feature do SAE é tão
boa quanto a direção média?" (a pergunta do AxBench). A variante com várias features fica em aberto (ver acima:
o FGAA não se aplica diretamente). Planejamento e resultados: notebooks de steering.
""")

# 20
md(r"""<a id="20"></a>
## 20. Steering com a feature × steering com a direção média: qual a diferença? O que fizemos antes é descartado?

> **Atenção (06/10/2026):** os resultados do DeepSeek nesta pergunta são do **SAE antigo (camada 15)**, que foi descartado e
> será refeito do zero a partir de uma nova varredura de camadas (pergunta 6). Valem como histórico do método, não como resultado final.

**Nada é descartado.** O steering com a feature 19219 no Llama (`03-steering-sae-v3.ipynb`) continua sendo o
resultado do método com SAE. A direção média é um experimento **a mais**, de comparação.

| | **feature do SAE** (o que já fizemos) | **direção média** (diff-of-means) |
|---|---|---|
| de onde vem a seta $d$ | uma linha do **decoder do SAE**: uma das 32.768 "peças" em que o SAE decompõe o estado interno, escolhida pelas métricas (cosseno com a direção, Cohen's d, AUC) | **sem SAE**: média do estado interno na posição do veredito nos claims NEI **menos** a média nos SUP/REF (metade de treino), normalizada |
| o que capta | só o que está **nessa feature** (no Llama, ~70% da direção de abstenção; no DeepSeek, ~45%) | o sinal de abstenção **inteiro**, esteja numa feature ou espalhado |
| vantagem | **interpretável**: dá para ver o que a feature é (viewer, exemplos) | método mais simples e mais forte da literatura (Arditi et al., 2024; Panickssery et al., 2024; AxBench, Wu et al., 2025) |
| no steering | x ← x + α·d, mesma grade α/R, mesmas 3 condições, mesmos 500 claims | **idem — só troca d** |

**Não é "escolher a feature pela média":** a direção média não é uma feature; ela nem passa pelo SAE.

**A pergunta que a comparação responde:** a feature interpretável funciona tão bem quanto a direção média?
No Llama (feature forte) esperamos resultados parecidos; no DeepSeek (abstenção espalhada, pergunta 19), a
direção deve funcionar melhor que uma feature só.

**Status (01/10/2026):** implementado (`--direction abstention` em `fcsae/steering/factcheck_sae.py`; direção
em `results/<preset>/abstention_direction.safetensors`, AUC held-out 0,796 no Llama e 0,779 no DeepSeek, iguais
às da análise). A grade foi iniciada no DeepSeek e **pausada** pelo Lucas no mesmo dia, para priorizar os SAEs
dos outros modelos (ver se a abstenção espalhada se repete). Retomar: `bash scripts/run_dirabst_queue.sh`.
""")

# 21
md(r"""<a id="21"></a>
## 21. Quantas features o SAE deve ter (32.768)? Como justificar esse hiperparâmetro?

O número de features — o **tamanho do dicionário**, *n* (ou *width*) — é um **hiperparâmetro**: escolhido antes do
treino, não aprendido. O encoder leva o resíduo (d = 4.096) a *n* números e o decoder faz o caminho de volta. Usamos
**n = 32.768 = 8 × d** ("8× overcomplete"), com **k = 32** features ativas por token (outro hiperparâmetro).

**O que a literatura diz** (verificado nas fontes em 01/10/2026; conferir as citações no PDF antes de usar no texto):
- **Não existe um n "ótimo" derivado de princípios.** Gao et al. (2024, arXiv:2406.04093) ajustam leis de escala:
  o erro sempre cai com n maior, e o número de tokens até convergir cresce ~Θ(n^0,6); mas chamam a comparação de
  n diferentes num orçamento fixo de *"arguably unprincipled"*. Ou seja: **n se justifica por convenção e por
  testes de robustez**, não por uma fórmula.
- **32.768 com k = 32 é a convenção que seguimos:** o SAE oficial da OpenAI (repositório do Gao) para o GPT-2 é
  `v5_32k`: *"2**15 = 32768"* features com *"TopK(32)"*; o **Llama Scope** (He et al., 2024, arXiv:2410.20526) treina
  para o **Llama-3.1-8B** SAEs *"8x (32K features) and 32x (128K features)"* — inclusive o **L15R-8x**, mesma família,
  mesma camada (15) e mesmo tamanho que o nosso; O'Brien et al. (2024, arXiv:2411.11296) usam *"k = 32 and an
  expansion factor of 8"*. Gemma Scope (Lieberum et al., 2024, arXiv:2408.05147) publica de 16K a 1M.
- **Quantidade de dados:** ~115 milhões de tokens / 32.768 ≈ **3.500 tokens por feature** (conta nossa). É
  comparável ao único trabalho de SAE de domínio que achamos — O'Neill et al. (2025, arXiv:2508.09363), dados
  médicos, ~49 milhões de tokens, de 2^14 a 2^20 features — e bem abaixo do Gemma Scope (4–16 bilhões). Nenhum
  trabalho dá uma regra explícita de "tokens por feature"; reportamos a convergência e as features mortas (nb02).
- **O tamanho muda o tipo de feature (*splitting*):** Bricken et al. (2023) foram de 512 a 131.072 features e
  mostram conceitos se dividindo (*"the number of features specific to base64 contexts goes from 1 to 3 to many
  more"*); Templeton et al. (2024): *"San Francisco"* vira 2 features no SAE de 4M e 11 no de 34M. Tamanhos
  diferentes dão *"different 'resolutions'"* do mesmo objeto.
- **A absorção piora com n maior:** Chanin et al. (2024, arXiv:2409.14507): *"absorption rate increases with higher
  sparsity and higher SAE width"*; SAEBench (Karvonen et al., 2025, arXiv:2503.09532), com 4k/16k/65k: *"Absorption
  scores worsen with increased dictionary size for all architectures except Matryoshka"*.
- **Testar um segundo tamanho é prática comum:** Bricken e Templeton comparam tamanhos; o Gemma Scope publica uma
  suíte de *feature splitting* (16K a 524K); o Llama Scope compara 8× e 32× na camada 15.
- **Semente:** Paulo & Belrose (2025, arXiv:2501.16615): só ~30% das features se repetem entre sementes, e SAEs TopK
  são *"more seed-dependent"* — controlar a semente ao comparar tamanhos.
- **Alternativa ao problema de splitting/absorção — Matryoshka SAE** (Bussmann et al., 2025, arXiv:2503.17547):
  treina dicionários **aninhados** de tamanhos crescentes, forçando os menores a reconstruir sozinhos; no SAEBench
  é a única arquitetura cuja absorção quase não piora com o tamanho.

**Como justificamos n = 32.768:** seguimos a convenção de Gao et al. (release oficial com n = 32.768, k = 32) e do
Llama Scope (8× no Llama-3.1-8B, camada 15). Como a escolha é convencional e afeta se um conceito aparece numa
feature ou em várias, **testamos a robustez** com um segundo tamanho (n = 8.192, mesmo k) e, idealmente, uma segunda
semente — ver pergunta 19 e o experimento correspondente.
""")

# 22
md(r"""<a id="22"></a>
## 22. No DeepSeek, quais features controlam a abstenção? O que a triagem de steering mostrou?

> **Atenção (06/10/2026):** os resultados do DeepSeek nesta pergunta são do **SAE antigo (camada 15)**, que foi descartado e
> será refeito do zero a partir de uma nova varredura de camadas (pergunta 6). Valem como histórico do método, não como resultado final.

**O experimento (02–04/10/2026).** Steering, **sem evidência**, com cada uma das 5 features de maior |Cohen's d|
do SAE do DeepSeek (pergunta 19), mesma fórmula e mesmos 500 claims do Llama (`03-steering-sae-v3.ipynb`). Primeiro
uma triagem em α/R = ±0,5; depois a grade completa (−1 a +1) de cada uma. Rótulo lido com o 1º rótulo escrito
(§3.1 do nb03). Resultados em `results/v3_deepseek/steer_sae/none__feat<id>__b<α/R>.json`.

**Triagem (taxa de NEI; sem steering = 0,546; ≥ 93% de respostas coerentes em todas):**

| feature | tipo (sinal do d) | α/R = −0,5 | α/R = +0,5 | variação (+0,5 − −0,5) | controla? |
|---|---|---|---|---|---|
| **25349** | espelho (d < 0) | 0,608 | **0,340** | **−0,27** | sim — o maior efeito |
| **32642** | sobe com NEI (d > 0) | **0,394** | 0,610 | **+0,22** | sim |
| **18658** | sobe com NEI (d > 0) | 0,446 | 0,624 | +0,18 | sim |
| 19789 | espelho (d < 0) | 0,524 | 0,406 | −0,12 | só de um lado |
| 7531 | sobe com NEI (d > 0) | 0,508 | 0,482 | −0,03 | **não** |

**O que isso mostra** (Lucas, 04/10/2026):
1. **Três features controlam a abstenção nos dois sentidos**, cada uma no sentido esperado pelo tipo: nas que
   **sobem com NEI** (32642 e 18658), **aumentar** a feature dá **mais** abstenção; na **espelho** (25349),
   **aumentar** dá **menos** abstenção.
2. **A 25349 tem o maior efeito:** com +0,5, o NEI cai de 0,546 para **0,340**, e a acurácia sobe de 0,296 para
   0,380.
3. **A 7531, que liderava as métricas de correlação (maior cosseno e maior Cohen's d), é a única sem efeito.** Isso
   reforça que **correlação não garante controle** e que só o steering separa uma coisa da outra — o mesmo ponto de
   Arad et al. (2025, arXiv:2505.20063): as features que detectam um conceito nem sempre são as que mudam a saída.
4. **No DeepSeek, o controle da abstenção está distribuído em várias features.** É coerente com a análise de
   espalhamento (pergunta 19): no Llama uma feature (19219) faz tudo; no DeepSeek são várias, cada uma com efeito
   parcial. Isso abre a pergunta de se **combinar** essas features (p.ex. 32642 diminuída + 25349 aumentada) daria
   um controle tão forte quanto o da 19219 no Llama.

**Detalhes das grades completas já fechadas:**
- **7531:** NEI entre 0,48 e 0,55 nas doses em que o modelo responde bem; em ±1 degenera (≈300 respostas em
  *loop*), o que infla o NEI.
- **32642:** monotônica — −0,5: 0,394 · 0: 0,546 · +0,5: 0,610 · +0,75: 0,690 · +1: 0,822 (84% coerentes); em −0,75
  e −1 o modelo degenera (loops).
- **19789:** espelho só do lado positivo — +0,75: **0,268** (95% coerentes), o maior "menos abstenção" visto; do
  lado negativo fica em 0,52–0,55 até degenerar em −1.

- **18658 (grade completa, 05/10/2026):** −1: **0,128** · −0,75: 0,394 · −0,5: 0,446 · −0,25: 0,514 · 0: 0,546 · +0,25: 0,568 ·
  +0,5: 0,624 · +0,75: **0,738** · +1: 0,938 — monotônica e **coerente em ≥ 95% de −1 a +0,75** (78% em +1, com loops): a
  maior faixa controlável do DeepSeek e **a melhor candidata** até aqui (era só a 4ª em Cohen's d e a 21ª em cosseno).
- **19789 (grade completa):** espelho só do lado positivo; +0,75: 0,268; lado negativo sem efeito até degenerar.
- **Ordem das candidatas:** a lista das 5 testadas veio do Cohen's d na **metade de treino**; com **todos os dados**, as 4
  primeiras se mantêm, mas da 5ª em diante os valores empatam (0,413–0,423) e a ordem muda: 3424 vira a 5ª, 5393 a 6ª e a
  25349 a 7ª. A 3424 e a 5393 **não foram testadas**.

**Status (05/10/2026, decisão do Lucas):** DeepSeek pausado para seguir com o Qwen3. **Pendentes para o final** (LEIA-ME,
seção "PENDÊNCIAS PARA O FINAL"): doses ±0,25/±0,75/±1 da 25349; 3424 e 5393; steering **com evidência** (Single e
Iterative) na melhor feature; combinação de features. Publicado: painéis de steering das 5 features no viewer e nota no card
do SAE do DeepSeek.
""")

# 23
md(r"""<a id="23"></a>
## 23. Como o cosseno entre uma feature e a direção de abstenção é calculado? O que ele significa?

O estado interno do modelo na posição do veredito é um vetor de 4.096 números — um ponto num espaço de 4.096
direções. O cosseno compara **duas setas** nesse espaço:

1. **A direção de abstenção** $v$ (sem SAE): $v = \text{média}(h \mid \text{NEI}) - \text{média}(h \mid \text{SUP/REF})$,
   na metade de treino das decisões — aponta "de decidir para abster". Antes da comparação, $v$ é **centralizada**
   (subtrai-se a média dos seus 4.096 componentes), porque o SAE recebe as entradas centralizadas e normalizadas
   (Gao et al., 2024, §2.1).
2. **A direção da feature** $d_j$: a linha $j$ do **decoder** do SAE — a seta que a feature acrescenta ao estado
   interno quando dispara (a reconstrução é $\hat{x} = \sum_j z_j\, d_j + b$).

$$\cos(d_j, v) = \frac{d_j \cdot v}{\lVert d_j \rVert\, \lVert v \rVert}$$

| cosseno | ângulo | leitura |
|---|---|---|
| 1 | 0° | apontam para o mesmo lado |
| 0 | 90° | não têm relação |
| −1 | 180° | apontam para lados opostos (feature "espelho") |

**Nossos valores:** 19219 (Llama) 0,72 (~44°); 7531 (DeepSeek) 0,35 (~70°); 19789 (DeepSeek) −0,45.

**Escala de referência:** em 4.096 dimensões, duas direções ao acaso têm cosseno perto de 0 (desvio típico
≈ 1/√4.096 ≈ 0,016). Então 0,35 já é muito acima do acaso, e 0,72 é excepcional.

**Por que usamos:** o steering empurra o modelo **ao longo de** $d_j$ (Ferrando et al., 2025, Eq. 4); se $d_j$ aponta
para o mesmo lado de $v$, espera-se que mexa na abstenção. É uma **expectativa**, não uma garantia: no DeepSeek a
7531 tem o maior cosseno positivo e não controla a abstenção (pergunta 22).

> Código: `fcsae/analysis/abstention.py` (cosseno de todas as features) e `fcsae/analysis/selection_map.py`
> (dados do gráfico "Cohen's d × cosine" do viewer).
""")

md(r"""<a id="24"></a>
## 24. O que é a "ativação" de uma feature (o eixo "how strongly the feature fires" do viewer)?

**Em uma frase:** é um número ≥ 0 que diz **o quanto a feature está presente** no estado interno do modelo no
momento em que ele vai escrever o veredito. 0 = a feature não disparou; quanto maior, mais forte ela disparou.

**Passo a passo, para um claim:**

1. O modelo lê o prompt (claim + evidências). Na posição do veredito, na camada escolhida (Llama: 15), pegamos o
   estado interno $x$: um vetor de 4.096 números.
2. **Normalização** (Gao et al., 2024, §2.1): tiramos a média dos 4.096 números e dividimos pelo tamanho do vetor,
   de modo que o estado vira uma seta de **comprimento 1**.
3. O **encoder** do SAE calcula um número para cada uma das 32.768 features:
   $\text{pré}_j = (x - b_{dec}) \cdot W_{enc,j} + b_{enc,j}$.
4. **TopK:** só as **32 maiores** sobrevivem (passando por ReLU, então ficam ≥ 0); as outras 32.736 viram 0.
   O número que sobra para a feature $j$ é a **ativação** $z_j$.

**Como interpretar a escala:** como a entrada tem comprimento 1 e cada direção do decoder $d_j$ também tem
comprimento 1, a reconstrução é $\hat{x} = \sum_j z_j\, d_j + b_{dec}$ — ou seja, a ativação é **quanto da direção
$d_j$ entra na soma**. Uma ativação de 0,16 significa, grosso modo, que a feature contribui com um "pedaço" de
comprimento 0,16 num estado de comprimento 1 (aproximado, porque as direções das features não são perpendiculares
entre si). Por isso os valores são pequenos (0 a ~0,4 na 19219) e **só fazem sentido comparados dentro da mesma
feature** — 0,16 numa feature não é "a mesma força" que 0,16 em outra.

**No gráfico do viewer:** cada claim tem uma ativação para a feature; agrupamos os claims em 10 faixas de ativação de
mesma largura e mostramos, em cada faixa, a % de claims NEI. Os claims com ativação 0 (a feature não ficou entre as
32 do TopK) ficam fora das barras e são contados no título do eixo. Na 19219: de 6% de NEI na faixa mais fraca a
97% nas mais fortes; a partir de 0,16, 67% dos claims são NEI (37.933 de 56.699).

> Código: `fcsae/sae/model.py` (`normalize_input`, `encode_pre`, `_topk`).
""")

md(r"""<a id="25"></a>
## 25. Qual a diferença entre "feature" e "ativação"?

- **Feature = o QUÊ.** Um "detector" fixo que o SAE aprendeu no treino: há 32.768, cada um com um número (19219,
  14953…). É **o mesmo para todos os claims**: a mesma direção $d_j$ no espaço interno do modelo, o mesmo significado.
- **Ativação = O QUANTO.** A leitura desse detector **num claim específico**, $z_j \geq 0$. Muda de claim para claim
  (pergunta 24).

**Analogia — mesa de som com 32.768 canais:** cada canal é uma feature (o canal 19219 é sempre o mesmo
"instrumento"); o volume desse canal em cada música (cada claim) é a ativação. A cada momento, só 32 canais podem estar
com volume acima de zero (TopK).

| | feature 19219 |
|---|---|
| claim com evidência clara | ativação ≈ 0,02 (quase muda) |
| claim sem evidência que decida | ativação ≈ 0,25 (forte) |
| claim em que ela não entrou no TopK | ativação 0 |

**No viewer:** a página é de uma **feature**; o gráfico mostra como a **ativação** dela varia entre os 598.930 claims
(e quantos são NEI em cada faixa); "fires on 97,1% of answers" = ativação > 0 em 97,1% dos claims. O **steering**
empurra a **direção da feature** no estado do modelo — como subir ou baixar o volume desse canal na mão.
""")

md(r"""<a id="26"></a>
## 26. Como a ativação é medida a partir da direção? Como sei se ela é forte ou fraca?

**1. A medida é uma projeção.** Para cada feature $j$, o encoder do SAE tem um vetor $w_j$ (4.096 números) — uma
"régua" apontada numa direção. A ativação é o quanto o estado (normalizado) avança ao longo dessa régua:

$$z_j = (x - b_{dec}) \cdot w_j + b_{enc,j}, \quad \text{depois ReLU e TopK (32 maiores)}$$

Exemplo em 2 dimensões, estado $x = (0{,}8;\ 0{,}6)$:

| régua da feature | conta | ativação |
|---|---|---|
| → (1; 0) | 0,8·1 + 0,6·0 | 0,8 (aponta bastante para lá) |
| ↑ (0; 1) | 0,8·0 + 0,6·1 | 0,6 |
| ← (−1; 0) | −0,8 → ReLU | 0 (aponta para o lado oposto) |

Intuição: é a "sombra" do estado sobre a régua — longa se apontam para o mesmo lado, nula se apontam para lados
opostos. Como o estado tem comprimento 1 (pergunta 24), os valores são pequenos.

**2. Forte ou fraca é relativo à própria feature** — não há limiar universal. Na 19219 (Llama), entre os claims em
que ela dispara:

| ativação | posição |
|---|---|
| ≤ 0,069 | metade dos claims (mediana) — típica/fraca |
| ≤ 0,158 | 90% dos claims |
| > 0,256 | só o 1% mais forte |
| 0,394 | máximo |

No gráfico do viewer o eixo x vai do mínimo ao máximo **daquela** feature: quanto mais à direita, mais forte.

> Código: `fcsae/sae/model.py` (`encode_pre`, `_topk`). Quantis calculados do `hist_by_label` do JSON da feature no
> dataset do viewer.
""")

md(r"""<a id="27"></a>
## 27. O que é, em detalhe, a "régua" de uma feature (o vetor do encoder)?

A régua da feature $j$ é a **coluna $j$ da matriz do encoder** $W_{enc}$: 4.096 números, um por dimensão do estado.

**1. É aprendida, não escrita.** O treino tem uma só tarefa: reconstruir o estado do modelo usando 32 features por vez.
Para cada feature o gradiente ajusta duas setas: a **régua** $w_j$ (lê: "o quanto ela está presente?") e a **peça**
$d_j$ (escreve: "o que somo de volta?"). A régua começa como cópia da peça (inicialização amarrada,
`fcsae/sae/model.py`) e se ajusta. O **significado vem depois**, olhando em quais claims ela dá valores altos.

**2. Nenhum número sozinho tem significado.** Na 19219 (Llama), as maiores componentes em módulo estão nas
dimensões 1291 (−0,205), 1747 (−0,168), 772 (+0,145), mas as 50 maiores somam só **15%** do peso ($\sum w^2$): é
um **padrão espalhado**, não um "neurônio da abstenção".

**3. A conta completa:** (i) normaliza o estado (comprimento 1); (ii) produto escalar com a régua;
(iii) soma o viés $b_{enc,19219} = -0{,}048$ (um "custo de entrada"); (iv) ReLU; (v) TopK — só as 32 maiores das
32.768 ficam. O que sobra é a ativação.

**4. Régua ≠ peça, e é normal.** $\cos(w_{19219}, d_{19219}) = 0{,}52$ (~58°); a mediana entre todas as 32.768
features é **0,54** (percentil 10: 0,47). Como há 32.768 features em 4.096 dimensões, elas se sobrepõem; a régua
aprende a apontar um pouco de lado para não confundir a feature com as vizinhas (analogia: o microfone que grava o
violino é inclinado para longe do violoncelo ao lado).

| | régua $w_j$ (encoder) | peça $d_j$ (decoder) |
|---|---|---|
| função | ler → ativação | escrever → o que a feature soma ao estado |
| usada em | ativações, gráfico do viewer, AUC, Cohen's d | steering ($x \leftarrow x + \alpha d_j$), cosseno com a direção de abstenção (pergunta 23) |

> Números: `checkpoints/v3_llama/sae.safetensors` (`W_enc[:, 19219]`, `b_enc[19219]`, `W_dec[19219]`).
""")

md(r"""<a id="28"></a>
## 28. O que o steering da feature 12482 mostrou no Qwen3?

Mesmo protocolo do Llama (500 claims do AVeriTeC dev; rótulo = `first_label`; α/R de −1 a +1; Iterative só até ±0,75
— as doses ±1 ficaram pendentes, LEIA-ME #9). Porcentagem de cada veredito respondido:

| α/R | Sem evidência (acc · SUP/REF/NEI) | Single (acc · SUP/REF/NEI) | Iterative (acc · SUP/REF/NEI) |
|---|---|---|---|
| −0,75 | 0,244 · 100/0/0 | 0,244 · 100/0/0 | 0,244 · 100/0/0 |
| −0,50 | 0,244 · 100/0/0 | 0,272 · 96/4/0 | 0,300 · 93/7/0 |
| −0,25 | 0,244 · 100/0/0 | 0,288 · 94/6/0 | 0,334 · 89/11/0 |
| **0** | **0,450 · 0/70/30** | **0,400 · 13/30/57** | **0,452 · 22/37/42** |
| +0,25 | 0,524 · 0/84/16 | 0,310 · 0/43/57 | 0,366 · 0/58/42 |
| +0,50 | 0,556 · 0/89/11 | 0,362 · 0/49/51 | 0,398 · 0/62/38 |
| +0,75 | 0,516 · 0/81/19 | 0,354 · 0/49/51 | 0,388 · 0/61/39 |

**Leitura (preliminar):** diminuir a 12482 faz o modelo responder **SUPPORTED a quase tudo** — sem evidência, ele chega
a escrever *"the task assumes that the evidence is sufficient to support the claim"*. Aumentar elimina o SUP, mas o NEI
**não sobe** (com evidência fica em ~40–57%; sem evidência até cai): o que cresce é o REF. Ou seja, a 12482 parece
codificar **"a evidência não sustenta a alegação"**, e não a abstenção em si — diferente da 19219 do Llama, que aumenta e
diminui o NEI. Nenhuma dose supera o modelo sem intervenção em acurácia com evidência.

**Cuidado com α/R = ±1:** só ~52% das respostas são coerentes em +1 (texto degenera; resposta sem rótulo vira NEI pela
regra) — não interpretar o NEI dessas células.

> Dados: `results/v3_qwen3_l22/steer_sae/`; viewer: página da feature 12482 (Qwen3-8B · layer 22), tabela por dose e
> "browse all 500 claims".
> Tabela completa (grade inteira, regra de rótulo v2, dose também em α/R_c): **nb03 §5.5**.
""")

md(r"""<a id="29"></a>
## 29. O que deu errado no treino do SAE do Gemma3 (perda "nan") e como foi corrigido?

**O que aconteceu (11/10/2026).** A coleta das ativações do Gemma-3-4B (camada 19, 598.930 prompts) terminou sem erro,
mas no treino do SAE a perda virou `nan` entre os passos 600 e 700 (FVU vinha caindo normalmente: 0,26 → 0,14).

**Por quê.** O Gemma3 tem uma *massive activation* (Sun et al., 2024): a **dimensão 443** do resíduo vale ≈ 32.768 em
**100% dos tokens** e sozinha responde por **98,8% da norma²** do vetor (as outras 2.559 dimensões ficam na casa das
dezenas). As ativações eram gravadas em **float16**, cujo maior valor representável é **65.504**; o maior valor
observado foi 50.432, e em alguns tokens (0 a 57 por shard de 500 mil) o valor passou do teto e virou `inf`. Um único
`inf` num lote contamina a perda e os pesos → `nan`.

| modelo (camada) | maior \|valor\| nas decisões | teto do float16 |
|---|---|---|
| Llama-3.1-8B (15) | 5,4 | 65.504 |
| Qwen3-8B (22) | 88 | 65.504 |
| **Gemma-3-4B (19)** | **50.432** (e `inf` em alguns tokens) | 65.504 |

Llama e Qwen3 **não** foram afetados (além dos valores acima, os dois treinos terminaram com perda finita após duas
passadas completas pelos dados). A **varredura de camadas** do Gemma3 também não: ela usa as ativações direto do
modelo, sem passar pelo float16.

**Como foi corrigido.** As ativações dos Gemma passaram a ser gravadas em **bfloat16** (`dtype_store="bfloat16"` nos
presets `v3_gemma3`/`v3_gemma4`; `STORE_DTYPE` em `fcsae/harvest/harvest.py`). O bfloat16 é o formato em que o próprio
modelo calcula (o `compute_dtype` da quantização nf4), vai até ~10³⁸ e, portanto, não estoura nem perde informação em
relação ao que o modelo produz. Todas as etapas que leem as ativações já convertiam para float32 antes de calcular;
Llama, Qwen3 e DeepSeek continuam em float16 (sem mudança). Teste com 400 prompts: 0 valores não finitos e treino
finito. A coleta em float16 (589 GB) e o checkpoint corrompido foram apagados, e o ciclo foi relançado (~15 h de
coleta + treino).

**Ponto em aberto para o steering dos Gemma.** A dose é α/R, com R = norma média do resíduo na decisão. No Gemma3 essa
norma (~33.000) é quase toda da dimensão 443, que é praticamente constante; α/R no Gemma não seria comparável ao dos
outros modelos. A forma de calcular R para os Gemma será decidida antes do steering.
""")

md(r"""<a id="30"></a>
## 30. "Ativação" do modelo × ativação de uma feature: de onde até onde vai cada número?

A palavra "ativação" é usada em dois sentidos neste projeto:

**1. Ativação do modelo (resíduo).** O estado interno num token: um **vetor** (2.560 números no Gemma3; 4.096 no
Llama/Qwen3/DeepSeek). Cada número é real (positivo ou negativo) e **não tem faixa fixa**: o resíduo é uma **soma** —
cada camada soma sua contribuição ao vetor da camada anterior (conexão residual) — e nenhuma função o "aperta" (as
normalizações RMSNorm só normalizam a *entrada* de cada subcamada). Por isso os valores crescem com a profundidade
(Llama: norma média 0,77 na camada 0 → 139 na última). Faixa observada nas posições de decisão:

| modelo (camada) | maior \|valor\| |
|---|---|
| Llama (15) | 5,4 |
| Qwen3 (22) | 88 |
| Gemma3 (19) | dezenas na maioria das dimensões; **dim. 443 ≈ 32.768** (até 50.432) — *massive activation* |

Foi um desses números (Gemma3, dim. 443) que passou do teto do float16 (65.504) e virou `inf` (pergunta 29).

**2. Ativação de uma feature do SAE.** Um **único número ≥ 0** por feature (pergunta 24), pequeno (0 a ~0,4 na
19219), porque o SAE recebe o resíduo **normalizado** para comprimento 1. Não corre risco de estourar.

**Relação:** modelo → resíduo (vetor sem limite) → normalização → SAE → ativações das features (≥ 0, pequenas).
""")

md(r"""<a id="31"></a>
## 31. Por que a dose do steering passou a usar a norma centrada ($R_c$) e não a norma média?

A dose é $\alpha/R$: o deslocamento $\alpha$ (a direção da feature tem comprimento 1) dividido por uma escala $R$ do
resíduo na posição de decisão. Até o Qwen3 usávamos a **norma média**, $\mathbb{E}\lVert x\rVert$. Ela mistura a parte
do vetor que é **igual em todas as alegações** (a média $\mu$, que não diz nada sobre a alegação) com a **variação**
entre alegações. No Gemma3 a parte comum é enorme (dim. 443 ≈ 32.768, *massive activations*; Sun et al., 2024), e
$\alpha/R$ com a norma média empurraria o Gemma3 muito mais, em relação à sua variação real, do que os outros modelos.

**Decisão (Lucas, 12/10/2026):** $R_c = \frac{1}{N}\sum_i \lVert x_i - \mu\rVert$ para todos os modelos.

| modelo | norma média | $R_c$ | razão |
|---|---|---|---|
| Llama (15) | 8,38 | 1,60 | 5,2 |
| Qwen3 (22) | 131,81 | 22,70 | 5,8 |
| Gemma3 (19) | 32.958 | 839 | 39,3 |

Llama e Qwen3 foram rodados com a norma média (−1 a +1); a conversão é exata (×5,2 e ×5,8), sem refazer nada. Gemma3:
grade 0, ±1,5, ±3, ±4,5, ±6 em $\alpha/R_c$. Texto na dissertação: Metodologia, seção de Activation Steering
(Tabela `tab:steering_scale`). Código: `residual_norm(cfg, kind)` e `--norm centered` em `fcsae/steering/factcheck_sae.py`.
""")

md(r"""<a id="32"></a>
## 32. "Resíduo" é o mesmo que *residual stream*?

Sim. "Resíduo" foi usado como abreviação de ***residual stream*** ("*stream* residual" na dissertação): o vetor de cada
posição que atravessa o modelo da entrada à saída; cada camada lê dele e **soma** a sua contribuição (conexão residual;
Elhage et al., 2021). É nele que as ativações do SAE são coletadas e é nele que o steering soma a direção da feature.
A "dimensão do resíduo" $d$ é o tamanho desse vetor: 4.096 (Llama, Qwen3, DeepSeek, Ministral) e 2.560 (Gemma3, Gemma4);
o dicionário do SAE tem $8 \times d$ features. Daqui em diante: usar "*residual stream*" (preferência do Lucas, 14/10/2026).
""")

md(r"""<a id="33"></a>
## 33. Por que a feature do Qwen3 (12482) gera REF/SUP no steering, e não NEI? Fizemos algo errado?

Não foi erro de código: a 12482 codifica **"a alegação não é sustentada"** (REF **ou** NEI), e não a abstenção em si.
Ativação média por rótulo-ouro (598.930 decisões de treino):

| feature | SUP | REF | NEI | REF ÷ SUP | NEI ÷ SUP |
|---|---|---|---|---|---|
| Llama 19219 | 0,059 | 0,072 | **0,126** | 1,2 | **2,1** |
| **Qwen3 12482** | 0,067 | **0,161** | **0,174** | **2,4** | 2,6 |
| Gemma3 12903 | 0,0073 | 0,0082 | **0,0120** | 1,1 | **1,6** |

19219 e 12903 são específicas de NEI (e controlam o NEI nos dois sentidos); a 12482 dispara quase igual em REF e NEI —
por isso, no steering, diminuí-la leva tudo a SUPPORTED e aumentá-la troca SUP por REF. O padrão é o mesmo no VitaminC
e no FEVER (não é artefato de um conjunto).

**Por que ela foi escolhida:** o critério compara NEI × (SUP+REF) juntos; como SUP é o grupo maior, uma feature de "não
sustentado" ainda tem d de Cohen alto — a própria direção de abstenção (diff-of-means) carrega esse viés. Um critério
mais específico seria o menor entre d(NEI×SUP) e d(NEI×REF). **Decisão do Lucas (14/10/2026): manter a 12482 e não
refazer a triagem.** Para a dissertação: exemplo de que detecção ≠ controle (Arad et al., 2025) e de uma limitação do
critério de seleção.
""")

md(r"""<a id="34"></a>
## 34. No viewer, por que só os "top 20" e por que "not shown: 17.614 claims"? Esses dados são ignorados?

Não são ignorados nas análises.

- **"Not shown: N claims where it does not fire"** = claims com ativação **exatamente 0** (a feature não ficou entre as
  32 do TopK naquele claim). Na 19219 são 17.614 de 598.930 (2,9%). Ficam fora do **desenho** porque, tendo todos o
  mesmo valor (0), virariam uma "torre" em x = 0 que esmaga a escala das curvas. **AUC, d de Cohen e NEI no top-200 usam
  os 598.930 claims** (com 0 nesses casos).
- **Top 20** = exemplos para **ler** (inspeção qualitativa), convenção dos viewers de SAE (sae-viewer da OpenAI,
  Neuronpedia). Guardar os 598.930 claims × 32.768 features seria inviável numa página web. As curvas/histogramas usam
  todos os claims em que a feature dispara; as métricas de seleção usam todos os claims.
""")

md(r"""<a id="35"></a>
## 35. Como a literatura escolhe a feature de um SAE para um comportamento? E o nosso método?

**Nosso método:** (1) **escolha** = 1º no cosseno com a direção de abstenção e 1º no d de Cohen (NEI × SUP/REF);
(2) **caracterização** = AUC, % de NEI no top-200 e leitura dos exemplos (não entram na escolha); (3) **confirmação
causal** = steering.

| abordagem | o que faz | exemplos |
|---|---|---|
| 1. Contraste entre grupos | ativação da feature em dois conjuntos; escolhe as que mais separam | Ferrando et al. (2025): *separation score* conhecidas × desconhecidas — **= nosso d de Cohen** |
| 2. Alinhamento com direção | cosseno decoder × direção de diferença de médias | decomposição de direções como a de recusa (Arditi et al., 2024) — **= nosso cosseno** |
| 3. Sondas esparsas | classificador linear esparso sobre as features; melhor feature sozinha ("1D probe") | Gao et al. (2024); Kantamneni et al. (2025); Gurnee et al. (2023) |
| 4. Atribuição (efeito na saída) | *attribution patching*: efeito de cada feature numa métrica da saída (ex.: logit de NEI) | Marks et al. (2024), *Sparse Feature Circuits* |
| 5. Descrição / inspeção | exemplos que mais ativam; descrições automáticas por LLM | Bricken et al. (2023); Bills et al. (2023); Paulo et al. (2024) |

**Lições:** Arad et al. (2025) — detectar ≠ controlar; para controle, critérios de efeito na saída (tipo 4) tendem a ser
melhores (é o caso da 12482 do Qwen3; pergunta 33). AxBench (Wu et al., 2025) — para steering, diff-of-means costuma
ser competitiva com features de SAE. Nosso método (1+2, confirmado por steering) é padrão; a abordagem 4 fica como
trabalho futuro/limitação.
""")

md(r"""<a id="36"></a>
## 36. De onde até onde vão o d de Cohen, o cosseno e a AUC?

| medida | vai de | "sem relação" | sinal / extremos |
|---|---|---|---|
| AUC | 0 a 1 | 0,5 | 1 = separa NEI perfeitamente; 0 = separa ao contrário |
| cosseno | **−1 a 1** | 0 | 1 = mesma direção da abstenção; −1 = oposta ("espelho") |
| d de Cohen | **sem limite** (−∞ a +∞) | 0 | + = dispara mais em NEI; − = mais em SUP/REF |

O d é a diferença de médias em **desvios padrão**; leitura usual (Cohen, 1988): 0,2 pequeno, 0,5 médio, 0,8 grande.
Nossos valores: 1,276 (Llama 19219), 1,078 (Qwen3 12482), 0,928 (Gemma3 12903) — todos "grandes".
Com dois grupos normais, **AUC ≈ Φ(d/√2)**: d = 0,5 → 0,64; 0,8 → 0,71; 1,276 → 0,82 (medida na 19219: 0,783; as
ativações são ≥ 0 e assimétricas, não normais).
""")

md(r"""<a id="37"></a>
## 37. O que o steering da feature 12903 mostrou no Gemma3?

Grade completa (27 células; 500 claims do AVeriTeC dev; dose em $\alpha/R_c$, $R_c = 839$; regra de rótulo v2):

| α/R_c | sem evidência: acc · NEI | Single: acc · NEI | Iterative: acc · NEI |
|---|---|---|---|
| −3 | **0,642** · 1,6% | 0,606 · 5,8% | **0,642** · 1,6% |
| −1,5 | 0,592 · 7,6% | **0,608** · 4,0% | 0,628 · 0,6% |
| **0** | **0,306 · 48,0%** | **0,590 · 6,2%** | **0,624 · 0,8%** |
| +1,5 | 0,104 · 96,0% | 0,530 · 15,2% | 0,602 · 3,0% |
| +3 | 0,056 · 71,4% | 0,466 · 32,0% | 0,478 · 25,6% |
| +4,5 | 0,170 · 47,6% | 0,272 · 59,8% | 0,254 · 66,2% |
| ±6 | texto quebrado (~99% sem rótulo) | idem | idem |

**Leitura:** a 12903 **controla a abstenção nos dois sentidos** (como a 19219 do Llama e ao contrário da 12482 do
Qwen3): aumentá-la sobe o NEI nas três condições; diminuí-la tira a abstenção — sem evidência a acurácia **dobra**
(0,306 → 0,642); com evidência o ganho é pequeno (0,590 → 0,608; 0,624 → 0,642) porque o Gemma3 quase não abstém.
Faixa útil −3 a +4,5; em ±6 o texto degenera. **Veredito no viewer (Lucas, 18/10/2026): "✓ controls abstention".**
Detalhes: nb03 §5.4.
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "00-faq.ipynb")
nbf.write(nb, out)
print("escrito:", out, "| células:", len(cells))
