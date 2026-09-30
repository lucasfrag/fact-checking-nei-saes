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
6. [Como descobrimos que a camada 19 é a melhor pro Qwen3?](#6)
7. [O que é o SAE v3 e o que significa "seguir o Gao et al. (2024)"?](#7)
8. [Por que as features mortas caíram de 33–48% para ~0%?](#8)
9. [Por que a AUC da direção caiu no v3 (0,839 → 0,796)?](#9)
10. [A feature 19219 "liga" só quando o modelo abstém?](#10)
11. [Como é feita a validação — como escolhemos o α do steering?](#11)
12. [O que significa "todas as linhas usam os mesmos 250 claims da metade de teste"?](#12)
13. [O que é CAA?](#13)
14. [Como decidimos se o steering "melhorou" ou "piorou"?](#14)
15. [Por que o critério de qualidade do steering mudou de "legível" para "coerente"?](#15)
16. [O que significa a fórmula do steering x ← x + α·d₁₉₂₁₉ — e de onde ela vem?](#16)
17. [Que defeito foi achado na regra de rótulo do pipeline, e o que ele muda?](#17)
18. [Por que a grade de steering foi cortada em |α/R| ≤ 1?](#18)

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
## 6. Como descobrimos que a camada 19 é a melhor pro Qwen3?

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

**O que já seguíamos desde o v1:** TopK (k=32), 32.768 latentes, AuxK (α=1/32, k_aux=512),
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
"significativo". Por isso as **curvas** (§5.1) e a tabela por célula (§5.2) usam os **500**.
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
## 14. Como decidimos se o steering "melhorou" ou "piorou"?

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

**Referências:** McNemar, Q. (1947). *Psychometrika, 12*(2), 153–157 · Dietterich, T. G. (1998).
Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.
*Neural Computation, 10*(7), 1895–1923.

> Onde ver: `03-steering-sae-v3.ipynb`, §5.0.
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

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "00-faq.ipynb")
nbf.write(nb, out)
print("escrito:", out, "| células:", len(cells))
