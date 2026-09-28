# -*- coding: utf-8 -*-
"""Gera notebooks/03-steering-sae-v3.ipynb — steering com a feature 19219 do SAE v3 (Llama)
no passo de veredito do fact-checking (AVeriTeC dev): desenho, justificativas com referência,
fidelidade ao pipeline e resultados (lidos das células já computadas; reexecutar atualiza).
Fonte única de verdade do notebook — rode e depois:
    jupyter nbconvert --to notebook --execute --inplace notebooks/03-steering-sae-v3.ipynb
Dados: results/v3_llama/steer_sae/<cond>__<direção>__b<α/R>[__test100].json  (nos nomes de arquivo, o valor é α/R)
Código do experimento: fcsae/steering/factcheck_sae.py (+ scripts/run_steer_sae.sh)
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(t): cells.append(new_markdown_cell(t))
def code(t): cells.append(new_code_cell(t))

md(r"""# 03 · Steering com a feature 19219 do SAE v3 — fact-checking no AVeriTeC

**Dissertação de mestrado (PPGCC/PUCRS).** Estágio 2 da RQ2: *existe uma feature responsável
pela abstenção que possa ser manipulada para reduzir a abstenção sem depender de mais
evidência?* Aqui **manipulamos a feature 19219** (identificada no `02-sae-v3.ipynb`) durante o
veredito e medimos o efeito com as **mesmas métricas do Seminário de Andamento** (SA).

> Modelo: **Llama-3.1-8B-Instruct** (4-bit nf4), camada 15, SAE `v3_llama`.
> **Estado:** em execução. As seções 1–4 descrevem o desenho; a seção 5 lê os resultados que
> já existem (reexecutar o notebook atualiza).
>
> 👉 **Resposta direta — melhorou ou piorou?** Ver **§5.0**: uma tabela única comparando
> Single-step, Iterative e cada combinação de steering, com o veredito de cada uma.
""")

md(r"""## 1. Pergunta e hipóteses

O SA (§6.2) propõe dois testes com a feature, em dois cenários de evidência:
- **diminuir** a feature — o modelo passa a decidir (abstém menos) sem ganhar evidência nova?
- **aumentar** a feature — o modelo passa a abster mesmo quando decidiria?

**Hipóteses registradas antes de ver os resultados:**
- **H1 (causal):** aumentar a feature 19219 aumenta a taxa de NEI; diminuí-la reduz — nas
  três condições de evidência.
- **H2 (especificidade):** o mesmo steering com **features aleatórias** do SAE, de mesmo
  tamanho, **não** produz o mesmo efeito na taxa de NEI.
- **H3 (aplicação):** como o modelo **super-abstém** (prevê NEI em ~40% dos claims; o gold tem
  7%), diminuir a feature com evidência **aumenta a acurácia** sem piorar muito o erro das
  predições decididas.
- **H4 (sem evidência — hipótese do Lucas):** sem evidência, o modelo **não abstém sempre**:
  tenta decidir com o **conhecimento interno** (paramétrico). Se isso for verdade, a abstenção
  sem evidência é bem menor que 100% já no α = 0.
""")

md(r"""## 2. Desenho experimental

### 2.1 Condições de evidência (3, não 4)

Na hora do veredito, Single-step e Iterative **só diferem pela evidência** que cada pipeline
recuperou. Sem evidência, o prompt é o mesmo nos dois — então "sem evidência" é **uma**
condição só:

| condição | o que o modelo vê no prompt de veredito |
|---|---|
| **sem evidência** (`none`) | claim + data + autor; blocos de evidência e de Q&A **vazios** |
| **Single-step** (`single`) | a evidência e os Q&A que o pipeline **Single-step HF** recuperou |
| **Iterative** (`iterative`) | a evidência e os Q&A que o pipeline **Iterative HF** recuperou |

Para cada condição: **α = 0** (sem manipulação, o baseline), α < 0 (**diminuir** a feature) e
α > 0 (**aumentar**).

### 2.2 Como a feature é manipulada

Somamos ao *residual stream* da camada 15, em todas as posições, um "empurrão" na direção da
feature — **exatamente a Eq. 4 de Ferrando et al. (2025)**, com a mesma notação:

$$x \leftarrow x + \alpha \cdot d_{19219}$$

> **De onde vem a fórmula.** Ferrando et al. (2025), *"Do I Know This Entity? Knowledge Awareness
> and Hallucinations in Language Models"* (ICLR 2025), Eq. 4: $x_{new} \leftarrow x + \alpha\, d_j$ —
> *"increasing/decreasing the activation value of an SAE latent, a_j(x), is equivalent to doing
> activation steering (Turner et al., 2023) with the decoder latent vector"*. Somar um vetor ao
> residual stream para guiar o modelo é o *activation addition* de **Turner et al. (2023,
> ActAdd)**; somar numa camada só, em todas as posições, segue **Arditi et al. (2024, Eq. 3)**:
> $x^{(l)} \leftarrow x^{(l)} + r^{(l)}$. A única diferença é o índice: o $d_j$ deles é o nosso
> $d_{19219}$ (a feature $j$ = 19219).

**Símbolo por símbolo:**

| símbolo | o que é | tipo | valor |
|---|---|---|---|
| $x$ | o **residual stream** na camada 15, num token — a representação interna do modelo naquele ponto | vetor de 4.096 números | muda a cada token |
| $\alpha$ | o **coeficiente de steering** — o "botão de volume": quanto e em que sentido empurrar (**não é bias**). Mesmo α de Ferrando et al. (2025) | um número | α < 0 diminui a feature; α > 0 aumenta; α = 0 = sem steering |
| $d_{19219}$ | a **direção da feature 19219**: a linha dela no **decoder** do SAE, normalizada para comprimento 1 | vetor de 4.096 números | fixo |
| $\leftarrow$ | "**substitui** $x$ por…" | — | — |
| $R$ *(para interpretar α)* | o **tamanho típico de $x$**: norma média do residual stream na posição da decisão (camada 15, nas 598.930 decisões do harvest) | um número | **8,375** |
| $\alpha/R$ *(para interpretar α)* | α em **múltiplos do tamanho do resíduo** | um número | grade de −2 a +2 |

**O que a fórmula faz, em 2 passos:**
1. **Monta o empurrão** $\alpha \cdot d_{19219}$: um número multiplicando um vetor de comprimento 1.
   O resultado é um vetor que **aponta na direção da feature** (ou na oposta, se α < 0) e tem
   **comprimento |α|**.
2. **Soma o empurrão ao residual stream** e o modelo continua o processamento a partir do $x$
   modificado. A única operação sobre $x$ é essa **soma**.

**Como ler o tamanho de α — α/R.** O valor absoluto de α depende da escala do resíduo de cada
modelo. Por isso reportamos também **α/R**: o α em múltiplos do tamanho típico do resíduo — como
Ferrando et al. (2025, nota 5) descrevem o α que escolheram: *"corresponds to around two times the
norm of the residual stream"* (ou seja, α/R ≈ 2). A grade é definida em α/R e convertida em α
por α = (α/R) × 8,375.

**Exemplo com α/R = −0,25 (α = −2,09):** o empurrão tem comprimento 2,09. Como o $x$ tem tamanho
típico de 8,4, deslocamos o $x$ em **~25% do tamanho dele**, no sentido **contrário** à feature de
abstenção — e a abstenção cai. Com α/R = +0,25 (α = +2,09), mesmo tamanho, no sentido **da**
feature — e a abstenção sobe. α/R = 1 (α = 8,375) empurra 100% do tamanho do resíduo.

**Por que reportar α/R e não só α:** α/R é interpretável ("quanto do tamanho do resíduo") e
comparável entre modelos — o resíduo do Qwen3 é ~10× maior que o do Llama, então o mesmo α teria
efeitos muito diferentes.

**Onde acontece:** na **saída da camada 15**, em **todos os tokens** (os do prompt e cada token
gerado). No código (`fcsae/steering/factcheck_sae.py`): `vec = α · d` (com α = β·R, onde `β` é o
nome interno de α/R nos arquivos de resultado) e, num *hook* da camada, `saída + vec`.

**Nota técnica:** o SAE v3 opera no espaço normalizado da entrada (média 0, norma 1; ver
`02-sae-v3.ipynb`), mas a direção $d_{19219}$ é aplicada ao resíduo **bruto** — a escala certa vem
de α, que é medido na escala do resíduo bruto (via R).

**Por quê (literatura):**
- **Manipular uma feature = somar a direção dela no decoder.** Ferrando et al. (2025, Eq. 4):
  *"increasing/decreasing the activation value of an SAE latent, a_j(x), is equivalent to
  doing activation steering (Turner et al., 2023) with the decoder latent vector, i.e.
  updating the residual stream as follows: x_new ← x + α·d_j"*. Eles usam exatamente isso para
  **induzir e suprimir a recusa por desconhecimento de entidades** — fenômeno próximo do nosso
  (a 19219 ativa em claims sobre entidades obscuras e trivialidades, ver `02-sae-v3.ipynb`).
- **Coeficiente relativo à norma do resíduo.** Ferrando et al. (2025, nota 5): o coeficiente
  escolhido *"corresponds to around two times the norm of the residual stream"*. Reportar α também
  como múltiplo de R (α/R) torna a escala interpretável e comparável entre modelos.
- **Uma camada, todas as posições.** Arditi et al. (2024, Eq. 3): *"for activation addition,
  we intervene only at layer l, and across all token positions"*.

### 2.3 Grade de α e como o α reportado é escolhido

**Grade planejada:** α/R ∈ {−2, −1,5, −1, −0,75, −0,5, −0,25, 0, 0,25, 0,5, 0,75, 1, 1,5, 2}.
**Grade executada:** α/R ∈ {−1, −0,75, −0,5, −0,25, 0, 0,25, 0,5, 0,75, 1} — ou seja,
α ∈ {−8,38, −6,28, −4,19, −2,09, 0, +2,09, +4,19, +6,28, +8,38} —, nas três condições e nos 500
claims → **curva dose-resposta**.

**Por que a grade foi cortada em |α/R| ≤ 1 (decisão de 28/09/2026, com os dados sem evidência):**
em |α/R| = 1 o modelo **já degenera** — com α/R = +1, 91% das respostas entram em *loop*
(*"I have no information about this claim. I have no evidence…"* repetido) até o limite de 1.536
tokens; com α/R = −1, 30%. Doses maiores só degenerariam mais, nenhum critério de qualidade as
escolheria, e cada célula levaria ~7 h. É o mesmo procedimento do CAA (Panickssery et al., 2024):
explorar, observar a degradação do texto e **limitar a faixa** do multiplicador.

**Por quê reportar a curva (e não um ponto só):** o efeito do steering depende da dose e varia
muito entre exemplos — Tan et al. (2024) mostram que *"steerability is highly variable across
different inputs"*. E multiplicadores grandes **degradam o texto**: Panickssery et al. (2024)
— o método **CAA** (*Contrastive Activation Addition*), que soma às ativações a diferença média
entre exemplos com e sem um comportamento, multiplicada por um fator — *"find that steering with larger multipliers results in a degradation in the quality of
the open-ended text … Therefore, we choose to limit the multiplier range to strike a balance
between effectively steering the model's behavior and maintaining the overall quality"*.

**Escolha do α para as tabelas — sem ajustar no conjunto de teste.** Dividimos o AVeriTeC dev
em duas metades fixas: **validação** = `claim_id` par (250) e **teste** = `claim_id` ímpar (250).
- **Critério (sem usar rótulos):** em cada condição e em cada sentido (diminuir / aumentar),
  o **maior |α| cujas respostas continuam coerentes** em ≥ 80% dos claims **da validação**
  ("coerente" = a resposta contém um dos 4 rótulos, escrito como rótulo (maiúsculas), **ou** uma recusa por falta de informação
  escrita por extenso — ver §2.3.1). Não usar a acurácia evita escolher o α que "dá o melhor
  resultado" no próprio conjunto avaliado.
- **Reporte:** as métricas desse α na metade de **teste**, contra o α = 0 na mesma metade.
- **Referência:** selecionar o vetor/coeficiente num conjunto de **validação** é o protocolo de
  Arditi et al. (2024, §2.3: *"by evaluating each candidate vector over validation sets"*) e de
  Ferrando et al. (2025, nota 5: *"We use a validation set to select an appropriate steering
  coefficient α"*).
- **Escolha nossa (sem referência direta):** o limiar de 80%. É uma operacionalização do
  "limite de qualidade" do CAA; a curva completa permite ver o efeito de outros limiares.

#### 2.3.1 Correção do critério de qualidade (27/09/2026, depois das primeiras células)

**O que mudou.** O critério original era *"a resposta contém um dos 4 rótulos"* (**com
rótulo**). Na célula *sem evidência, α/R = +0,25* (feature **aumentada**), 26% das respostas não
tinham rótulo — mas **não eram o modelo quebrando**: eram **abstenções escritas por extenso**,
p.ex. *"I'm unable to find any information about the claim…"* (79 casos), *"Unfortunately, I
don't have any evidence to review… The task requires evidence…"*. É justamente o comportamento
que a manipulação deveria induzir. Com o critério original, o sentido "aumentar" seria
penalizado por produzir o que queremos medir.

**Novo critério (coerente)** = contém um dos 4 rótulos escrito como rótulo (MAIÚSCULAS, por extenso) **ou** uma **recusa por falta de
informação** detectada por correspondência de frases — o mesmo procedimento de Ferrando et al.
(2025, §5): *"we define a set of common knowledge refusal completions and detect if any of these
occur with string matching, e.g. 'Unfortunately, I don't have access to real-time
information…'"*. Uma recusa de **segurança** (*"I cannot provide information or guidance on
illegal or harmful activities"*) **não** conta: Ferrando et al. separam a recusa por
conhecimento da recusa *"due to safety concerns"*.

**Salvaguardas contra escolha conveniente:**
1. a lista de frases (`REFUSAL_PATTERNS` em `fcsae/steering/factcheck_sae.py`) foi definida a
   partir das duas primeiras células sem evidência e **congelada** para todas as células;
2. reportamos **os dois critérios** lado a lado (coerente e com rótulo) e qual α cada um escolhe;
3. na regra de rótulo do pipeline essas respostas já viravam NEI, então as métricas (acurácia,
   F1, NEI) **não mudam** — muda só qual α é considerado "sem quebrar o modelo".

### 2.4 Controle com features aleatórias

Nos dois α escolhidos (diminuir e aumentar), repetimos o steering trocando $d_{19219}$ pela
direção de **10 features aleatórias do mesmo SAE** (mesmo α, mesma camada), nas três condições,
em **100 claims da metade de teste**.

**Por quê:** separa o efeito **da feature** do efeito de **qualquer perturbação** de mesmo
tamanho no resíduo. É o controle de Ferrando et al. (2025): *"the average refusal rate after
steering with 10 differents random latents, using the same configuration (layer and steering
coefficient)"* (e *"a random baseline where we apply a randomly sampled SAE latent with the
same coefficient"*), aplicado num conjunto de 100 perguntas. Turner et al. (2023, Apêndice G)
fazem o análogo com vetores aleatórios *"with similar magnitudes to the steering vectors"*.

### 2.5 Métricas (as do SA)

- **Rótulo de cada resposta:** o **1º rótulo que o modelo escreve** (métrica principal; §3.1) e,
  ao lado, o da regra do pipeline original;
- **Acurácia** (4 rótulos) e **macro-F1** do SA = média de F1(SUP), F1(REF), F1(NEI);
- **F1 por classe**, incluindo a **CONF**;
- **taxa de NEI** (SA, Tabela 3);
- **erro condicionado a não-NEI** (SA, Tabela 4). O próprio SA §6.2 define o critério:
  *"We consider the feature to genuinely cause this behavior if the error rate does not worsen
  more than it already does under iteration for that specific model."* No HF, a iteração piora
  esse erro do Llama em **+0,004** (0,304 → 0,308; `01-single-vs-iterative-it.ipynb`, Tabela 4);
- **coerência** (fração de respostas com rótulo ou recusa por falta de informação, §2.3.1) e
  **com rótulo** (critério original) — controle de qualidade do steering.
""")

md(r"""## 3. Fidelidade ao pipeline (o baseline)

Para que "α = 0" seja o veredito do pipeline, reconstruímos **exatamente** o que o pipeline HF
fazia no passo de veredito (`shared/modules/verdict/llm_verdict.py` e `shared/llm/hf_llm.py`
do código original):

| item | como está no experimento | verificação |
|---|---|---|
| prompt de veredito | `LLMVerdict.build_prompt`, com a evidência final e os Q&A de cada passo, lidos do `averitec_dev.jsonl` | **idêntico byte a byte** ao do código original em 1.000/1.000 prompts |
| *system prompt* | o do `HFLLM` | idêntico |
| geração | chat template, `enable_thinking=False`, *greedy*, até **1.536** tokens | = `HF_MAX_NEW_TOKENS` do pipeline |
| rótulo | `LLMVerdict.normalize`: "REFUTE"/"FALSE" em qualquer ponto → REF; depois SUPPORT, CONFLICT, NOT ENOUGH; **sem rótulo → NEI** | cópia fiel — **mas tem um defeito** (§3.1); a métrica principal usa o 1º rótulo |

**Duas diferenças registradas:**
1. **Sem evidência, o pipeline não chamava o LLM** — devolvia NEI direto (`if not evidence:
   verdict = NEI`). Na nossa condição `none`, **o modelo é chamado**, com o mesmo template e os
   blocos de evidência e Q&A vazios. É o que permite testar a H4: o "≈100% de NEI sem
   evidência" que aparecia antes era esse atalho do código, não o comportamento do modelo.
2. **A rodada do pipeline foi feita em outro ambiente** (03/09/2026). Mesmo com prompt idêntico,
   pequenas diferenças numéricas do 4-bit podem mudar vereditos limítrofes. Por isso o
   **baseline dos experimentos é o nosso α = 0** (mesmo ambiente do steering), e reportamos a
   concordância dele com as predições do pipeline.

**Um tropeço (corrigido):** a primeira rodada usava só 64 tokens de geração. Isso **cortava** o
raciocínio do modelo antes do rótulo (sem evidência, o Llama costuma começar com *"To determine
the final verdict of the claim, I'll analyze…"*) e, como a normalização do pipeline lê o texto
inteiro, explicava a maior parte das divergências com o pipeline: com 64 tokens a concordância
era de 94,8% (single) e 94,6% (iterative); com 1.536 tokens, 22 dos 26 claims divergentes do
single passaram a concordar. Todas as células usam 1.536 tokens; as de 64 foram arquivadas.
""")

md(r"""### 3.1 Um defeito na regra de rótulo do pipeline (achado em 28/09/2026)

**O que é.** A regra que o pipeline original usa para extrair o rótulo da resposta
(`LLMVerdict.normalize`) procura as palavras **nesta ordem**: primeiro "REFUTE" ou "FALSE" **em
qualquer ponto do texto**; só depois "SUPPORT", "CONFLICT" e "NOT ENOUGH". Mas a explicação típica
de uma abstenção é:

> *"**NOT ENOUGH EVIDENCE**\n\nThe evidence provided does not contain any direct evidence to
> support or **refute** the claim…"*

A expressão *"support or refute"* contém "refute" → a resposta é registrada como **REFUTED**,
embora o modelo tenha dito NOT ENOUGH EVIDENCE.

**Onde está o código e qual versão cada rodada usou (verificado em 28/09/2026).** O código do pipeline
**não está** no projeto atual (`old_method/` guarda só as saídas). Ele existe (i) na cópia antiga do
projeto em `/mnt/d/Projetos WSL/mestrado/`, que ainda tem o histórico Git, e (ii) no repositório público
`github.com/lucasfrag/iterative-question-generation`. Cada rodada gravou o commit do código no
`config.json`:

| rodadas | commit | arquivo | `normalize` |
|---|---|---|---|
| HF (este notebook) | `dc19749` (03/09/2026) | `iterative-question-generation/modules/verdict/llm_verdict.py`, linhas 85–104 | a regra descrita acima (arquivo **idêntico** ao lido) |
| Ollama (SA / ENIAC) | `53829d5` e `809230a` | `modules/verdict/llm_verdict.py` (repositório público), linhas 85–104 | **a mesma regra** |

Nos três commits, `pipeline/pipeline_factory.py` usa `LLMVerdict` no Single-step (linha 48) e no
Iterative (linha 82). Ou seja: **os números do SA e do artigo do ENIAC também foram extraídos com essa
regra**. O que não dá para medir é o **tamanho** do efeito no Ollama, porque aqueles JSONL não guardam o
texto das respostas (e o formato das respostas do Ollama pode ser diferente do HF).

**Tamanho (baseline Single-step, sem steering, 500 claims):** 76 respostas que começam com "NOT
ENOUGH EVIDENCE" são contadas como REF (74 pelo "refute" de *"support or refute"*, 2 por
"false"); em 45 delas o gold é REF, então o erro de leitura **vira acerto**. Resultado:

| baseline Single-step | regra do pipeline | 1º rótulo que o modelo escreve |
|---|---|---|
| acurácia | 0,434 | **0,360** |
| taxa de NEI | 0,422 | **0,620** |

**Por que isso também afeta o steering:** o steering muda o **tamanho da explicação**. Mais texto →
mais chance de aparecer "refute" → a regra do pipeline mistura o efeito do steering com um erro de
leitura. Foi o que fez "aumentar NEI" parecer não aumentar a abstenção com evidência.

**Exemplos e contagens de cada tipo de resposta: §4.2.**

**Decisão (28/09/2026):** a **métrica principal** deste notebook usa o **rótulo que o modelo
produziu**, ignorando a explicação (`first_label` em `fcsae/steering/factcheck_sae.py`).

**O que conta como rótulo:** a expressão completa **escrita como rótulo, em MAIÚSCULAS** —
`SUPPORTED`, `REFUTED`, `NOT ENOUGH EVIDENCE`, `CONFLICTING EVIDENCE`. Verificamos nas células que o
modelo sempre escreve o veredito assim; as ocorrências em minúsculas são ele **falando** dos rótulos
(*"I cannot assess whether the claim is supported, refuted, or…"*), e palavras soltas como *"support or
refute"* não são veredito. (Uma 1ª versão desta leitura procurava as palavras sem diferenciar
maiúsculas e transformava *"no evidence to support or refute"* em SUP — corrigido ao ver os exemplos
da §4.2.)

**A leitura, em 3 passos:**
1. se a resposta **começa com um rótulo**, usa esse rótulo e ignora o resto — é o caso de quase todas
   as respostas **com evidência** (98–100% por célula; 81% em α/R = +0,5);
2. se **não começa** (sem evidência o modelo às vezes raciocina antes: *"To determine the final
   verdict… the claim is REFUTED"*), usa o **primeiro rótulo que aparece** no texto;
3. se **não há rótulo nenhum** (p.ex. as recusas por extenso *"I'm unable to find any information…"*),
   conta como **NEI** — o mesmo padrão do pipeline, e semanticamente correto (são abstenções).

Quando a resposta começa com um rótulo, os passos 1 e 2 dão sempre o mesmo resultado (verificado:
100% de concordância em todas as células). As métricas com a **regra do
pipeline** aparecem ao lado, para comparação com o `01-single-vs-iterative-it.ipynb`. Como todas as
células guardam o texto completo das respostas, **nada precisou ser re-rodado**.

**Consequência para a RQ1:** os números dos pipelines no notebook 01 foram calculados com essa
regra — ver a seção sobre o defeito no `01-single-vs-iterative-it.ipynb`.
""")

md(r"""## 4. Carregamento e métricas""")

code(r"""import json, glob, os, re, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

D = "../results/v3_llama/steer_sae"
CLS = ["SUP", "REF", "NEI", "CONF"]
COND = {"none": "sem evidência", "single": "Single-step", "iterative": "Iterative"}
THR = 0.80                                  # coerência mínima (§2.3)
from fcsae.steering.factcheck_sae import coherent, readable as has_label, first_label, REFUSAL_PATTERNS

def f1(pred, gold, c):
    tp = sum(p == c and g == c for p, g in zip(pred, gold)); fp = sum(p == c and g != c for p, g in zip(pred, gold))
    fn = sum(p != c and g == c for p, g in zip(pred, gold))
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0

def metrics(pred, gold, readable):
    F = {c: f1(pred, gold, c) for c in CLS}
    nn = [(p, g) for p, g in zip(pred, gold) if p != "NEI"]
    return {"n": len(pred), "acc": np.mean([p == g for p, g in zip(pred, gold)]),
            "macroF1": np.mean([F["SUP"], F["REF"], F["NEI"]]), **{f"F1_{c}": F[c] for c in CLS},
            "nei_rate": np.mean([p == "NEI" for p in pred]),
            "err_nonNEI": np.mean([p != g for p, g in nn]) if nn else np.nan,
            "legivel": np.mean(readable)}

cells = []
for p in sorted(glob.glob(f"{D}/*.json")):
    m = re.match(r"(\w+?)__(\w+?)__b([+-][\d.]+)(__test\d+)?\.json", os.path.basename(p))
    if not m: continue
    r = json.load(open(p, encoding="utf-8")); r["subset"] = m.group(4) or ""
    r["label_ok"] = [has_label(x) for x in r["raw"]]          # critério original (§2.3.1)
    r["pred_pipeline"] = r["pred"]                             # regra do pipeline (normalize)
    r["pred"] = [first_label(x) for x in r["raw"]]             # MÉTRICA PRINCIPAL: 1º rótulo (§3.1)
    r["readable"] = [coherent(x) for x in r["raw"]]           # critério principal: coerente
    cells.append(r)
print(f"células prontas: {len(cells)}")
for r in cells:
    print(f"  {r['cond']:10s} {r['direction']:10s} α/R={r['beta']:+.2f} {r['subset']}  (n={len(r['ids'])})")""")

md(r"""### 4.1 Fidelidade do baseline (α = 0) ao pipeline

**Como ler.** Para Single e Iterative: **concordância** = fração dos 500 claims em que o nosso
α = 0 dá o mesmo rótulo que o pipeline HF original — as duas lidas com a **mesma regra** (a do
pipeline), para medir a fidelidade da reconstrução. Ao lado, as métricas do nosso α = 0 com a
**leitura correta** (1º rótulo, §3.1), que é o baseline dos experimentos.""")

code(r"""rows = []
for r in cells:
    if r["beta"] == 0 and r["cond"] in ("single", "iterative") and not r["subset"]:
        ours = metrics(r["pred_pipeline"], r["gold"], r["readable"]); pipe = metrics(r["pipeline_pred"], r["gold"], [True] * len(r["gold"]))
        fl = metrics(r["pred"], r["gold"], r["readable"])
        agree = np.mean([p == q for p, q in zip(r["pred_pipeline"], r["pipeline_pred"])])   # mesma regra dos dois lados
        rows.append([COND[r["cond"]], agree, ours["acc"], pipe["acc"], ours["nei_rate"], pipe["nei_rate"], fl["acc"], fl["nei_rate"]])
pd.DataFrame(rows, columns=["condição", "concordância", "acc (nosso α=0, regra pipeline)", "acc (pipeline)", "NEI (nosso α=0, regra pipeline)",
                            "NEI (pipeline)", "acc (1º rótulo)", "NEI (1º rótulo)"]).round(3) \
  if rows else "ainda não há células de α=0 com evidência (1.536 tokens)"
""")

md(r"""### 4.2 Como o modelo responde — tipos de resposta, contagens e exemplos

Cada resposta é classificada num tipo, **a partir do texto que o modelo gerou**:

| tipo | o que é | exemplo típico |
|---|---|---|
| **A. começa com o rótulo** | rótulo na 1ª linha, explicação depois | *"REFUTED\n\nThe evidence provided…"* |
| **B. raciocina antes** | começa explicando e dá o rótulo no meio/fim | *"To determine the final verdict… the claim is REFUTED"* |
| **C. recusa por extenso** | abstém **sem** usar o rótulo | *"I'm unable to find any information about…"* |
| **D. outra, sem rótulo** | texto sem conclusão, recusa de segurança, etc. | *"I cannot provide information… illegal or harmful…"* |

Duas contagens à parte:
- **defeito da regra (§3.1):** respostas cujo rótulo produzido é **NEI** mas que a regra do pipeline
  lê como **REF** (por causa de "refute"/"false" na explicação);
- **em loop:** respostas que atingem o limite de geração (1.536 tokens) sem terminar — estimadas por
  tamanho (> 5.000 caracteres ≈ 1.500 tokens).

**Como ler a tabela.** Uma linha por célula (condição × α/R); cada coluna é o **número de respostas**
(de 500) de cada tipo. A + B + C + D = 500.""")

code(r"""from fcsae.steering.factcheck_sae import refusal, normalize
from IPython.display import Markdown, display
START = {"NOT ENOUGH EVIDENCE": "NEI", "CONFLICTING EVIDENCE": "CONF", "SUPPORTED": "SUP", "REFUTED": "REF"}

def starts_with_label(t):
    u = (t or "").strip().upper().lstrip("*#: ")
    return any(u.startswith(k) for k in START)

def kind(t):
    if starts_with_label(t): return "A"
    if has_label(t): return "B"
    if refusal(t): return "C"
    return "D"

rows = []
for r in sorted([c for c in cells if not c["subset"]], key=lambda r: (list(COND).index(r["cond"]), r["beta"])):
    ks = [kind(x) for x in r["raw"]]
    bug = sum(fl == "NEI" and normalize(x) == "REF" for x, fl in zip(r["raw"], r["pred"]))
    loop = sum(len(x) > 5000 for x in r["raw"])
    rows.append([COND[r["cond"]], r["beta"], ks.count("A"), ks.count("B"), ks.count("C"), ks.count("D"), bug, loop])
K = pd.DataFrame(rows, columns=["condição", "α/R", "A. começa c/ rótulo", "B. raciocina antes", "C. recusa por extenso",
                                "D. outra, sem rótulo", "NEI lido como REF (defeito)", "em loop"])
K.set_index(["condição", "α/R"])""")

md(r"""**Exemplos reais de cada tipo.** Para cada tipo, as primeiras ocorrências (na ordem dos claims)
nas células onde ele é mais comum. Em cada exemplo: o claim, o rótulo **gold**, o rótulo que a
**regra do pipeline** extrai e o **rótulo produzido** (nossa leitura), seguidos do começo da resposta do
modelo, exatamente como ele escreveu.""")

code(r"""claims_txt = {}
try:
    from fcsae.steering.factcheck_sae import load_claims
    claims_txt = {c["id"]: c["claim"] for c in load_claims("llama3.1:8b")}
except Exception as e:
    print("(sem texto dos claims:", e, ")")

def show(title, cond, beta, pred_fn, n=2, maxc=420):
    r = next((c for c in cells if c["cond"] == cond and c["beta"] == beta and not c["subset"]), None)
    if r is None:
        display(Markdown(f"#### {title}\n_célula {COND[cond]}, α/R = {beta:+.2f} ainda não rodou_")); return
    idx = [i for i, x in enumerate(r["raw"]) if pred_fn(x, r["pred"][i])][:n]
    out = [f"#### {title}\n*célula: {COND[cond]}, α/R = {beta:+.2f} — {sum(pred_fn(x, p) for x, p in zip(r['raw'], r['pred']))} de {len(r['raw'])} respostas*\n"]
    for i in idx:
        x = r["raw"][i]; cid = r["ids"][i]
        txt = x[:maxc].replace("\n", " ⏎ ") + (" […]" if len(x) > maxc else "")
        out.append(f"- **claim {cid}:** {claims_txt.get(cid, '')[:160]}  \n  gold **{r['gold'][i]}** · regra do pipeline **{normalize(x)}** · "
                   f"rótulo produzido **{r['pred'][i]}**  \n  > {txt}")
    display(Markdown("\n".join(out)))

show("A. Começa com o rótulo (e explica depois)", "single", 0.0, lambda x, p: starts_with_label(x))
show("B. Raciocina antes e dá o rótulo no meio/fim", "none", 0.0, lambda x, p: kind(x) == "B")
show("C. Recusa por extenso (abstém sem usar o rótulo)", "none", 0.5, lambda x, p: kind(x) == "C")
show("D. Outra, sem rótulo", "none", 0.5, lambda x, p: kind(x) == "D", n=1)
show("Defeito da regra: começa com NOT ENOUGH EVIDENCE, o pipeline lê REFUTED", "single", 0.0,
     lambda x, p: p == "NEI" and normalize(x) == "REF", n=3)
show("Em loop (atinge o limite de geração)", "none", 1.0, lambda x, p: len(x) > 5000, n=1, maxc=600)""")

md(r"""👉 **Leitura.**
- **Com evidência**, o modelo quase sempre **começa com o rótulo** (tipo A) — é aí que a leitura
  "pegar o rótulo do começo e ignorar a explicação" é exata.
- **Sem evidência**, ele muitas vezes **raciocina antes** (tipo B) e, com a feature aumentada, passa a
  **recusar por extenso** (tipo C) — abstenções que não usam o rótulo.
- O **defeito da regra do pipeline** aparece já no baseline com evidência: respostas que começam com
  *"NOT ENOUGH EVIDENCE"* e são contadas como REFUTED por causa de *"support or refute"* na explicação.
""")

md(r"""## 5. Resultados

### 5.0 Melhorou ou piorou? — comparação direta

**O que está sendo comparado.** Todas as linhas usam **os mesmos 250 claims da metade de teste**
(`claim_id` ímpar), para que a comparação seja justa:
- **Pipeline Single-step / Iterative** — as predições originais dos pipelines HF
  (`01-single-vs-iterative-it.ipynb`), só como referência;
- para cada condição de evidência (**sem evidência**, **Single-step**, **Iterative**): o nosso
  veredito **sem steering** (α = 0) e com a feature 19219 **diminuída** (menos NEI) e
  **aumentada** (mais NEI), no α escolhido na validação (§2.3).

**Como o veredito é decidido.** Cada manipulação é comparada com o α = 0 **da mesma condição**
(mesmo prompt, mesma evidência; só muda o steering):
- métrica principal: **acurácia** (a do SA);
- **▲ MELHOROU** / **▼ PIOROU** só se a diferença for **estatisticamente significativa**
  (p < 0,05); senão, **= sem diferença significativa**;
- teste: **McNemar exato**, que compara dois classificadores **nos mesmos exemplos** — o teste
  recomendado por Dietterich (1998) para esse caso, por ter baixo erro do tipo I
  [McNemar, 1947; Dietterich, 1998].
- abaixo do veredito, o que mudou em NEI, F1 por classe e erro fora de NEI.""")

code(r"""from scipy.stats import binomtest
from IPython.display import Markdown, display

def mcnemar_p(pa, pb, gold):
    # McNemar exato: só contam os claims em que exatamente um dos dois acerta
    b = sum((x == g) and (y != g) for x, y, g in zip(pa, pb, gold))
    c = sum((x != g) and (y == g) for x, y, g in zip(pa, pb, gold))
    return binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0

def test_half(r, key="pred"):
    idx = [i for i, cid in enumerate(r["ids"]) if cid % 2 == 1]
    return [r[key][i] for i in idx], [r["gold"][i] for i in idx], [r["readable"][i] for i in idx] if key == "pred" else [True] * len(idx)

cells500 = [r for r in cells if not r["subset"] and (r["direction"] == "none" or r["direction"].startswith("feat"))]
def cell(cond, b): return next((r for r in cells500 if r["cond"] == cond and r["beta"] == b), None)
def pick(cond, sign):   # maior |α| com coerência >= THR na VALIDAÇÃO (id par)
    ok = []
    for r in cells500:
        if r["cond"] == cond and np.sign(r["beta"]) == sign:
            val = [r["readable"][i] for i, cid in enumerate(r["ids"]) if cid % 2 == 0]
            if np.mean(val) >= THR: ok.append(r["beta"])
    return max(ok, key=abs) if ok else None

COLS = ["acc", "macroF1", "F1_SUP", "F1_REF", "F1_NEI", "F1_CONF", "nei_rate", "err_nonNEI", "legivel"]
rows, lines = [], []
for cond in ["single", "iterative"]:            # referência: pipelines originais
    r0 = cell(cond, 0.0)
    if r0:
        p, g, rd = test_half(r0, "pipeline_pred"); m = metrics(p, g, rd)
        rows.append([f"Pipeline {COND[cond]} (nb 01, regra original)", "—", "—"] + [m[k] for k in COLS] + ["referência", np.nan])
for cond in COND:
    r0 = cell(cond, 0.0)
    if not r0: continue
    p0, g, rd0 = test_half(r0); m0 = metrics(p0, g, rd0)
    rows.append([COND[cond], "sem steering", 0.0] + [m0[k] for k in COLS] + ["baseline", np.nan])
    for sense, sign in [("diminuir NEI", -1), ("aumentar NEI", 1)]:
        b = pick(cond, sign)
        if b is None:
            lines.append(f"- **{COND[cond]} + {sense}:** ainda sem células suficientes."); continue
        p1, _, rd1 = test_half(cell(cond, b)); m1 = metrics(p1, g, rd1); pv = mcnemar_p(p1, p0, g)
        d = m1["acc"] - m0["acc"]
        verdict = ("▲ MELHOROU" if d > 0 else "▼ PIOROU") if pv < 0.05 else "= sem diferença significativa"
        rows.append([COND[cond], sense, b] + [m1[k] for k in COLS] + [verdict, pv])
        lines.append(f"- **{COND[cond]} + {sense} (α = {b*8.375:+.2f}; α/R = {b:+.2f}): {verdict}** — acurácia {m0['acc']:.3f} → {m1['acc']:.3f} "
                     f"({100*d:+.1f} pts, p = {pv:.3g}). Taxa de NEI {m0['nei_rate']:.3f} → {m1['nei_rate']:.3f}; "
                     f"F1 NEI {m0['F1_NEI']:.3f} → {m1['F1_NEI']:.3f}; erro fora de NEI {m0['err_nonNEI']:.3f} → {m1['err_nonNEI']:.3f}; "
                     f"coerente {m1['legivel']:.0%}.")
display(Markdown("#### Veredito por manipulação (metade de teste, 250 claims)\n\n" + ("\n".join(lines) if lines else "_ainda sem células de steering_")))
S = pd.DataFrame(rows, columns=["condição", "manipulação", "α/R", "Acc", "F1", "F1 SUP", "F1 REF", "F1 NEI", "F1 CONF",
                                "NEI", "erro não-NEI", "coerente", "veredito", "p (McNemar)"])
S.set_index(["condição", "manipulação"]).round(3)""")

md(r"""👉 **Como interpretar.**
- **"Diminuir NEI" que melhora a acurácia** com evidência é o resultado esperado pela H3: o
  modelo super-abstém (gold tem 7% de NEI), então reduzir a abstenção converte NEIs errados em
  decisões certas. O **erro fora de NEI** diz se o preço foi decidir errado mais vezes —
  o critério do SA é que ele **não piore mais do que a iteração já piora** (+0,004 no Llama HF).
- **"Aumentar NEI"** deve **piorar** a acurácia (mais abstenção num conjunto quase sem NEI) —
  é o teste causal da H1 no outro sentido, não uma tentativa de melhorar.
- **Sem evidência**, "diminuir NEI" mostra quanto o modelo consegue decidir **só com o
  conhecimento interno** (H4). Lembrete: responder sempre "REFUTED" daria 0,61 de acurácia
  neste conjunto (61% do gold é REF), então acurácia alta sem evidência precisa ser lida junto
  com o F1 por classe.""")

md(r"""### 5.1 Curvas dose-resposta (feature 19219, 500 claims)

**Como ler este gráfico.** Uma coluna por condição de evidência. Eixo **X** = α/R (α em múltiplos da norma do resíduo; negativo =
feature diminuída; positivo = aumentada; 0 = sem manipulação). *Linha de cima:* **taxa de NEI**
(laranja) e **acurácia** (azul). *Linha de baixo:* **coerência** (cinza escuro:
respostas com rótulo ou recusa por falta de informação), **com rótulo** (cinza claro tracejado,
o critério original), o limiar de 80% (tracejado) e o **erro fora de NEI** (vermelho). A taxa de
NEI inclui as recusas escritas por extenso, porque a regra do pipeline as converte em NEI.""")

code(r"""feat = [r for r in cells if r["direction"] in ("none",) or r["direction"].startswith("feat")]
feat = [r for r in feat if not r["subset"]]
if not feat:
    print("ainda sem células")
else:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for j, cond in enumerate(COND):
        rs = sorted([r for r in feat if r["cond"] == cond], key=lambda r: r["beta"])
        if not rs: axes[0, j].set_title(f"{COND[cond]} (sem dados ainda)"); continue
        B = [r["beta"] for r in rs]; M = [metrics(r["pred"], r["gold"], r["readable"]) for r in rs]
        a = axes[0, j]; a.plot(B, [m["nei_rate"] for m in M], "o-", color="#f0883e", label="taxa de NEI")
        a.plot(B, [m["acc"] for m in M], "s-", color="#2a78d6", label="acurácia"); a.axvline(0, color="gray", lw=.8)
        a.set_ylim(0, 1); a.set_title(COND[cond]); a.grid(alpha=.3)
        b = axes[1, j]; b.plot(B, [m["legivel"] for m in M], "o-", color="#555", label="coerente")
        b.plot(B, [np.mean(r["label_ok"]) for r in rs], "o--", color="#aaa", label="com rótulo (critério original)")
        b.plot(B, [m["err_nonNEI"] for m in M], "^-", color="#d9534f", label="erro fora de NEI")
        b.axhline(THR, color="gray", ls="--", lw=1); b.axvline(0, color="gray", lw=.8); b.set_ylim(0, 1.02); b.grid(alpha=.3)
        b.set_xlabel("α/R  (α = (α/R) × 8,375)")
    axes[0, 0].legend(); axes[1, 0].legend(); plt.tight_layout(); plt.show()""")

md(r"""### 5.2 Tabela completa por célula (500 claims)

**Como ler.** Uma linha por condição × α; colunas no formato do SA (Tabela 1) + taxa de NEI,
erro fora de NEI, coerência (critério principal) e "com rótulo" (critério original).""")

code(r"""rows = []
for r in sorted(feat, key=lambda r: (list(COND).index(r["cond"]), r["beta"])):
    m = metrics(r["pred"], r["gold"], r["readable"]); mp = metrics(r["pred_pipeline"], r["gold"], r["readable"])
    rows.append([COND[r["cond"]], r["beta"], m["acc"], m["macroF1"], m["F1_SUP"], m["F1_REF"], m["F1_NEI"], m["F1_CONF"],
                 m["nei_rate"], m["err_nonNEI"], m["legivel"], np.mean(r["label_ok"]), mp["acc"], mp["nei_rate"]])
pd.DataFrame(rows, columns=["condição", "α/R", "Acc", "F1", "F1 SUP", "F1 REF", "F1 NEI", "F1 CONF", "NEI", "erro não-NEI", "coerente", "com rótulo", "Acc (regra pipeline)", "NEI (regra pipeline)"]) \
  .set_index(["condição", "α/R"]).round(3) if rows else "ainda sem células"
""")

md(r"""### 5.3 α escolhido na validação e resultado na metade de teste

**Como ler.** Para cada condição e sentido, o α escolhido pelo critério da §2.3 (maior |α| com
coerência ≥ 80% na validação, `claim_id` par) e as métricas na metade de **teste**
(`claim_id` ímpar) contra o α = 0 na mesma metade.""")

code(r"""def half(r, parity):
    idx = [i for i, cid in enumerate(r["ids"]) if cid % 2 == parity]
    return [r["pred"][i] for i in idx], [r["gold"][i] for i in idx], [r["readable"][i] for i in idx]

rows, chosen = [], {}
for cond in COND:
    rs = {r["beta"]: r for r in feat if r["cond"] == cond}
    if 0.0 not in rs: continue
    base = metrics(*half(rs[0.0], 1))
    rows.append([COND[cond], "baseline", 0.0] + [base[k] for k in ("acc", "macroF1", "F1_NEI", "F1_CONF", "nei_rate", "err_nonNEI", "legivel")])
    for sense, sign in [("diminuir", -1), ("aumentar", 1)]:
        ok = [b for b, r in rs.items() if np.sign(b) == sign and np.mean(half(r, 0)[2]) >= THR]
        if not ok: continue
        b = max(ok, key=abs); chosen[(cond, sense)] = b; m = metrics(*half(rs[b], 1))
        rows.append([COND[cond], sense, b] + [m[k] for k in ("acc", "macroF1", "F1_NEI", "F1_CONF", "nei_rate", "err_nonNEI", "legivel")])
T = pd.DataFrame(rows, columns=["condição", "manipulação", "α/R", "Acc", "F1", "F1 NEI", "F1 CONF", "NEI", "erro não-NEI", "coerente"])
alt = {}
for cond in COND:
    rs = {r["beta"]: r for r in feat if r["cond"] == cond}
    for sense, sign in [("diminuir", -1), ("aumentar", 1)]:
        ok = [b for b, r in rs.items() if np.sign(b) == sign and
              np.mean([r["label_ok"][i] for i, cid in enumerate(r["ids"]) if cid % 2 == 0]) >= THR]
        if ok: alt[(cond, sense)] = max(ok, key=abs)
print("α/R escolhidos (validação) — critério COERENTE (principal):", chosen if chosen else "ainda sem células suficientes")
print("α/R que o critério original (COM RÓTULO) escolheria:      ", alt if alt else "ainda sem células suficientes")
T.set_index(["condição", "manipulação"]).round(3) if rows else None""")

md(r"""### 5.4 Controle: feature 19219 × 10 features aleatórias (100 claims de teste)

**Como ler.** Para cada condição e α escolhido: a taxa de NEI e a acurácia com a feature 19219
e a média (± desvio) com as 10 features aleatórias, nos mesmos 100 claims. Se a feature é
específica, as aleatórias devem ficar perto do α = 0.""")

code(r"""rand = [r for r in cells if r["direction"].startswith("rand") and r["subset"]]
if not rand:
    print("controle aleatório ainda não rodado (roda depois da grade da feature)")
else:
    rows = []
    for cond in COND:
        for b in sorted({r["beta"] for r in rand if r["cond"] == cond}):
            R_ = [r for r in rand if r["cond"] == cond and r["beta"] == b]; ids = set(R_[0]["ids"])
            fr = next((r for r in feat if r["cond"] == cond and r["beta"] == b), None)
            b0 = next((r for r in feat if r["cond"] == cond and r["beta"] == 0.0), None)
            sub = lambda r: ([p for i, p in zip(r["ids"], r["pred"]) if i in ids], [g for i, g in zip(r["ids"], r["gold"]) if i in ids])
            nei = [np.mean([p == "NEI" for p in r["pred"]]) for r in R_]; acc = [np.mean([p == g for p, g in zip(r["pred"], r["gold"])]) for r in R_]
            fp, fg = sub(fr) if fr else ([], []); bp, bg = sub(b0) if b0 else ([], [])
            rows.append([COND[cond], b, len(R_), np.mean([p == "NEI" for p in bp]) if bp else np.nan,
                         np.mean([p == "NEI" for p in fp]) if fp else np.nan, f"{np.mean(nei):.3f} ± {np.std(nei):.3f}",
                         np.mean([p == g for p, g in zip(fp, fg)]) if fp else np.nan, f"{np.mean(acc):.3f} ± {np.std(acc):.3f}"])
    display(pd.DataFrame(rows, columns=["condição", "α/R", "nº aleatórias", "NEI α=0", "NEI feature 19219", "NEI aleatórias",
                                        "acc feature 19219", "acc aleatórias"]).round(3))""")

md(r"""## 6. Limitações previstas

1. **Um modelo só (Llama).** DeepSeek e Qwen3 entram quando os SAEs v3 deles estiverem prontos.
2. **Baseline num ambiente diferente do pipeline original** — mitigado usando o nosso α = 0
   como referência e reportando a concordância (§4.1).
3. **A regra de rótulo do pipeline** transforma respostas sem rótulo em NEI e pega "FALSE" em
   qualquer ponto do texto como REF. Mantemos a regra por fidelidade, e reportamos a
   coerência e "com rótulo" à parte.
4. **Limiar de coerência (80%)** e a **lista de frases de recusa** são escolhas nossas (a lista foi definida depois das 2 primeiras células sem evidência e congelada — §2.3.1); a curva completa e o critério original mostram a sensibilidade.
5. **Controle aleatório em 100 claims**, como em Ferrando et al. (2025), não nos 500.
""")

md(r"""## Referências

- **Arditi, A., et al. (2024).** Refusal in Language Models Is Mediated by a Single Direction. *NeurIPS 2024. arXiv:2406.11717.*
- **Ferrando, J., Obeso, O., Rajamanoharan, S., & Nanda, N. (2025).** Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models. *ICLR 2025. arXiv:2411.14257.*
- **O'Brien, K., et al. (2024).** Steering Language Model Refusal with Sparse Autoencoders. *arXiv:2411.11296.*
- **Panickssery, N., et al. (2024).** Steering Llama 2 via Contrastive Activation Addition. *arXiv:2312.06681.*
- **Tan, D., et al. (2024).** Analysing the Generalisation and Reliability of Steering Vectors. *arXiv:2407.12404.*
- **Turner, A. M., et al. (2023).** Steering Language Models With Activation Engineering. *arXiv:2308.10248.*
- **Dietterich, T. G. (1998).** Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation, 10*(7), 1895–1923.
- **McNemar, Q. (1947).** Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. *Psychometrika, 12*(2), 153–157.
- **Fraga, L. M., & Kupssinskü, L. S. (2026).** Less is More: Improved Fact-Checking on AVeriTeC with Iterative Question Generation. *ENIAC 2026* (Seminário de Andamento, PPGCC/PUCRS).
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "03-steering-sae-v3.ipynb")
nbf.write(nb, out)
print("escrito:", out, "| células:", len(cells))
