# Datasets de fact-checking — levantamento e seleção (treino vs teste)

Levantamento da literatura para escolher datasets do estudo de **feature de abstenção**
com SAE de domínio. Critérios:
- **TREINO do SAE:** claim + evidência em **texto (inline/renderável)**, de preferência
  com **classe NEI** (not enough info), e **volume**.
- **TESTE / generalização:** claim + evidência + **rótulo-ouro com NEI**, volume menor
  OK, **domínio/estilo diverso** (transferência out-of-distribution).
- **Não serve:** claim-only sem documento de evidência (ex.: LIAR), sem noção de
  verificabilidade/NEI, ou evidência só em **tabelas**.

## Tabela-resumo

| Nome | Tamanho | Domínio | Tem NEI? | Evidência | HF id | Veredito |
|---|---|---|---|---|---|---|
| **VitaminC** | ~489k pares | Wikipedia (edits) | Sim | sentença **inline** | `tals/vitaminc` | **TREINO** (em uso) |
| **FEVER** | ~185k | Wikipedia | Sim | ponteiros→dump; **inline** via copenlu | `fever/fever`, `copenlu/fever_gold_evidence` | **TREINO** (em uso) |
| FEVEROUS | 87k | Wikipedia (sent+tabelas) | Sim | sentenças + tabelas (dump) | `fever/feverous` | TREINO 2º (trabalhoso) |
| **AVeriTeC** | 4.568 | Web aberta | Sim (NEE+Conflicting) | Q&A **inline** | `chenxwh/AVeriTeC`, `pminervini/averitec` | **TESTE** (em uso) |
| **Climate-FEVER** | 1.535 / 7.675 pares | Clima | Sim | 5 sentenças **inline** | `tdiggelm/climate_fever` | **TESTE** ⭐ |
| **SciFact** | 1.409 | Biomédico (abstracts) | Sim (NOINFO) | abstracts (join) | `allenai/scifact` | **TESTE** ⭐ |
| **HealthVer** | 1.855 / 14k pares | Saúde/COVID | Sim (NEUTRAL) | sentenças **inline** | GitHub sarrouti | **TESTE** ⭐ |
| PUBHEALTH | ~11.8k | Saúde pública | Parcial (unproven) | `main_text` inline | `ImperialCollegeLondon/health_fact` | TESTE marginal |
| X-Fact | 31.189 | Multilíngue (25 lg) | Sim (UNVERIFIABLE) | snippets recuperados | GitHub utahnlp | TESTE marginal |
| DanFEVER | 6.407 | Dinamarquês | Sim | estilo FEVER (dump) | figshare | TESTE marginal |
| WICE | ~1.9k (incerto) | Wikipedia + web citada | Parcial | artigos web inline | GitHub ryokamoi | TESTE marginal |
| QuanTemp | 15.514 | Web (numérico) | Parcial (Conflicting) | snippets web | GitHub | TESTE marginal |
| CHEF | ~10k | Chinês | Sim | recuperada da web | GitHub THU-BPM | TESTE marginal |
| HoVer | ~26k | Wikipedia (multi-hop) | **Não** | ponteiros→dump | — | Não-usável |
| FaVIQ | ~188k | Wikipedia (QA) | **Não** | passagens Wiki | GitHub faviq | Não-usável (sem NEI) |
| COVID-Fact | 4.086 | COVID (Reddit) | **Não** | inline | GitHub asaakyan | Não-usável |
| MultiFC | ~34.9k | Web (26 sites) | **Não** (165 rótulos) | snippets web | `pszemraj/multi_fc` | Não-usável |
| LIAR | 12.836 | Política | **Não** | **nenhuma** | várias | Não-usável |
| LIAR-PLUS | ~12.8k | Política | **Não** | justificativa (vaza veredito) | várias | Não-usável |
| TabFact | 117.854 | Wikipedia (tabelas) | **Não** | **tabelas** | wenhuchen (GitHub) | Não-usável |
| SciTab | ~1.2k | Tabelas científicas | Sim | **tabelas** | GitHub | Não-usável (tabela) |
| RAWFC/LIAR-RAW | 2.012 / ~12.8k | Snopes/Política | **Não** | raw reports inline | GitHub | Não-usável |

## Recomendações

**Treino (além de VitaminC + FEVER):** as opções limpas são escassas.
- **FEVEROUS (porção de sentenças)** é o único grande com NEI + evidência textual, mas
  exige montar o dump e descartar tabelas. Adicionar só se quiser mais volume/estilo.
- Evitar: FaVIQ e HoVer (grandes, mas **sem NEI** — não ensinam abstenção); tabulares.

**Teste / generalização (além de AVeriTeC) — o maior valor:**
1. **SciFact** — científico/biomédico; tem NOINFO; evidência = abstracts.
2. **Climate-FEVER** — clima; NEI nativo; 5 sentenças inline no HF (*plug-and-play*).
3. **HealthVer** — saúde/COVID; classe NEUTRAL ≈ abstenção; evidência inline.
- Secundários: PUBHEALTH, X-Fact/DanFEVER (multilíngue), WICE/QuanTemp (estilos distintos).

Rationale: SciFact + Climate-FEVER + HealthVer dão **três domínios OOD** (científico,
clima, saúde), todos com classe de abstenção genuína e evidência textual — o setup ideal
para testar se a feature de abstenção generaliza.

**A validar antes de usar cada um:** confirmar no HF viewer se a evidência-texto vem
inline ou exige join com corpus (confirmado inline: VitaminC, Climate-FEVER,
`copenlu/fever_gold_evidence`; exigem montagem: FEVER padrão, FEVEROUS, SciFact).

## Fontes
FEVER (NAACL'18), FEVEROUS (NeurIPS'21), VitaminC (NAACL'21), HoVer (EMNLP'20),
SciFact (EMNLP'20), HealthVer (EMNLP'21), PUBHEALTH (EMNLP'20), COVID-Fact (ACL'21),
AVeriTeC (NeurIPS'23), MultiFC (EMNLP'19), Climate-FEVER (NeurIPS CCAI'20), LIAR
(ACL'17), LIAR-PLUS (FEVER-WS'18), X-Fact (ACL'21), CHEF (NAACL'22), DanFEVER
(NoDaLiDa'21), TabFact (ICLR'20), SciTab (EMNLP'23), FaVIQ (ACL'22), WICE (EMNLP'23),
QuanTemp (SIGIR'24).
