# RailVision AI — Guia Operacional de Anotação v2.0

> **Estado:** **Ativa (v2.0 — Revisão WS0)** · Substitui o rascunho pré-P1 para governar a expansão `open-top-v2` e qualquer curadoria *Human-in-the-Loop* (HITL).
> Os manifestos e contagens históricas da Fase 3.1 (`specs/data-manifests/open-top-split-v1.json`) permanecem preservados intactos.

---

## 1. Taxonomia Única e Política de Negativos

Decisão arquitetural confirmada pelo usuário:

| Classe ID | Nome canônico | Uso no detector | Regra |
|---|---|---|---|
| `0` | `open_top_freight_wagon` | **Classe única monitorada** (gôndolas abertas e *hoppers* abertos de carga a granel) | Anotar todo vagão aberto visível dentro da regra geométrica da Seção 2 |
| *(nenhuma)* | *Hard negatives* explícitos | Locomotivas, vagões fechados, vagões-tanque, pranchas/contêineres, trens de passageiros, equipamentos de via, fundo sem trem | **NÃO anotar caixa.** Manter imagem com arquivo `.txt` vazio como negativo explícito |

> Não criar classes adicionais (`locomotive`, `closed_wagon`, etc.) nesta iteração; o objetivo do detector é delimitar com máxima precisão apenas `open_top_freight_wagon`.

---

## 2. Regra Geométrica da *Bounding Box* (`open_top_freight_wagon`)

Na auditoria da Fase 3.1 (`phase3_1-open-top-report.md`), verificou-se que caixas apertadas apenas na caçamba superior contra GT cobrindo o vagão inteiro reduziram o IoU para `0,35–0,45`, causando falso positivo e falso negativo simultâneos.

Para eliminar ambiguidade entre anotadores e pré-anotação assistida:
1. **Corpo completo visível:** a *bounding box* deve envolver o **vagão aberto completo visível** — borda superior da caçamba (e pilha visível da carga, se houver), laterais, longarina/chassi inferior e truques/rodados visíveis.
2. **Acopladores / engates:** encerrar a caixa na face externa do vagão/testeria, evitando invadir o corpo do vagão adjacente.
3. **Vagões parcialmente fora de quadro ou ocluídos:** anotar apenas se `>= 30%` do vagão estiver visível e inequivocamente identificável como vagão aberto (*open-top*); não chutar caixas em vagões distantes sem silhueta clara de vagão aberto.
4. **Oclusão entre vagões adjacentes:** em ângulos oblíquos onde a lateral inferior está coberta pela via adjacente ou vegetação, ajustar a caixa até o limite visível do vagão (sem extrapolar para além da imagem).

---

## 3. Identidade por Vídeo (`wagon_identity`) e GT de Tracking (`MOT 1.1`)

- `track_id`: inteiro `>= 1` dentro do clipe anotado (`MOT 1.1`), temporário e válido apenas naquela sequência.
- `sequence_index`: ordem de passagem observada no vídeo (`1, 2, ...`).
- `physical_id`: **`null`** no MVP (OCR deferido).
- Cada frame de um clipe GT de tracking deve preservar IDs consistentes mesmo após oclusão curta por poste/pórtico (marcar `track_fragmentation_suspected` se houver dúvida na avaliação automática).

---

## 4. Isolamento Obrigatório de Benchmarks e Vídeos Próprios

1. **Benchmarks congelados (uso estritamente proibido em `train` ou `val`):**
   - `vid_09` + `vid_16` (`camp_vfli_cosne`): **congelados exclusivamente no split `test`** para medir generalização para *hoppers* europeus inéditos e benchmarking MOT.
   - `vid_01` (`vid_01_langer_unbeladener_g_terzug_von_oben_k_l`, 4K aéreo vazio): **reservado exclusivamente como benchmark negativo independente (`test_negative_only`)** para medir regressão em falsos positivos (`0 FP` exigido no Portão G3).
   - `specs/data-manifests/open-top-split-v1.json` e `data/processed/open-top-yolo-v1/` permanecem preservados intactos.
2. **Vídeos próprios (`source_type = user_owned`):**
   - Inclusão no dataset exige registro explícito em `specs/data-manifests/user-videos-manifest.json` com `origin`, `campaign`, `provenance` e `target_split`.
   - Uploads exploratórios realizados no Laboratório de Visão ou na tela de Inspeções **nunca** entram automaticamente em treino.
   - Um mesmo vídeo/campanha ou SHA-256 jamais pode aparecer em mais de um split.
3. **Pré-anotação assistida e curadoria HITL:**
   - Pré-anotações geradas localmente (ex.: `exp04c` + modelo *open-vocabulary* local) e frames capturados via `curate-frame` exigem **100% de revisão humana no CVAT** (`--confirm-human-reviewed`) antes da ingestão no dataset versionado.

---

## 5. Protocolo de Material (`material`) e Nível de Carregamento (`visual_fill`) — Fase 5 (Planejado)

> **Status:** Pendente de expansão de fontes (WS4). Enquanto modelos da Fase 5 não estiverem treinados e validados, o sistema grava `not_evaluated`.

### Classes provisórias de material (admissão condicionada a `>= 2` campanhas independentes)
- `iron_ore`, `coal`, `limestone`, `sand`, `other_material` (+ rejeição `unknown`; **vagão `empty` exige `material = null`**).
- **`label_basis` obrigatório por amostra:** `visual_only`, `source_metadata` ou `expert_confirmed`.

### Níveis ordinais de preenchimento visual (`visual_fill`)
> Percentual numérico (`estimated_fill_percentage = null`) e volume físico (`physical_volume_verified = false`) estão fora de escopo no MVP.

| Nível ordinal | Definição operacional visual (câmera elevada/oblíqua com interior visível) | Regra com `material` |
|---|---|---|
| `empty` | Fundo/estrutura interna visivelmente expostos, sem carga a granel visível (ou apenas resíduo mínimo) | `material = null` |
| `low` | Carga ocupando porção inferior do interior, muito abaixo da borda superior (~até 25% da altura útil visível) | Material ou `unknown` |
| `medium` | Carga na faixa intermediária da parede interna visível, com folga nítida até a borda (~25% a ~50%) | Material ou `unknown` |
| `high` | Carga próxima às bordas superiores do vagão (~50% a ~75%) | Material ou `unknown` |
| `very_high` | Carga atingindo o topo da borda ou formando pilha/coroamento (*heap*) acima das laterais (> ~75%) | Material ou `unknown` |

---

## 6. Regras de Amostragem, Qualidade e Split

- **Deduplicação intra-campanha:** amostrar por deslocamento/evento e aplicar filtragem *dHash* antes de extrair frames para o CVAT.
- **Teto por campanha:** nenhuma campanha isolada deve dominar o conjunto de treino (teto alvo `<= 30–35%` das caixas positivas na expansão v2).
- **Split estrito por grupo (`campaign` / `author_or_owner`):** zero vazamento entre `train`, `val` e `test`.
