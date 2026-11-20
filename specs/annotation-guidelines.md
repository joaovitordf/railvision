# RailVision AI — Guia Operacional de Anotação (Rascunho — Aguardando Portão P1)

> **Estado:** Rascunho operacional preparado na Fase 2. **Não iniciar anotação manual em escala** antes da aprovação do usuário no Portão **P1** (definição da camada de escopo viável A, B ou C).

---

## 1. Taxonomia de Vagões e Objetos

Para alinhar a detecção no RailVision AI ao escopo da v1 (*open-top freight wagons only*, Q3):

| Classe alvo | Uso no RailVision AI | Observação sobre TWIST |
|---|---|---|
| `freight_wagon_open` | Classe primária de interesse (vagões abertos de carga a granel: gôndolas, hoppers abertos) | Mapear apenas vagões abertos após inspeção de `Wagon` vs. `Freight Car` |
| `freight_wagon_other` | Vagões fechados, tanques, pranchas/contêineres (opcional como classe negativa/auxiliar) | Não entra em análise de material/carregamento |
| `locomotive` | Locomotivas (opcional para contextualizar início/fim da composição) | Não entra em análise de carga |

---

## 2. Identidade por Vídeo (`wagon_identity`)

- `track_id`: identificador **temporário** válido apenas dentro da mesma execução/vídeo.
- `sequence_index`: ordem de passagem observada no vídeo (`1, 2, ...`).
- `physical_id`: **`null`** no MVP (leitura de código físico por OCR está fora do escopo, Q14).
- **Fragmentação / Troca de ID:** se um mesmo vagão receber dois `track_id`s ou ocorrer *ID switch*, marcar o resultado para revisão (`track_fragmentation_suspected`).

---

## 3. Protocolo de Material (`material`) e Origem do Rótulo (`label_basis`)

### Classes iniciais provisórias (sujeitas à regra de admissão pós-auditoria)
- `iron_ore`
- `coal`
- `limestone`
- `sand`
- `other_material` *(somente se houver exemplos reais rotulados e visualmente coerentes)*

### Regras estritas de consistência
1. **Vagão vazio (`loading_level = empty`):** `material` **deve ser `null`** (nunca `unknown` nem classe de material).
2. **`unknown`:** reservado prioritariamente para **rejeição/abstenção do modelo** por baixa confiança, oclusão ou evidência visual insuficiente quando há carga visível.
3. **`label_basis` obrigatório por amostra:**
   - `visual_only`: material inferido apenas pela aparência visual no vídeo público (confiabilidade limitada; registrar em análises).
   - `source_metadata`: material descrito explicitamente pelo autor/descrição confiável da fonte.
   - `expert_confirmed`: confirmado por especialista ou fonte operacional verificada.

---

## 4. Protocolo de Preenchimento Visual (`visual_fill`) — Categorias Ordinais

> No MVP, **percentual numérico (`estimated_fill_percentage = null`) e volume/tonelagem física (`physical_volume_verified = false`) estão fora de escopo**. A referência visual abaixo serve apenas para consistência entre anotadores.

| Nível ordinal | Definição operacional visual (para câmera elevada/oblíqua com interior visível) | Regra com `material` |
|---|---|---|
| `empty` | Fundo/estrutura interna do vagão claramente expostos, sem carga a granel visível (ou apenas resíduo mínimo aderido ao piso) | `material = null` |
| `low` | Carga visível ocupando porção inferior do interior, muito abaixo da borda superior (referência visual aproximada: até ~25% da altura útil visível da caçamba) | Material ou `unknown` |
| `medium` | Carga ocupa faixa intermediária da parede interna visível, com folga nítida até a borda superior (referência aproximada: ~25% a ~50%) | Material ou `unknown` |
| `high` | Carga próxima às bordas superiores do vagão, sem ultrapassar ou com topo moderado (referência aproximada: ~50% a ~75%) | Material ou `unknown` |
| `very_high` | Carga atingindo o topo da borda ou formando pilha/coroamento (*heap*) acima do nível das laterais (referência aproximada: > ~75%) | Material ou `unknown` |

*(Caso os vídeos não mostrem as paredes internas do vagão, reduzir para `empty` vs. `loaded_coarse` ou suspender estimativa de nível na Camada B/C conforme decisão em P1.)*

---

## 5. Regras de Amostragem, Qualidade e Split

- **Deduplicação:** nunca anotar todos os frames consecutivos; amostrar por evento/deslocamento de vagão e checar *near-duplicates*.
- **Flags de incerteza:** marcar amostras com *motion blur*, sombra forte, oclusão parcial ou ângulo raso (`partial_occlusion`, `low_visibility`).
- **Revisão humana:** 100% de toda pré-anotação assistida por modelo deve passar por validação humana.
- **Split por grupo (obrigatório):** separar conjuntos de treino, validação e teste estritamente por **vídeo / sessão / autor-fonte**, garantindo zero vazamento entre splits.
