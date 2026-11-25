# RailVision AI — Especificação SDD de Reestruturação da Interface e Experiência (U0–U5)

- **Status:** Aprovada para execução (Fases U0–U5)
- **Data:** 2026-11-24
- **Relacionado a:** [requirements.md](requirements.md), [architecture.md](architecture.md), [ADR-0006](adr/ADR-0006-ui-redesign-and-annotation-studio-mvp.md), [openapi/railvision-v1.yaml](openapi/railvision-v1.yaml)

---

## 1. Auditoria da Interface Anterior (Baseline U0)

A inspeção do frontend Angular (`frontend/src/app/`) e dos fluxos ligados ao backend FastAPI identificou os seguintes problemas concretos:

1. **Bug estrutural na sidebar recolhida:** em [sidebar.component.ts](../frontend/src/app/core/layout/sidebar/sidebar.component.ts) e [shell.css](../frontend/src/styles/shell.css), `.rv-sidebar__header` mantinha o bloco da marca (`34px` + gap) lado a lado com `.rv-desktop-collapse-btn` dentro de `72px` recolhidos sem `overflow: hidden` controlado, comprimindo/deslocando o botão de alternância para fora da área útil. Em paralelo, `.rv-mobile-menu-btn` no header superior ficava oculto em desktop (`>= 1024px`) e o estado `sidebarCollapsed` residia localmente em `ShellComponent` sem persistência.
2. **Página inicial (`OverviewPageComponent`) sobrecarregada de metadados técnicos:** além dos 4 KPIs operacionais e inspeções recentes, exibia um banner permanente com `SHA-256`, `CUDAExecutionProvider` e duas tabelas inteiras (`Model Registry (Checkpoints)` e `Modelos registrados`), competindo com o uso operacional.
3. **Navegação fragmentada e redundância entre telas:** `Relatórios` ocupava item próprio na sidebar apesar de apenas listar inspeções concluídas para exportar JSON/CSV; `Vision Lab` ainda residia sob `features/onnx-lab` com nomenclatura histórica (`OnnxLabPageComponent`, 585 linhas) e fazia chamadas duplicadas de `/models/checkpoints` via `RuntimeStatusService` e `OnnxLabApiService`.
4. **Componentes monolíticos:** `InspectionDetailPageComponent` concentrava vídeo, canvas, resumo, vagões, observações e polling de job em **816 linhas**; `OnnxLabPageComponent` concentrava execução, comparação A/B e histórico em **585 linhas**; `LabRunViewerComponent` tinha **527 linhas**.
5. **Ausência de ambiente nativo de Anotação & Extração de Frames:** o fluxo dependia exclusivamente da exportação de inbox (`curate-frame`) + scripts CLI + CVAT externo para revisar qualquer caixa ou extrair frames deduplicados de vídeos próprios.

---

## 2. Nova Arquitetura de Informação e Navegação

Fluxo central do produto: **Acompanhar → Inspecionar → Anotar → Experimentar**.

| Área na Sidebar (Exatamente 4 itens) | Rota Principal | Sub-rotas e Compatibilidade | Papel Principal |
| :--- | :--- | :--- | :--- |
| **1. Visão geral** | `/overview` | `''`, `'**'` -> `/overview` | KPIs reais, atalhos rápidos, inspeções recentes e atividades reais |
| **2. Inspeções** | `/inspections` | `/inspections/:id`, `/inspections/:id/report`, `/reports` -> `/inspections?tab=reports` | Envio/filtro de vídeos, detalhe em 5 abas e exportação JSON/CSV |
| **3. Anotações** | `/annotations` | `/annotations/:projectId` | Studio MVP: projetos por campanha, extração/dedup de frames, editor `<canvas>`, pré-anotação IA, revisão humana e YOLO 1.1 `.zip` |
| **4. Laboratório** | `/lab` | `/lab/runs/:runId`, `/lab/models`, `/vision-lab` -> `/lab`, `/onnx-lab` -> `/lab` | Inferência imagem/vídeo, métricas reais, comparação A/B, histórico e Model Registry |

---

## 3. Requisitos Funcionais (Rastreáveis)

| ID | Requisito | Critério de Aceitação Verificável |
| :--- | :--- | :--- |
| `UFR-01` | Sidebar com exatamente 4 áreas principais | `NAVIGATION_ITEMS` possui 4 itens (`/overview`, `/inspections`, `/annotations`, `/lab`); rotas antigas redirecionam sem quebrar |
| `UFR-02` | Estado centralizado e controle confiável da sidebar | `LayoutStateService` persiste estado recolhido em `localStorage`; botão 3-linhas da topbar alterna desktop expandido/recolhido (e drawer mobile `<1024px`) sem desaparecer; tooltips no modo recolhido |
| `UFR-03` | Identidade visual dedicada e dual theme | Sidebar azul-marinho escura (`#0b192c`), área clara neutra (`#f8fafc`), azul primário (`#2563eb`), suporte completo claro/escuro (`data-theme`) |
| `UFR-04` | Visão Geral limpa e 100% real | Exibe 4 contagens reais, atalhos para Inspeções/Anotações/Laboratório, últimas inspeções e atividades reais cronológicas; zero tabelas de checkpoints/SHA no dashboard |
| `UFR-05` | Inspeções com busca, filtro, ordenação, duração e vagões | `GET /api/v1/inspections` entrega `durationMs` e `wagonCount`; listagem suporta busca, filtro por status, ordenação por data e aba integrada `Relatórios` |
| `UFR-06` | Detalhe de inspeção modularizado em abas | Abas `Resumo`, `Vídeo`, `Vagões`, `Observações`, `Relatório`; vídeo com overlay sincronizado + navegação por observações; material/carregamento `not_evaluated` nunca exibidos como classificação real |
| `UFR-07` | Laboratório unificado com 4 abas | Abas `Inferência`, `Comparação A/B`, `Histórico`, `Modelos`; seleção de checkpoint e runtime real, tracking opt-in distinto da detecção, métricas p50/p95 e preview A/B lado a lado |
| `UFR-08` | Model Registry e diagnóstico movidos para `/lab/models` | Tabela de checkpoints, Val/Test mAP50, SHA-256, `model_versions` e diagnóstico CUDA/ONNX acessíveis em `/lab/models`, com atalho "Testar modelo" |
| `UFR-09` | Projetos de Anotação isolados por campanha/origem | `/api/v1/annotations/projects` persiste metadados em `data/processed/annotation-studio/{id}/` sem alterar `data/processed/open-top-yolo-v1/` nem campanhas congeladas |
| `UFR-10` | Upload multi-imagem, upload de vídeo e inbox do Lab | Upload múltiplo de imagens com miniatura/SHA-256, upload de vídeo com metadados PyAV e importação deduplicada de `curation-inbox` |
| `UFR-11` | Extração configurável e deduplicada de frames de vídeo | Preview candidato com `sampleFps`, `maxFrames`, `dedupThreshold` e seleção manual dos frames a extrair em resolução original preservando `frameIndex` e `timestampMs` |
| `UFR-12` | Editor de bounding boxes em tela ampla (`<canvas>`) | Desenhar, selecionar, mover, redimensionar em 8 alças, excluir, zoom no cursor, pan, undo/redo, atalhos de teclado, estado sem objetos (`N`), taxonomia inicial `open_top_freight_wagon` |
| `UFR-13` | Pré-anotação assistida por IA sem sobrescrever revisão humana | Sugestões via `VisionRuntimeManager` (`exp04c` padrão) marcadas como `ai_suggestion` + `humanReviewed=false`; editor permite aceitar, editar ou rejeitar caixas |
| `UFR-14` | Curadoria por estados e exportação/importação CVAT YOLO 1.1 | Fluxo `pending -> pre_annotated -> in_review -> reviewed -> approved / rejected`; exportação `.zip` bloqueada por padrão se nenhum item tiver `humanReviewed=true` (`409`); importação `.zip` YOLO 1.1 suportada |

---

## 4. Requisitos Não Funcionais

- `UNFR-01` (**Fidelidade dos dados):** zero métricas inventadas, placeholders simulando sucesso ou contagens sintéticas; campos `not_evaluated` rotulados explicitamente.
- `UNFR-02` (**Preservação científica):** `vid_01`, `vid_09`, `vid_16` e `camp_vfli_cosne` protegidos contra ingestão silenciosa (`reservedBenchmarkWarning` em uploads/anotações); nenhum checkpoint ou experimento Fase 3/3.1/4 alterado.
- `UNFR-03` (**Modularidade e manutenção):** componentes de página abaixo de ~350 linhas sempre que possível, delegando lógica HTTP/negócio a serviços e subcomponentes coesos.
- `UNFR-04` (**Dependências enxutas):** reutilização de Angular 22 + Angular Material + Canvas 2D nativo no editor, evitando dependências pesadas externas para canvas/anotação.
- `UNFR-05` (**Acessibilidade e atalhos):** contraste adequado em claro/escuro, `:focus-visible`, `aria-label`, Escape fechando drawers/seleções e atalhos documentados no editor.
- `UNFR-06` (**Contratos OpenAPI e validação):** novas rotas `/api/v1/annotations/*` e campos aditivos de `/api/v1/inspections` documentados em [railvision-v1.yaml](openapi/railvision-v1.yaml) com erros RFC 9457.

---

## 5. Plano de Etapas (U0–U5)

- **Etapa U0:** Especificação SDD e `ADR-0006`.
- **Etapa U1:** Design tokens, shell centralizado (`LayoutStateService`), sidebar fixa de 4 áreas, topbar e redirecionamentos compatíveis.
- **Etapa U2:** Backend aditivo de inspeções (`durationMs`, `wagonCount`) + nova Visão Geral limpa + Inspeções com aba Relatórios + Detalhe modular em 5 abas.
- **Etapa U3:** Unificação `/lab` (`Inferência`, `Comparação A/B`, `Histórico`, `Modelos`) e remoção das duplicações `reports`/`onnx-lab`.
- **Etapa U4:** Módulo backend `/api/v1/annotations/*` + Annotation Studio MVP no frontend (Projetos, Extrator deduplicado de frames, Editor `<canvas>`, Pré-anotação IA e YOLO 1.1 `.zip`).
- **Etapa U5:** Suíte completa (`pip check`, `pytest` ML, `pytest` Backend + PostgreSQL, Vitest Frontend, `ng build` produção), verificação ao vivo com capturas reais no navegador e relatório final de 14 seções.
