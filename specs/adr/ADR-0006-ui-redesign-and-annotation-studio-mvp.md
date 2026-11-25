# ADR-0006: Reestruturação da Interface (4 Áreas) e Annotation Studio MVP Nativo

- **Status:** Aceita e em implementação (Etapas U0–U5)
- **Data:** 2026-11-24
- **Substitui parcialmente:** [ADR-0005](ADR-0005-annotation-studio-evaluation.md) *(somente quanto ao adiamento do micro-revisor de imagem única; mantém o editor temporal completo de vídeo/tracking fora do MVP e preserva compatibilidade CVAT)*
- **Relacionado a:** [ui-redesign-sdd.md](../ui-redesign-sdd.md), [ADR-0004](ADR-0004-model-registry-and-hitl-curation.md)

## Contexto

A plataforma evoluiu de protótipo de migração Java -> Python para uso contínuo na expansão do conjunto de dados (WS2). Contudo, a interface Angular anterior sofria de quatro problemas estruturais:
1. O menu lateral acumulava áreas de suporte (`Relatórios`) enquanto escondia a necessidade real de revisão de frames/caixas; o botão de recolhimento sofria clipping ao comprimir a sidebar para `72px`.
2. A página inicial expunha hashes SHA-256, `CUDAExecutionProvider` e duas tabelas de modelos que pertencem ao ambiente de experimentação.
3. Os componentes `InspectionDetailPageComponent` (816 linhas) e `OnnxLabPageComponent` (585 linhas) concentravam múltiplas responsabilidades.
4. A revisão humana dependia 100% de rodar scripts CLI e subir o CVAT mesmo para revisar poucas dezenas de frames extraídos ou sugeridos por `exp04c`.

## Decisões Arquiteturais

### 1. Navegação enxuta em 4 áreas principais
Consolidar o menu lateral exclusivamente em:
- **Visão geral (`/overview`)**: estado operacional e atalhos reais;
- **Inspeções (`/inspections`)**: histórico operacional, envio de vídeo, detalhes em 5 abas (`Resumo`, `Vídeo`, `Vagões`, `Observações`, `Relatório`) e aba integrada de relatórios agregados (`?tab=reports`);
- **Anotações (`/annotations`)**: workspace de projetos, extração deduplicada de frames de vídeo e editor de bounding boxes com pré-anotação assistida;
- **Laboratório (`/lab`)**: inferência exploratória imagem/vídeo, comparação A/B, histórico e `Model Registry` (`/lab/models`).

Rotas legadas (`/reports`, `/vision-lab`, `/onnx-lab`) são mantidas via redirecionamento direto para evitar links quebrados.

### 2. Estado centralizado da shell via `LayoutStateService`
Em vez de estado isolado em `ShellComponent` e botões concorrentes, um serviço root (`LayoutStateService`) gerencia `collapsed` (persistido em `localStorage`), `mobileOpen`, `isMobile` (`matchMedia`) e `fullBleed` (ativado automaticamente no editor de anotações). O botão de menu de três linhas permanece sempre visível na barra superior como controle primário previsível em qualquer resolução.

### 3. Armazenamento isolado em filesystem para o Annotation Studio (`data/processed/annotation-studio/`)
- **Alternativa rejeitada:** adicionar 4 tabelas novas no PostgreSQL para projetos/itens/caixas de anotação antes da estabilização do workflow de WS2. Isso exigiria nova migração Alembic para rascunhos altamente mutáveis que já possuem representação canônica em arquivos YOLO/JSON no pipeline ML.
- **Decisão adotada:** persistência atômica por projeto em `data/processed/annotation-studio/{project_id}/` (`project.json` + imagens limpas + miniaturas + vídeos + exports `.zip`), completamente isolada de `data/processed/open-top-yolo-v1/` e de `data/raw/`. A promoção para datasets oficiais continua exigindo revisão humana explícita (`humanReviewed = true`) e ingestão validada.

### 4. Editor nativo com HTML5 `<canvas>` + PointerEvents (sem nova biblioteca externa)
Conforme avaliado na Opção A da `ADR-0005`, utilizamos Canvas 2D nativo + `PointerEvents` acoplado a Angular Signals (`AnnotationCanvasComponent`). Benefícios: zero novas dependências npm no `package.json`, compatibilidade garantida com o build Angular 22 / Rolldown no Windows, zero sobrecarga de bundle e controle preciso sobre coordenadas YOLO normalizadas (`[x, y, width, height]`), alças de redimensionamento, zoom ancorado no cursor e distinção visual entre sugestões de IA e caixas confirmadas por humanos.

## Limite Explícito do MVP vs. CVAT
- **Incluído no MVP (U4):** criação de projetos por campanha/origem, upload multi-imagem, upload de vídeo com extração configurável e deduplicação por diferença visual média (`mean abs diff`), editor de bounding boxes de imagem completa com undo/redo/zoom/pan, pré-anotação via `VisionRuntimeManager` (`exp04c` padrão) e exportação/importação CVAT YOLO 1.1 `.zip` com trava `humanReviewed`.
- **Diferido (permanece no CVAT até justificativa pós-G3):** interpolação de caixas quadro a quadro em vídeo e edição de trajetórias temporais (`MOT 1.1`).

## Consequências
- **Positivas:** fluxo completo *Acompanhar -> Inspecionar -> Anotar -> Experimentar* em interface única; redução drástica de atrito para triagem de falsos positivos/negativos e extração de frames de vídeos próprios; componentes modulares menores e testáveis.
- **Cuidados:** manter sincronia rigorosa entre o contrato OpenAPI (`railvision-v1.yaml`), backend `/api/v1/annotations/*` e as interfaces TypeScript em `annotations.models.ts`.
