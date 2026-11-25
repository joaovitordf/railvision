# ADR-0005: Avaliação Arquitetural de um Futuro Annotation Studio Nativo no RailVision

- **Status:** Avaliada — **Não implementar no WS0/WS1; manter CVAT como editor principal no WS2–WS3**
- **Data:** 2026-11-24
- **Relacionado a:** [ADR-0004](ADR-0004-model-registry-and-hitl-curation.md), [annotation-guidelines.md](../annotation-guidelines.md), [model-improvement-protocol.md](../experiments/model-improvement-protocol.md)

## Contexto

Avalia-se se o RailVision deve incorporar um editor próprio de anotações (imagens, vídeos com tracking e gestão de curadoria) no frontend Angular + backend FastAPI, ou continuar utilizando o **CVAT** apoiado pelas ferramentas construídas no WS1.

## Análise das 6 Questões-Chave

### 1. O WS1 já resolve parte suficiente desse problema?
**Parcialmente — resolve toda a camada de orquestração e pré-anotação, mas não a edição interativa de caixas.**
- **Já resolvido no WS1:** registro auditado de vídeos próprios ([register_user_video.py](../../ml/scripts/register_user_video.py)), extração deduplicada ([extract_dataset_packs.py](../../ml/scripts/extract_dataset_packs.py)), pré-anotação local GPU/CPU compatível com CVAT YOLO 1.1 e MOT 1.1 + ranking de incerteza ([preannotate_cvat.py](../../ml/scripts/preannotate_cvat.py)), captura de frames limpos no Vision Lab (`POST /api/v1/vision-lab/runs/{id}/curate-frame`), e bloqueio rígido de ingestão sem revisão humana ([validate_cvat_exports.py](../../ml/scripts/validate_cvat_exports.py)).
- **Não resolvido no WS1 (intencionalmente):** arrastar, redimensionar, criar ou remover bounding boxes diretamente no navegador e ajustar trajetórias multi-frame.

### 2. Vale desenvolver um editor próprio ou continuar utilizando o CVAT?
**Continuar utilizando o CVAT como ferramenta principal durante o WS2 e WS3.**
- Reproduzir um editor completo de imagem + vídeo + tracking no Angular introduziria meses de engenharia de UI e dívida técnica (sincronização precisa de seek/frame em HTML5 `<video>`, interpolação linear de tracks, undo/redo, atalhos, zoom/pan subpixel e controle de estados por frame) **sem melhorar diretamente mAP50, MOTA ou IDF1**.
- O gargalo atual do projeto é qualidade/diversidade de dados e validação do modelo (Portão G3), não falta de interface de desenho.

### 3. Qual seria o menor MVP capaz de economizar trabalho manual de verdade?
Um **Micro-Revisor de Imagem Única (`HITL Queue Reviewer`)** focado apenas na fila `data/processed/curation-inbox/` e nos pacotes de alta incerteza (`uncertainty_ranking.json`), contendo apenas:
1. Listagem dos frames pendentes agrupados por `campaign_id` / `target_split`.
2. Renderização da imagem limpa + sobreposição de caixas YOLO em `<canvas>`.
3. Ações mínimas por frame: **Aprovar sugestão**, **Ajustar/Mover/Redimensionar/Criar/Excluir caixa única-classe (`open_top_freight_wagon`)**, **Marcar frame como negativo vazio**, **Rejeitar para descarte**.
4. Gravação do `.txt` YOLO revisado + atualização explícita de `"humanReviewed": true` com revisor e timestamp, mantendo exportação/importação YOLO 1.1 e CVAT.

> **Fora desse MVP:** timeline de vídeo quadro a quadro e interpolação de tracking — manter vídeos longos e MOT 1.1 no CVAT.

### 4. Quais funcionalidades exigiriam mais esforço e risco?
1. **Edição de vídeo e trajetórias (maior custo):** navegadores decodificam vídeo por timestamps/keyframes, não por `frame_index` exato garantido em todos codecs; exigir seek frame-a-frame fiel demanda proxy intra-frame ou extração de todos os frames para JPEG em disco.
2. **Propagação e correção de IDs de tracking (merge/split de `trackId`, keyframe interpolation, oclusão parcial):** equivaleria a reescrever o módulo mais complexo do CVAT.
3. **Ergonomia de rotulagem rápida:** zoom centrado no cursor, pan com botão do meio, undo/redo em lote, atalhos de teclado e prevenção de perda de progresso exigem engenharia dedicada para serem mais rápidos do que abrir o CVAT.

### 5. Quais bibliotecas Angular seriam adequadas?
- **Opção A (Recomendada para MVP leve): `konva` + wrapper fino em Angular Signals (`HTML5 Canvas` gerenciado)** ou **Canvas 2D nativo com transformações de matriz (`zoom/pan`)** — bundle pequeno, sem conflito com Angular 22 standalone/signals/zoneless, controle total sobre coordenadas normalizadas YOLO e zero estilo opinionado de DOM.
- **Opção B: `@annotorious/annotorious` (OpenSeadragon/Image)** — boa opção para imagens estáticas e retângulos rápidos, porém menos flexível caso queira acoplar metadados específicos do RailVision (confiança do modelo, razão de incerteza, badge de benchmark reservado).
- **Opção C: `fabric`** — madura, mas bundle maior e API orientada a estado mutável fora do modelo reativo de Angular Signals.

### 6. Como reaproveitar Model Registry e a infraestrutura do WS1?
- **Catálogo de modelos (`ModelRegistryService` + `GET /api/v1/models/checkpoints`):** seleciona qual checkpoint gera pré-anotações sob demanda e grava `modelId`, `modelSha256` e thresholds em cada item da fila.
- **Pipeline de pré-anotação (`VisionRuntimeManager` + `preannotate_cvat.py`):** gera caixas iniciais, ranking de incerteza (`uncertainty_ranking.json`) e empacotamento CVAT YOLO 1.1 `.zip`.
- **Inbox de curadoria e governança (`curate-frame` + `validate_cvat_exports.py`):** toda imagem editada continuaria respeitando separação por `campaign_id` / `target_split`, aviso `reservedBenchmarkWarning` e exigência de revisão humana 100% concluída antes da ingestão em `data/processed/`.

## Decisão

Não implementar o Annotation Studio no WS0/WS1. Utilizar o fluxo **Vision Lab (`curate-frame`) + `preannotate_cvat.py` + CVAT + `validate_cvat_exports.py --confirm-human-reviewed`** no WS2, e reavaliar somente o **MVP de revisão em imagem única** após o Portão G3 caso o volume de triagem recorrente em campo justifique evitar abertura frequente do CVAT.
