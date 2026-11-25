# ADR-0004: Model Registry, Comparação de Execuções e Curadoria HITL

- **Status:** Aceita (WS1)
- **Data:** 2026-11-24
- **Relacionado a:** [model-improvement-protocol.md](../experiments/model-improvement-protocol.md), [annotation-guidelines.md](../annotation-guidelines.md), [railvision-v1.yaml](../openapi/railvision-v1.yaml)

## Contexto

Com a migração para Python + FastAPI concluída e o endurecimento do runtime (WS0), o projeto entra no ciclo iterativo de melhoria do detector `open_top_freight_wagon`. O laboratório precisa comparar múltiplos checkpoints (ex.: `exp04a`, `exp04b`, `exp04c`, futuros `exp05`/`exp06`) de forma reproduzível em uma única GPU local de 8 GB VRAM (`NVIDIA RTX 4060 Ti`) e encaminhar casos difíceis para revisão humana no CVAT sem contaminar splits de treino/teste.

## Decisões

1. **Model Registry em `artifacts/models/registry.json`:**
   - Fonte de verdade declarativa com caminhos relativos à raiz do repositório, SHA-256, experimento de origem, arquitetura, resolução de entrada, taxonomia (`{"0": "open_top_freight_wagon"}`), status (`active`, `candidate`, `archived`) e métricas registradas em `val`/`test`.
   - `exp04c_yolo26n_opentop_640` permanece como checkpoint `active` padrão até que um candidato seja promovido explicitamente após cumprir critérios em `val` e Portão G3.
   - Checkpoints legados de outra taxonomia (TWIST 2 classes: `exp01`–`exp03`) não são expostos automaticamente no seletor do detector de vagões graneleiros.
   - Endpoint de consulta `GET /api/v1/models/checkpoints` independente de banco PostgreSQL para operar também em modo laboratório local.

2. **Gerenciamento de VRAM em Multi-Modelo (`VisionRuntimeManager`):**
   - Cache LRU estrito de **no máximo 1 modelo PyTorch em `cuda:0` por vez**.
   - Ao trocar de `modelId` em CUDA, o modelo anterior é removido da cache e `torch.cuda.empty_cache()` é invocado sob lock para evitar OOM em 8 GB de VRAM.

3. **Rotas canônicas `/api/v1/vision-lab/*` e alias `/api/v1/onnx-spike/*`:**
   - Todos os endpoints do laboratório passam a ser expostos sob `/api/v1/vision-lab/*` (e `/vision-lab` no frontend), mantendo `/api/v1/onnx-spike/*` e `/onnx-lab` como aliases compatíveis.
   - Adição de seleção explícita de `?modelId=` em `/status`, `/analyze-image` e `/analyze-video`, e parâmetro opcional `ProcessInspectionRequest` em `POST /api/v1/inspections/{id}/process`.

4. **Comparação A/B de Execuções (`GET /api/v1/vision-lab/compare?runA=&runB=`):**
   - Calcula lado a lado métricas de modelo, runtime, dispositivo, thresholds, total de detecções, tracks únicos, FPS, latência média/p95 e deltas explícitos, incluindo comparação frame a frame quando ambas as execuções são vídeos.

5. **Curadoria HITL (`POST /api/v1/vision-lab/runs/{runId}/curate-frame`):**
   - Extrai **sempre o frame limpo do arquivo original** (`original.<ext>` via PyAV/OpenCV, nunca o vídeo anotado) para `data/processed/curation-inbox/<YYYY-MM-DD>/`.
   - Gera par `<run>_f<frame>.jpg` + `<run>_f<frame>.txt` (pré-anotação YOLO `0 cx cy w h`) + `<run>_f<frame>.meta.json` com `"human_reviewed": false` e `"auto_train_eligible": false`.
   - Nenhum arquivo enviado para a caixa de curadoria entra automaticamente em `data/processed/open-top-yolo-v1/` ou em qualquer manifest de treino sem revisão humana explícita e passagem pelo validador de governança.
