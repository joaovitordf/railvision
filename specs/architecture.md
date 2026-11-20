# RailVision AI — Arquitetura do Sistema e Diretrizes de UI

**Documento fonte:** Especificação Unificada (atualizada pela ADR-0003)  
**Última atualização:** 2026-11-20  

## 1. Visão Arquitetural

O sistema adota um **monorepo local modular** composto por:
- **`frontend/`**: aplicação Angular 22.x (*standalone components* + Angular Material/CDK) servida em `localhost:4200`.
- **`backend/`**: **backend unificado Python 3.12 + FastAPI + SQLAlchemy 2.0 + Alembic**, único proprietário de regras de negócio, armazenamento local, processamento de mídia via **PyAV**, inferência/rastreamento e schema PostgreSQL, escutando em `127.0.0.1:8080`.
- **`database`**: PostgreSQL `17.12-alpine` em Docker Compose local (`127.0.0.1:5432`) com volume persistente e *healthcheck*.
- **`ml/`**: ferramental offline de auditoria, preparação de datasets, treinamento e exportação ONNX compartilhando o ambiente `ml/.venv` com versões fixadas.

## 2. Módulos Internos do Backend (`backend/app`)

| Módulo | Responsabilidade |
|---|---|
| `core` | Configurações tipadas (`pydantic-settings`), resolução de caminhos, schema `CamelModel` e erros RFC 9457 (`ProblemDetail`) |
| `db` | Base declarativa SQLAlchemy 2.0, modelos ORM espelhando `inspections`, `processing_jobs`, `model_versions`, `wagon_results` e `wagon_observations` + Alembic |
| `health` | Probes `/health` e `/actuator/health` com verificação real de storage e disponibilidade CUDA |
| `media` | Validação de extensão/assinatura binária, escrita em stream com SHA-256, extração de metadados PyAV e streaming HTTP (`200` / `Range 206` / `416`) |
| `inspections` | Ciclo de vida de inspeções (`/api/v1/inspections*`) |
| `jobs` | Fila assíncrona (`ThreadPoolExecutor`), `VisionInspectionInferenceEngine` (PyTorch CUDA + PyAV + ByteTrack), recuperação no startup (`PROCESS_INTERRUPTED`) e consulta `/api/v1/jobs/{id}` |
| `results` | Persistência/consulta de vagões, observações e versões de modelos reais (`/api/v1/models`) |
| `vision` | Pré/pós-processamento determinístico, gerenciador explícito de runtimes (`pytorch_cuda`, `pytorch_cpu`, `onnx_cpu`), `ByteTrackSession` por vídeo e encoding MP4/H.264 controlado (`libx264`, CRF 23) |
| `lab` | Endpoints `/api/v1/onnx-spike/*` para testes locais ad-hoc de imagens e vídeos com comparação entre runtimes e resoluções `1080p` / `native` |

## 3. Fluxo de Processamento Assíncrono e Recuperação

1. `POST /api/v1/inspections` valida, armazena o vídeo em `data/processed/app-storage/uploads/<uuid>.<ext>`, extrai metadados técnicos reais via PyAV (`duration_ms`, `width_px`, `height_px`, `fps`, `codec`) e cria a inspeção (`UPLOADED`).
2. `POST /api/v1/inspections/{id}/process` verifica ausência de job ativo (`QUEUED`/`RUNNING`), grava `ProcessingJob(QUEUED, engine_type='VISION_PYTORCH_CUDA')`, faz commit antes de despachar e retorna `202 Accepted`.
3. `JobExecutionManager` transita para `RUNNING`, decodifica o vídeo na resolução original com PyAV, executa detecção no runtime configurado (padrão `pytorch_cuda` na GPU RTX 4060 Ti, sem fallback silencioso) e rastreamento `ByteTrack`, gravando observações reais com `material='not_evaluated'` e `loading_level='not_evaluated'` até o treinamento da Fase 5.
4. **Recuperação ao reiniciar:** `recover_interrupted_jobs` no lifespan do FastAPI marca jobs `RUNNING`/`QUEUED` interrompidos como `FAILED` (`PROCESS_INTERRUPTED`) e garante o registro dos modelos ativos não-placeholder em `model_versions`.

## 4. Schema PostgreSQL (Migrations Alembic + Coexistência)

- `inspections`: metadados técnicos reais do vídeo (`duration_ms`, `width_px`, `height_px`, `fps`, `codec`, checksum SHA-256, `status`, `latest_job_id`).
- `processing_jobs`: histórico de execuções por inspeção, `engine_type`, `is_demo_simulation`, `error_code`, `error_message`.
- `model_versions`: catálogo de versões ativas (`task`, `name`, `version`, `runtime`, `weights_sha256`, `artifact_ref`, `is_demo_placeholder`).
- `wagon_results` e `wagon_observations`: consolidação por `track_id`, timestamps reais, bounding boxes normalizadas e sinalização explícita `not_evaluated` / `review_required=true` quando aplicável.

## 5. Diretrizes de UI e Laboratório de Visão (`frontend/src/app/`)

- **Identidade visual e shell:** Angular 22.x + Angular Material, 2 colunas (*sidebar* com badge dinâmico GPU/CPU, *topbar*, área principal e *footer*).
- **Páginas principais:** `Visão geral` (exibindo hardware/runtime ativo real), `Inspeções`, `Detalhe da inspeção` (player `HTTP Range` + overlays sincronizados por timestamp), `Relatórios` e `Laboratório de Visão` (`/onnx-lab`, com seleção entre `PyTorch CUDA`, `PyTorch CPU`, `ONNX Runtime CPU`, saída `1080p` / `native` via `libx264` e `ByteTrack` habilitável).
