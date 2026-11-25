# Log de Migração — Backend Python + FastAPI (RailVision AI)

Este documento registra a execução incremental da migração do backend (Java 21 + Spring Boot -> Python 3.12 + FastAPI em `backend/`), preservando integralmente o frontend Angular e todos os artefatos das Fases 3, 3.1 e 4.

> **Nota histórica:** durante as etapas M0–M4 o código Python ficou temporariamente no diretório transitório `backend-py/` com os scripts `start-py.ps1`/`stop-py.ps1`; na Etapa **M5**, `backend-py/` foi promovido para `backend/` e o antigo projeto Java (`backend/pom.xml`) foi removido definitivamente.

## Etapa M0 — Fundação Isolada, Dependências Fixadas, ORM e Alembic (`CONCLUÍDA`)

### 1. Reaproveitamento e Fixação de Dependências (`ml/.venv`)

- Interpretador: Python `3.12.8` em `ml/.venv`.
- Dependências de ML/visão preservadas sem atualização/downgrade:
  - `torch==2.9.1+cu128` (`cuda: True`, dispositivo `NVIDIA GeForce RTX 4060 Ti`)
  - `torchvision==0.24.1+cu128`
  - `ultralytics==8.4.215`
  - `numpy==2.5.5`
  - `opencv-python-headless==5.1.0.95` / `cv2==5.1.0`
  - `onnx==1.23.2`
  - `onnxruntime==1.32.0`
  - `pytest==9.2.0`
- Dependências do backend instaladas com versões fixadas em [ml/pyproject.toml](../../ml/pyproject.toml) (`[project.optional-dependencies].backend`) e [backend/pyproject.toml](../../backend/pyproject.toml):
  - `alembic==1.20.1`
  - `av==19.0.2`
  - `fastapi==0.165.0`
  - `psycopg[binary]==3.3.9`
  - `pydantic==2.14.4`
  - `pydantic-settings==2.16.3`
  - `python-multipart==0.0.34`
  - `SQLAlchemy==2.1.9`
  - `uvicorn==0.54.0`
- Integridade do ambiente validada com `pip check` (`No broken requirements found.`).

### 2. Estrutura Criada (Promovida para `backend/` em M5)

- Configuração e contratos base:
  - [backend/pyproject.toml](../../backend/pyproject.toml)
  - [app/core/paths.py](../../backend/app/core/paths.py) — resolução unificada de caminhos relativos à raiz do repositório e suporte legado `../`.
  - [app/core/config.py](../../backend/app/core/config.py) — `Settings` (`pydantic-settings`) lendo `.env` na raiz e sobrescritas `RAILVISION_*`.
  - [app/core/schemas.py](../../backend/app/core/schemas.py) — `CamelModel` com `to_camel`, `populate_by_name=True` e `from_attributes=True`.
- Persistência e migrações:
  - [app/db/base.py](../../backend/app/db/base.py) — `Base` SQLAlchemy 2.0 com convenção de nomes compatível com PostgreSQL/Flyway.
  - [app/db/models.py](../../backend/app/db/models.py) — mapeamento ORM 1:1 das 5 tabelas de `V1__initial_schema.sql` (`inspections`, `processing_jobs`, `model_versions`, `wagon_results`, `wagon_observations`), FKs `ON DELETE CASCADE`, restrições `UNIQUE` e índices.
  - [app/db/session.py](../../backend/app/db/session.py) — `create_db_engine`, `get_session_factory`, `get_db_session`.
  - [alembic.ini](../../backend/alembic.ini), [alembic/env.py](../../backend/alembic/env.py), [alembic/script.py.mako](../../backend/alembic/script.py.mako) — coexistência explícita com Flyway (`include_object` ignora `flyway_schema_history`) e suporte a `-x schema=<nome>` para validação isolada.
  - [alembic/versions/0001_initial_schema.py](../../backend/alembic/versions/0001_initial_schema.py) — DDL equivalente a `V1__initial_schema.sql` + `downgrade` reversível.
- Aplicação FastAPI e pacotes modulares:
  - [app/main.py](../../backend/app/main.py) — `create_app()` com CORS configurado e rotas modulares em `app/modules/{health,media,inspections,jobs,results,vision,lab}/`.

### 3. Sonda PyAV / Decodificação 4K e Codecs H.264

- `av==19.0.2` listou os seguintes codecs relacionados a H.264 em `av.codecs_available`: `h264`, `h264_amf`, `h264_cuvid`, `h264_d3d12va`, `h264_mf`, `h264_nvenc`, `h264_qsv`, `libx264`, `libx264rgb`.
- Abertura do encoder `av.codec.Codec("libx264", "w")` validada para escrita (controle explícito de CRF/preset sem fallback silencioso).
- Decodificação do caso negativo de referência `data/raw/public-videos/vid_01_langer_unbeladener_g_terzug_von_oben_k_l.webm` (`3840x2160`, codec `vp9`, ~29.97 fps) validada com geração de frame `rgb24` de shape `(2160, 3840, 3)`.

### 4. Resultados dos Testes Executados na Etapa M0

- Container PostgreSQL (`docker compose up -d --wait postgres`, imagem `postgres:17.12-alpine` em `127.0.0.1:5432`) iniciado com status `healthy`.
- `backend/tests` (`pytest -v`): **6 passed** (`test_settings_and_paths_resolution`, `test_camel_model_serialization_and_orm_attributes`, `test_orm_metadata_matches_flyway_v1_contract`, `test_fastapi_health_and_cors`, `test_cuda_and_pyav_4k_decode_plus_h264_encoder_probe`, `test_postgresql_readonly_public_and_isolated_alembic_roundtrip`).
- `ml/tests` (`pytest ml\tests -v`): **17 passed**.
- `pip check`: `No broken requirements found.`

### 5. Arquivos Modificados Fora do Diretório Transitório

- [ml/pyproject.toml](../../ml/pyproject.toml) — adicionada seção opcional `backend`.
- [.gitignore](../../.gitignore) — adicionadas regras para cache e `.venv` de Python.

---

## Etapa M1 — Tratamento de Erros, Mídia, Inspeções, Resultados, Modelos e Saúde (`CONCLUÍDA`)

- **Erros RFC 9457 (`application/problem+json`):** [app/core/exceptions.py](../../backend/app/core/exceptions.py) (`EMPTY_FILE`, `UNSUPPORTED_EXTENSION`, `INVALID_VIDEO_SIGNATURE`, `INVALID_FILENAME`, `413`, `404`, `409`, `416`, `503`).
- **Mídia local + PyAV:** [validator.py](../../backend/app/modules/media/validator.py), [probe.py](../../backend/app/modules/media/probe.py), [storage.py](../../backend/app/modules/media/storage.py) com streaming HTTP completo (`200`) / `Range` (`206` e `416`).
- **Domínios e Saúde:** [inspections/](../../backend/app/modules/inspections/router.py), [results/](../../backend/app/modules/results/router.py) e [health/](../../backend/app/modules/health/router.py).

---

## Etapa M2 — Fila Assíncrona de Jobs, Paridade DemoFixture e Recuperação de Stale Jobs (`CONCLUÍDA`)

- [engine.py](../../backend/app/modules/jobs/engine.py), [worker.py](../../backend/app/modules/jobs/worker.py), [service.py](../../backend/app/modules/jobs/service.py), [router.py](../../backend/app/modules/jobs/router.py) e [recovery.py](../../backend/app/modules/jobs/recovery.py).

---

## Etapa M3 — Runtimes de Visão (PyTorch CUDA, Fallback Explícito CPU, ONNX CPU), Pipeline 1080p/4K PyAV e ByteTrack (`CONCLUÍDA`)

- [preprocess.py](../../backend/app/modules/vision/preprocess.py), [runtimes.py](../../backend/app/modules/vision/runtimes.py), [renderer.py](../../backend/app/modules/vision/renderer.py), [tracker.py](../../backend/app/modules/vision/tracker.py) e [video_pipeline.py](../../backend/app/modules/vision/video_pipeline.py).
- **Paridade medida nos 8 frames de referência da Fase 4** ([parity_reference.json](../../artifacts/runs/phase4-onnx-spike/parity_reference.json)):
  - `pytorch_cuda`: `8/8` correspondências exatas de contagem, `minIoU = 0.999206`, `maxCoordDiffPx = 0.15889`, `maxConfDiff = 0.001104`.
  - `pytorch_cpu`: `8/8` correspondências exatas, `minIoU = 0.999998`, `maxCoordDiffPx = 0.00032`, `maxConfDiff = 0.000004`.
  - `onnx_cpu`: `8/8` correspondências exatas, `minIoU = 1.0`, `maxCoordDiffPx = 0.0`, `maxConfDiff = 0.0`.

---

## Etapa M4 — Laboratório `/api/v1/onnx-spike`, Atualização Compatível do Angular e Scripts (`CONCLUÍDA`)

- **API do laboratório:** [lab/](../../backend/app/modules/lab/router.py) e [app/main.py](../../backend/app/main.py).
- **Frontend Angular:** [onnx-lab.models.ts](../../frontend/src/app/features/onnx-lab/models/onnx-lab.models.ts), [onnx-lab-api.service.ts](../../frontend/src/app/features/onnx-lab/services/onnx-lab-api.service.ts) e [onnx-lab-page.component.ts](../../frontend/src/app/features/onnx-lab/pages/onnx-lab-page/onnx-lab-page.component.ts).
- Na etapa M4, scripts temporários `start-py.ps1` e `stop-py.ps1` coexistiram com `start.ps1` e `stop.ps1` até o cutover em M5.

### Resumo das Verificações na Etapa M4

- `backend/tests`: **17 passed** (`6` M0 + `3` M1/M2 + `6` M3 + `2` M4).
- `ml/tests`: **17 passed**.
- Frontend Angular: `ng test --watch=false` (**5 test files / 10 tests passed**) e `ng build` OK.

---

## Etapa M5 — Remoção do Backend Java, Cutover para `backend/`, Motor Real em GPU e Atualização Completa da UI (`CONCLUÍDA`)

- **Remoção e promoção:** antigo backend Java 21 / Spring Boot (`backend/pom.xml`) removido definitivamente; diretório transitório `backend-py/` promovido para [backend/app/main.py](../../backend/app/main.py); [start.ps1](../../start.ps1), [stop.ps1](../../stop.ps1) e [.gitignore](../../.gitignore) unificados para Python + FastAPI + Alembic + Angular + PostgreSQL.
- **Inferência real nas inspeções (`VisionInspectionInferenceEngine`):** [engine.py](../../backend/app/modules/jobs/engine.py) com `pytorch_cuda` (`PyTorch CUDA (NVIDIA GeForce RTX 4060 Ti)`), PyAV e `ByteTrack`.
- **Frontend Angular atualizado:** [overview-page.component.ts](../../frontend/src/app/features/overview/pages/overview-page/overview-page.component.ts), [sidebar.component.ts](../../frontend/src/app/core/layout/sidebar/sidebar.component.ts), [footer.component.ts](../../frontend/src/app/core/layout/footer/footer.component.ts), [inspection-detail-page.component.ts](../../frontend/src/app/features/inspection-details/pages/inspection-detail-page/inspection-detail-page.component.ts), [reports-page.component.ts](../../frontend/src/app/features/reports/pages/reports-page/reports-page.component.ts) e [onnx-lab-page.component.ts](../../frontend/src/app/features/onnx-lab/pages/onnx-lab-page/onnx-lab-page.component.ts).
- **Documentação SDD atualizada:** [README.md](../../README.md), [architecture.md](../architecture.md), [implementation-plan.md](../implementation-plan.md), [ADR-0003](../adr/ADR-0003-python-fastapi-backend-migration.md), [project-overview.md](../project-overview.md), [requirements.md](../requirements.md), [licenses.md](../licenses.md), [cv-and-measurement.md](../cv-and-measurement.md) e [railvision-v1.yaml](../openapi/railvision-v1.yaml).

### Resumo das Verificações Finais (Etapa M5)

- `backend/tests` (`pytest -v`): **18 passed** em `15.77s`.
- `ml/tests` (`pytest ml\tests -v`): **17 passed** em `2.84s`.
- `pip check`: `No broken requirements found.`
- Frontend Angular: `ng test --watch=false` (**5 test files / 10 tests passed**) e `ng build`.
