# Log de Migração — Backend Python + FastAPI (RailVision AI)

Este documento registra a execução incremental da migração do backend (`backend/` Java 21 + Spring Boot -> `backend-py/` Python 3.12 + FastAPI), preservando integralmente o backend Java, o frontend Angular e todos os artefatos das Fases 3, 3.1 e 4 até validação completa.

## Etapa M0 — Fundação Isolada, Dependências Fixadas, ORM e Alembic (`CONCLUÍDA`)

### 1. Reaproveitamento e Fixação de Dependências (`ml/.venv`)

- Interpretador: Python `3.12.8` em `ml/.venv`.
- Dependências de ML/visão preservadas sem atualização/downgrade:
  - `torch==2.9.1+cu128` (`cuda: True`, dispositivo `NVIDIA GeForce RTX 4060 Ti`)
  - `torchvision==0.24.1+cu128`
  - `ultralytics==8.4.215`
  - `numpy==2.5.5`
  - `opencv-python==5.1.0.95` (`cv2==5.1.0`)
  - `onnx==1.23.2`
  - `onnxruntime==1.32.0`
  - `pytest==9.2.0`
- Dependências do backend instaladas com versões fixadas em `ml/pyproject.toml` (`[project.optional-dependencies].backend`) e `backend-py/pyproject.toml`:
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

### 2. Estrutura Criada em `backend-py/`

- Configuração e contratos base:
  - [pyproject.toml](../../backend-py/pyproject.toml)
  - [app/core/paths.py](../../backend-py/app/core/paths.py) — resolução unificada de caminhos relativos à raiz do repositório e suporte legado `../` do `backend/`.
  - [app/core/config.py](../../backend-py/app/core/config.py) — `Settings` (`pydantic-settings`) lendo `.env` na raiz e sobrescritas `RAILVISION_*`.
  - [app/core/schemas.py](../../backend-py/app/core/schemas.py) — `CamelModel` com `to_camel`, `populate_by_name=True` e `from_attributes=True`.
- Persistência e migrações:
  - [app/db/base.py](../../backend-py/app/db/base.py) — `Base` SQLAlchemy 2.0 com convenção de nomes compatível com PostgreSQL/Flyway.
  - [app/db/models.py](../../backend-py/app/db/models.py) — mapeamento ORM 1:1 das 5 tabelas de `V1__initial_schema.sql` (`inspections`, `processing_jobs`, `model_versions`, `wagon_results`, `wagon_observations`), FKs `ON DELETE CASCADE`, restrições `UNIQUE` (`inspections_stored_filename_key`, `uq_model_versions_task_name_version`) e os 4 índices.
  - [app/db/session.py](../../backend-py/app/db/session.py) — `create_db_engine`, `get_session_factory`, `get_db_session`.
  - [alembic.ini](../../backend-py/alembic.ini), [alembic/env.py](../../backend-py/alembic/env.py), [alembic/script.py.mako](../../backend-py/alembic/script.py.mako) — coexistência explícita com Flyway (`include_object` ignora `flyway_schema_history`) e suporte a `-x schema=<nome>` para validação isolada.
  - [alembic/versions/0001_initial_schema.py](../../backend-py/alembic/versions/0001_initial_schema.py) — DDL equivalente a `V1__initial_schema.sql` + `downgrade` reversível.
- Aplicação FastAPI e pacotes modulares reservados para M1–M4:
  - [app/main.py](../../backend-py/app/main.py) — `create_app()` com CORS configurado e `GET /health` (`{"status": "ok", "stage": "M0"}`).
  - Placeholders modulares: `app/modules/{health,media,inspections,jobs,results,vision,lab}/__init__.py`.

### 3. Sonda PyAV / Decodificação 4K e Codecs H.264

- `av==19.0.2` listou os seguintes codecs relacionados a H.264 em `av.codecs_available`: `h264`, `h264_amf`, `h264_cuvid`, `h264_d3d12va`, `h264_mf`, `h264_nvenc`, `h264_qsv`, `libx264`, `libx264rgb`.
- Abertura do encoder `av.codec.Codec("libx264", "w")` validada para escrita (controle explícito de CRF/preset sem fallback silencioso).
- Decodificação do caso negativo de referência `data/raw/public-videos/vid_01_langer_unbeladener_g_terzug_von_oben_k_l.webm` (`3840x2160`, codec `vp9`, ~29.97 fps) validada com geração de frame `rgb24` de shape `(2160, 3840, 3)`.

### 4. Resultados dos Testes Executados na Etapa M0

- Container PostgreSQL (`docker compose up -d --wait postgres`, imagem `postgres:17.12-alpine` em `127.0.0.1:5432`) iniciado com status `healthy`.
- `backend-py/tests` (`pytest -v`): **6 passed** (`test_settings_and_paths_resolution`, `test_camel_model_serialization_and_orm_attributes`, `test_orm_metadata_matches_flyway_v1_contract`, `test_fastapi_health_and_cors`, `test_cuda_and_pyav_4k_decode_plus_h264_encoder_probe`, `test_postgresql_readonly_public_and_isolated_alembic_roundtrip`).
  - O teste no PostgreSQL ativo inspecionou o schema `public` apenas em modo leitura (inalterado antes e depois), executou `alembic upgrade head` -> escrita/leitura ORM -> `alembic downgrade base` em um schema temporário isolado (`railvision_m0_alembic_test_<hex>`), removendo o schema temporário ao final.
- `ml/tests` (`pytest ml\tests -v`): **17 passed**.
- `pip check`: `No broken requirements found.`

### 5. Arquivos Modificados Fora de `backend-py/`

- [ml/pyproject.toml](../../ml/pyproject.toml) — adicionada seção opcional `backend`.
- [.gitignore](../../.gitignore) — adicionadas regras para cache e `.venv` de `backend-py/`.

---

## Etapa M1 — Tratamento de Erros, Mídia, Inspeções, Resultados, Modelos e Saúde (`CONCLUÍDA`)

- **Erros RFC 9457 (`application/problem+json`):** [app/core/exceptions.py](../../backend-py/app/core/exceptions.py) espelha `GlobalExceptionHandler.java` (`EMPTY_FILE`, `UNSUPPORTED_EXTENSION`, `INVALID_VIDEO_SIGNATURE`, `INVALID_FILENAME`, `413`, `404`, `409`, `416`, `503`).
- **Mídia local + PyAV:** [validator.py](../../backend-py/app/modules/media/validator.py) valida extensões (`mp4`, `mov`, `mkv`, `webm`, `avi`) e assinaturas de container; [probe.py](../../backend-py/app/modules/media/probe.py) extrai `width_px`, `height_px`, `fps`, `duration_ms` e `codec` via PyAV; [storage.py](../../backend-py/app/modules/media/storage.py) realiza escrita em stream com SHA-256 e streaming HTTP completo (`200`) / `Range` (`206` e `416`).
- **Domínios e Saúde:** [inspections/](../../backend-py/app/modules/inspections/router.py), [results/](../../backend-py/app/modules/results/router.py) e [health/](../../backend-py/app/modules/health/router.py) implementam `/api/v1/inspections*`, `/api/v1/models`, `/health` e `/actuator/health`, mantendo a regra de vagões `empty` com `material = null`.

---

## Etapa M2 — Fila Assíncrona de Jobs, Paridade DemoFixture e Recuperação de Stale Jobs (`CONCLUÍDA`)

- [engine.py](../../backend-py/app/modules/jobs/engine.py) porta integralmente `DemoFixtureInferenceEngine` (`DEMO_FIXTURE_ENGINE`, 4 modelos placeholder e 3 vagões de demonstração).
- [worker.py](../../backend-py/app/modules/jobs/worker.py) gerencia fila assíncrona em `ThreadPoolExecutor` com transições transacionais `QUEUED -> RUNNING -> COMPLETED/FAILED`.
- [service.py](../../backend-py/app/modules/jobs/service.py), [router.py](../../backend-py/app/modules/jobs/router.py) e [recovery.py](../../backend-py/app/modules/jobs/recovery.py) entregam `POST /api/v1/inspections/{id}/process` (`202` / `409` em concorrência), `GET /api/v1/jobs/{jobId}` e recuperação automática no startup (`PROCESS_INTERRUPTED`).

---

## Etapa M3 — Runtimes de Visão (PyTorch CUDA, Fallback Explícito CPU, ONNX CPU), Pipeline 1080p/4K PyAV e ByteTrack (`CONCLUÍDA`)

- [preprocess.py](../../backend-py/app/modules/vision/preprocess.py), [runtimes.py](../../backend-py/app/modules/vision/runtimes.py), [renderer.py](../../backend-py/app/modules/vision/renderer.py), [tracker.py](../../backend-py/app/modules/vision/tracker.py) e [video_pipeline.py](../../backend-py/app/modules/vision/video_pipeline.py).
- **Runtime padrão:** `pytorch_cuda` (`PyTorch CUDA (NVIDIA GeForce RTX 4060 Ti)`), com opções explícitas `pytorch_cpu` e `onnx_cpu` (`CPUExecutionProvider`). Ausência de CUDA falha explicitamente (`503 RuntimeUnavailableException`), permitindo fallback para CPU somente com `allowCpuFallback=true` registrado em `fallbackOccurred=true`.
- **Paridade medida nos 8 frames de referência da Fase 4** ([parity_reference.json](../../artifacts/runs/phase4-onnx-spike/parity_reference.json)):
  - `pytorch_cuda`: `8/8` correspondências exatas de contagem, `minIoU = 0.999206`, `maxCoordDiffPx = 0.15889`, `maxConfDiff = 0.001104`.
  - `pytorch_cpu`: `8/8` correspondências exatas, `minIoU = 0.999998`, `maxCoordDiffPx = 0.00032`, `maxConfDiff = 0.000004`.
  - `onnx_cpu`: `8/8` correspondências exatas, `minIoU = 1.0`, `maxCoordDiffPx = 0.0`, `maxConfDiff = 0.0`.
- **Vídeos 4K / 1080p + ByteTrack:** decodificação do original em resolução nativa via PyAV; encoding controlado em MP4/H.264 com `libx264` (`CRF 23`, `yuv420p`, `movflags=+faststart`) sem fallback silencioso; padrão `1080p` (`1920x1080`) e opção `native` (`3840x2160` no `vid_01`, ambos validados com `0` detecções para vagões graneleiros); ByteTrack opcional (`enableTracking=true`) validado no `vid_15` atribuindo `trackId` persistente.

---

## Etapa M4 — Laboratório `/api/v1/onnx-spike`, Atualização Compatível do Angular e Scripts `start-py.ps1` / `stop-py.ps1` (`CONCLUÍDA`)

- **API do laboratório:** [lab/](../../backend-py/app/modules/lab/router.py) e [app/main.py](../../backend-py/app/main.py) preservam todos os endpoints `/api/v1/onnx-spike/*`, mantendo leitura do histórico existente e adicionando seleção de runtime, resolução de saída e tracking via ByteTrack.
- **Frontend Angular:** [onnx-lab.models.ts](../../frontend/src/app/features/onnx-lab/models/onnx-lab.models.ts), [onnx-lab-api.service.ts](../../frontend/src/app/features/onnx-lab/services/onnx-lab-api.service.ts) e [onnx-lab-page.component.ts](../../frontend/src/app/features/onnx-lab/pages/onnx-lab-page/onnx-lab-page.component.ts) mantêm compatibilidade retroativa com contratos Java e habilitam seletor de runtime, resolução (`1080p` / `native`), `libx264` CRF e ByteTrack quando servidos pelo FastAPI.
- **Operação local paralela:** adicionados [start-py.ps1](../../start-py.ps1) (com detecção de schema Flyway existente para executar `alembic stamp 0001_initial_schema` antes de `alembic upgrade head` sem perder dados) e [stop-py.ps1](../../stop-py.ps1), preservando integralmente [start.ps1](../../start.ps1), [stop.ps1](../../stop.ps1) e [backend/](../../backend/pom.xml).

### Resumo Final das Verificações

- `backend-py/tests` (`pytest -v`): **17 passed** (`6` M0 + `3` M1/M2 em schema isolado + `6` M3 paridade/vídeo/ByteTrack + `2` M4 API/histórico), com schema PostgreSQL `public` inalterado nos testes.
- `ml/tests` (`pytest ml\tests -v`): **17 passed**.
- `pip check`: `No broken requirements found.`
- Frontend Angular: `ng test --watch=false` (**5 test files / 10 tests passed**) e `ng build` concluídos com sucesso.
- Teste ao vivo `uvicorn app.main:app` contra o PostgreSQL ativo: `GET /health`, `/actuator/health`, `/api/v1/onnx-spike/status`, `/api/v1/inspections` (`LEN=1`) e `/api/v1/models` (`LEN=4`) retornaram HTTP `200`.

---

## Etapa M5 — Remoção do Backend Java, Cutover para `backend/`, Motor Real em GPU e Atualização Completa da UI (`CONCLUÍDA`)

- **Remoção e promoção:** backend Java 21 / Spring Boot removido definitivamente após encerramento do processo na porta `8080`; diretório `backend-py/` promovido para [backend/](../../backend/app/main.py); [start.ps1](../../start.ps1), [stop.ps1](../../stop.ps1) e [.gitignore](../../.gitignore) unificados para Python + FastAPI + Alembic + Angular + PostgreSQL.
- **Inferência real nas inspeções (`VisionInspectionInferenceEngine`):** substituído o motor simulado padrão (`DEMO_FIXTURE_ENGINE`) pelo pipeline real em [engine.py](../../backend/app/modules/jobs/engine.py) utilizando `pytorch_cuda` (`PyTorch CUDA (NVIDIA GeForce RTX 4060 Ti)`), decodificação PyAV em resolução nativa e rastreamento `ByteTrack`.
  - Campos sem modelo treinado (`material_class`, `loading_level`) passaram a gravar `not_evaluated` com `review_required=true`, sem dados inventados.
  - Startup do FastAPI registra de forma idempotente os modelos reais em `model_versions` (`isDemoPlaceholder=false`); `GET /api/v1/models` oculta placeholders legados por padrão sem remover linhas antigas do banco.
- **Frontend Angular atualizado:** removidas todas as referências ativas a Java, `Spike Fase 4` e banners estáticos de Fase 1 em [overview-page.component.ts](../../frontend/src/app/features/overview/pages/overview-page/overview-page.component.ts), [sidebar.component.ts](../../frontend/src/app/core/layout/sidebar/sidebar.component.ts), [footer.component.ts](../../frontend/src/app/core/layout/footer/footer.component.ts), [inspection-detail-page.component.ts](../../frontend/src/app/features/inspection-details/pages/inspection-detail-page/inspection-detail-page.component.ts), [reports-page.component.ts](../../frontend/src/app/features/reports/pages/reports-page/reports-page.component.ts) e [onnx-lab-page.component.ts](../../frontend/src/app/features/onnx-lab/pages/onnx-lab-page/onnx-lab-page.component.ts), exibindo o dispositivo GPU/CPU real obtido da API, badges `not_evaluated` e overlays sincronizados por timestamp.
- **Documentação SDD atualizada:** [README.md](../../README.md), [architecture.md](../architecture.md), [implementation-plan.md](../implementation-plan.md), [ADR-0003](../adr/ADR-0003-python-fastapi-backend-migration.md), [project-overview.md](../project-overview.md), [requirements.md](../requirements.md), [licenses.md](../licenses.md), [cv-and-measurement.md](../cv-and-measurement.md) e [railvision-v1.yaml](../openapi/railvision-v1.yaml). Artefatos e relatórios históricos das Fases 3, 3.1 e 4 em `specs/experiments/`, `specs/adr/ADR-0001..0002` e `artifacts/runs/` preservados intactos.

### Resumo das Verificações Finais (Etapa M5)

- `backend/tests` (`pytest -v`): **18 passed** em `15.77s` (incluindo `test_real_vision_inspection_workflow_gpu_and_bytetrack` ponta a ponta em GPU + schema isolado).
- `ml/tests` (`pytest ml\tests -v`): **17 passed** em `2.84s`.
- `pip check`: `No broken requirements found.`
- Frontend Angular: `ng test --watch=false` (**5 test files / 10 tests passed**) e `ng build`.
