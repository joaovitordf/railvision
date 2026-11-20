# ADR-0003: Unificação do Backend em Python 3.12 + FastAPI com PyTorch CUDA, PyAV e Alembic

- **Status:** Aceita e Implementada (Etapas M0–M5 concluídas)
- **Data:** 2026-11-20
- **Supersede parcialmente:** [ADR-0002](ADR-0002-gate-p2-java-onnx-runtime-decision.md) (mantida intacta apenas como registro histórico do Spike da Fase 4)

## Contexto

Durante a Fase 4 validou-se a exequibilidade técnica do backend Java 21 + ONNX Runtime CPU + JavaCV. Entretanto:
1. O host local dispõe de GPU dedicada (`NVIDIA GeForce RTX 4060 Ti 8 GB`) já operacional via PyTorch CUDA (`torch==2.9.1+cu128`) em `ml/.venv`, enquanto no Java a inferência permanecia limitada à CPU (`CPUExecutionProvider`).
2. A evolução da plataforma concentra esforços em visão computacional (detecção, ByteTrack/BoT-SORT, futuros classificadores de material e carregamento ordinal e encoding controlado de vídeo). Manter dois runtimes de servidor (Java 21 + serviço auxiliar Python) introduzia duplicação desnecessária de contratos, pré/pós-processamento e decodificação de mídia.

## Decisão

1. **Unificar o backend em Python 3.12 + FastAPI (`backend/`):**
   - Remover o backend Java 21 / Spring Boot após a validação completa das etapas M0–M4 (`Etapa M5`).
   - Preservar os contratos REST `/api/v1/*`, payloads JSON camelCase (`CamelModel`), códigos de erro RFC 9457 (`application/problem+json`) e compatibilidade integral com o frontend Angular e o schema PostgreSQL existente.
2. **Persistência e Migrações:**
   - SQLAlchemy 2.0 + `psycopg[binary]` + Alembic (`0001_initial_schema.py`), mantendo compatibilidade com bancos existentes criados pelo schema V1 sem perda de dados.
3. **Visão Computacional e Mídia em Produção Local:**
   - `pytorch_cuda` como runtime padrão em `GPU`, com fallback explícito (somente quando autorizado) para `pytorch_cpu` e seleção opcional de `onnx_cpu` para comparações.
   - `PyAV` (`av==19.0.2`) decodificando fontes originais sem redução destrutiva (incluindo 4K `3840x2160`) e codificando MP4/H.264 controlado via `libx264` (`CRF 23`, padrão `1080p` com opção `native`).
   - Substituição definitiva do antigo motor simulado nas novas inspeções pelo `VisionInspectionInferenceEngine` (detecção real + ByteTrack), marcando material e carregamento honestamente como `not_evaluated` até o treino dos modelos da Fase 5.

## Consequências

- **Positivas:** backend único, acesso direto à GPU RTX 4060 Ti, ganho expressivo de velocidade no processamento 4K/1080p, eliminação da duplicação Java/Python de pré-processamento e tracking.
- **Negativas / Trade-offs:** uso compartilhado do ambiente `ml/.venv` exige fixação rigorosa das versões em `pyproject.toml` e validação com `pip check`.
- **Preservação:** relatórios e artefatos de benchmark das Fases 3, 3.1 e 4 em `specs/experiments/` e `artifacts/runs/` permanecem preservados integralmente para referência científica e comparação histórica.
