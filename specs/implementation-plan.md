# RailVision AI — Plano de Implementação Incremental

**Documento fonte:** Especificação Unificada (atualizada pela ADR-0003)  
**Última atualização:** 2026-11-20  

## Portões de Decisão

| Portão | Descrição | Status atual |
|---|---|---|
| **P0** | Confirmação da estrutura e raiz do repositório | **Concluído** |
| **P1** | Auditoria dos datasets e triagem de vídeos públicos (define camada A/B/C) | **Aprovado (Camada C+: detecção + tracking PoC na Fase 3; Fase 5 pendente de mais fontes)** |
| **P2** | Arquitetura definitiva de backend e inferência local em produção de desenvolvimento | **Atualizado pela ADR-0003 — Backend único Python 3.12 + FastAPI + PyTorch CUDA + PyAV (Java/Spring Boot descomissionado em M5; Fase 4 Java/ONNX preservada em `specs/experiments/` e `artifacts/runs/`)** |

## Fases e Entregáveis

| Fase | Nome | Dependências | Status |
|---|---|---|---|
| **0** | Bootstrap seguro do repositório (`specs/`, `ml/`, `docker-compose.yml`, `.gitignore`, `.env.example`) | P0 | **Concluído** |
| **1** | *Walking skeleton* Angular + Backend + PostgreSQL (upload, job assíncrono, Range media, fixtures históricos `isExample=true`) | Fase 0 | **Concluído** |
| **2** | Auditoria dos datasets (`TWIST`, `UFOP`) e triagem licenciada de vídeos públicos | Fase 0 | **Concluído** |
| **3** | Baseline CV em Python (`YOLO26n` vs. `YOLO26s` + ablação *near-dedup* + `ByteTrack`/`BoT-SORT` PoC em Camada C+) | Fase 2 / P1 | **Concluído (PoC Camada C+ — baseline preservado)** |
| **3.1** | Dataset manual especializado `open_top_freight_wagon` (divisão por campanha) + *fine-tuning* `YOLO26n` + benchmark anotado `ByteTrack` vs. `BoT-SORT` (`5 fps` / CVAT) | Fase 3 | **Concluído** |
| **4** | Spike ONNX + benchmark comparativo CPU e Visualizador local `/onnx-lab` | Fases 1 e 3.1 | **Concluído (artefatos preservados em `artifacts/runs/phase4-onnx-spike/`)** |
| **M0–M5** | Migração integral para backend único Python 3.12 + FastAPI + PyTorch CUDA + PyAV + ByteTrack e remoção do Java | Fase 4 | **Concluído (ver ADR-0003 e `specs/migration/fastapi-migration-log.md`)** |
| **5** | Modelos especializados de material e nível ordinal de carregamento | Expansão de fontes de carga | **Pendente (aguardando novas fontes públicas com carga visível)** |
| **6** | Integração da classificação de material e nível ordinal real sobre `VisionInspectionInferenceEngine` | Fases M0–M5 e 5 | **Parcial (detecção + ByteTrack reais integrados; material/carregamento marcados como `not_evaluated`)** |
| **7** | Finalização do MVP, relatórios completos, testes ponta a ponta e medição de hardware | Fase 6 | Pendente |
