# ADR-0001: Monólito Modular Spring Boot, Java 21 e Fronteira de Inferência

- **Status:** Aceito (Fase 0/1) — Decisão definitiva do runtime de inferência condicionada ao Spike da Fase 4 (Portão P2)
- **Data:** 2026-11-19

## Contexto

O RailVision AI exige uma aplicação web local simples (Angular + backend + PostgreSQL) aliada a um fluxo de Visão Computacional treinado em Python (PyTorch / Ultralytics YOLO26). Precisamos evitar complexidade operacional prematura (microsserviços, brokers, múltiplos bancos) e respeitar o ambiente verificado na máquina do usuário (Amazon Corretto 21 já instalado, sem JDK 25 no sistema).

## Decisão

1. **Monólito modular Spring Boot (`4.1.2`) + Java 21 (`Corretto 21.0.5`) + Maven Wrapper (`mvnw`):**
   - Mantém um único processo de backend para API REST, validação de mídia, migrations Flyway e orquestração assíncrona de jobs (`concurrency = 1` inicial).
   - Não instalar JDK 25 por padrão; reutilizar o Corretto 21 validado sem alterar `PATH`/`JAVA_HOME` globais do Windows.
2. **Fronteira `InferenceEngine` no backend Java:**
   - Toda chamada de análise de vídeo passa pela interface `ai.railvision.backend.inference.InferenceEngine`.
   - Na **Fase 1**, o adaptador ativo é `DemoFixtureInferenceEngine`, claramente sinalizado com `isDemoSimulation = true` e registros `isExample = true`.
   - Na **Fase 4 (Portão P2)**, será executado um spike comparando ONNX Runtime Java + decodificação de vídeo no Windows contra a referência Python. Caso o spike atenda aos critérios de paridade, latência e manutenção, a inferência permanecerá embutida em Java; caso contrário, `ml/serving/` (FastAPI) atuará exclusivamente como adaptador de inferência interno atrás de `InferenceEngine`, sem acesso ao PostgreSQL.
3. **Licenciamento Ultralytics (AGPL-3.0):**
   - Por tratar-se de projeto de portfólio/estudo (Q13), a baseline `YOLO26` sob AGPL-3.0 é aceita para pesquisa e desenvolvimento local, mantendo `specs/licenses.md` atualizado e revisando obrigações caso haja publicação ou uso comercial futuro.

## Consequências

- **Positivas:** execução local trivial (`docker compose` apenas para PostgreSQL + backend + frontend), zero acoplamento da UI ou schema de banco ao runtime de ML, e transparência total entre dados simulados e inferência real.
- **Riscos monitorados:** paridade de pré/pós-processamento ONNX e tracking em Java (mitigados pelo spike dedicado da Fase 4 e opção documentada de fallback).
