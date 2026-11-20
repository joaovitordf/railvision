# RailVision AI — Visão Geral do Projeto

**Documento fonte:** Especificação Unificada v1.2  
**Última atualização:** 2026-11-19  

## 1. Objetivo

Construir incrementalmente uma aplicação local que processe vídeos ferroviários pré-gravados (câmeras RGB convencionais), detecte e rastreie vagões de carga abertos (*open-top*), identifique o material visível quando houver evidência suficiente e estime visualmente o nível ordinal de carregamento, utilizando modelos especializados treinados de forma reproduzível e resultados mensuráveis.

**Objetivo secundário:** aprofundamento prático em Visão Computacional, Deep Learning, engenharia de software e arquitetura de sistemas (portfólio e estudo; observar restrições de licenças como AGPL-3.0).

## 2. Escopo do MVP vs. Itens Deferidos

### No escopo do MVP
- Aplicação local de usuário único, sem login, perfis ou autenticação.
- Uma inspeção corresponde a **um vídeo pré-gravado** (limites de tamanho/duração configuráveis após auditoria de mídia na Fase 2).
- Apenas vagões de carga abertos (*open-top*).
- Detecção e rastreamento por vídeo (`track_id` temporário dentro da inspeção + `sequence_index` na composição).
- Classificação de material apenas com amostras representativas e suporte explícito a rejeição/`unknown`; vagão vazio tem `material = null`.
- Estimativa visual **ordinal** de nível de carregamento (`empty`, `low`, `medium`, `high`, `very_high`).
- Consolidação temporal por vagão, registro de discordâncias e flags de revisão humana.
- Interface web Angular para upload, acompanhamento de processamento, reprodução com overlays e visualização/relatório simples.

### Fora do escopo do MVP (deferido)
- Streaming de câmera ao vivo e sincronização de múltiplas câmeras.
- Percentual numérico de preenchimento (`estimated_fill_percentage = null` no MVP), volume físico ou tonelagem.
- OCR de código/identificador físico permanente do vagão.
- Processamento distribuído, brokers (`Kafka`/`RabbitMQ`), Kubernetes, deploy em nuvem.
- Múltiplos vídeos por inspeção ou imagens avulsas.
- Autenticação/multi-tenancy.

## 3. Camadas de Contingência de Escopo (*Feasibility Tiers*)

Caso a auditoria de vídeos públicos (Fase 2 / Portão P1) demonstre visibilidade insuficiente da superfície da carga:
- **Camada A:** Detecção + tracking + material + nível ordinal de carregamento (MVP completo).
- **Camada B:** Detecção + tracking + vagão vazio/carregado e nível ordinal grosseiro; material somente se houver amostras suficientes.
- **Camada C:** Detecção + tracking + contagem/sequência de vagões (sem análise de carga).

*(Critérios numéricos de transição permanecem `null` até a auditoria de dados.)*

## 4. Ambiente Local Verificado (2026-11-20)

- **SO:** Windows
- **CPU:** AMD Ryzen 7 5700X3D 8-Core
- **RAM:** 31,9 GB
- **GPU:** NVIDIA GeForce RTX 4060 Ti (`8188 MiB` reportados via `nvidia-smi`, usada por padrão pelo runtime PyTorch CUDA do backend FastAPI)
- **Backend / ML venv (`ml/.venv`):** Python `3.12.8`, FastAPI `0.165.0`, SQLAlchemy `2.1.9`, Alembic `1.20.1`, PyAV `19.0.2`, PyTorch `2.9.1+cu128`, Ultralytics `8.4.215`, ONNX Runtime `1.32.0`
- **Node.js / npm:** `v24.15.0` / `11.12.1`
- **Docker:** `29.6.2` / Docker Compose `v5.3.1` (PostgreSQL `17.12-alpine`)
