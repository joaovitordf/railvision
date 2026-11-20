# RailVision AI — Requisitos e Definições de Domínio

**Documento fonte:** Especificação Unificada v1.2  
**Última atualização:** 2026-11-19  

## 1. Definições de Domínio

| Termo | Definição operacional |
|---|---|
| `visual_fill` | Categoria ordinal estimada a partir do que é visível na imagem; **não** representa volume físico real ou massa. |
| `empty` | Vagão sem carga visível (`loading_level = empty`). Exige obrigatoriamente `material = null` (não aplicável), nunca `unknown`. |
| `unknown` | Há carga visível, mas sem evidência visual suficiente para classificação segura (oclusão, iluminação, baixa confiança). |
| `other_material` | Carga identificada fora das classes treinadas. Mantida separada de `unknown`; só entra no treino se houver exemplos reais rotulados e coerentes. |
| `wagon_identity` | Composta por `track_id` (temporário, válido apenas naquela execução/inspeção) + `sequence_index` (ordem observada). `physical_id` é `null` no MVP (OCR deferido). |
| `label_basis` | Origem do rótulo em dataset customizado: `visual_only`, `source_metadata` ou `expert_confirmed`. |

## 2. Requisitos Funcionais (Rastreáveis)

| ID | Requisito | Fase alvo | Critério de aceitação |
|---|---|---|---|
| `FR-01` | Criar inspeção com upload de 1 vídeo pré-gravado e metadados básicos | Fase 1 | Vídeo validado (extensão + assinatura binária + limite) e salvo fora da pasta pública; erro retorna RFC 9457 |
| `FR-02` | Listar inspeções e exibir detalhe individual | Fase 1 | Histórico consultável e página de detalhe exibindo status e vídeo |
| `FR-03` | Disparar processamento assíncrono e consultar status do job | Fase 1 | `POST /process` retorna `202 Accepted` sem bloquear HTTP; `GET /jobs/{id}` informa progresso/status/erro sanitizado |
| `FR-04` | Reproduzir vídeo no navegador com suporte a busca (*seek*) | Fase 1 | Endpoint `/media` suporta cabeçalho `Range` (`206 Partial Content` e `Accept-Ranges: bytes`) |
| `FR-05` | Recuperar jobs interrompidos e permitir reprocessamento explícito | Fase 1 | Jobs `RUNNING`/`QUEUED` após reinício são marcados como `FAILED` (`PROCESS_INTERRUPTED`) e podem ser reprocessados |
| `FR-06` | Transparência explícita de simulações/escopo de modelos | Fase 1 / M5 | Registros históricos de demonstração preservam `isExample=true`/`isDemoSimulation=true`; inferência real marca material e carregamento como `not_evaluated` enquanto os modelos da Fase 5 estiverem pendentes |
| `FR-07` | Auditar datasets públicos e vídeos licenciados antes de treino | Fase 2 | `dataset-audit.md` preenchido com contagens medidas, ângulos e manifestos de licença |
| `FR-08` | Detectar e rastrear vagões abertos em vídeo preservando timestamps | Fases 3–6 | Detector + tracker associados ao `track_id`, mantendo `sequence_index` e sinalizando suspeita de fragmentação |
| `FR-09` | Classificar material visível com suporte a rejeição (`unknown`) | Fases 5–6 | Apenas classes viáveis treinadas; vagão vazio recebe `material = null` |
| `FR-10` | Estimar nível ordinal de carregamento sem inventar % ou tonelagem | Fases 5–6 | `estimated_fill_percentage = null` e `physical_volume_verified = false` no MVP |
| `FR-11` | Consolidar observações por vagão com discordâncias e flags de revisão | Fase 6 | `WagonResult` persiste contagem, distribuição de votos, discordâncias e `review_reasons` |
| `FR-12` | Gerar visualização/relatório simples da inspeção | Fase 7 | Exportação e resumo consolidado consistentes com o banco |

## 3. Requisitos Não Funcionais

- `NFR-01`: Aplicação local de usuário único; bind padrão em `127.0.0.1` / `localhost`, CORS restrito à origem do frontend.
- `NFR-02`: Python + FastAPI + SQLAlchemy/Alembic como único backend de negócio e único proprietário do schema PostgreSQL (ver ADR-0003).
- `NFR-03`: Processamento assíncrono com concorrência inicial `1` para limitar uso de memória/GPU.
- `NFR-04`: Tratamento de erros conforme **RFC 9457** (`application/problem+json`), sem expor caminhos absolutos do servidor ou *stack traces*.
- `NFR-05`: Prevenção contra *path traversal* e armazenamento com nomes internos gerados (UUID).
- `NFR-06`: Dados brutos, vídeos, pesos (`.pt`, `.onnx`, `.ckpt`) e `.env` ignorados pelo Git.
- `NFR-07`: Reprodutibilidade em ML (logs reais, seeds, splits sem vazamento, hardware e versões documentados).
- `NFR-08`: Limiares numéricos de ML e latência permanecem `null` até medição de baseline real.
