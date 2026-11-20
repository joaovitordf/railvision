# RailVision AI — Módulo ML Offline (`ml/`)

Este diretório abriga exclusivamente rotinas **offline** em Python (3.12+):

- Auditoria e triagem de datasets e fontes de vídeo públicas (com verificação prévia de licença);
- Preparação de dados, deduplicação e splits agrupados por vídeo/sessão/fonte;
- Treinamento, avaliação, análise de erros e geração de model cards;
- Exportação de modelos validados para ONNX.

## Regras arquiteturais

1. `ml/` **não** é um backend de negócio e **não** acessa diretamente o PostgreSQL da aplicação.
2. Não instalar pacotes globalmente no sistema; usar sempre um ambiente virtual local em `ml/.venv/` (ignorado pelo Git).
3. Dependências pesadas (PyTorch, Ultralytics, OpenCV) serão instaladas e pinadas a partir das Fases 2 e 3, após verificação de compatibilidade CUDA/driver via `nvidia-smi` no host Windows.
4. `ml/serving/` (FastAPI) **não existe por padrão**; só será criado caso o spike Java/ONNX da Fase 4 falhe e exista ADR formal aprovando o fallback.

## Criação do ambiente virtual (Fases 2–3)

```powershell
cd ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
