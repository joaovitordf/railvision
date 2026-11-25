# RailVision AI

Sistema local de visão computacional para monitoramento ferroviário em vagões abertos (*open-top*): detecção e rastreamento de vagões em vídeos pré-gravados, com infraestrutura preparada para classificação de material e estimativa ordinal do nível de carregamento.

> **Estado atual do projeto:** Backend unificado em **Python 3.12 + FastAPI** (`backend/`) com inferência real em **PyTorch + Ultralytics CUDA (`NVIDIA GeForce RTX 4060 Ti`)**, opções explícitas em CPU (`pytorch_cpu`, `onnx_cpu`), decodificação/codificação controlada em MP4/H.264 (`libx264`) via **PyAV**, e rastreamento **ByteTrack**.  
> O detector atual (`exp04c_yolo26n_opentop_640`) ainda possui caráter experimental (PoC) e **não está aprovado para produção**. Classificação de material e nível ordinal de carregamento permanecem explicitamente marcados como `not_evaluated` (sem dados inventados) até o treinamento dos modelos da Fase 5.

---

## Estrutura do Monorepo

```text
railvision/
├── frontend/             # Aplicação web Angular 22 (Visão geral, Inspeções, Anotações Studio e Laboratório/Treinamento)
├── backend/              # Backend único Python 3.12 + FastAPI + SQLAlchemy 2.0 + Alembic + PyAV + Visão CUDA/ONNX
├── ml/                   # Ferramental offline (venv compartilhado ml/.venv, auditoria, pré-anotação, treino, avaliação, export ONNX)
├── specs/                # Documentação SDD, contratos OpenAPI, ADRs (ADR-0001..ADR-0006), relatórios e manifestos de dados
├── data/                 # [LOCAL / IGNORADO PELO GIT] Vídeos, datasets brutos e armazenamentos locais
├── artifacts/            # [LOCAL / IGNORADO PELO GIT] Checkpoints (.pt), modelos (.onnx) e logs de execução
├── docker-compose.yml    # Serviço local PostgreSQL 17
├── start.ps1             # Inicialização unificada (PostgreSQL + Alembic + FastAPI + Angular)
├── stop.ps1              # Encerramento limpo dos serviços locais
├── .env.example          # Variáveis de ambiente de exemplo (sem segredos)
└── .gitignore
```

---

## Pré-requisitos verificados no ambiente

- **Python / ML venv:** Python `3.12.8` em `ml/.venv` (`torch==2.9.1+cu128`, `ultralytics==8.4.215`, `fastapi==0.165.0`, `SQLAlchemy==2.1.9`, `alembic==1.20.1`, `av==19.0.2`, `onnxruntime==1.32.0`)
- **GPU Local (padrão):** NVIDIA GeForce RTX 4060 Ti 8 GB com suporte CUDA verificado via PyTorch (`cu128`)
- **Node.js / npm:** Node `v24.15.0`, npm `11.12.1`
- **Docker:** Docker Engine + Docker Compose v5 (PostgreSQL `17.12-alpine`)

---

## Execução rápida (um comando)

> **Primeira vez neste projeto?** Siga primeiro o guia passo a passo completo em [SETUP.md](SETUP.md).

Na raiz do repositório em PowerShell (com Docker Desktop aberto e dependências do frontend já instaladas via `npm --prefix frontend install`):

```powershell
.\start.ps1
```

O script `start.ps1`:
- Carrega o `.env` local (criando a partir de `.env.example` se ainda não existir) e valida Docker, `ml/.venv`, `backend/app/main.py` e `frontend/node_modules`;
- Sobe e aguarda o PostgreSQL (`railvision-postgres`) ficar saudável;
- Sincroniza/aplica migrations Alembic (`alembic upgrade head`);
- Inicia o backend FastAPI (`127.0.0.1:8080`) e o frontend Angular (`http://localhost:4200`) evitando instâncias duplicadas caso já estejam rodando;
- Grava logs em `artifacts/runs/local-dev/` e exibe os logs ao vivo no terminal até `Ctrl+C`.

Para iniciar todos os serviços em segundo plano e retornar imediatamente ao terminal:

```powershell
.\start.ps1 -NoFollow
```

Para encerrar corretamente Frontend, Backend FastAPI e PostgreSQL (preservando o volume de dados `railvision_pgdata`):

```powershell
.\stop.ps1
```

---

## Como executar manualmente por serviço (Windows / PowerShell)

### 1. Configurar variáveis de ambiente

Copie `.env.example` para `.env` na raiz (caso ainda não exista) e ajuste se necessário:

```powershell
Copy-Item .env.example .env
```

### 2. Subir o PostgreSQL

```powershell
docker compose up -d --wait postgres
docker compose ps postgres
```

O container `railvision-postgres` (`postgres:17.12-alpine`) escuta apenas em `127.0.0.1:5432` com volume persistente `railvision_pgdata`.

### 3. Executar o Backend (Python + FastAPI + Alembic)

```powershell
Push-Location backend
..\ml\.venv\Scripts\python.exe -m alembic upgrade head
Pop-Location
.\ml\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8080
```

Para executar a suíte de testes do backend (usa schemas temporários isolados no PostgreSQL, sem alterar `public`):

```powershell
Push-Location backend
..\ml\.venv\Scripts\pytest.exe -v
Pop-Location
```

Documentação interativa da API (Swagger UI): `http://127.0.0.1:8080/docs`  
Health checks: `http://127.0.0.1:8080/health` e `http://127.0.0.1:8080/actuator/health`

### 4. Executar o Frontend (Angular)

```powershell
npm --prefix frontend install
npm --prefix frontend start
```

Acesse `http://localhost:4200`. O proxy de desenvolvimento (`frontend/proxy.conf.json`) redireciona `/api` e `/actuator` para `http://127.0.0.1:8080`.

Para build e testes unitários do frontend (Vitest):

```powershell
npm --prefix frontend run build
npm --prefix frontend test -- --watch=false
```

---

## Backup e Higiene de Dados

- As pastas `data/` e `artifacts/`, assim como arquivos de vídeo, datasets, pesos (`.pt`, `.onnx`, `.ckpt`) e o arquivo `.env`, são **ignorados pelo Git**.
- Caso deseje preservar vídeos enviados em `data/processed/app-storage/`, execuções do laboratório em `data/processed/onnx-spike-runs/`, datasets ou modelos treinados, faça backup manual dessas pastas locais.
- Consulte `specs/` para a especificação técnica completa, ADRs (`ADR-0001` a `ADR-0006`), plano de interface (`specs/ui-redesign-sdd.md`), histórico da migração (`specs/migration/fastapi-migration-log.md`) e auditoria de dados.
