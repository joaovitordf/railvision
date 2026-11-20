# Guia de Setup do Zero — RailVision AI

Este guia explica passo a passo como configurar e executar o **RailVision AI** em uma máquina limpa (Windows 10/11 com PowerShell).

Para detalhes de arquitetura, métricas e histórico de experimentos, consulte o [README.md](README.md) e a pasta `specs/`.

---

## 1. Pré-requisitos

Instale e valide na máquina antes de iniciar:

| Ferramenta | Versão recomendada | Verificação rápida |
|---|---|---|
| **Python** | `3.12.x` (`>=3.12,<3.13`) | `python --version` |
| **Node.js & npm** | `Node 22+` / `npm 10+` | `node -v; npm -v` |
| **Docker Desktop** | Docker Engine + Compose | `docker info` |
| **Git** | Qualquer versão recente | `git --version` |
| **GPU NVIDIA (opcional/recomendada)** | Driver NVIDIA atualizado (ex.: RTX 4060 Ti 8 GB) | `nvidia-smi` |

> **Nota sobre GPU vs. CPU:** o runtime padrão configurado no backend é `pytorch_cuda` (GPU NVIDIA local). Caso a máquina de destino não possua GPU NVIDIA com CUDA, configure `RAILVISION_DEFAULT_RUNTIME=pytorch_cpu` ou `onnx_cpu` no arquivo `.env`.

---

## 2. Clonar o repositório e configurar variáveis de ambiente

```powershell
git clone <URL_DO_REPOSITORIO>
cd railvision
Copy-Item .env.example .env
```

Edite o `.env` caso queira alterar portas ou senha local do PostgreSQL:

```dotenv
POSTGRES_DB=railvision
POSTGRES_USER=railvision
POSTGRES_PASSWORD=defina_uma_senha_local_aqui
POSTGRES_PORT=5432

RAILVISION_SERVER_PORT=8080
RAILVISION_STORAGE_ROOT=./data/processed/app-storage
RAILVISION_UPLOAD_MAX_SIZE_MB=500
RAILVISION_CORS_ALLOWED_ORIGIN=http://localhost:4200
```

> Nunca suba `.env`, vídeos ou pesos de modelo para o Git (todos estão protegidos pelo `.gitignore`).

---

## 3. Criar o ambiente virtual Python (`ml/.venv`) e instalar dependências

O backend FastAPI e as ferramentas de ML compartilham o mesmo ambiente virtual em `ml/.venv` para evitar duplicar PyTorch/CUDA:

```powershell
python -m venv ml\.venv
.\ml\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 3.1. Instalar PyTorch + Ultralytics + ONNX

**Com GPU NVIDIA (CUDA 12.8 — padrão deste projeto):**

```powershell
.\ml\.venv\Scripts\python.exe -m pip install -e "ml[audit,train,onnx,backend]" --extra-index-url https://download.pytorch.org/whl/cu128
```

*(Caso todas as dependências com hashes/URL padrão precisem de resolução no índice CUDA acima, valide ao final com `pip check`.)*

### 3.2. Validar integridade do ambiente Python e CUDA

```powershell
.\ml\.venv\Scripts\python.exe -m pip check
.\ml\.venv\Scripts\python.exe -c "import torch, av, fastapi; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'av:', av.__version__, 'fastapi:', fastapi.__version__)"
```

---

## 4. Instalar dependências do Frontend Angular

```powershell
npm --prefix frontend install
```

---

## 5. Subir PostgreSQL e inicializar o banco de dados

Com o **Docker Desktop** aberto:

```powershell
docker compose up -d --wait postgres
```

Aplique o schema inicial via Alembic:

```powershell
Push-Location backend
..\ml\.venv\Scripts\python.exe -m alembic upgrade head
Pop-Location
```

---

## 6. Posicionar pesos do modelo (não versionados no Git)

Os arquivos binários de modelo (`.pt` / `.onnx`) são ignorados pelo `.gitignore`. Para rodar detecção em inspeções e no Laboratório de Visão:

- Coloque os pesos PyTorch em:
  `artifacts/runs/detect/exp04c_yolo26n_opentop_freeze10/weights/best.pt`
- (Opcional — necessário apenas para comparar o runtime `onnx_cpu`) coloque ou exporte o modelo ONNX em:
  `artifacts/models/onnx/exp04c_yolo26n_opentop_640.onnx` e seu `.meta.json`:

```powershell
.\ml\.venv\Scripts\python.exe ml\scripts\export_and_verify_onnx.py --weights artifacts\runs\detect\exp04c_yolo26n_opentop_freeze10\weights\best.pt
```

*(Caso os pesos estejam em outro diretório, defina `RAILVISION_PYTORCH_MODEL_PATH` e `RAILVISION_ONNX_MODEL_PATH` no `.env`.)*

---

## 7. Executar toda a aplicação (um comando)

Na raiz do repositório:

```powershell
.\start.ps1
```

Ou em segundo plano:

```powershell
.\start.ps1 -NoFollow
```

Endpoints locais após a inicialização:
- **Interface Web (Angular):** `http://localhost:4200`
- **API FastAPI:** `http://127.0.0.1:8080`
- **Swagger / OpenAPI Docs:** `http://127.0.0.1:8080/docs`
- **Health Checks:** `http://127.0.0.1:8080/health` e `http://127.0.0.1:8080/actuator/health`

Para encerrar tudo mantendo os dados do PostgreSQL no volume Docker `railvision_pgdata`:

```powershell
.\stop.ps1
```

---

## 8. Executar suítes de testes

```powershell
# Testes de backend + banco isolado + paridade de visão (requer PostgreSQL Docker ativo)
Push-Location backend
..\ml\.venv\Scripts\pytest.exe -v
Pop-Location

# Testes do ferramental offline ML
.\ml\.venv\Scripts\pytest.exe ml\tests -v

# Testes unitários e build de produção do Frontend Angular
npm --prefix frontend test -- --watch=false
npm --prefix frontend run build
```

---

## 9. Solução de problemas comuns

- **Porta `8080` ou `4200` ocupada:** execute `.\stop.ps1` ou verifique o processo ativo com `Get-NetTCPConnection -LocalPort 8080,4200 -State Listen`.
- **Erro `503 CUDA não está disponível` no runtime `pytorch_cuda`:** o sistema evita fallback silencioso por segurança. Confirme `torch.cuda.is_available()` ou selecione `pytorch_cpu` / `onnx_cpu` no Laboratório de Visão (ou defina `RAILVISION_DEFAULT_RUNTIME=pytorch_cpu` no `.env`).
- **Docker não responde:** abra o Docker Desktop e aguarde `docker info` retornar sem erros antes de rodar `.\start.ps1`.
