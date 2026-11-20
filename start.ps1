[CmdletBinding()]
param(
    [switch]$NoFollow
)

$ErrorActionPreference = 'Stop'

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StateDir = Join-Path $RootDir 'artifacts\runs\local-dev'
$PidsFile = Join-Path $StateDir 'pids.json'
$BackendOutLog = Join-Path $StateDir 'backend.out.log'
$BackendErrLog = Join-Path $StateDir 'backend.err.log'
$FrontendOutLog = Join-Path $StateDir 'frontend.out.log'
$FrontendErrLog = Join-Path $StateDir 'frontend.err.log'
$ComposeFile = Join-Path $RootDir 'docker-compose.yml'

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

function Write-Info([string]$Message) {
    Write-Host "[RailVision] $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "[RailVision] $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "[RailVision] $Message" -ForegroundColor Yellow
}

function Write-Err([string]$Message) {
    Write-Host "[RailVision] ERRO: $Message" -ForegroundColor Red
}

function Import-DotEnv([string]$EnvPath, [string]$ExamplePath) {
    if (-not (Test-Path $EnvPath)) {
        if (Test-Path $ExamplePath) {
            Copy-Item -Path $ExamplePath -Destination $EnvPath
            Write-Warn "Arquivo .env criado a partir de .env.example."
        } else {
            Write-Err "Arquivo .env e .env.example nao encontrados na raiz."
            exit 1
        }
    }

    foreach ($rawLine in Get-Content -Path $EnvPath -Encoding UTF8) {
        $line = $rawLine.Trim()
        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith('#')) {
            continue
        }
        $eqIndex = $line.IndexOf('=')
        if ($eqIndex -le 0) {
            continue
        }
        $key = $line.Substring(0, $eqIndex).Trim()
        $val = $line.Substring($eqIndex + 1).Trim()
        if (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'"))) {
            $val = $val.Substring(1, $val.Length - 2)
        }
        [System.Environment]::SetEnvironmentVariable($key, $val, 'Process')
    }
}

function Get-ListeningPid([int]$Port) {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) {
        return [int]$conn.OwningProcess
    }
    return $null
}

function Get-ProcessSummary([int]$TargetPid) {
    $proc = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if ($proc) {
        return "$($proc.ProcessName) (PID $TargetPid)"
    }
    return "PID $TargetPid"
}

function Test-BackendReady([int]$Port) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $body = & curl.exe -sS --max-time 2 "http://127.0.0.1:$Port/actuator/health" 2>$null
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($code -ne 0 -or [string]::IsNullOrWhiteSpace($body)) {
        return $false
    }
    return ($body -match '"status"\s*:\s*"UP"' -and $body -match '"python-fastapi"')
}

function Test-FrontendReady([int]$Port) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $body = & curl.exe -sS --max-time 2 "http://localhost:$Port" 2>$null
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($code -ne 0 -or [string]::IsNullOrWhiteSpace($body)) {
        return $false
    }
    return ($body -match 'RailVision AI')
}

function Show-LogTail([string]$Title, [string]$OutPath, [string]$ErrPath, [int]$Lines = 25) {
    Write-Host "--- $Title (ultimas linhas) ---" -ForegroundColor Yellow
    if (Test-Path $OutPath) {
        Get-Content -Path $OutPath -Tail $Lines -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" }
    }
    if (Test-Path $ErrPath) {
        $errLines = Get-Content -Path $ErrPath -Tail $Lines -ErrorAction SilentlyContinue
        if ($errLines) {
            Write-Host "--- $Title (stderr) ---" -ForegroundColor Red
            $errLines | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        }
    }
}

function Read-AppendedLines([string]$Path, [hashtable]$Offsets, [string]$Tag, [string]$Color) {
    if (-not (Test-Path $Path)) {
        return
    }
    $fs = $null
    $sr = $null
    try {
        $fs = [System.IO.FileStream]::new(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite
        )
        $pos = if ($Offsets.ContainsKey($Path)) { [long]$Offsets[$Path] } else { 0L }
        if ($fs.Length -lt $pos) {
            $pos = 0L
        }
        $null = $fs.Seek($pos, [System.IO.SeekOrigin]::Begin)
        $sr = [System.IO.StreamReader]::new($fs, [System.Text.Encoding]::UTF8, $true)
        while (-not $sr.EndOfStream) {
            $line = $sr.ReadLine()
            if ($null -ne $line) {
                Write-Host "[$Tag] $line" -ForegroundColor $Color
            }
        }
        $Offsets[$Path] = $fs.Position
    } catch {
        # Ignorar leitura concorrente momentanea
    } finally {
        if ($sr) { $sr.Dispose() }
        elseif ($fs) { $fs.Dispose() }
    }
}

# 1. Carregar .env na sessao atual
Import-DotEnv -EnvPath (Join-Path $RootDir '.env') -ExamplePath (Join-Path $RootDir '.env.example')

$PostgresPort = if ($env:POSTGRES_PORT) { [int]$env:POSTGRES_PORT } else { 5432 }
$BackendPort = if ($env:RAILVISION_SERVER_PORT) { [int]$env:RAILVISION_SERVER_PORT } else { 8080 }
$FrontendPort = 4200

# 2. Validar pre-requisitos
Write-Info "Verificando pre-requisitos..."

$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& docker info *> $null
$dockerOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevPref
if (-not $dockerOk) {
    Write-Err "Docker Engine nao esta acessivel. Inicie o Docker Desktop e tente novamente."
    exit 1
}

$PythonExe = Join-Path $RootDir 'ml\.venv\Scripts\python.exe'
if (-not (Test-Path $PythonExe)) {
    Write-Err "Ambiente virtual Python nao encontrado em $PythonExe."
    exit 1
}

$BackendApp = Join-Path $RootDir 'backend\app\main.py'
if (-not (Test-Path $BackendApp)) {
    Write-Err "Aplicacao FastAPI nao encontrada em $BackendApp."
    exit 1
}

$nodeModulesPath = Join-Path $RootDir 'frontend\node_modules'
if (-not (Test-Path $nodeModulesPath)) {
    Write-Err "Dependencias do frontend ausentes. Execute primeiro: npm --prefix frontend install"
    exit 1
}

$existingState = $null
if (Test-Path $PidsFile) {
    try {
        $existingState = Get-Content -Path $PidsFile -Raw | ConvertFrom-Json
    } catch {
        $existingState = $null
    }
}

# 3. Iniciar PostgreSQL via Docker Compose
Write-Info "Iniciando PostgreSQL (Docker Compose)..."
$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& docker compose -f $ComposeFile up -d postgres
$composeExit = $LASTEXITCODE
$ErrorActionPreference = $prevPref
if ($composeExit -ne 0) {
    Write-Err "Falha ao subir o container PostgreSQL via Docker Compose."
    exit 1
}

$pgHealthy = $false
for ($i = 0; $i -lt 60; $i++) {
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $healthStatus = (& docker inspect --format '{{.State.Health.Status}}' railvision-postgres 2>$null)
    $ErrorActionPreference = $prevPref
    if ($healthStatus -eq 'healthy') {
        $pgHealthy = $true
        break
    }
    Start-Sleep -Seconds 1
}
if (-not $pgHealthy) {
    Write-Err "PostgreSQL nao ficou saudavel dentro de 60 segundos."
    & docker compose -f $ComposeFile logs --tail 30 postgres
    exit 1
}
Write-Ok "PostgreSQL pronto (127.0.0.1:$PostgresPort, status: healthy)."

# 4. Sincronizar historico Alembic e aplicar migracoes pendentes
Write-Info "Verificando schema Alembic..."
Push-Location (Join-Path $RootDir 'backend')
try {
    $needsStamp = & $PythonExe -c "import sqlalchemy as sa; from app.core.config import get_settings; e = sa.create_engine(get_settings().database_url); insp = sa.inspect(e); print('STAMP' if (insp.has_table('inspections') and not insp.has_table('alembic_version')) else 'OK'); e.dispose()"
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Falha ao inspecionar estado de migracao no PostgreSQL."
        exit 1
    }
    if ($needsStamp -eq 'STAMP') {
        Write-Info "Schema existente detectado; registrando baseline Alembic 0001_initial_schema sem recriar tabelas..."
        & $PythonExe -m alembic stamp 0001_initial_schema
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Falha ao registrar baseline Alembic."
            exit 1
        }
    }
    & $PythonExe -m alembic upgrade head
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Falha ao executar alembic upgrade head."
        exit 1
    }
} finally {
    Pop-Location
}

# 5. Iniciar Backend Python + FastAPI
$backendPidToSave = $null
$existingBackendListener = Get-ListeningPid -Port $BackendPort
if ($existingBackendListener) {
    if (Test-BackendReady -Port $BackendPort) {
        Write-Ok "Backend FastAPI ja esta em execucao em http://127.0.0.1:$BackendPort ($(Get-ProcessSummary $existingBackendListener)) - reutilizando instancia."
        if ($existingState -and $existingState.backendPid) {
            $tracked = Get-Process -Id ([int]$existingState.backendPid) -ErrorAction SilentlyContinue
            if ($tracked) { $backendPidToSave = [int]$existingState.backendPid }
        }
        if (-not $backendPidToSave) { $backendPidToSave = $existingBackendListener }
    } else {
        Write-Err "Porta $BackendPort ocupada por outro processo: $(Get-ProcessSummary $existingBackendListener)."
        exit 1
    }
} else {
    Write-Info "Iniciando Backend Python + FastAPI em http://127.0.0.1:$BackendPort..."
    Set-Content -Path $BackendOutLog -Value $null -Encoding UTF8
    Set-Content -Path $BackendErrLog -Value $null -Encoding UTF8

    $backendProc = Start-Process -FilePath $PythonExe `
        -ArgumentList '-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', "$BackendPort" `
        -WorkingDirectory $RootDir `
        -RedirectStandardOutput $BackendOutLog `
        -RedirectStandardError $BackendErrLog `
        -WindowStyle Hidden `
        -PassThru

    $backendPidToSave = $backendProc.Id
    $backendUp = $false
    for ($i = 0; $i -lt 60; $i++) {
        if ($backendProc.HasExited) {
            Write-Err "Backend FastAPI encerrou prematuramente (codigo $($backendProc.ExitCode))."
            Show-LogTail -Title 'Backend' -OutPath $BackendOutLog -ErrPath $BackendErrLog -Lines 40
            exit 1
        }
        if (Test-BackendReady -Port $BackendPort) {
            $backendUp = $true
            break
        }
        Start-Sleep -Seconds 1
    }
    if (-not $backendUp) {
        Write-Err "Backend FastAPI nao respondeu UP em /actuator/health dentro de 60 segundos."
        Show-LogTail -Title 'Backend' -OutPath $BackendOutLog -ErrPath $BackendErrLog -Lines 40
        & cmd.exe /c "taskkill /PID $($backendProc.Id) /T /F >nul 2>&1"
        exit 1
    }
    Write-Ok "Backend FastAPI pronto em http://127.0.0.1:$BackendPort."
}

# 6. Iniciar Frontend Angular
$frontendPidToSave = $null
$existingFrontendListener = Get-ListeningPid -Port $FrontendPort
if ($existingFrontendListener) {
    if (Test-FrontendReady -Port $FrontendPort) {
        Write-Ok "Frontend ja esta em execucao em http://localhost:$FrontendPort ($(Get-ProcessSummary $existingFrontendListener)) - reutilizando instancia."
        if ($existingState -and $existingState.frontendPid) {
            $tracked = Get-Process -Id ([int]$existingState.frontendPid) -ErrorAction SilentlyContinue
            if ($tracked) { $frontendPidToSave = [int]$existingState.frontendPid }
        }
        if (-not $frontendPidToSave) { $frontendPidToSave = $existingFrontendListener }
    } else {
        Write-Err "Porta $FrontendPort ocupada por outro processo: $(Get-ProcessSummary $existingFrontendListener)."
        exit 1
    }
} else {
    Write-Info "Iniciando Frontend Angular em http://localhost:$FrontendPort..."
    Set-Content -Path $FrontendOutLog -Value $null -Encoding UTF8
    Set-Content -Path $FrontendErrLog -Value $null -Encoding UTF8
    $env:NG_CLI_ANALYTICS = 'false'

    $frontendProc = Start-Process -FilePath 'cmd.exe' `
        -ArgumentList '/c', 'npm.cmd --prefix frontend start' `
        -WorkingDirectory $RootDir `
        -RedirectStandardOutput $FrontendOutLog `
        -RedirectStandardError $FrontendErrLog `
        -WindowStyle Hidden `
        -PassThru

    $frontendPidToSave = $frontendProc.Id
    $frontendUp = $false
    for ($i = 0; $i -lt 90; $i++) {
        if ($frontendProc.HasExited) {
            Write-Err "Frontend encerrou prematuramente (codigo $($frontendProc.ExitCode))."
            Show-LogTail -Title 'Frontend' -OutPath $FrontendOutLog -ErrPath $FrontendErrLog -Lines 40
            exit 1
        }
        if (Test-FrontendReady -Port $FrontendPort) {
            $frontendUp = $true
            break
        }
        Start-Sleep -Seconds 1
    }
    if (-not $frontendUp) {
        Write-Err "Frontend nao respondeu em http://localhost:$FrontendPort dentro de 90 segundos."
        Show-LogTail -Title 'Frontend' -OutPath $FrontendOutLog -ErrPath $FrontendErrLog -Lines 40
        & cmd.exe /c "taskkill /PID $($frontendProc.Id) /T /F >nul 2>&1"
        exit 1
    }
    Write-Ok "Frontend pronto em http://localhost:$FrontendPort."
}

# 7. Salvar estado de PIDs para stop.ps1
$statePayload = [ordered]@{
    backendPid   = $backendPidToSave
    frontendPid  = $frontendPidToSave
    backendPort  = $BackendPort
    frontendPort = $FrontendPort
    startedAt    = (Get-Date).ToString('o')
}
$statePayload | ConvertTo-Json | Set-Content -Path $PidsFile -Encoding UTF8

Write-Host ""
Write-Host "================ RailVision AI em execucao ================" -ForegroundColor Green
Write-Host "  Frontend:   http://localhost:$FrontendPort"
Write-Host "  Backend:    http://127.0.0.1:$BackendPort"
Write-Host "  Docs API:   http://127.0.0.1:$BackendPort/docs"
Write-Host "  Health:     http://127.0.0.1:$BackendPort/actuator/health"
Write-Host "  PostgreSQL: 127.0.0.1:$PostgresPort"
Write-Host "  Logs:       $StateDir"
Write-Host "  Encerrar:   .\stop.ps1"
Write-Host "===========================================================" -ForegroundColor Green
Write-Host ""

Show-LogTail -Title 'Backend' -OutPath $BackendOutLog -ErrPath $BackendErrLog -Lines 8
Show-LogTail -Title 'Frontend' -OutPath $FrontendOutLog -ErrPath $FrontendErrLog -Lines 8

if ($NoFollow) {
    Write-Info "Modo -NoFollow ativo. Todos os servicos permanecem ativos em segundo plano."
    exit 0
}

Write-Info "Acompanhando logs ao vivo (pressione Ctrl+C para sair do visualizador; use .\stop.ps1 para parar os servicos)..."
$offsets = @{}
foreach ($p in @($BackendOutLog, $BackendErrLog, $FrontendOutLog, $FrontendErrLog)) {
    if (Test-Path $p) {
        $offsets[$p] = (Get-Item $p).Length
    }
}

try {
    while ($true) {
        Read-AppendedLines -Path $BackendOutLog -Offsets $offsets -Tag 'BACKEND' -Color 'Cyan'
        Read-AppendedLines -Path $BackendErrLog -Offsets $offsets -Tag 'BACKEND-ERR' -Color 'Red'
        Read-AppendedLines -Path $FrontendOutLog -Offsets $offsets -Tag 'FRONTEND' -Color 'Magenta'
        Read-AppendedLines -Path $FrontendErrLog -Offsets $offsets -Tag 'FRONTEND-ERR' -Color 'Red'
        Start-Sleep -Milliseconds 400
    }
} finally {
    Write-Host ""
    Write-Warn "Visualizador de logs encerrado. Os servicos continuam rodando; execute .\stop.ps1 para encerra-los."
}
