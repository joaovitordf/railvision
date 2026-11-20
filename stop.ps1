[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StateDir = Join-Path $RootDir 'artifacts\runs\local-dev'
$PidsFile = Join-Path $StateDir 'pids.json'
$ComposeFile = Join-Path $RootDir 'docker-compose.yml'

function Write-Info([string]$Message) {
    Write-Host "[RailVision] $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "[RailVision] $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "[RailVision] $Message" -ForegroundColor Yellow
}

function Stop-ProcessTree([int]$TargetPid, [string]$Label) {
    if ($TargetPid -le 0) {
        return
    }
    $proc = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if (-not $proc) {
        Write-Info "$Label (PID $TargetPid) ja nao estava em execucao."
        return
    }
    Write-Info "Encerrando $Label (arvore de processos PID $TargetPid)..."
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & cmd.exe /c "taskkill /PID $TargetPid /T /F >nul 2>&1"
    $ErrorActionPreference = $prev
    Start-Sleep -Milliseconds 300
    Write-Ok "$Label encerrado."
}

function Stop-RailVisionPortFallback([int]$Port, [string]$Label) {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $conns) {
        $listeningPid = [int]$conn.OwningProcess
        if ($listeningPid -le 0) {
            continue
        }
        $cim = Get-CimInstance Win32_Process -Filter "ProcessId = $listeningPid" -ErrorAction SilentlyContinue
        $cmdLine = if ($cim -and $cim.CommandLine) { $cim.CommandLine } else { '' }
        if ($cmdLine -match 'railvision|uvicorn') {
            Stop-ProcessTree -TargetPid $listeningPid -Label "$Label (porta $Port)"
        } else {
            Write-Warn "Porta $Port ainda ocupada por processo nao identificado como RailVision (PID $listeningPid); mantido por seguranca."
        }
    }
}

$backendPort = 8080
$frontendPort = 4200
$envPath = Join-Path $RootDir '.env'
if (Test-Path $envPath) {
    foreach ($rawLine in Get-Content -Path $envPath -Encoding UTF8) {
        $line = $rawLine.Trim()
        if ($line -match '^RAILVISION_SERVER_PORT\s*=\s*(\d+)$') {
            $backendPort = [int]$Matches[1]
        }
    }
}

if (Test-Path $PidsFile) {
    try {
        $state = Get-Content -Path $PidsFile -Raw | ConvertFrom-Json
        if ($state.frontendPort) { $frontendPort = [int]$state.frontendPort }
        if ($state.backendPort) { $backendPort = [int]$state.backendPort }

        if ($state.frontendPid) {
            Stop-ProcessTree -TargetPid ([int]$state.frontendPid) -Label 'Frontend Angular'
        }
        if ($state.backendPid) {
            Stop-ProcessTree -TargetPid ([int]$state.backendPid) -Label 'Backend FastAPI'
        }
    } catch {
        Write-Warn "Nao foi possivel ler pids.json; verificando portas configuradas."
    }
} else {
    Write-Info "Nenhum arquivo pids.json encontrado; verificando processos do RailVision nas portas $frontendPort e $backendPort."
}

Start-Sleep -Milliseconds 400
Stop-RailVisionPortFallback -Port $frontendPort -Label 'Frontend Angular'
Stop-RailVisionPortFallback -Port $backendPort -Label 'Backend FastAPI'

if (Test-Path $PidsFile) {
    Remove-Item -Path $PidsFile -Force -ErrorAction SilentlyContinue
}

# Encerrar container PostgreSQL mantendo o volume persistente railvision_pgdata
$prev = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$pgRunning = & docker ps -q --filter "name=^railvision-postgres$" 2>$null
if (-not [string]::IsNullOrWhiteSpace($pgRunning)) {
    Write-Info "Encerrando PostgreSQL (Docker Compose)..."
    & docker compose -f $ComposeFile stop postgres
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "PostgreSQL encerrado (volume de dados preservado)."
    } else {
        Write-Warn "Falha ao parar o container PostgreSQL."
    }
} else {
    Write-Info "PostgreSQL (railvision-postgres) ja esta parado."
}
$ErrorActionPreference = $prev

Write-Ok "Encerramento do RailVision AI concluido."
