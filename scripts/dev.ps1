<#
.SYNOPSIS
    Start all three module services plus the gateway (and optionally the frontend).

.DESCRIPTION
    Each service keeps its own venv and can still be started by hand exactly as
    before - this script is a convenience, not a new requirement.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 -WithFrontend
#>
[CmdletBinding()]
param(
    [switch]$WithFrontend,
    [switch]$Reload,
    [switch]$Bootstrap,
    [switch]$SkipHealthWait,
    [int]$HealthTimeoutSec = 180
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root '.logs'

function Write-Step($message) { Write-Host "==> $message" -ForegroundColor Cyan }
function Write-Warn($message) { Write-Host "  ! $message" -ForegroundColor Yellow }
function Write-Ok($message)   { Write-Host "  + $message" -ForegroundColor Green }
function Write-Bad($message)  { Write-Host "  x $message" -ForegroundColor Red }

# NOTE: ocr-service is `app:app` on port 8000 - a flat app.py, not app.main:app,
# and not port 8001 as the repo-root README claims.
$Services = @(
    [pscustomobject]@{ Name = 'ocr';       Dir = 'services\ocr-service';       Module = 'app:app';      Port = 8000; Health = 'http://127.0.0.1:8000/';              NeedsEnv = $true  }
    [pscustomobject]@{ Name = 'reasoning'; Dir = 'services\reasoning-service'; Module = 'app.main:app'; Port = 8002; Health = 'http://127.0.0.1:8002/health';        NeedsEnv = $true  }
    [pscustomobject]@{ Name = 'feedback';  Dir = 'services\feedback-service';  Module = 'app.main:app'; Port = 8003; Health = 'http://127.0.0.1:8003/health';        NeedsEnv = $true  }
    [pscustomobject]@{ Name = 'gateway';   Dir = 'services\gateway';           Module = 'app.main:app'; Port = 8080; Health = 'http://127.0.0.1:8080/health';        NeedsEnv = $false }
)

# -- Preflight ------------------------------------------------------------

Write-Step 'Preflight'

$busy = @()
foreach ($service in $Services) {
    $listening = Get-NetTCPConnection -LocalPort $service.Port -State Listen -ErrorAction SilentlyContinue
    if ($listening) {
        $owner = (Get-Process -Id $listening[0].OwningProcess -ErrorAction SilentlyContinue).ProcessName
        $busy += "  port $($service.Port) ($($service.Name)) is already in use by $owner (PID $($listening[0].OwningProcess))"
    }
}
if ($busy.Count -gt 0) {
    Write-Bad 'Ports already in use:'
    $busy | ForEach-Object { Write-Host $_ }
    Write-Host ''
    Write-Host 'Run .\scripts\stop-dev.ps1 first, or stop those processes manually.'
    exit 1
}

foreach ($service in $Services) {
    $dir = Join-Path $Root $service.Dir
    if (-not (Test-Path $dir)) { Write-Bad "Missing directory: $dir"; exit 1 }

    $python = Join-Path $dir '.venv\Scripts\python.exe'
    if (-not (Test-Path $python)) {
        if (-not $Bootstrap) {
            Write-Bad "$($service.Name): no virtualenv at $dir\.venv"
            Write-Host "      Re-run with -Bootstrap to create it, or set it up manually:"
            Write-Host "      cd $dir; py -3 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
            exit 1
        }
        Write-Step "$($service.Name): creating virtualenv"
        & py -3 -m venv (Join-Path $dir '.venv')
        & $python -m pip install --quiet --upgrade pip
        & $python -m pip install --quiet -r (Join-Path $dir 'requirements.txt')
        Write-Ok "$($service.Name): virtualenv ready"
    }
    $service | Add-Member -NotePropertyName Python -NotePropertyValue $python -Force

    # reasoning-service raises at IMPORT time without OPENAI_API_KEY, so a missing
    # .env there is a hard failure rather than a late 503.
    if ($service.NeedsEnv -and -not (Test-Path (Join-Path $dir '.env'))) {
        Write-Warn "$($service.Name): no .env file - copy .env.example to .env and fill it in."
        if ($service.Name -eq 'reasoning') {
            Write-Bad 'reasoning-service fails at import without OPENAI_API_KEY. Aborting.'
            Write-Host "      Copy-Item $dir\.env.example $dir\.env   # then set OPENAI_API_KEY"
            exit 1
        }
    }
}

if (-not (Test-Path (Join-Path $Root 'services\gateway\.env'))) {
    Write-Warn 'gateway: no .env - using built-in defaults (ocr 8000 / reasoning 8002 / feedback 8003).'
}

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# -- Launch ------------------------------------------------------------

Write-Step 'Starting services'
if ($Reload) {
    Write-Warn '--reload is on: the gateway loses its in-memory jobs on every file save.'
}

$started = @()
foreach ($service in $Services) {
    $dir = Join-Path $Root $service.Dir
    # Not $args - that is a PowerShell automatic variable.
    $uvicornArgs = "-m uvicorn $($service.Module) --host 127.0.0.1 --port $($service.Port)"
    # The gateway's job store lives in this process - never more than one worker.
    if ($service.Name -eq 'gateway') { $uvicornArgs += ' --workers 1' }
    if ($Reload) { $uvicornArgs += ' --reload' }

    $process = Start-Process -FilePath 'powershell' -ArgumentList @(
        '-NoExit', '-Command', "`$host.UI.RawUI.WindowTitle = '$($service.Name) :$($service.Port)'; Set-Location '$dir'; & '$($service.Python)' $uvicornArgs"
    ) -PassThru
    $started += [pscustomobject]@{ name = $service.Name; pid = $process.Id; port = $service.Port }
    Write-Ok "$($service.Name) starting on :$($service.Port) (PID $($process.Id))"
}

if ($WithFrontend) {
    $frontend = Join-Path $Root 'frontend'
    $process = Start-Process -FilePath 'powershell' -ArgumentList @(
        '-NoExit', '-Command', "`$host.UI.RawUI.WindowTitle = 'frontend :5173'; Set-Location '$frontend'; npm run dev"
    ) -PassThru
    $started += [pscustomobject]@{ name = 'frontend'; pid = $process.Id; port = 5173 }
    Write-Ok "frontend starting on :5173 (PID $($process.Id))"
}

$started | ConvertTo-Json | Out-File -FilePath (Join-Path $LogDir 'pids.json') -Encoding utf8

# -- Health wait ------------------------------------------------------------

if (-not $SkipHealthWait) {
    Write-Step "Waiting for services (up to ${HealthTimeoutSec}s)"
    Write-Host '      reasoning-service takes longest - it imports LangGraph and LangChain.'

    $deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
    $pending = [System.Collections.ArrayList]@($Services)

    while ($pending.Count -gt 0 -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        foreach ($service in @($pending)) {
            try {
                $response = Invoke-WebRequest -Uri $service.Health -UseBasicParsing -TimeoutSec 3
                if ($response.StatusCode -eq 200) {
                    Write-Ok "$($service.Name) is up"
                    $pending.Remove($service)
                }
            } catch { }
        }
    }

    foreach ($service in $pending) {
        Write-Bad "$($service.Name) did not become healthy at $($service.Health)"
    }

    if ($pending.Count -eq 0) {
        Write-Step 'Gateway view of the three modules'
        try {
            Invoke-RestMethod -Uri 'http://127.0.0.1:8080/health/services' -TimeoutSec 10 |
                ConvertTo-Json -Depth 6 | Write-Host
        } catch {
            Write-Warn "Could not read /health/services: $_"
        }
    }
}

# -- Banner ------------------------------------------------------------

Write-Host ''
Write-Host '  Gateway    http://127.0.0.1:8080/docs   <- the one call' -ForegroundColor White
Write-Host '  OCR        http://127.0.0.1:8000/docs'
Write-Host '  Reasoning  http://127.0.0.1:8002/docs'
Write-Host '  Feedback   http://127.0.0.1:8003/docs'
if ($WithFrontend) { Write-Host '  Frontend   http://localhost:5173' -ForegroundColor White }
Write-Host ''
Write-Host '  The first evaluation can take 2-4 minutes: the feedback model runs on a' -ForegroundColor DarkGray
Write-Host '  Hugging Face Space that cold-starts. Later runs are 60-90 seconds.' -ForegroundColor DarkGray
Write-Host ''
Write-Host '  Stop everything:  .\scripts\stop-dev.ps1' -ForegroundColor DarkGray
Write-Host ''
