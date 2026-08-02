<#
.SYNOPSIS
    Stop everything dev.ps1 started.

.DESCRIPTION
    Uses .logs\pids.json when available, and falls back to whatever is listening
    on the known ports (which also catches services started by hand).
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $Root '.logs\pids.json'
$Ports = @(8000, 8002, 8003, 8090, 5173)

function Write-Ok($message)  { Write-Host "  + $message" -ForegroundColor Green }
function Write-Info($message) { Write-Host "  - $message" -ForegroundColor DarkGray }

$stopped = 0

if (Test-Path $PidFile) {
    $entries = Get-Content $PidFile -Raw | ConvertFrom-Json
    foreach ($entry in @($entries)) {
        $process = Get-Process -Id $entry.pid -ErrorAction SilentlyContinue
        if ($process) {
            Stop-Process -Id $entry.pid -Force -ErrorAction SilentlyContinue
            Write-Ok "stopped $($entry.name) (PID $($entry.pid))"
            $stopped++
        }
    }
    Remove-Item $PidFile -ErrorAction SilentlyContinue
}

# Catch anything still holding a port - e.g. a service started manually, or a
# uvicorn child that outlived its launcher window.
foreach ($port in $Ports) {
    $listening = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($connection in @($listening)) {
        $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
        if ($process) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            Write-Ok "stopped $($process.ProcessName) on port $port (PID $($process.Id))"
            $stopped++
        }
    }
}

if ($stopped -eq 0) { Write-Info 'Nothing was running.' }
