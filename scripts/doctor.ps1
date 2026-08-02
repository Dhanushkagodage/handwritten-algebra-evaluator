<#
.SYNOPSIS
    Diagnose a broken local setup. Read-only - changes nothing.

.DESCRIPTION
    Checks every prerequisite the integrated pipeline needs and prints exactly
    which one is missing. Run this first whenever the frontend reports
    "Couldn't reach the gateway".

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\doctor.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
$problems = New-Object System.Collections.ArrayList

function Write-Head($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Write-Ok($t)   { Write-Host "  OK    $t" -ForegroundColor Green }
function Write-Bad($t)  { Write-Host "  FAIL  $t" -ForegroundColor Red; [void]$problems.Add($t) }
function Write-Warn($t) { Write-Host "  warn  $t" -ForegroundColor Yellow }

$Services = @(
    [pscustomobject]@{ Name='ocr';       Dir='services\ocr-service';       Port=8000; Health='http://127.0.0.1:8000/';       Keys=@('OPENAI_API_KEY') }
    [pscustomobject]@{ Name='reasoning'; Dir='services\reasoning-service'; Port=8002; Health='http://127.0.0.1:8002/health'; Keys=@('OPENAI_API_KEY') }
    [pscustomobject]@{ Name='feedback';  Dir='services\feedback-service';  Port=8003; Health='http://127.0.0.1:8003/health'; Keys=@('SPACE_ID','API_KEY') }
    [pscustomobject]@{ Name='gateway';   Dir='services\gateway';           Port=8080; Health='http://127.0.0.1:8080/health'; Keys=@() }
)

# ------------------------------------------------------------------
Write-Head 'Environment'
Write-Host "  PowerShell  $($PSVersionTable.PSVersion)"
if ($PSVersionTable.PSVersion.Major -le 5) {
    Write-Host "              (Windows PowerShell - .ps1 files must be ASCII or carry a BOM)"
}
$py = (Get-Command py -ErrorAction SilentlyContinue)
if ($py) { Write-Ok "py launcher found" } else { Write-Bad "py launcher not found - install Python 3.10+" }
$npm = (Get-Command npm -ErrorAction SilentlyContinue)
if ($npm) { Write-Ok "npm found" } else { Write-Bad "npm not found - install Node 18+" }

# ------------------------------------------------------------------
Write-Head 'Scripts parse correctly'
Get-ChildItem (Join-Path $PSScriptRoot '*.ps1') | ForEach-Object {
    $errors = $null
    [System.Management.Automation.PSParser]::Tokenize((Get-Content $_.FullName -Raw), [ref]$errors) | Out-Null
    if ($errors.Count -eq 0) { Write-Ok $_.Name }
    else { Write-Bad "$($_.Name) has $($errors.Count) parse error(s) - first at line $($errors[0].Token.StartLine). Re-pull; it must be ASCII or BOM-encoded." }
}

# ------------------------------------------------------------------
Write-Head 'Per-service setup'
foreach ($s in $Services) {
    $dir = Join-Path $Root $s.Dir
    Write-Host "  [$($s.Name)]"

    if (-not (Test-Path $dir)) { Write-Bad "$($s.Name): directory missing - did the git pull succeed?"; continue }

    $python = Join-Path $dir '.venv\Scripts\python.exe'
    if (Test-Path $python) { Write-Ok "$($s.Name): virtualenv present" }
    else { Write-Bad "$($s.Name): no .venv - run scripts\dev.ps1 -Bootstrap" }

    $envFile = Join-Path $dir '.env'
    if (-not (Test-Path $envFile)) {
        if ($s.Keys.Count -gt 0) { Write-Bad "$($s.Name): no .env - copy .env.example to .env and fill it in" }
        else { Write-Warn "$($s.Name): no .env (optional - built-in defaults apply)" }
    } else {
        $content = Get-Content $envFile -Raw
        foreach ($key in $s.Keys) {
            $match = [regex]::Match($content, "(?m)^\s*$key\s*=\s*(.*)$")
            if (-not $match.Success) { Write-Bad "$($s.Name): .env is missing $key" }
            else {
                $value = $match.Groups[1].Value.Trim()
                if (-not $value) { Write-Bad "$($s.Name): $key is empty" }
                elseif ($value -match 'REPLACE_ME|your_.*_here|xxx') { Write-Bad "$($s.Name): $key still holds a placeholder" }
                else { Write-Ok "$($s.Name): $key set (length $($value.Length))" }
            }
        }
    }
}

# ------------------------------------------------------------------
Write-Head 'Frontend'
$fe = Join-Path $Root 'frontend'
if (Test-Path (Join-Path $fe 'node_modules')) { Write-Ok 'node_modules present' }
else { Write-Bad 'frontend\node_modules missing - run "npm install" in frontend' }

$feEnv = Join-Path $fe '.env'
if (Test-Path $feEnv) {
    $line = (Get-Content $feEnv -Raw | Select-String -Pattern '(?m)^\s*VITE_GATEWAY_URL\s*=\s*(.*)$')
    $val = if ($line) { $line.Matches[0].Groups[1].Value.Trim() } else { '' }
    if ($val) { Write-Host "  VITE_GATEWAY_URL = $val  (calls the gateway directly; CORS applies)" }
    else { Write-Ok 'VITE_GATEWAY_URL blank - uses the Vite dev proxy (recommended)' }
} else {
    Write-Ok 'no frontend\.env - uses the Vite dev proxy (recommended)'
}

$viteConfig = Get-Content (Join-Path $fe 'vite.config.ts') -Raw
if ($viteConfig -match "proxy") { Write-Ok 'vite.config.ts defines the /api proxy' }
else { Write-Bad 'vite.config.ts has no /api proxy - re-pull the branch' }

# ------------------------------------------------------------------
Write-Head 'Are the services actually running?'
foreach ($s in $Services) {
    $listening = Get-NetTCPConnection -LocalPort $s.Port -State Listen -ErrorAction SilentlyContinue
    if (-not $listening) {
        Write-Bad "$($s.Name): nothing listening on :$($s.Port)"
        continue
    }
    try {
        Invoke-RestMethod -Uri $s.Health -TimeoutSec 5 | Out-Null
        Write-Ok "$($s.Name): responding on :$($s.Port)"
    } catch {
        Write-Bad "$($s.Name): port :$($s.Port) is open but $($s.Health) failed - $($_.Exception.Message)"
    }
}

# ------------------------------------------------------------------
Write-Head 'Gateway view of the three modules'
try {
    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/health/services' -TimeoutSec 20
    $health.services.PSObject.Properties | ForEach-Object {
        if ($_.Value.status -eq 'up') { Write-Ok ("{0,-10} {1}" -f $_.Name, $_.Value.url) }
        else { Write-Bad ("{0,-10} {1} - {2}" -f $_.Name, $_.Value.url, $_.Value.error) }
    }
} catch {
    Write-Bad "gateway /health/services unreachable - the gateway is not running on :8080"
}

# ------------------------------------------------------------------
Write-Host ''
if ($problems.Count -eq 0) {
    Write-Host '  Everything checks out. If the browser still fails, hard-refresh it (Ctrl+F5).' -ForegroundColor Green
} else {
    Write-Host "  $($problems.Count) problem(s) found:" -ForegroundColor Red
    $problems | ForEach-Object { Write-Host "    - $_" }
    Write-Host ''
    Write-Host '  Usual fix:  powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 -Bootstrap -WithFrontend'
}
Write-Host ''
