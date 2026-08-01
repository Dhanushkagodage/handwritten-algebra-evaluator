<#
.SYNOPSIS
    End-to-end smoke test against a running stack.

.DESCRIPTION
    Starts a job on the gateway, polls it to completion, and prints the graded
    result. Verifies the whole OCR -> reasoning -> feedback chain for real.

.EXAMPLE
    .\scripts\smoke.ps1 -AnswerImage .\samples\answer.jpg -SchemeImage .\samples\scheme.jpg
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$AnswerImage,
    [Parameter(Mandatory = $true)][string]$SchemeImage,
    [string]$QuestionText = '',
    [string]$GatewayUrl = 'http://127.0.0.1:8080',
    [int]$TimeoutSec = 360
)

$ErrorActionPreference = 'Stop'

foreach ($path in @($AnswerImage, $SchemeImage)) {
    if (-not (Test-Path $path)) { Write-Host "Not found: $path" -ForegroundColor Red; exit 1 }
}

Write-Host '==> Checking the gateway can see all three modules' -ForegroundColor Cyan
$health = Invoke-RestMethod -Uri "$GatewayUrl/health/services" -TimeoutSec 15
$health.services.PSObject.Properties | ForEach-Object {
    $colour = if ($_.Value.status -eq 'up') { 'Green' } else { 'Red' }
    Write-Host ("  {0,-10} {1,-5} {2}" -f $_.Name, $_.Value.status, $_.Value.url) -ForegroundColor $colour
}
if ($health.status -ne 'ok') {
    Write-Host 'Some services are down — start them with .\scripts\dev.ps1' -ForegroundColor Red
    exit 1
}

Write-Host '==> Starting an evaluation' -ForegroundColor Cyan
$form = @{
    answer_images        = Get-Item $AnswerImage
    marking_scheme_image = Get-Item $SchemeImage
    question_text        = $QuestionText
}
$job = Invoke-RestMethod -Uri "$GatewayUrl/api/v1/jobs" -Method Post -Form $form -TimeoutSec 120
Write-Host "  job $($job.job_id)"

$deadline = (Get-Date).AddSeconds($TimeoutSec)
$lastStage = ''
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 1500
    $status = Invoke-RestMethod -Uri "$GatewayUrl$($job.poll_url)" -TimeoutSec 20

    if ($status.stage -ne $lastStage) {
        Write-Host "  [$($status.stage)] $($status.stage_message)" -ForegroundColor DarkGray
        $lastStage = $status.stage
    }

    if ($status.status -eq 'succeeded') {
        Write-Host '==> Result' -ForegroundColor Cyan
        Write-Host "  Score      $($status.result.final_score) / $($status.result.total_marks)"
        Write-Host "  Question   $($status.result.question_text)"
        Write-Host "  Timings    $($status.result.timings_ms | ConvertTo-Json -Compress)"
        Write-Host ''
        foreach ($step in $status.result.step_feedback) {
            Write-Host "  Step $($step.step_number) [$($step.validity)] $($step.marks_awarded) marks — $($step.expression)"
            Write-Host "      correct : $($step.what_is_correct)"
            if ($step.what_is_missing)    { Write-Host "      missing : $($step.what_is_missing)" }
            if ($step.why_marks_reduced)  { Write-Host "      why     : $($step.why_marks_reduced)" }
            Write-Host "      improve : $($step.how_to_improve)"
        }
        Write-Host ''
        Write-Host "  Overall    $($status.result.overall_feedback)"
        foreach ($warning in $status.result.warnings) {
            Write-Host "  ! $warning" -ForegroundColor Yellow
        }
        exit 0
    }

    if ($status.status -in @('failed', 'cancelled')) {
        Write-Host "==> $($status.status.ToUpper())" -ForegroundColor Red
        Write-Host "  stage   $($status.error.stage)"
        Write-Host "  code    $($status.error.error_code)"
        Write-Host "  message $($status.error.message)"
        exit 1
    }
}

Write-Host "Timed out after ${TimeoutSec}s. Job $($job.job_id) may still be running." -ForegroundColor Red
exit 1
