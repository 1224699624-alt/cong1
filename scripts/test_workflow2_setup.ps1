param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$SessionId = ""
)

$ErrorActionPreference = "Continue"

function Test-ItemExists {
    param(
        [string]$Label,
        [string]$Path
    )
    if (Test-Path $Path) {
        Write-Host "[OK] $Label -> $Path"
        return $true
    }
    else {
        Write-Warning "[MISS] $Label -> $Path"
        return $false
    }
}

function Test-CommandExists {
    param([string]$Name)
    $Cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($Cmd) {
        Write-Host "[OK] command '$Name' -> $($Cmd.Source)"
        return $true
    }
    else {
        Write-Warning "[MISS] command '$Name'"
        return $false
    }
}

$PromptFile = Join-Path $RepoRoot "research-workflow\WORKFLOW2_AUTORUN_PROMPT.md"
$GuideFile = Join-Path $RepoRoot "research-workflow\WORKFLOW_2_AUTORUN.md"
$LoopScript = Join-Path $RepoRoot "scripts\run_workflow2_loop.ps1"
$NotifyScript = Join-Path $RepoRoot "scripts\send_workflow2_notification.ps1"
$InstallTaskScript = Join-Path $RepoRoot "scripts\install_workflow2_task.ps1"
$StatusFile = Join-Path $RepoRoot "research-workflow\refine-logs\AUTONOMOUS_LOOP_STATUS.md"
$TrackerFile = Join-Path $RepoRoot "research-workflow\refine-logs\EXPERIMENT_TRACKER.md"
$ResultsFile = Join-Path $RepoRoot "research-workflow\refine-logs\EXPERIMENT_RESULTS.md"
$RepairPlanFile = Join-Path $RepoRoot "research-workflow\refine-logs\NEXT_CANDIDATE_REPAIR_PLAN.md"

$AllOk = $true

$AllOk = (Test-ItemExists "Repo root" $RepoRoot) -and $AllOk
$AllOk = (Test-ItemExists "Workflow 2 guide" $GuideFile) -and $AllOk
$AllOk = (Test-ItemExists "Workflow 2 prompt" $PromptFile) -and $AllOk
$AllOk = (Test-ItemExists "Workflow 2 loop script" $LoopScript) -and $AllOk
$AllOk = (Test-ItemExists "Workflow 2 install-task script" $InstallTaskScript) -and $AllOk
$AllOk = (Test-ItemExists "Workflow 2 notify script" $NotifyScript) -and $AllOk
$AllOk = (Test-ItemExists "Autonomous loop status" $StatusFile) -and $AllOk
$AllOk = (Test-ItemExists "Experiment tracker" $TrackerFile) -and $AllOk
$AllOk = (Test-ItemExists "Experiment results" $ResultsFile) -and $AllOk
$AllOk = (Test-ItemExists "Repair plan" $RepairPlanFile) -and $AllOk
$AllOk = (Test-CommandExists "codex") -and $AllOk

try {
    $null = & codex exec --help 2>$null
    Write-Host "[OK] codex exec is available"
}
catch {
    Write-Warning "[MISS] codex exec failed: $($_.Exception.Message)"
    $AllOk = $false
}

try {
    $null = & codex resume --help 2>$null
    Write-Host "[OK] codex resume is available"
}
catch {
    Write-Warning "[MISS] codex resume failed: $($_.Exception.Message)"
    $AllOk = $false
}

try {
    $null = & powershell -NoProfile -ExecutionPolicy Bypass -File $LoopScript -RepoRoot $RepoRoot -MaxPasses 1 -DryRun 2>$null
    Write-Host "[OK] Workflow 2 loop dry-run invocation works"
}
catch {
    Write-Warning "[MISS] Workflow 2 loop dry-run failed: $($_.Exception.Message)"
    $AllOk = $false
}

if ($SessionId) {
    Write-Host "[INFO] Resume mode target session: $SessionId"
}
else {
    Write-Host "[INFO] No SessionId supplied. Default recommendation is to use non-interactive exec mode."
}

$DaemonLogDir = Join-Path $RepoRoot "research-workflow\daemon-logs"
if (-not (Test-Path $DaemonLogDir)) {
    New-Item -ItemType Directory -Force -Path $DaemonLogDir | Out-Null
    Write-Host "[OK] Created daemon log dir -> $DaemonLogDir"
}
else {
    Write-Host "[OK] Daemon log dir -> $DaemonLogDir"
}

if ($AllOk) {
    Write-Host ""
    Write-Host "Workflow 2 setup check PASSED."
    Write-Host "Recommended start command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File `"$LoopScript`" -RepoRoot `"$RepoRoot`" -IntervalMinutes 20 -MaxPasses 999 -Model `"5.6-sol`""
    Write-Host "Recommended smoke test:"
    Write-Host "powershell -ExecutionPolicy Bypass -File `"$LoopScript`" -RepoRoot `"$RepoRoot`" -MaxPasses 1 -DryRun"
    exit 0
}
else {
    Write-Host ""
    Write-Warning "Workflow 2 setup check FAILED. Fix missing items before relying on unattended autorun."
    exit 1
}
