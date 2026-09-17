param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$RemoteHost = "10.1.115.157",
    [string]$RemoteUser = "shenzeyu",
    [string]$RemotePassword = "40904090TWO",
    [string]$RunId = "R011",
    [int]$TailLines = 30
)

$ErrorActionPreference = "Continue"

function Read-TextFile {
    param([string]$Path)
    if (Test-Path $Path) {
        return Get-Content -Raw $Path
    }
    return ""
}

function Show-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("==== {0} ====" -f $Title) -ForegroundColor Cyan
}

function Get-RunLine {
    param(
        [string]$TrackerText,
        [string]$TargetRunId
    )
    foreach ($line in ($TrackerText -split "`r?`n")) {
        if ($line -like "| $TargetRunId *") {
            return $line
        }
    }
    return ""
}

function Get-RunStatus {
    param([string]$RunLine)
    if (-not $RunLine) { return "" }
    $parts = $RunLine.Split('|') | ForEach-Object { $_.Trim() }
    if ($parts.Count -ge 9) {
        return $parts[8]
    }
    return ""
}

function Get-RunNotes {
    param([string]$RunLine)
    if (-not $RunLine) { return "" }
    $parts = $RunLine.Split('|') | ForEach-Object { $_.Trim() }
    if ($parts.Count -ge 10) {
        return $parts[9]
    }
    return ""
}

$StatusFile = Join-Path $RepoRoot "research-workflow\refine-logs\AUTONOMOUS_LOOP_STATUS.md"
$TrackerFile = Join-Path $RepoRoot "research-workflow\refine-logs\EXPERIMENT_TRACKER.md"
$ResultsFile = Join-Path $RepoRoot "research-workflow\refine-logs\EXPERIMENT_RESULTS.md"
$SummaryFile = Join-Path $RepoRoot "research-workflow\refine-logs\PROGRESS_SUMMARY_CN.md"
$ReviewDoc = Join-Path $RepoRoot "review-stage\AUTO_REVIEW.md"
$ReviewState = Join-Path $RepoRoot "review-stage\REVIEW_STATE.json"

$StatusText = Read-TextFile $StatusFile
$TrackerText = Read-TextFile $TrackerFile
$ResultsText = Read-TextFile $ResultsFile
$SummaryText = Read-TextFile $SummaryFile

$RunLine = Get-RunLine -TrackerText $TrackerText -TargetRunId $RunId
$RunStatus = Get-RunStatus -RunLine $RunLine
$RunNotes = Get-RunNotes -RunLine $RunLine

Write-Host "Workflow status overview" -ForegroundColor Yellow
Write-Host ("RepoRoot: {0}" -f $RepoRoot)
Write-Host ("RunId: {0}" -f $RunId)

Show-Section "Loop Status"
if ($StatusText) {
    Write-Host $StatusText
}
else {
    Write-Host "AUTONOMOUS_LOOP_STATUS.md not found"
}

Show-Section "Tracker Line"
if ($RunLine) {
    Write-Host $RunLine
    Write-Host ""
    Write-Host ("Run status: {0}" -f $RunStatus)
    Write-Host ("Run notes: {0}" -f $RunNotes)
}
else {
    Write-Host ("Run {0} not found in tracker" -f $RunId)
}

Show-Section "Readable Summary"
if ($SummaryText) {
    Write-Host $SummaryText
}
elseif ($ResultsText) {
    Write-Host "PROGRESS_SUMMARY_CN.md not found. Fallback to first 80 lines of EXPERIMENT_RESULTS.md"
    ($ResultsText -split "`r?`n" | Select-Object -First 80) | ForEach-Object { Write-Host $_ }
}
else {
    Write-Host "No summary file found"
}

Show-Section "Review Stage"
if (Test-Path $ReviewDoc) {
    Write-Host "AUTO_REVIEW.md exists"
    Write-Host ("Path: {0}" -f $ReviewDoc)
}
else {
    Write-Host "AUTO_REVIEW.md not found. auto-review-loop has not started yet."
}

if (Test-Path $ReviewState) {
    Write-Host ("REVIEW_STATE.json: {0}" -f $ReviewState)
}

Show-Section "Remote tmux and log"
$Py = @"
import paramiko
HOST='$RemoteHost'
USER='$RemoteUser'
PASSWORD='$RemotePassword'
RUN_ID='$RunId'
TAIL_LINES=$TailLines
session_map = {
    'R011': 'r011_keepbone_v3',
}
log_map = {
    'R011': '/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/bridge_logs/r011_keepbone_v3.log',
}
SESSION = session_map.get(RUN_ID, '')
LOG = log_map.get(RUN_ID, '')
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASSWORD, timeout=20)
cmds = []
if SESSION:
    cmds.append("echo tmux-session:")
    cmds.append(f"tmux ls 2>/dev/null | grep {SESSION} || true")
if LOG:
    cmds.append("echo log-tail:")
    cmds.append(f"tail -n {TAIL_LINES} {LOG} 2>/dev/null || true")
stdin, stdout, stderr = client.exec_command("\\n".join(cmds), timeout=60)
print(stdout.read().decode('utf-8', errors='replace'))
err = stderr.read().decode('utf-8', errors='replace')
if err:
    print(err)
client.close()
"@

try {
    @($Py) | python - 2>&1 | ForEach-Object { Write-Host $_ }
}
catch {
    Write-Host ("Remote query failed: {0}" -f $_.Exception.Message)
}

Show-Section "Suggested order"
Write-Host "1. AUTONOMOUS_LOOP_STATUS.md"
Write-Host "2. EXPERIMENT_TRACKER.md"
Write-Host "3. PROGRESS_SUMMARY_CN.md"
Write-Host "4. remote tmux session and log"
Write-Host "5. if review started, read review-stage/AUTO_REVIEW.md"
