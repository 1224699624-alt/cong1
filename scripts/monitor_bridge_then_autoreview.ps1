param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$Model = "5.6-sol",
    [string]$TriggerPromptFile = "",
    [string]$StatusFile = "",
    [string]$TrackerFile = "",
    [string]$ResultsFile = "",
    [string]$RunId = "R011",
    [string]$RemoteHost = "10.1.115.157",
    [string]$RemoteUser = "shenzeyu",
    [string]$RemotePassword = "40904090TWO",
    [int]$PollSeconds = 300,
    [int]$MaxPolls = 288,
    [string]$WebhookUrl = "",
    [string]$SyncScript = ""
)

$ErrorActionPreference = "Stop"

if (-not $TriggerPromptFile) {
    $TriggerPromptFile = Join-Path $RepoRoot "research-workflow\AUTO_REVIEW_TRIGGER_PROMPT.md"
}
if (-not $StatusFile) {
    $StatusFile = Join-Path $RepoRoot "research-workflow\refine-logs\AUTONOMOUS_LOOP_STATUS.md"
}
if (-not $TrackerFile) {
    $TrackerFile = Join-Path $RepoRoot "research-workflow\refine-logs\EXPERIMENT_TRACKER.md"
}
if (-not $ResultsFile) {
    $ResultsFile = Join-Path $RepoRoot "research-workflow\refine-logs\EXPERIMENT_RESULTS.md"
}
if (-not $SyncScript) {
    $SyncScript = Join-Path $RepoRoot "scripts\sync_aris_bridge_state.py"
}

$LoopLogDir = Join-Path $RepoRoot "research-workflow\daemon-logs"
New-Item -ItemType Directory -Force -Path $LoopLogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LoopLog = Join-Path $LoopLogDir "bridge_then_autoreview_$Timestamp.log"
$NotifyScript = Join-Path $RepoRoot "scripts\send_workflow2_notification.ps1"

function Write-LoopLog {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $Line | Tee-Object -FilePath $LoopLog -Append
}

function Send-Notification {
    param([string]$Title, [string]$Message)
    if (-not (Test-Path $NotifyScript)) {
        return
    }
    try {
        if (-not $WebhookUrl -and -not $env:WORKFLOW2_WEBHOOK_URL) {
            Write-Host "[INFO] No webhook configured. Skipping notification."
            return
        }
        $Args = @(
            '-NoProfile'
            '-ExecutionPolicy'
            'Bypass'
            '-File'
            $NotifyScript
            '-Title'
            $Title
            '-Message'
            $Message
        )
        if ($WebhookUrl) {
            $Args += @('-WebhookUrl', $WebhookUrl)
        }
        & powershell @Args 2>&1 | Tee-Object -FilePath $LoopLog -Append
    }
    catch {
        Write-LoopLog ("Notification failed: " + $_.Exception.Message)
    }
}

function Get-RunPidFromTracker {
    param([string]$Path, [string]$TargetRunId)
    if (-not (Test-Path $Path)) {
        return ""
    }
    $text = Get-Content -Raw $Path
    foreach ($line in ($text -split "`r?`n")) {
        if ($line -like "| $TargetRunId *") {
            if ($line -match 'pid `(?<pid>\d+)`') {
                return $Matches["pid"]
            }
        }
    }
    return ""
}

function Test-RunComplete {
    param([string]$Path, [string]$TargetRunId)
    if (-not (Test-Path $Path)) {
        return $false
    }
    $text = Get-Content -Raw $Path
    foreach ($line in ($text -split "`r?`n")) {
        if ($line -like "| $TargetRunId *") {
            $parts = $line.Split('|') | ForEach-Object { $_.Trim() }
            if ($parts.Count -ge 9) {
                return ($parts[8] -eq 'DONE')
            }
            return $false
        }
    }
    return $false
}

function Test-ReviewStageStarted {
    param([string]$RepoRootPath)
    $reviewDoc = Join-Path $RepoRootPath "review-stage\AUTO_REVIEW.md"
    return (Test-Path $reviewDoc)
}

function Sync-ArisState {
    if (-not (Test-Path $SyncScript)) {
        Write-LoopLog "Sync script not found. Skipping local md sync."
        return
    }
    try {
        & python $SyncScript `
            --repo-root $RepoRoot `
            --remote-host $RemoteHost `
            --remote-user $RemoteUser `
            --remote-password $RemotePassword 2>&1 | Tee-Object -FilePath $LoopLog -Append
    }
    catch {
        Write-LoopLog ("State sync failed: " + $_.Exception.Message)
    }
}

function Invoke-CodexExec {
    param(
        [string]$PromptText,
        [string]$WorkingRoot,
        [string]$SelectedModel
    )
    $oldPref = $ErrorActionPreference
    $hasNativePref = Test-Path Variable:PSNativeCommandUseErrorActionPreference
    if ($hasNativePref) {
        $oldNativePref = $PSNativeCommandUseErrorActionPreference
        $script:PSNativeCommandUseErrorActionPreference = $false
    }
    $ErrorActionPreference = "Continue"
    try {
        $execArgs = @('exec', '-', '-C', $WorkingRoot, '-s', 'danger-full-access', '--dangerously-bypass-approvals-and-sandbox')
        if ($SelectedModel -and $SelectedModel -ne 'gpt-5') {
            $execArgs += @('-m', $SelectedModel)
        }
        $output = @($PromptText | codex @execArgs 2>&1)
        $output | Tee-Object -FilePath $LoopLog -Append | Out-Null
        $exitCode = $LASTEXITCODE
        $joined = ($output | Out-String)
        if (($joined -match "model is not supported") -and $SelectedModel) {
            Write-LoopLog ("Model '{0}' not supported in current Codex account. Retrying without explicit -m." -f $SelectedModel)
            $retryArgs = @('exec', '-', '-C', $WorkingRoot, '-s', 'danger-full-access', '--dangerously-bypass-approvals-and-sandbox')
            $retryOutput = @($PromptText | codex @retryArgs 2>&1)
            $retryOutput | Tee-Object -FilePath $LoopLog -Append | Out-Null
            return $LASTEXITCODE
        }
        return $exitCode
    }
    finally {
        $ErrorActionPreference = $oldPref
        if ($hasNativePref) {
            $script:PSNativeCommandUseErrorActionPreference = $oldNativePref
        }
    }
}

if (-not (Test-Path $TriggerPromptFile)) {
    throw "Trigger prompt file not found: $TriggerPromptFile"
}

$TriggerPrompt = Get-Content -Raw $TriggerPromptFile

Write-LoopLog "Bridge-to-auto-review monitor starting."
Write-LoopLog "RunId=$RunId"
Write-LoopLog "TrackerFile=$TrackerFile"
Write-LoopLog "ResultsFile=$ResultsFile"
Write-LoopLog "SyncScript=$SyncScript"
Send-Notification -Title "Bridge Monitor Started" -Message "Monitoring $RunId for bridge completion. Log: $LoopLog"

for ($Poll = 1; $Poll -le $MaxPolls; $Poll++) {
    Write-LoopLog "Poll $Poll started."
    Sync-ArisState

    if (Test-ReviewStageStarted -RepoRootPath $RepoRoot) {
        Write-LoopLog "review-stage already exists. Assuming Workflow 2 has started previously. Exiting monitor."
        exit 0
    }

    if (Test-RunComplete -Path $TrackerFile -TargetRunId $RunId) {
        Write-LoopLog "$RunId is marked DONE in tracker. Triggering upstream auto-review flow."
        Send-Notification -Title "Bridge Complete" -Message "$RunId is DONE. Triggering upstream auto-review-loop."
        Sync-ArisState

        $ExitCode = Invoke-CodexExec -PromptText $TriggerPrompt -WorkingRoot $RepoRoot -SelectedModel $Model
        Write-LoopLog "Auto-review trigger exited with code $ExitCode."
        Sync-ArisState
        if ($ExitCode -eq 0) {
            Send-Notification -Title "Auto Review Triggered" -Message "Upstream Workflow 2 trigger for $RunId finished with code 0."
            exit 0
        }
        else {
            Send-Notification -Title "Auto Review Trigger Failed" -Message "Upstream Workflow 2 trigger for $RunId failed with code $ExitCode. Log: $LoopLog"
            exit $ExitCode
        }
    }

    $TrackedPid = Get-RunPidFromTracker -Path $TrackerFile -TargetRunId $RunId
    if ($TrackedPid) {
        Write-LoopLog ("Current tracked pid for {0}: {1}" -f $RunId, $TrackedPid)
        $Py = @"
import paramiko
HOST='$RemoteHost'; USER='$RemoteUser'; PASSWORD='$RemotePassword'; PID='$TrackedPid'
client=paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy()); client.connect(HOST, username=USER, password=PASSWORD, timeout=20)
stdin, stdout, stderr = client.exec_command(f"ps -p {PID} -o pid=,etimes=,cmd= || true", timeout=30)
print(stdout.read().decode('utf-8', errors='replace'))
err = stderr.read().decode('utf-8', errors='replace')
if err:
    print(err)
client.close()
"@
        @($Py) | python - 2>&1 | Tee-Object -FilePath $LoopLog -Append | Out-Null
    }
    else {
        Write-LoopLog "No pid parsed for $RunId from tracker yet."
    }

    if ($Poll -lt $MaxPolls) {
        Write-LoopLog "Sleeping for $PollSeconds seconds before next poll."
        Start-Sleep -Seconds $PollSeconds
    }
}

Write-LoopLog "MaxPolls reached without bridge completion."
Send-Notification -Title "Bridge Monitor Timeout" -Message "Monitoring for $RunId reached MaxPolls without completion. Log: $LoopLog"
exit 1
