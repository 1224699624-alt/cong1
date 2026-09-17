param(
    [string]$RepoRoot = (Get-Location).Path,
    [int]$IntervalMinutes = 20,
    [int]$MaxPasses = 999,
    [string]$Model = "5.6-sol",
    [string]$PromptFile = "",
    [switch]$DryRun,
    [switch]$UseResume,
    [string]$SessionId = "",
    [string]$WebhookUrl = "",
    [switch]$NotifyOnEveryPass
)

$ErrorActionPreference = "Stop"

if (-not $PromptFile) {
    $PromptFile = Join-Path $RepoRoot "research-workflow\WORKFLOW2_AUTORUN_PROMPT.md"
}

if (-not (Test-Path $PromptFile)) {
    throw "Prompt file not found: $PromptFile"
}

$LogDir = Join-Path $RepoRoot "research-workflow\daemon-logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LoopLog = Join-Path $LogDir "workflow2_loop_$Timestamp.log"
$NotifyScript = Join-Path $RepoRoot "scripts\send_workflow2_notification.ps1"

function Write-LoopLog {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $Line | Tee-Object -FilePath $LoopLog -Append
}

function Send-Workflow2Notification {
    param(
        [string]$Message,
        [string]$Title = "Workflow 2"
    )

    if (-not (Test-Path $NotifyScript)) {
        return
    }

    try {
        if (-not $WebhookUrl -and -not $env:WORKFLOW2_WEBHOOK_URL) {
            Write-Host "[INFO] No webhook configured. Skipping notification."
            return
        }
        & powershell -NoProfile -ExecutionPolicy Bypass -File $NotifyScript `
            -Message $Message `
            -Title $Title `
            -WebhookUrl $WebhookUrl 2>&1 | Tee-Object -FilePath $LoopLog -Append
    }
    catch {
        Write-LoopLog ("Notification failed: " + $_.Exception.Message)
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

function Invoke-CodexResume {
    param(
        [string]$SessionToResume,
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
        $resumeArgs = @('resume', $SessionToResume, $PromptText, '-C', $WorkingRoot, '-a', 'never', '-s', 'danger-full-access')
        if ($SelectedModel -and $SelectedModel -ne 'gpt-5') {
            $resumeArgs += @('-m', $SelectedModel)
        }
        $output = @(& codex @resumeArgs 2>&1)
        $output | Tee-Object -FilePath $LoopLog -Append | Out-Null
        $exitCode = $LASTEXITCODE
        $joined = ($output | Out-String)
        if (($joined -match "model is not supported") -and $SelectedModel) {
            Write-LoopLog ("Model '{0}' not supported in current Codex account. Retrying resume without explicit -m." -f $SelectedModel)
            $retryArgs = @('resume', $SessionToResume, $PromptText, '-C', $WorkingRoot, '-a', 'never', '-s', 'danger-full-access')
            $retryOutput = @(& codex @retryArgs 2>&1)
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

$PromptText = Get-Content -Raw $PromptFile
Write-LoopLog "Workflow 2 loop starting."
Write-LoopLog "RepoRoot=$RepoRoot"
Write-LoopLog "PromptFile=$PromptFile"
Write-LoopLog ("Mode=" + $(if ($UseResume) { "resume" } else { "exec" }))
Write-LoopLog "DryRun=$DryRun"
Write-LoopLog "NotifyOnEveryPass=$NotifyOnEveryPass"
if ($WebhookUrl -or $env:WORKFLOW2_WEBHOOK_URL) {
    Send-Workflow2Notification -Title "Workflow 2 Started" -Message "Autorun loop started at $RepoRoot. Log: $LoopLog"
}

for ($Pass = 1; $Pass -le $MaxPasses; $Pass++) {
    Write-LoopLog "Pass $Pass started."

    try {
        if ($DryRun) {
            Write-LoopLog "DryRun enabled. Skipping Codex invocation for this pass."
            $ExitCode = 0
        }
        else {
            if ($UseResume) {
                if (-not $SessionId) {
                    throw "UseResume was specified but SessionId is empty."
                }

                $ExitCode = Invoke-CodexResume -SessionToResume $SessionId -PromptText $PromptText -WorkingRoot $RepoRoot -SelectedModel $Model
            }
            else {
                $ExitCode = Invoke-CodexExec -PromptText $PromptText -WorkingRoot $RepoRoot -SelectedModel $Model
            }
        }

        Write-LoopLog "Pass $Pass exited with code $ExitCode."
        if ($NotifyOnEveryPass -and ($WebhookUrl -or $env:WORKFLOW2_WEBHOOK_URL)) {
            Send-Workflow2Notification -Title "Workflow 2 Pass $Pass" -Message "Pass $Pass exited with code $ExitCode. Log: $LoopLog"
        }
    }
    catch {
        Write-LoopLog ("Pass $Pass threw exception: " + $_.Exception.Message)
        if ($WebhookUrl -or $env:WORKFLOW2_WEBHOOK_URL) {
            Send-Workflow2Notification -Title "Workflow 2 Exception" -Message "Pass $Pass threw exception: $($_.Exception.Message)`nLog: $LoopLog"
        }
    }

    if ($Pass -lt $MaxPasses) {
        Write-LoopLog "Sleeping for $IntervalMinutes minutes before next pass."
        Start-Sleep -Seconds ($IntervalMinutes * 60)
    }
}

Write-LoopLog "Workflow 2 loop finished."
if ($WebhookUrl -or $env:WORKFLOW2_WEBHOOK_URL) {
    Send-Workflow2Notification -Title "Workflow 2 Finished" -Message "Autorun loop finished. Log: $LoopLog"
}
