param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$TaskName = "YOLO_SAM_Workflow2_Autorun",
    [int]$IntervalMinutes = 20,
    [string]$Model = "5.6-sol",
    [int]$MaxPasses = 999999,
    [switch]$DryRun,
    [switch]$UseResume,
    [string]$SessionId = "",
    [string]$WebhookUrl = "",
    [switch]$NotifyOnEveryPass
)

$ErrorActionPreference = "Stop"

$LoopScript = Join-Path $RepoRoot "scripts\run_workflow2_loop.ps1"
if (-not (Test-Path $LoopScript)) {
    throw "Loop script not found: $LoopScript"
}

if ($UseResume -and -not $SessionId) {
    throw "UseResume was specified but SessionId is empty."
}

$ResumeArgs = ""
if ($UseResume) {
    $ResumeArgs = " -UseResume -SessionId `"$SessionId`""
}

$DryRunArgs = ""
if ($DryRun) {
    $DryRunArgs = " -DryRun"
}

$NotifyArgs = ""
if ($NotifyOnEveryPass) {
    $NotifyArgs += " -NotifyOnEveryPass"
}
if ($WebhookUrl) {
    $NotifyArgs += " -WebhookUrl `"$WebhookUrl`""
}

$ArgString = @(
    "-NoProfile"
    "-ExecutionPolicy Bypass"
    "-File `"$LoopScript`""
    "-RepoRoot `"$RepoRoot`""
    "-IntervalMinutes $IntervalMinutes"
    "-MaxPasses $MaxPasses"
    "-Model `"$Model`""
    $ResumeArgs
    $DryRunArgs
    $NotifyArgs
) -join " "

$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $ArgString
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Days 3650)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Force | Out-Null

Write-Host "Installed scheduled task: $TaskName"
Write-Host "RepoRoot: $RepoRoot"
Write-Host "LoopScript: $LoopScript"
Write-Host "Start command:"
Write-Host "powershell.exe $ArgString"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Start-ScheduledTask -TaskName `"$TaskName`""
Write-Host "  Get-ScheduledTask -TaskName `"$TaskName`" | Get-ScheduledTaskInfo"
Write-Host "  Unregister-ScheduledTask -TaskName `"$TaskName`" -Confirm:`$false"
