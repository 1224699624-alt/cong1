$ErrorActionPreference = "Stop"
$Root = "G:\gutou\YOLO+SAM"
$Python = "E:\360Downloads\anaconda3\envs\YOLOSAM\python.exe"
$Output = Join-Path $Root "outputs\ram_w600\r284_overlap_prior"
$LogDir = Join-Path $Root "outputs\bridge_logs"
New-Item -ItemType Directory -Force $Output, $LogDir | Out-Null
$Stdout = Join-Path $LogDir "r284_ramw600_overlap_prior_local.stdout.log"
$Stderr = Join-Path $LogDir "r284_ramw600_overlap_prior_local.stderr.log"
$Arguments = @(
    "scripts\train_r284_ramw600_overlap_prior.py",
    "--dataset-root", "G:\gutou\RAM-W600",
    "--output", "outputs\ram_w600\r284_overlap_prior",
    "--size", "384",
    "--batch-size", "2",
    "--baseline-epochs", "30",
    "--improved-epochs", "20"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
@{
    experiment = "R284"
    pid = $Process.Id
    stdout = $Stdout
    stderr = $Stderr
    output = $Output
    started = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Output "local_process.json")
Write-Output "R284 PID=$($Process.Id)"
