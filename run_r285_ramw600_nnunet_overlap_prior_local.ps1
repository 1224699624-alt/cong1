$ErrorActionPreference = "Stop"
$Root = "G:\gutou\YOLO+SAM"
$Python = "E:\360Downloads\anaconda3\envs\unet\python.exe"
$Output = Join-Path $Root "outputs\ram_w600\r285_nnunet_overlap_prior"
$LogDir = Join-Path $Root "outputs\bridge_logs"
New-Item -ItemType Directory -Force $Output, $LogDir | Out-Null
$Stdout = Join-Path $LogDir "r285_ramw600_nnunet_overlap_prior_local.stdout.log"
$Stderr = Join-Path $LogDir "r285_ramw600_nnunet_overlap_prior_local.stderr.log"
$Arguments = @(
    "scripts\train_r284_ramw600_overlap_prior.py",
    "--dataset-root", "G:\gutou\RAM-W600",
    "--output", "outputs\ram_w600\r285_nnunet_overlap_prior",
    "--architecture", "nnunet",
    "--size", "384",
    "--batch-size", "1",
    "--baseline-epochs", "30",
    "--improved-epochs", "20",
    "--seed", "285"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
@{
    experiment = "R285"
    pid = $Process.Id
    stdout = $Stdout
    stderr = $Stderr
    output = $Output
    architecture = "nnUNet v2 PlainConvUNet 2D multilabel"
    started = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Output "local_process.json")
Write-Output "R285 PID=$($Process.Id)"
