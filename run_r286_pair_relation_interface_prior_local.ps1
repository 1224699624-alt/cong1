$ErrorActionPreference = "Stop"
$Root = "G:\gutou\YOLO+SAM"
$Python = "E:\360Downloads\anaconda3\envs\unet\python.exe"
$Output = Join-Path $Root "outputs\ram_w600\r286_pair_relation_interface_prior"
$LogDir = Join-Path $Root "outputs\bridge_logs"
New-Item -ItemType Directory -Force $Output, $LogDir | Out-Null
$Stdout = Join-Path $LogDir "r286_pair_relation_interface_prior_local.stdout.log"
$Stderr = Join-Path $LogDir "r286_pair_relation_interface_prior_local.stderr.log"
$Arguments = @(
    "scripts\train_r286_pair_relation_interface_prior.py",
    "--dataset-root", "G:\gutou\RAM-W600",
    "--baseline-checkpoint", "outputs\ram_w600\r285_nnunet_overlap_prior\baseline_best.pth",
    "--output", "outputs\ram_w600\r286_pair_relation_interface_prior",
    "--size", "384",
    "--batch-size", "1",
    "--pair-count", "15",
    "--interface-radius", "7",
    "--head-epochs", "8",
    "--finetune-epochs", "20",
    "--seed", "286"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
@{
    experiment = "R286"
    pid = $Process.Id
    stdout = $Stdout
    stderr = $Stderr
    output = $Output
    architecture = "pair-specific gap-overlap-uncertain local-interface nnUNet"
    started = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Output "local_process.json")
Write-Output "R286 PID=$($Process.Id)"
