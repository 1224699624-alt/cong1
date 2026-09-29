$ErrorActionPreference = "Stop"
$Root = "G:\gutou\YOLO+SAM"
$Python = "E:\360Downloads\anaconda3\envs\unet\python.exe"
$Output = Join-Path $Root "outputs\ram_w600\r288_directional_pair_gate_dev"
$LogDir = Join-Path $Root "outputs\bridge_logs"
New-Item -ItemType Directory -Force $Output, $LogDir | Out-Null
$Stdout = Join-Path $LogDir "r288_directional_pair_gate_dev.stdout.log"
$Stderr = Join-Path $LogDir "r288_directional_pair_gate_dev.stderr.log"
$Arguments = @(
    "scripts\train_r286_pair_relation_interface_prior.py",
    "--dataset-root", "G:\gutou\RAM-W600",
    "--baseline-checkpoint", "outputs\ram_w600\r285_nnunet_overlap_prior\baseline_best.pth",
    "--output", "outputs\ram_w600\r288_directional_pair_gate_dev",
    "--experiment-name", "R288", "--directional-confidence-gate",
    "--size", "384", "--batch-size", "1", "--pair-count", "15",
    "--interface-radius", "7", "--head-epochs", "12", "--finetune-epochs", "0",
    "--boundary-weight", "0.01", "--selection-metric", "overlap_nsd_2px",
    "--skip-test", "--seed", "288"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
@{
    experiment = "R288"
    pid = $Process.Id
    stdout = $Stdout
    stderr = $Stderr
    output = $Output
    official_test_locked = $true
    started = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Output "local_process.json")
Write-Output "R288 PID=$($Process.Id)"
