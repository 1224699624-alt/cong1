$ErrorActionPreference = "Stop"
$Root = "G:\gutou\YOLO+SAM"
$Python = "E:\360Downloads\anaconda3\envs\unet\python.exe"
$Output = Join-Path $Root "outputs\tsrs\r289_pair_gap_head_dev"
$LogDir = Join-Path $Root "outputs\bridge_logs"
$Checkpoint = Join-Path $Root "outputs\nnunet\r275_mature_control\nnUNet_results\Dataset202_TSRS_RSNAEpiphysis2D\nnUNetTrainerR275ControlMature__nnUNetPlans__2d\fold_0\checkpoint_best.pth"
New-Item -ItemType Directory -Force $Output, $LogDir | Out-Null
$Stdout = Join-Path $LogDir "r289_tsrs_pair_gap_head_dev.stdout.log"
$Stderr = Join-Path $LogDir "r289_tsrs_pair_gap_head_dev.stderr.log"
$Arguments = @(
    "scripts\train_r289_tsrs_pair_gap_head.py",
    "--dataset-root", "data\raw\TSRS_RSNA-Epiphysis",
    "--checkpoint", $Checkpoint,
    "--output", "outputs\tsrs\r289_pair_gap_head_dev",
    "--pair-count", "20", "--crop-size", "512", "--gap-radius", "8",
    "--epochs", "4", "--samples-per-epoch", "800",
    "--lr", "0.0002", "--seed", "289"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
@{
    experiment = "R289"
    pid = $Process.Id
    stdout = $Stdout
    stderr = $Stderr
    output = $Output
    clean_test_v2_locked = $true
    mask_editing = $false
    started = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Output "local_process.json")
Write-Output "R289 PID=$($Process.Id)"
