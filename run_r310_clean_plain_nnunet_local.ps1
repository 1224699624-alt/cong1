$ErrorActionPreference = "Stop"

$Root = "G:\gutou\YOLO+SAM"
$Python = "E:\360Downloads\anaconda3\envs\unet\python.exe"
$Bin = "E:\360Downloads\anaconda3\envs\unet\Scripts"
$Run = Join-Path $Root "outputs\nnunet\r310_clean_plain_nnunet"
$Data = Join-Path $Run "data"
$Bundle = Join-Path $Root "outputs\artifact_bundles\r260_recovered_from_49284_20260724"
$SourceCheckpoint = Join-Path $Bundle "outputs\nnunet\r202_mature_import\checkpoint_best.pth"
$SourcePlan = Join-Path $Bundle "outputs\nnunet\r260_mature_prior\data\nnUNet_preprocessed\Dataset204_TSRS_RSNAEpiphysisMaturePrior2D\nnUNetPlans.json"

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $Run | Out-Null
$Env:Path = "$Bin;$Env:Path"
$Env:CUDA_VISIBLE_DEVICES = "0"
$Env:PYTHONHASHSEED = "260"
$Env:nnUNet_compile = "false"
$Env:nnUNet_n_proc_DA = "0"
$Env:nnUNet_raw = Join-Path $Data "nnUNet_raw"
$Env:nnUNet_preprocessed = Join-Path $Data "nnUNet_preprocessed"
$Env:nnUNet_results = Join-Path $Run "nnUNet_results"
$Env:R310_SOURCE_CHECKPOINT = $SourceCheckpoint
$Env:R310_EXPECTED_SOURCE_SHA256 = "65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7"

function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Command $($Arguments -join ' ')" }
}

Write-Output "[R310] start $([DateTimeOffset]::Now.ToString('o'))"
& nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
$Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceCheckpoint).Hash.ToLowerInvariant()
if ($Hash -ne $Env:R310_EXPECTED_SOURCE_SHA256) { throw "R202 checkpoint hash mismatch: $Hash" }

Invoke-Checked $Python @(
    "scripts/prepare_r310_clean_plain_nnunet.py",
    "--selection", "configs/r308_tsrs_clean_panel_selection.json",
    "--source-plan", $SourcePlan,
    "--output", $Data, "--overwrite"
)
Invoke-Checked (Join-Path $Bin "nnUNetv2_extract_fingerprint.exe") @(
    "-d", "310", "-np", "4", "--verify_dataset_integrity", "--clean"
)
Invoke-Checked (Join-Path $Bin "nnUNetv2_preprocess.exe") @(
    "-d", "310", "-plans_name", "nnUNetPlans", "-c", "2d", "-np", "4"
)
$SplitSource = Join-Path $Env:nnUNet_raw "Dataset310_TSRSPlainCleanPanels2D\splits_final.json"
$SplitTarget = Join-Path $Env:nnUNet_preprocessed "Dataset310_TSRSPlainCleanPanels2D\splits_final.json"
Copy-Item -Force $SplitSource $SplitTarget

$TrainerDir = & $Python -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss')"
Copy-Item -Force "scripts\nnunet_trainers\nnUNetTrainerR310CleanPlain.py" $TrainerDir
Invoke-Checked $Python @("-m", "py_compile", "scripts/nnunet_trainers/nnUNetTrainerR310CleanPlain.py")

$Model = Join-Path $Env:nnUNet_results "Dataset310_TSRSPlainCleanPanels2D\nnUNetTrainerR310CleanPlain__nnUNetPlans__2d"
$Fold = Join-Path $Model "fold_0"
New-Item -ItemType Directory -Force -Path $Fold | Out-Null
Copy-Item -Force $SourceCheckpoint (Join-Path $Fold "checkpoint_r202_frozen.pth")
Copy-Item -Force (Join-Path $Env:nnUNet_preprocessed "Dataset310_TSRSPlainCleanPanels2D\nnUNetPlans.json") (Join-Path $Model "plans.json")
Copy-Item -Force (Join-Path $Env:nnUNet_raw "Dataset310_TSRSPlainCleanPanels2D\dataset.json") (Join-Path $Model "dataset.json")

Invoke-Checked (Join-Path $Bin "nnUNetv2_predict.exe") @(
    "-i", (Join-Path $Run "imagesVal"), "-o", (Join-Path $Run "pred_frozen"),
    "-d", "310", "-c", "2d", "-f", "0", "-tr", "nnUNetTrainerR310CleanPlain",
    "-chk", "checkpoint_r202_frozen.pth"
)
Invoke-Checked $Python @(
    "scripts/convert_nnunet_predictions.py", "--pred-dir", (Join-Path $Run "pred_frozen"),
    "--mask-dir", (Join-Path $Run "masks_frozen"), "--expected-count", "94", "--strip-prefix", "val_"
)
Invoke-Checked (Join-Path $Bin "nnUNetv2_train.exe") @("310", "2d", "0", "-tr", "nnUNetTrainerR310CleanPlain")
Invoke-Checked (Join-Path $Bin "nnUNetv2_predict.exe") @(
    "-i", (Join-Path $Run "imagesVal"), "-o", (Join-Path $Run "pred_clean"),
    "-d", "310", "-c", "2d", "-f", "0", "-tr", "nnUNetTrainerR310CleanPlain",
    "-chk", "checkpoint_best.pth"
)
Invoke-Checked $Python @(
    "scripts/convert_nnunet_predictions.py", "--pred-dir", (Join-Path $Run "pred_clean"),
    "--mask-dir", (Join-Path $Run "masks_clean"), "--expected-count", "94", "--strip-prefix", "val_"
)
Invoke-Checked $Python @(
    "scripts/evaluate_r201_single_split.py", "--gt-dir", (Join-Path $Run "cleanValGT"),
    "--pred-dir", (Join-Path $Run "masks_frozen"),
    "--output", "outputs/analysis/r310_frozen_plain_nnunet_clean_val94_r201.json", "--expected-count", "94"
)
Invoke-Checked $Python @(
    "scripts/evaluate_r201_single_split.py", "--gt-dir", (Join-Path $Run "cleanValGT"),
    "--pred-dir", (Join-Path $Run "masks_clean"),
    "--output", "outputs/analysis/r310_clean_finetune_plain_nnunet_val94_r201.json", "--expected-count", "94"
)
$Vis = Join-Path $Root "outputs\visualizations\r310_plain_nnunet_frozen_vs_clean_val94"
Invoke-Checked $Python @(
    "scripts/render_r259_nnunet_prior_comparison.py",
    "--image-dir", "data/raw/TSRS_RSNA-Epiphysis/val", "--gt-dir", (Join-Path $Run "cleanValGT"),
    "--baseline-dir", (Join-Path $Run "masks_frozen"), "--improved-dir", (Join-Path $Run "masks_clean"),
    "--selection-csv", "outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv",
    "--output-dir", $Vis, "--num-cases", "12", "--xray-label", "Original X-ray",
    "--baseline-label", "Frozen standard nnU-Net", "--improved-label", "Standard nnU-Net + clean fine-tune"
)
Write-Output "[R310] complete $([DateTimeOffset]::Now.ToString('o'))"
