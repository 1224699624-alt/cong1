param([switch]$ResumeAfterPreprocess)

$ErrorActionPreference = "Stop"

$Root = "G:\gutou\YOLO+SAM"
$Python = "E:\360Downloads\anaconda3\envs\unet\python.exe"
$Bin = "E:\360Downloads\anaconda3\envs\unet\Scripts"
$Run = Join-Path $Root "outputs\nnunet\r308_clean_panels_r260"
$Data = Join-Path $Run "data"
$Log = Join-Path $Root "outputs\bridge_logs\r308_clean_panels_r260_local.log"
$Bundle = Join-Path $Root "outputs\artifact_bundles\r260_recovered_from_49284_20260724"
$CheckpointDir = Join-Path $Bundle "outputs\pair_prior\r258b_prediction_relation_prior_full_20260715_133745"
$SourceCheckpoint = Join-Path $Bundle "outputs\nnunet\r260_mature_prior\nnUNet_results\Dataset204_TSRS_RSNAEpiphysisMaturePrior2D\nnUNetTrainerR260MaturePrior__nnUNetPlans__2d\fold_0\checkpoint_best.pth"
$SourcePlan = Join-Path $Bundle "outputs\nnunet\r260_mature_prior\data\nnUNet_preprocessed\Dataset204_TSRS_RSNAEpiphysisMaturePrior2D\nnUNetPlans.json"
$FrozenMasks = Join-Path $Bundle "outputs\ablations\r260_mature_prior\TSRS_RSNA-Epiphysis\val\masks"

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path (Split-Path $Log), $Run | Out-Null
$Env:Path = "$Bin;$Env:Path"
$Env:CUDA_VISIBLE_DEVICES = "0"
$Env:PYTHONHASHSEED = "260"
$Env:nnUNet_compile = "false"
$Env:nnUNet_n_proc_DA = "0"
$Env:nnUNet_raw = Join-Path $Data "nnUNet_raw"
$Env:nnUNet_preprocessed = Join-Path $Data "nnUNet_preprocessed"
$Env:nnUNet_results = Join-Path $Run "nnUNet_results"
$Env:R306_SOURCE_CHECKPOINT = $SourceCheckpoint
$Env:R306_EXPECTED_SOURCE_SHA256 = "7d1d8d2916115c2e29696b2a76881da2a7a78f3db91a5f6a9491297369052a6e"

function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Command $($Arguments -join ' ')"
    }
}

Write-Output "[R308] start $([DateTimeOffset]::Now.ToString('o'))"
& nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

$Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceCheckpoint).Hash.ToLowerInvariant()
if ($Hash -ne $Env:R306_EXPECTED_SOURCE_SHA256) { throw "R260 checkpoint hash mismatch: $Hash" }

if (-not $ResumeAfterPreprocess) {
    $PriorRoot = Join-Path $Root "outputs\priors\r259_frozen_relation"
    $PriorCount = (Get-ChildItem -Recurse -File -Filter "*.npy" -Path $PriorRoot -ErrorAction SilentlyContinue).Count
    if ($PriorCount -ne 971) {
        Write-Output "[R308] rebuilding frozen R259 prior maps (found $PriorCount/971)"
        Invoke-Checked $Python @(
            "scripts/build_r259_frozen_prior_maps.py",
            "--checkpoint-dir", $CheckpointDir,
            "--output-root", $PriorRoot,
            "--batch-size", "32",
            "--device", "cuda"
        )
    }

    Invoke-Checked $Python @(
        "scripts/prepare_r308_clean_panel_r260.py",
        "--selection", "configs/r308_tsrs_clean_panel_selection.json",
        "--prior-root", $PriorRoot,
        "--frozen-mask-root", $FrozenMasks,
        "--source-plan", $SourcePlan,
        "--output", $Data,
        "--overwrite"
    )

    Invoke-Checked (Join-Path $Bin "nnUNetv2_extract_fingerprint.exe") @(
        "-d", "309", "-np", "4", "--verify_dataset_integrity", "--clean"
    )
    Invoke-Checked (Join-Path $Bin "nnUNetv2_preprocess.exe") @(
        "-d", "309", "-plans_name", "nnUNetPlans", "-c", "2d", "-np", "4"
    )
    $SplitSource = Join-Path $Env:nnUNet_raw "Dataset309_TSRSR260CleanPanels2D\splits_final.json"
    $SplitTarget = Join-Path $Env:nnUNet_preprocessed "Dataset309_TSRSR260CleanPanels2D\splits_final.json"
    Copy-Item -Force $SplitSource $SplitTarget
} else {
    Write-Output "[R308] resume from completed preprocessing with single-threaded augmentation"
}

$TrainerDir = & $Python -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss')"
Copy-Item -Force "scripts\nnunet_trainers\nnUNetTrainerR306R260InstanceAux.py" $TrainerDir
Invoke-Checked $Python @("-m", "py_compile", "scripts/nnunet_trainers/nnUNetTrainerR306R260InstanceAux.py")

Invoke-Checked (Join-Path $Bin "nnUNetv2_train.exe") @(
    "309", "2d", "0", "-tr", "nnUNetTrainerR308CleanContinue"
)
Invoke-Checked (Join-Path $Bin "nnUNetv2_predict.exe") @(
    "-i", (Join-Path $Run "imagesValPrior"),
    "-o", (Join-Path $Run "pred_clean"),
    "-d", "309", "-c", "2d", "-f", "0",
    "-tr", "nnUNetTrainerR308CleanContinue", "-chk", "checkpoint_best.pth"
)
Invoke-Checked $Python @(
    "scripts/convert_nnunet_predictions.py",
    "--pred-dir", (Join-Path $Run "pred_clean"),
    "--mask-dir", (Join-Path $Run "masks_clean"),
    "--expected-count", "94", "--strip-prefix", "val_"
)
Invoke-Checked $Python @(
    "scripts/evaluate_r201_single_split.py",
    "--gt-dir", (Join-Path $Run "cleanValGT"),
    "--pred-dir", (Join-Path $Run "frozenR260Masks"),
    "--output", "outputs/analysis/r308_frozen_r260_clean_val94_r201.json",
    "--expected-count", "94"
)
Invoke-Checked $Python @(
    "scripts/evaluate_r201_single_split.py",
    "--gt-dir", (Join-Path $Run "cleanValGT"),
    "--pred-dir", (Join-Path $Run "masks_clean"),
    "--output", "outputs/analysis/r308_clean_finetune_clean_val94_r201.json",
    "--expected-count", "94"
)

$Vis = Join-Path $Root "outputs\visualizations\r308_frozen_vs_clean_finetune_val94"
Invoke-Checked $Python @(
    "scripts/render_r259_nnunet_prior_comparison.py",
    "--image-dir", "data/raw/TSRS_RSNA-Epiphysis/val",
    "--gt-dir", (Join-Path $Run "cleanValGT"),
    "--baseline-dir", (Join-Path $Run "frozenR260Masks"),
    "--improved-dir", (Join-Path $Run "masks_clean"),
    "--selection-csv", "outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv",
    "--output-dir", $Vis, "--num-cases", "12",
    "--xray-label", "Original X-ray",
    "--baseline-label", "Frozen R260 best",
    "--improved-label", "R260 + clean-panel fine-tune"
)
Write-Output "[R308] complete $([DateTimeOffset]::Now.ToString('o'))"
