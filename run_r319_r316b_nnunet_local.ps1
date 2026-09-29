param([switch]$ResumeAtPreprocess)

$ErrorActionPreference = "Stop"
$Root = "G:\gutou\YOLO+SAM"
$Python = "E:\360Downloads\anaconda3\envs\unet\python.exe"
$Bin = "E:\360Downloads\anaconda3\envs\unet\Scripts"
$Run = Join-Path $Root "outputs\nnunet\r319_r316b_gated_prior"
$Data = Join-Path $Run "data"
$PriorRoot = Join-Path $Root "outputs\priors\r319_r316b_gated_relation"
$Bundle = Join-Path $Root "outputs\artifact_bundles\r260_recovered_from_49284_20260724"
$R202 = Join-Path $Bundle "outputs\nnunet\r202_mature_import\checkpoint_best.pth"
$SourcePlan = Join-Path $Bundle "outputs\nnunet\r260_mature_prior\data\nnUNet_preprocessed\Dataset204_TSRS_RSNAEpiphysisMaturePrior2D\nnUNetPlans.json"
$R316B = Join-Path $Root "outputs\artifact_bundles\r316b_full_quantile_remote_43758_20260727\safe_relation_seed3161_final.pt"
$Adapted = Join-Path $Run "r202_two_channel_zero_prior.pth"
$Dataset = "Dataset204_TSRS_RSNAEpiphysisMaturePrior2D"
$R260Masks = Join-Path $Bundle "outputs\ablations\r260_mature_prior\TSRS_RSNA-Epiphysis\val\masks"
Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $Run,outputs\bridge_logs | Out-Null
$env:Path = "$Bin;$env:Path"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONHASHSEED = "260"
$env:nnUNet_compile = "false"
$env:nnUNet_n_proc_DA = "0"
$env:nnUNet_raw = Join-Path $Data "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $Data "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $Run "nnUNet_results"
$env:R260_ADAPTED_CHECKPOINT = $Adapted
$env:R315_EXPECTED_DATASET = $Dataset

function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Command $($Arguments -join ' ')" }
}

if (-not $ResumeAtPreprocess) {
    # Build inference-safe full-data priors. No GT is read by this step.
    Invoke-Checked $Python @(
        "scripts/build_r316b_gated_prior_maps.py",
        "--checkpoint", $R316B,
        "--output-root", $PriorRoot,
        "--threshold", "0.65",
        "--device", "cuda"
    )

    # Build the exact two-channel R260 dataset, allowing a zero prior for images
    # where the conservative gate intentionally selects no candidate pair.
    Invoke-Checked $Python @(
        "scripts/prepare_r260_mature_prior_dataset.py",
        "--prior-root", $PriorRoot,
        "--nnunet-root", $Data,
        "--supplemental-label-root", "data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1",
        "--allow-zero-prior",
        "--overwrite"
    )
    Invoke-Checked (Join-Path $Bin "nnUNetv2_extract_fingerprint.exe") @("-d", "204", "-np", "4", "--verify_dataset_integrity", "--clean")
}
$PlanDir = Join-Path $env:nnUNet_preprocessed $Dataset
New-Item -ItemType Directory -Force -Path $PlanDir | Out-Null
Copy-Item -Force $SourcePlan (Join-Path $PlanDir "nnUNetPlans.json")
Copy-Item -Force (Join-Path $env:nnUNet_raw "$Dataset\dataset.json") (Join-Path $PlanDir "dataset.json")
Invoke-Checked (Join-Path $Bin "nnUNetv2_preprocess.exe") @("-d", "204", "-plans_name", "nnUNetPlans", "-c", "2d", "-np", "4")
Copy-Item -Force (Join-Path $Data "splits_final_r260.json") (Join-Path $PlanDir "splits_final.json")

# Exact R260 mature initialization and trainer.
Invoke-Checked $Python @("scripts/adapt_r202_checkpoint_two_channel.py", "--input", $R202, "--output", $Adapted)
$TrainerDir = & $Python -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss')"
Copy-Item -Force scripts\nnunet_trainers\nNUNetTrainerR260MaturePrior.py $TrainerDir
Invoke-Checked $Python @("scripts/prepare_r260_val_inputs.py", "--prior-root", (Join-Path $PriorRoot "val"), "--output", (Join-Path $Run "imagesValPrior"))
Invoke-Checked (Join-Path $Bin "nnUNetv2_train.exe") @("204", "2d", "0", "-tr", "nnUNetTrainerR260MaturePrior")
Invoke-Checked (Join-Path $Bin "nnUNetv2_predict.exe") @(
    "-i", (Join-Path $Run "imagesValPrior"),
    "-o", (Join-Path $Run "predictions"),
    "-d", "204", "-c", "2d", "-f", "0",
    "-tr", "nnUNetTrainerR260MaturePrior", "-chk", "checkpoint_best.pth"
)
Invoke-Checked $Python @(
    "scripts/convert_nnunet_predictions.py",
    "--pred-dir", (Join-Path $Run "predictions"),
    "--mask-dir", (Join-Path $Run "masks"),
    "--expected-count", "96", "--strip-prefix", "val_"
)
Invoke-Checked $Python @(
    "scripts/evaluate_r201_single_split.py",
    "--gt-dir", "data/raw/TSRS_RSNA-Epiphysis/val_labels",
    "--pred-dir", (Join-Path $Run "masks"),
    "--output", "outputs/analysis/r319_r316b_gated_nnunet_original_val_r201.json",
    "--expected-count", "96"
)
Invoke-Checked $Python @(
    "scripts/render_r259_nnunet_prior_comparison.py",
    "--image-dir", "data/raw/TSRS_RSNA-Epiphysis/val",
    "--gt-dir", "data/raw/TSRS_RSNA-Epiphysis/val_labels",
    "--baseline-dir", $R260Masks,
    "--improved-dir", (Join-Path $Run "masks"),
    "--selection-csv", "outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv",
    "--output-dir", "outputs/visualizations/r319_r260_vs_r316b_nnunet_original_val",
    "--num-cases", "12",
    "--baseline-label", "R260 mature prior nnU-Net",
    "--improved-label", "R316B gated prior nnU-Net"
)
