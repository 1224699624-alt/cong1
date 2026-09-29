$ErrorActionPreference = "Stop"
$Root = "G:\gutou\YOLO+SAM"
$Python = "E:\360Downloads\anaconda3\envs\unet\python.exe"
$Bin = "E:\360Downloads\anaconda3\envs\unet\Scripts"
$Run = Join-Path $Root "outputs\nnunet\r315_clean_refit_prior"
$Data = Join-Path $Run "data"
$Bundle = Join-Path $Root "outputs\artifact_bundles\r260_recovered_from_49284_20260724"
$R202 = Join-Path $Bundle "outputs\nnunet\r202_mature_import\checkpoint_best.pth"
$SourcePlan = Join-Path $Bundle "outputs\nnunet\r260_mature_prior\data\nnUNet_preprocessed\Dataset204_TSRS_RSNAEpiphysisMaturePrior2D\nnUNetPlans.json"
$Adapted = Join-Path $Run "r202_two_channel_zero_prior.pth"
$Dataset = "Dataset315_TSRSCleanRefitPrior2D"
Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $Run | Out-Null
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
function Invoke-Checked { param([string]$Command,[string[]]$Arguments); & $Command @Arguments; if($LASTEXITCODE -ne 0){throw "Command failed: $Command $($Arguments -join ' ')"} }
Invoke-Checked $Python @("scripts/adapt_r202_checkpoint_two_channel.py","--input",$R202,"--output",$Adapted)
Invoke-Checked $Python @("scripts/prepare_r308_clean_panel_r260.py","--selection","configs/r308_tsrs_clean_panel_selection.json","--prior-root","outputs/priors/r313_clean_relation","--frozen-mask-root","outputs/nnunet/r310_clean_plain_nnunet/masks_clean","--source-plan",$SourcePlan,"--output",$Data,"--dataset-id","315","--dataset-name","TSRSCleanRefitPrior2D","--experiment","R315","--overwrite")
Invoke-Checked (Join-Path $Bin "nnUNetv2_extract_fingerprint.exe") @("-d","315","-np","4","--verify_dataset_integrity","--clean")
Invoke-Checked (Join-Path $Bin "nnUNetv2_preprocess.exe") @("-d","315","-plans_name","nnUNetPlans","-c","2d","-np","4")
Copy-Item -Force (Join-Path $env:nnUNet_raw "$Dataset\splits_final.json") (Join-Path $env:nnUNet_preprocessed "$Dataset\splits_final.json")
$TrainerDir=& $Python -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss')"
Copy-Item -Force scripts\nnunet_trainers\nnUNetTrainerR260MaturePrior.py $TrainerDir
Invoke-Checked (Join-Path $Bin "nnUNetv2_train.exe") @("315","2d","0","-tr","nnUNetTrainerR260MaturePrior")
Invoke-Checked (Join-Path $Bin "nnUNetv2_predict.exe") @("-i",(Join-Path $Run "imagesValPrior"),"-o",(Join-Path $Run "pred"),"-d","315","-c","2d","-f","0","-tr","nnUNetTrainerR260MaturePrior","-chk","checkpoint_best.pth")
Invoke-Checked $Python @("scripts/convert_nnunet_predictions.py","--pred-dir",(Join-Path $Run "pred"),"--mask-dir",(Join-Path $Run "masks"),"--expected-count","94","--strip-prefix","val_")
Invoke-Checked $Python @("scripts/evaluate_r201_single_split.py","--gt-dir",(Join-Path $Run "cleanValGT"),"--pred-dir",(Join-Path $Run "masks"),"--output","outputs/analysis/r315_clean_refit_prior_nnunet_val94_r201.json","--expected-count","94")
Invoke-Checked $Python @("scripts/render_r259_nnunet_prior_comparison.py","--image-dir","data/raw/TSRS_RSNA-Epiphysis/val","--gt-dir",(Join-Path $Run "cleanValGT"),"--baseline-dir","outputs/nnunet/r310_clean_plain_nnunet/masks_clean","--improved-dir",(Join-Path $Run "masks"),"--selection-csv","outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv","--output-dir","outputs/visualizations/r315_clean_plain_vs_refit_prior_nnunet_val94","--num-cases","12","--baseline-label","Clean standard nnU-Net","--improved-label","Clean-refit-prior nnU-Net")
