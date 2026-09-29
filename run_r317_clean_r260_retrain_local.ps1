$ErrorActionPreference = "Stop"
$Root = "G:\gutou\YOLO+SAM"
$PriorPy = "E:\360Downloads\anaconda3\envs\unet\python.exe"
$NnPy = "E:\360Downloads\anaconda3\envs\unet\python.exe"
$NnBin = "E:\360Downloads\anaconda3\envs\unet\Scripts"
$PriorRun = Join-Path $Root "outputs\pair_prior\r317_clean_r260_prior"
$PriorResult = Join-Path $Root "outputs\analysis\r317_clean_r260_prior.json"
$PairCsv = Join-Path $Root "outputs\analysis\r317_clean_r260_prior_pairs.csv"
$PriorMaps = Join-Path $Root "outputs\priors\r317_clean_r260_relation"
$UnetRun = Join-Path $Root "outputs\experiments\r317_clean_r260_unet"
$NnRun = Join-Path $Root "outputs\nnunet\r317_clean_r260_nnunet"
$Selection = Join-Path $Root "configs\r308_tsrs_clean_panel_selection.json"
$Bundle = Join-Path $Root "outputs\artifact_bundles\r260_recovered_from_49284_20260724"
$R202 = Join-Path $Bundle "outputs\nnunet\r202_mature_import\checkpoint_best.pth"
$SourcePlan = Join-Path $Bundle "outputs\nnunet\r260_mature_prior\data\nnUNet_preprocessed\Dataset204_TSRS_RSNAEpiphysisMaturePrior2D\nnUNetPlans.json"
$Adapted = Join-Path $NnRun "r202_two_channel_zero_prior.pth"
$Dataset = "Dataset317_TSRSCleanR260"
Set-Location -LiteralPath $Root
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONHASHSEED = "260"

# R260 prior protocol: same R258B architecture, fixed proposal provenance,
# fixed hyperparameters and three seeds; only the frozen 814/94 selection changes.
& $PriorPy scripts\train_r258b_prediction_relation_prior.py --selection $Selection --output-dir $PriorRun --result-json $PriorResult --pair-manifest-csv $PairCsv --workers 0
if ($LASTEXITCODE -ne 0) { throw "R317 R260 prior retraining failed" }
& $PriorPy scripts\build_r259_frozen_prior_maps.py --selection $Selection --r258b-result $PriorResult --checkpoint-dir $PriorRun --output-root $PriorMaps
if ($LASTEXITCODE -ne 0) { throw "R317 prior map generation failed" }

# Exact paired compact U-Net comparison.
& $PriorPy scripts\run_r312_clean_paired_unet_prior.py --experiment R317 --prior-root $PriorMaps --output-root $UnetRun
if ($LASTEXITCODE -ne 0) { throw "R317 U-Net failed" }
& $PriorPy scripts\evaluate_r201_single_split.py --gt-dir outputs\nnunet\r310_clean_plain_nnunet\cleanValGT --pred-dir (Join-Path $UnetRun "plain\masks") --output outputs\analysis\r317_clean_r260_plain_unet_r201.json --expected-count 94
& $PriorPy scripts\evaluate_r201_single_split.py --gt-dir outputs\nnunet\r310_clean_plain_nnunet\cleanValGT --pred-dir (Join-Path $UnetRun "prior\masks") --output outputs\analysis\r317_clean_r260_prior_unet_r201.json --expected-count 94

# Exact R260 mature nnU-Net protocol, initialized from the recovered R202 model.
New-Item -ItemType Directory -Force -Path $NnRun | Out-Null
$env:Path = "$NnBin;$env:Path"
$env:nnUNet_compile = "false"
$env:nnUNet_n_proc_DA = "0"
$env:nnUNet_raw = Join-Path $NnRun "data\nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $NnRun "data\nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $NnRun "nnUNet_results"
$env:R260_ADAPTED_CHECKPOINT = $Adapted
$env:R315_EXPECTED_DATASET = $Dataset
& $NnPy scripts\adapt_r202_checkpoint_two_channel.py --input $R202 --output $Adapted
& $NnPy scripts\prepare_r308_clean_panel_r260.py --selection $Selection --prior-root $PriorMaps --frozen-mask-root outputs\nnunet\r310_clean_plain_nnunet\masks_clean --source-plan $SourcePlan --output (Join-Path $NnRun "data") --dataset-id 317 --dataset-name TSRSCleanR260 --experiment R317 --overwrite
& (Join-Path $NnBin "nnUNetv2_extract_fingerprint.exe") -d 317 -np 4 --verify_dataset_integrity --clean
& (Join-Path $NnBin "nnUNetv2_preprocess.exe") -d 317 -plans_name nnUNetPlans -c 2d -np 4
$TrainerDir = & $NnPy -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss')"
Copy-Item -Force scripts\nnunet_trainers\nNUNetTrainerR260MaturePrior.py $TrainerDir
& (Join-Path $NnBin "nnUNetv2_train.exe") 317 2d 0 -tr nnUNetTrainerR260MaturePrior
& (Join-Path $NnBin "nnUNetv2_predict.exe") -i (Join-Path $NnRun "imagesValPrior") -o (Join-Path $NnRun "pred") -d 317 -c 2d -f 0 -tr nnUNetTrainerR260MaturePrior -chk checkpoint_best.pth
& $NnPy scripts\convert_nnunet_predictions.py --pred-dir (Join-Path $NnRun "pred") --mask-dir (Join-Path $NnRun "masks") --expected-count 94 --strip-prefix val_
& $NnPy scripts\evaluate_r201_single_split.py --gt-dir (Join-Path $NnRun "cleanValGT") --pred-dir (Join-Path $NnRun "masks") --output outputs\analysis\r317_clean_r260_prior_nnunet_r201.json --expected-count 94
& $NnPy scripts\render_r259_nnunet_prior_comparison.py --image-dir data\raw\TSRS_RSNA-Epiphysis\val --gt-dir (Join-Path $NnRun "cleanValGT") --baseline-dir outputs\nnunet\r310_clean_plain_nnunet\masks_clean --improved-dir (Join-Path $NnRun "masks") --selection-csv outputs\analysis\r255_close_gap_sanity_original_val_pairs.csv --output-dir outputs\visualizations\r317_clean_r260_plain_vs_prior_nnunet_val94 --num-cases 12 --baseline-label "Clean standard nnU-Net" --improved-label "Clean R260 prior nnU-Net"
