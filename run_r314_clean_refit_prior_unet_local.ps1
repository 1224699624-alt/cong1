$ErrorActionPreference = "Stop"
$Root = "G:\gutou\YOLO+SAM"
$Python = "E:\360Downloads\anaconda3\envs\YOLOSAM\python.exe"
$Run = Join-Path $Root "outputs\experiments\r314_clean_refit_prior_unet"
Set-Location -LiteralPath $Root
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONHASHSEED = "260"
& $Python scripts\run_r312_clean_paired_unet_prior.py --experiment R314 --prior-root outputs\priors\r313_clean_relation --output-root $Run
if ($LASTEXITCODE -ne 0) { throw "R314 U-Net training failed" }
& $Python scripts\evaluate_r201_single_split.py --gt-dir outputs\nnunet\r310_clean_plain_nnunet\cleanValGT --pred-dir (Join-Path $Run "plain\masks") --output outputs\analysis\r314_clean_plain_unet_val94_r201.json --expected-count 94
if ($LASTEXITCODE -ne 0) { throw "R314 plain R201 failed" }
& $Python scripts\evaluate_r201_single_split.py --gt-dir outputs\nnunet\r310_clean_plain_nnunet\cleanValGT --pred-dir (Join-Path $Run "prior\masks") --output outputs\analysis\r314_clean_refit_prior_unet_val94_r201.json --expected-count 94
if ($LASTEXITCODE -ne 0) { throw "R314 prior R201 failed" }
& $Python scripts\render_r259_nnunet_prior_comparison.py --image-dir data\raw\TSRS_RSNA-Epiphysis\val --gt-dir outputs\nnunet\r310_clean_plain_nnunet\cleanValGT --baseline-dir (Join-Path $Run "plain\masks") --improved-dir (Join-Path $Run "prior\masks") --selection-csv outputs\analysis\r255_close_gap_sanity_original_val_pairs.csv --output-dir outputs\visualizations\r314_clean_plain_vs_refit_prior_unet_val94 --num-cases 12 --baseline-label "Clean plain U-Net" --improved-label "Clean-refit-prior U-Net"
if ($LASTEXITCODE -ne 0) { throw "R314 visualization failed" }
