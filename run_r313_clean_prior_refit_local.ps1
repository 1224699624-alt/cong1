$ErrorActionPreference = "Stop"
$Root = "G:\gutou\YOLO+SAM"
$Python = "E:\360Downloads\anaconda3\envs\unet\python.exe"
$Run = Join-Path $Root "outputs\pair_prior\r313_clean_relation_prior"
$Result = Join-Path $Root "outputs\analysis\r313_clean_relation_prior.json"
$Pairs = Join-Path $Root "outputs\analysis\r313_clean_relation_prior_pairs.csv"
$Maps = Join-Path $Root "outputs\priors\r313_clean_relation"
Set-Location -LiteralPath $Root
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONHASHSEED = "2581"
& $Python scripts\train_r258b_prediction_relation_prior.py --selection configs\r308_tsrs_clean_panel_selection.json --output-dir $Run --result-json $Result --pair-manifest-csv $Pairs --workers 0
if ($LASTEXITCODE -ne 0) { throw "R313 clean prior training failed" }
& $Python scripts\build_r259_frozen_prior_maps.py --selection configs\r308_tsrs_clean_panel_selection.json --r258b-result $Result --checkpoint-dir $Run --output-root $Maps
if ($LASTEXITCODE -ne 0) { throw "R313 prior-map generation failed" }
