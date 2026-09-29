param([int]$PriorProcessId = 13924)
$ErrorActionPreference = "Stop"
$Root = "G:\gutou\YOLO+SAM"
Set-Location -LiteralPath $Root
$p = Get-Process -Id $PriorProcessId -ErrorAction SilentlyContinue
if ($p) { Wait-Process -Id $PriorProcessId }
$Manifest = Join-Path $Root "outputs\priors\r313_clean_relation\manifest.json"
if (-not (Test-Path -LiteralPath $Manifest)) { throw "R313 prior manifest missing; downstream training is locked" }
$m = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if ($m.splits.train.images -ne 814 -or $m.splits.val.images -ne 94 -or $m.splits.train.nonzero_maps -ne 814 -or $m.splits.val.nonzero_maps -ne 94) { throw "R313 prior coverage audit failed" }
& "E:\360Downloads\anaconda3\envs\YOLOSAM\python.exe" scripts\visualize_r313_prior_probability_maps.py
if ($LASTEXITCODE -ne 0) { throw "R313 prior heatmap visualization failed; downstream training is locked" }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "run_r314_clean_refit_prior_unet_local.ps1")
if ($LASTEXITCODE -ne 0) { throw "R314 failed; R315 is locked" }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "run_r315_clean_refit_prior_nnunet_local.ps1")
if ($LASTEXITCODE -ne 0) { throw "R315 failed" }
