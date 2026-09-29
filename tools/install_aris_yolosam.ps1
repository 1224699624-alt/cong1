#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$ProjectPath = (Resolve-Path ".").Path,
    [string]$ArisRepo = "",
    [switch]$Reconcile,
    [switch]$Uninstall,
    [switch]$DryRun,
    [switch]$NoDoc
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath {
    param([string]$Path)
    return (Resolve-Path $Path).Path
}

$ProjectPath = Resolve-FullPath $ProjectPath

if (-not $ArisRepo) {
    $candidates = @(
        (Join-Path $ProjectPath ".tmp\aris_repo"),
        "G:\gutou\.tmp\Auto-claude-code-research-in-sleep",
        "G:\gutou\YOLO+SAM\.tmp\aris_repo"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate "tools\install_aris.ps1")) {
            $ArisRepo = $candidate
            break
        }
    }
}

if (-not $ArisRepo) {
    throw "Cannot locate ARIS repo automatically. Pass -ArisRepo explicitly."
}

$ArisRepo = Resolve-FullPath $ArisRepo
$Installer = Join-Path $ArisRepo "tools\install_aris.ps1"

if (-not (Test-Path $Installer)) {
    throw "ARIS installer not found: $Installer"
}

$params = @{
    ProjectPath = $ProjectPath
    Platform = "codex"
    ArisRepo = $ArisRepo
}

if ($Reconcile) { $params.Reconcile = $true }
if ($Uninstall) { $params.Uninstall = $true }
if ($DryRun) { $params.DryRun = $true }
if ($NoDoc) { $params.NoDoc = $true }

Write-Host "Installing ARIS Codex workflow into $ProjectPath"
Write-Host "Using ARIS repo: $ArisRepo"

& $Installer @params
