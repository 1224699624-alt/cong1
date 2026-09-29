#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$ProjectPath = (Resolve-Path ".").Path,
    [string]$ArisRepo = ""
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $PSScriptRoot "install_aris_yolosam.ps1"

if (-not (Test-Path $scriptPath)) {
    throw "Missing installer wrapper: $scriptPath"
}

$params = @{
    ProjectPath = $ProjectPath
    Reconcile = $true
}

if ($ArisRepo) {
    $params.ArisRepo = $ArisRepo
}

& $scriptPath @params
