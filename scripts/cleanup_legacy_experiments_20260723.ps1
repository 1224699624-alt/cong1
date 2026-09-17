param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path.TrimEnd('\')
$ArchiveDir = Join-Path $Workspace 'outputs\archives'
$ManifestPath = Join-Path $ArchiveDir 'cleanup_manifest_20260723.json'
$LegacyArchive = Join-Path $ArchiveDir 'legacy_r271_r273_r274_metadata_20260723.tar.gz'
$R275Archive = Join-Path $ArchiveDir 'r275_superseded_best_checkpoints_20260723.tar.gz'

function Assert-InWorkspace([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not ($resolved -eq $Workspace -or $resolved.StartsWith($Workspace + '\', [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Refusing path outside workspace: $resolved"
    }
    return $resolved
}

function Get-PathBytes([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return [int64]0 }
    $item = Get-Item -LiteralPath $Path
    if ($item.PSIsContainer) {
        return [int64]((Get-ChildItem -LiteralPath $Path -Recurse -File -Force | Measure-Object Length -Sum).Sum)
    }
    return [int64]$item.Length
}

function Test-Tar([string]$Path) {
    & tar -tzf $Path *> $null
    if ($LASTEXITCODE -ne 0) { throw "Archive validation failed: $Path" }
}

function Copy-WithRelativePath([System.IO.FileInfo]$File, [string]$DestinationRoot) {
    $relative = $File.FullName.Substring($Workspace.Length + 1)
    $destination = Join-Path $DestinationRoot $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    Copy-Item -LiteralPath $File.FullName -Destination $destination -Force
}

$deleteRoots = @(
    'outputs\targets',
    'outputs\nnunet\r202',
    'outputs\nnunet\r273_local',
    'outputs\nnunet\r274_seam',
    'outputs\medsam_embedding_cache',
    'outputs\nnunet\r265_hierarchical_carpal',
    'outputs\priors\r265_hierarchical_carpal'
) | ForEach-Object { Join-Path $Workspace $_ }

$deleteFiles = @(
    'outputs\bridge_logs\r275_remote_sync_minimal.tar.gz',
    'tmp\r258b_seeta_payload.tar',
    'tmp\r260_raw_epiphysis.tar',
    '.tmp\r255_seeta_local_payload.tar.gz'
) | ForEach-Object { Join-Path $Workspace $_ }

$r275Roots = @(
    (Join-Path $Workspace 'outputs\nnunet\r275_mature_control')
    (Join-Path $Workspace 'outputs\nnunet\r275_mature_seam')
)
$r275Regenerable = @()
foreach ($root in $r275Roots) {
    if (Test-Path -LiteralPath $root) {
        $r275Regenerable += Get-ChildItem -LiteralPath $root -Recurse -File -Force |
            Where-Object { $_.Extension -in @('.npz', '.pkl', '.pyc') }
    }
}

$r275Best = @(
    Get-ChildItem -LiteralPath $r275Roots[0] -Recurse -File -Filter 'checkpoint_best.pth'
    Get-ChildItem -LiteralPath $r275Roots[1] -Recurse -File -Filter 'checkpoint_best.pth'
) | Where-Object { $_ -is [System.IO.FileInfo] }

$allDeleteCandidates = @($deleteRoots + $deleteFiles + $r275Regenerable.FullName + $r275Best.FullName) |
    Where-Object { Test-Path -LiteralPath $_ }
$beforeBytes = [int64](($allDeleteCandidates | ForEach-Object { Get-PathBytes $_ } | Measure-Object -Sum).Sum)

$preview = [ordered]@{
    date = '2026-07-23'
    mode = if ($Execute) { 'execute' } else { 'dry-run' }
    workspace = $Workspace
    protected_datasets = @(
        (Join-Path $Workspace 'data\raw\TSRS_RSNA-Epiphysis'),
        'G:\gutou\RAM-W600'
    )
    archive_outputs = @($LegacyArchive, $R275Archive)
    delete_roots = $deleteRoots
    delete_files = $deleteFiles
    r275_regenerable_file_count = $r275Regenerable.Count
    r275_superseded_best_checkpoint_count = $r275Best.Count
    candidate_bytes = $beforeBytes
}

if (-not $Execute) {
    $preview | ConvertTo-Json -Depth 5
    exit 0
}

New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null

# Archive lightweight R271/R273/R274 evidence. Large model/data/cache formats are
# deliberately excluded because the mature R275 checkpoints supersede them.
$stage = Join-Path $ArchiveDir '.legacy_metadata_staging_20260723'
if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath (Assert-InWorkspace $stage) -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
$metadataFiles = @()
foreach ($root in @(
    (Join-Path $Workspace 'outputs\nnunet\r273_local'),
    (Join-Path $Workspace 'outputs\nnunet\r274_seam')
)) {
    if (Test-Path -LiteralPath $root) {
        $metadataFiles += Get-ChildItem -LiteralPath $root -Recurse -File -Force |
            Where-Object { $_.Extension -in @('.json', '.txt', '.py') }
    }
}
$metadataFiles += Get-ChildItem -LiteralPath (Join-Path $Workspace 'outputs\targets') -Recurse -File -Filter 'manifest.json' -ErrorAction SilentlyContinue
$metadataFiles += Get-ChildItem -LiteralPath (Join-Path $Workspace 'research-workflow\refine-logs') -File |
    Where-Object { $_.Name -match '^R(271|273|274|275)' }
foreach ($file in $metadataFiles) { Copy-WithRelativePath $file $stage }
if (Test-Path -LiteralPath $LegacyArchive) { Remove-Item -LiteralPath (Assert-InWorkspace $LegacyArchive) -Force }
Push-Location $stage
try { & tar -czf $LegacyArchive .; if ($LASTEXITCODE -ne 0) { throw 'Legacy archive creation failed' } }
finally { Pop-Location }
Test-Tar $LegacyArchive
Remove-Item -LiteralPath (Assert-InWorkspace $stage) -Recurse -Force

# Archive the superseded R275 checkpoint_best files. checkpoint_final remains
# unpacked because it is the exact active baseline used by R296.
$r275ArchiveFiles = @($r275Best.FullName)
$r275ArchiveFiles += @(
    (Join-Path $Workspace 'outputs\analysis\r275_mature_control_original_val_r201.json')
    (Join-Path $Workspace 'outputs\analysis\r275_mature_seam_original_val_r201.json')
    (Join-Path $Workspace 'scripts\nnunet_trainers\nnUNetTrainerR275ControlMature.py')
    (Join-Path $Workspace 'scripts\nnunet_trainers\nnUNetTrainerR275SeamMature.py')
) | Where-Object { Test-Path -LiteralPath $_ }
if (Test-Path -LiteralPath $R275Archive) { Remove-Item -LiteralPath (Assert-InWorkspace $R275Archive) -Force }
Push-Location $Workspace
try {
    $relativeR275 = $r275ArchiveFiles | ForEach-Object { $_.Substring($Workspace.Length + 1) }
    & tar -czf $R275Archive @relativeR275
    if ($LASTEXITCODE -ne 0) { throw 'R275 archive creation failed' }
}
finally { Pop-Location }
Test-Tar $R275Archive

# Validate the existing R265 useful bundle before deleting its unpacked copy.
$r265Bundle = Join-Path $Workspace 'outputs\artifact_bundles\r265_useful_bundle.tar.gz'
Test-Tar $r265Bundle

# Every recursive target is resolved and constrained to this workspace first.
foreach ($path in $deleteRoots) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath (Assert-InWorkspace $path) -Recurse -Force
    }
}
foreach ($path in $deleteFiles) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath (Assert-InWorkspace $path) -Force
    }
}
foreach ($file in @($r275Regenerable + $r275Best)) {
    if (Test-Path -LiteralPath $file.FullName) {
        Remove-Item -LiteralPath (Assert-InWorkspace $file.FullName) -Force
    }
}

$archiveRows = @($LegacyArchive, $R275Archive, $r265Bundle) | ForEach-Object {
    [ordered]@{
        path = $_
        bytes = (Get-Item -LiteralPath $_).Length
        sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$afterData = [ordered]@{
    date = '2026-07-23'
    workspace = $Workspace
    deleted_candidate_bytes = $beforeBytes
    archives = $archiveRows
    preserved = @(
        'data/raw/TSRS_RSNA-Epiphysis',
        'G:/gutou/RAM-W600',
        'outputs/analysis',
        'outputs/ram_w600',
        'outputs/visualizations',
        'outputs/nnunet/r275_mature_control/**/checkpoint_final.pth',
        'outputs/nnunet/r275_mature_seam/**/checkpoint_final.pth',
        'outputs/nnunet/r293_dualbranch',
        'outputs/nnunet/r294_local_fusion',
        'outputs/nnunet/r296_plan_aligned_local_fusion'
    )
    clean_test_used = $false
}
$afterData | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
Write-Output ($afterData | ConvertTo-Json -Depth 6)
