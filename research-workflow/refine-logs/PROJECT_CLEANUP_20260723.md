# Project Cleanup 2026-07-23

Status: completed 2026-07-23; archive validation passed.

## Protected scope

- Original datasets are read-only and excluded from deletion.
- TSRS and RAM-W600 artifacts remain separated.
- All `outputs/analysis`, `outputs/ram_w600`, research logs, scripts, configs, and visualizations
  are protected.
- R275 final checkpoints and the lightweight R293--R296 evidence chain are protected.
- `clean-test-v2` remains locked and unused.

## Cleanup classes

1. Archive lightweight R271/R273/R274 metadata before deleting superseded preprocessed arrays and
   checkpoints.
2. Replace the oversized R275 transfer payload with a focused archive of superseded
   `checkpoint_best` files; retain the active `checkpoint_final` files unpacked.
3. Retain the validated R265 artifact bundle and remove only its unpacked duplicate.
4. Delete reproducible R271 target arrays, R202/R273/R274 preprocessing caches, R275 validation
   probability arrays, MedSAM embedding cache, and obsolete transfer payloads.

The machine-readable record is written to
`outputs/archives/cleanup_manifest_20260723.json` after successful execution.

## Execution record

- Candidate deleted bytes: `48,133,091,263` (~44.8 GiB).
- Legacy metadata archive: `outputs/archives/legacy_r271_r273_r274_metadata_20260723.tar.gz`
  (SHA256 `97f73bf30689190fefdd3f73e2d3af239d22fd4edfe6a1a2b67ece5561aac1a6`).
- R275 superseded-best archive: `outputs/archives/r275_superseded_best_checkpoints_20260723.tar.gz`
  (SHA256 `4121d84a15a63164be8dc43286c40f687b45725f984c539f7ec154ca7389455c`).
- Existing R265 bundle was revalidated before its unpacked duplicate was removed.
- TSRS Epiphysis, TSRS Articular-Surface, and RAM-W600 file counts and byte totals were unchanged.
- The independent integrity review returned `WARN`: no fake GT or self-normalized metrics were
  found; remaining warnings are single-seed scope, R296 metric-schema drift, and historically
  missing R244/R275 status notes. Those result JSON files and scripts remain protected.
