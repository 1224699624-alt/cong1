# Round 2 Review

**Overall score**: 8.13/10  
**Verdict**: MAJOR-MINOR REVISION / NOT READY

Resolved: training-only bone age, no semantic cross-case IDs, no rank/core BCE, continuous bridge risk, full-resolution loss, augmentation/X-ray reliability, per-image normalization.

Remaining blockers:

1. d_ij must be separated from context u_ij to avoid circular conditional definition.
2. widest-path must be restricted to a pair-local capsule and exclude third bones/cores/detours.
3. warm-up and 2-epoch active sanity protocol must be separated.
4. sidecar provenance, capsule equation, non-saturating robust BCE, q terms and alpha protocol need executable definitions.

