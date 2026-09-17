# R183 True R110 Train/Val Materialization

R183 completed on the remote server only.

It materialized true R110-style train/val masks from the `R100 + R108` patch/basic checkpoint instead of using the older proxy anchor from R177.

- Output experiment: `r110_r100_r108_patch_basic_trainval`
- Train masks: `875/875`
- Val masks: `96/96`
- Status: `ready`
- Remote mask root: `outputs/ablations_variants/r110_r100_r108_patch_basic_trainval/TSRS_RSNA-Epiphysis`

This reopens a stricter R110-anchored local repair route. Future train/val arbitrators should use this anchor, not `r100_like_r025b_r097_patch_basic_trainval`.
