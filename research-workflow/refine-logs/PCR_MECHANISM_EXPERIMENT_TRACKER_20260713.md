# PCR-Loss Mechanism Experiment Tracker

| ID | Milestone | Purpose | Backbone/Variant | Split | Seeds | Status |
|---|---|---|---|---|---:|---|
| PCR-M0-01 | M0 | reliability/valid-pair coverage audit | data only | original train/val | - | TODO |
| PCR-M0-02 | M0 | sentinel manifest freeze | image/GT-only selection | original val | - | TODO |
| PCR-M0-03 | M0 | loss/logger one-batch gradient test | classic U-Net | original train | 1 | TODO |
| PCR-M1-01 | M1 | vanilla development baseline | U-Net B0 | original train/val | 1 | TODO |
| PCR-M1-02 | M1 | full mechanism gate | U-Net B3 | original train/val | 1 | TODO |
| PCR-M2-01 | M2 | robust base ablation | U-Net B1 | original train/val | 3 | TODO |
| PCR-M2-02 | M2 | uniform pair ablation | U-Net B2 | original train/val | 3 | TODO |
| PCR-M2-03 | M2 | paired full seeds | U-Net B0/B3 | original train/val | 3 | TODO |
| PCR-M3-01 | M3 | strong medical transfer pilot | nnU-Net native continuation vs HA-PDSP+PCR | original train/val | 1 | COMPLETED_MIXED_NEGATIVE_R254 |
| PCR-M4-01 | M4 | pretrained encoder transfer | DINOv3-UNet B0/B3 | original train/val | 1→3 | TODO |
| PCR-M4-02 | M4 | non-U-Net transfer | DeepLabV3+/validated alternative B0/B3 | original train/val | 1→3 | TODO |
| PCR-M4-03 | M4 | optional prompted/transformer transfer | YOLO+SAM refiner or SegFormer B0/B3 | original train/val | 1→3 | TODO |
| PCR-M5-01 | M5 | config/protocol freeze | all accepted models | original val | - | TODO |
| PCR-M6-01 | M6 | one-shot final evaluation | accepted models | clean-test-v2/R201 | 3 | LOCKED |
| PCR-M6-02 | M6 | paper figures | pre-registered cases | final outputs | - | LOCKED |
| R261-M0 | M0 | conditional seam bank/maps and leakage audit | data only | original train/val | - | LOCAL_SANITY_PASSED |
| R261-M1 | M3 | mature nnU-Net conditional per-seam reasoning sanity | nnU-Net R202 initialization | original train/val | 261 | STOPPED_PRE_GPU_ANATOMY_GATE_FAILED |
| R261-M2 | M3 | full conditional per-seam reasoning | nnU-Net R202 initialization | original train/val | 261 | NOT_RUN |
| R262-M0 | M0 | broad palm Delaunay + image-valley prior visualization | frozen proposals + train-only bank | 12 original-val cases | - | FAILED_STRICT_METACARPAL_GATE_WRIST_EDGES |
| R263-M0 | M0 | strict metacarpal band + adjacent graph + image-valley prior visualization (max 6) | frozen proposals + train-only bank | 12 original-val cases | - | VISUAL_GATE_READY_NO_TRAINING |
| R264-M0 | M0 | carpal ossification cluster + adjacent graph + image-valley prior visualization (max 6) | frozen proposals + train-only bank | 12 original-val cases | - | VISUAL_GATE_READY_66_OF_72_NO_TRAINING |
| R265-M1 | M3 | hierarchical carpal relation prior (CC/CM/CR) + gradient-balanced relation loss | mature R202 nnU-Net initialization | original train/val | 265 | COMPLETED_NO_GO_BOUNDARY_AND_MERGE_WORSE |
