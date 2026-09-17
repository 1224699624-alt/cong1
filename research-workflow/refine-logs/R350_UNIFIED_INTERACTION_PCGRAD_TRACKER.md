# R350 Unified Interaction PCGrad Tracker

| Run | Dataset | Purpose | Split | Epochs | Status |
|---|---|---|---|---:|---|
| R350-S0 | synthetic | shared formula and emphasis smoke | none | 0 | PASSED |
| R350-TA | TSRS | identity/state/gradient audit | train batch | 0 | PASSED |
| R350-RA | RAM | identity/state/gradient audit | train batch | 0 | PASSED |
| R350-T | TSRS | unified three-state training | original-val | 30 | COMPLETED; R201 audited; overall no-go vs R317 anchor |
| R350-R | RAM | unified three-state training | validation | 30 | COMPLETED; epoch 29 selected; gate passed |

Test splits are locked throughout development.

Completion note (2026-08-21): neither TSRS clean-test-v2 nor RAM test was used.
