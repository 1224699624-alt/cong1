# R260 Mature nnU-Net Frozen-Prior Probe

- Baseline: frozen mature R202 one-channel nnU-Net checkpoint, SHA256 `65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7`, validation EMA Dice `0.917071`, with its original 96 original-val masks.
- Improved initialization: copy all R202 network weights; expand only the first convolution from one to two channels by copying X-ray weights unchanged and initializing the prior-channel weights to exactly zero.
- Data: original `TSRS_RSNA-Epiphysis` train/original-val X-rays plus the already frozen R259 prior maps. No clean-test-v2 or Articular-Surface.
- Prior/loss: unchanged R259 prior input and fixed `0.05` background-only separation loss. No threshold or loss-weight adjustment.
- Fine-tuning: maximum 60 epochs, standard 250 train/50 validation iterations, learning rate `1e-4`, PolyLR.
- Early stopping: minimum 15 completed epochs; stop after 10 consecutive epochs without EMA foreground-Dice improvement greater than `1e-4`.
- Inference: validation-best checkpoint only, all 96 original-val cases.
- Visualization: fixed R255 baseline-selected 12 hard cases; original raw X-ray, GT, mature R202 nnU-Net, mature-initialized prior+loss nnU-Net, with full image and close-gap zoom.

This remains an original-val qualitative probe and does not unlock clean-test-v2.
## Execution status (2026-07-15)

- Imported mature R202 checkpoint SHA256: `65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7`.
- Verified the 96 imported R202 original-val masks against the source copy; deterministic directory SHA256: `2f685e548cdbc853814eba7cd351b351210bc7411d3ce71a0501b90bfb3a0457`.
- Adaptation audit passed: 444 tensors unchanged, four aliases of the first input convolution expanded; X-ray channel is exact, prior channel is zero, and 16 segmentation-head tensors are inherited.
- Strict nnU-Net 2.8.1 initialization passed at `2026-07-15 18:09:04 +08:00`.
- Active SeetaCloud screen: `r260_mature_prior` on GPU 0 (RTX 4090D).
- Training split: 875 train / 96 original-val. `clean-test-v2` is not used.

## Final result

- Completed 29 epochs and stopped after 10 validation epochs without the required `1e-4` EMA-Dice improvement; best EMA foreground Dice was `0.916879535`.
- R201 original-val, mature R202 -> R260: Dice `0.897112 -> 0.902759`, Boundary IoU `0.239893 -> 0.244642`, Boundary F1 `0.379017 -> 0.385938`, gap FP rate `0.205547 -> 0.175200`, merge rate `0.572917 -> 0.500000`, Surface Dice 2px `0.654511 -> 0.667281`, HD95 `54.3414 -> 15.8641`, ASSD `14.1738 -> 4.3204`.
- Important trade-off: precision increased `0.886486 -> 0.911294`, while recall decreased `0.916238 -> 0.899632`; the gain is partly a more conservative foreground prediction and cannot yet be attributed solely to correct seam separation.
- Twelve fixed 1-4 px hard-pair panels were generated under `outputs/visualizations/r260_mature_nnunet_prior_original_val`; 10/12 improve whole-image Dice, but close-gap visual changes remain modest rather than decisive.
- `clean-test-v2` remained unused.
