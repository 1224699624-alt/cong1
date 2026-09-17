# Initial Experiment Results

**Date**: 2026-06-15 21:06  
**Plan**: `research-workflow/refine-logs/EXPERIMENT_PLAN.md`  
**Bridge Input**: `research-workflow/refine-logs/EXPERIMENT_BRIDGE_INPUT.md`

## Workflow Status

- Current workflow mode: continue from real server state, not blind retraining
- Reason: the main candidate `allstack_anatomy_roi_aic_refiner_sam_hqsam` already has:
  - checkpoint
  - threshold sweep
  - original test metrics
- Current active remote job:
  - host: `10.1.115.157`
  - repo: `/home/shenzeyu/workspace/YOLO_SAM_generic_src`
  - purpose: run `clean-test-v2` evaluation for `allstack_anatomy_roi_aic_refiner_sam_hqsam`
  - pid: `317519`
  - log: `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/bridge_logs/aic_refiner_clean_test_v2_20260615_2104.log`

## M0: Locked Baselines

### External anchor: ARAA best on clean-test-v2

- metrics path:
  - `/home/shenzeyu/workspace/ARAA-Net/outputs/tsrs_epiphysis_clean_test_v2_ckpt_sweep/epoch_98/metrics.json`
- Dice: `0.9117660066557425`
- IoU: `0.8384360930797206`
- Precision: `0.9101767600852156`
- Recall: `0.9146908686504701`
- Specificity: `0.9971764181974658`
- Boundary IoU: `0.2352651559145258`

### Current strongest internal anchor on clean-test-v2

- system: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam`
- metrics path:
  - `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/ablations_variants/allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam/TSRS_RSNA-Epiphysis_clean_test_v2/test/metrics.json`
- Dice: `0.9100290053191697`
- IoU: `0.8354338498823685`
- Precision: `0.9131011545806722`
- Recall: `0.9082009857713057`
- Specificity: `0.9971784330664731`
- Boundary IoU: `0.22848859791261497`

### Current strongest internal anchor on original test

- system: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam`
- metrics path:
  - `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/ablations/allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam/TSRS_RSNA-Epiphysis/test/metrics.json`
- Dice: `0.8936935131682259`
- IoU: `0.810590948170741`
- Precision: `0.8992150550562691`
- Recall: `0.8915754953748375`
- Specificity: `0.9970575205845561`
- Boundary IoU: `0.2159435973764147`

## M1: Main Candidate State

### Existing original-test result already found on server

- system: `allstack_anatomy_roi_aic_refiner_sam_hqsam`
- checkpoint:
  - `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/error_refiner/TSRS_RSNA-Epiphysis/best_allstack_anatomy_roi_aic_refiner_sam_hqsam.pt`
- threshold sweep:
  - `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/error_refiner/TSRS_RSNA-Epiphysis/threshold_sweep_val_allstack_anatomy_roi_aic_refiner_sam_hqsam.json`
- original test metrics:
  - `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/ablations/allstack_anatomy_roi_aic_refiner_sam_hqsam/TSRS_RSNA-Epiphysis/test/metrics.json`
- Dice: `0.8897098247956451`
- IoU: `0.8043424421111297`
- Precision: `0.8803708058316431`
- Recall: `0.9019376122732686`
- Specificity: `0.9965692091342953`
- Boundary IoU: `0.22374453240000397`

### Immediate interpretation before clean-test-v2 finishes

- On original test, `aic_refiner` is currently below the strongest internal anchor.
- The gap is:
  - Dice: about `-0.00398`
  - IoU: about `-0.00625`
  - Precision: about `-0.01884`
  - Recall: about `+0.01036`
  - Boundary IoU: about `+0.00780`
- This profile suggests:
  - `aic_refiner` is less conservative than the current best
  - it may recover more positives
  - but it currently loses too much precision on the noisier original test
- Final judgment is deferred until `clean-test-v2` completes.

## Active Job

### Running now

```bash
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export REFINER_EXP=allstack_anatomy_roi_aic_refiner_sam_hqsam
bash run_epiphysis_clean_test_keepbone_cutgap_refiner_v2.sh
```

### Expected output path

- `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/ablations_variants/allstack_anatomy_roi_aic_refiner_sam_hqsam/TSRS_RSNA-Epiphysis_clean_test_v2/test/metrics.json`

## Pending Next Steps

1. Wait for `clean-test-v2` evaluation to finish.
2. Compare `aic_refiner` against:
   - ARAA clean-test-v2
   - keepbone-cutgap-v2 clean-test-v2
3. If still not competitive, classify the candidate as `REJECT` or `ITERATE`.
4. If close enough, continue to hard-case validation on `2982/1475/3040/1433`.

## Provisional Status

- M0 baseline freeze: `DONE`
- M1 original test candidate readback: `DONE`
- M1 clean-test-v2 candidate run: `DONE`
- Promotion verdict: `REJECT` for mainline promotion, `ITERATE` for mechanism insight

## M1: clean-test-v2 Result

### Main candidate on clean-test-v2

- system: `allstack_anatomy_roi_aic_refiner_sam_hqsam`
- metrics path:
  - `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/ablations_variants/allstack_anatomy_roi_aic_refiner_sam_hqsam/TSRS_RSNA-Epiphysis_clean_test_v2/test/metrics.json`
- summary path:
  - `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/ablations_variants/TSRS_RSNA-Epiphysis_clean_test_v2_test_clean_test_v2_summary.md`
- Dice: `0.9058689991490124`
- IoU: `0.8288071109243607`
- Precision: `0.8964336332990822`
- Recall: `0.9169009242293409`
- Specificity: `0.9966923693406234`
- Boundary IoU: `0.23809236134009293`

### Comparison against locked anchors

- vs current strongest internal clean-test-v2 baseline:
  - Dice: `-0.004160`
  - IoU: `-0.006627`
  - Precision: `-0.016668`
  - Recall: `+0.008700`
  - Boundary IoU: `+0.009604`
- vs ARAA clean-test-v2:
  - Dice: `-0.005897`
  - IoU: `-0.009629`
  - Precision: `-0.013743`
  - Recall: `+0.002210`
  - Boundary IoU: `+0.002827`

### Metric-level interpretation

- `aic_refiner` does not meet the primary success gate.
- It is clearly below both:
  - `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam`
  - ARAA best `epoch_98`
- However, it shows a stable pattern:
  - recall is higher
  - boundary IoU is higher
  - precision is substantially lower
- This suggests the mechanism is not useless, but its current operating point is too aggressive for mainline promotion.

## M3: Hard-Case Validation

### Artifact directory

- `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/bridge_hardcase/aic_vs_keepbone_cutgap_v2`

### Generated files

- `1433_comparison.jpg`
- `1475_comparison.jpg`
- `2982_comparison.jpg`
- `3040_comparison.jpg`
- `hardcase_metrics.json`

### Per-sample comparison summary

- `1433`
  - keepbone-cutgap-v2 Dice `0.8503`, boundary IoU `0.1755`
  - aic-refiner Dice `0.8263`, boundary IoU `0.1501`
  - interpretation: `aic_refiner` is worse on both Dice and boundary quality
- `1475`
  - keepbone-cutgap-v2 Dice `0.7835`, boundary IoU `0.1708`
  - aic-refiner Dice `0.7221`, boundary IoU `0.1084`
  - interpretation: `aic_refiner` is clearly worse and loses substantial precision
- `2982`
  - keepbone-cutgap-v2 Dice `0.8871`, boundary IoU `0.2002`
  - aic-refiner Dice `0.8846`, boundary IoU `0.2155`
  - interpretation: Dice is slightly lower, but recall and boundary are somewhat better
- `3040`
  - keepbone-cutgap-v2 Dice `0.8274`, boundary IoU `0.3338`
  - aic-refiner Dice `0.8446`, boundary IoU `0.3665`
  - interpretation: `aic_refiner` is clearly better on this sample

### Hard-case conclusion

- The candidate is not uniformly bad.
- It appears to help some high-recall / gap-sensitive cases such as `3040`, and partly `2982`.
- But it regresses too strongly on `1433` and `1475`.
- So the visual evidence does not overturn the metric verdict.

## Final Bridge Verdict For This Candidate

- Mainline promotion verdict: `REJECT`
- Mechanism verdict: `ITERATE`

### Why `REJECT`

- clean-test-v2 Dice is below the internal best by about `0.00416`
- clean-test-v2 Dice is below ARAA by about `0.00590`
- original test is also below the internal best
- hard-case evidence is mixed rather than dominant

### Why still `ITERATE`

- boundary IoU is better than both the internal best and ARAA on clean-test-v2
- recall is also higher
- the failure mode looks more like over-aggressive positives / precision collapse than a useless direction

## Recommended Next Action

Do not run filtered-data control for this candidate, because the primary gate already failed.

The next experiment should reuse the useful part of this signal:

- keep the boundary / recall strength revealed by `aic_refiner`
- add stronger gap-protection or precision recovery
- treat `1433` and `1475` as priority repair cases
