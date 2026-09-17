# R283 Spatial/Confidence-Gated Support Plan

- Dataset: `TSRS_RSNA-Epiphysis` only, 875 train / 96 original-val.
- `clean-test-v2`: locked; no threshold search or model selection on it.
- Initialization: exact mature R275 two-channel adaptation used by R279-R282.
- Fixed seam coefficient: `0.05`.
- Fixed support coefficient: `0.00485`, inherited from R282's converged effective
  coefficient (`0.05 * 0.097`); no sweep and no dynamic GradNorm.
- Controlled change: support is restricted to a seam-adjacent shoulder that is:
  1. outside a 3 px seam-center safety buffer;
  2. inside an 8 px outer influence radius;
  3. at least 2 px inside the binary bone target;
  4. currently high-confidence foreground, with a detached smooth ramp from 0.70 to 0.90.
- Support is diagnostic-logged every epoch, including mass and overlap with the
  seam buffer. The primary gate remains the registered R201 original-val gate against
  mature R275, with gap FP, merge rate, Dice, Recall, Boundary IoU and Surface Dice.
