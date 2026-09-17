# R128 Ordered Instance Layout Audit

## Raw Numbers

- images audited: `971`
- contiguous id fraction: `0.840371`
- component-count median: `27.000`
- component-count q10/q90: `23.000` / `28.000`
- id-vs-PC1 inversion median/q90: `0.156695` / `0.192029`
- PC1 explained median/q10: `0.635144` / `0.574002`
- bbox-overlap pairs median/q90: `9.000` / `16.000`

## Decision

- `ordered_instance_signal_strong`: `False`
- `fixed_slot_count_safe`: `False`
- `single_axis_order_safe`: `False`
- `needs_variable_slots`: `True`

## Recommendation

Instance ids/layout are not stable enough for a naive fixed-slot model; use variable slots or manual protocol review first. A one-dimensional left-right or top-bottom order alone is risky.

## Component Count Histogram

- `2`: `2`
- `3`: `1`
- `4`: `1`
- `6`: `1`
- `7`: `1`
- `8`: `2`
- `9`: `1`
- `10`: `1`
- `12`: `3`
- `14`: `1`
- `15`: `2`
- `16`: `3`
- `17`: `3`
- `18`: `3`
- `19`: `7`
- `20`: `2`
- `21`: `10`
- `22`: `30`
- `23`: `63`
- `24`: `64`
- `25`: `37`
- `26`: `83`
- `27`: `440`
- `28`: `192`
- `29`: `18`
