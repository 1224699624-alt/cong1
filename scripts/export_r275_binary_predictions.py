"""Export matched R275 validation PNGs as binary original-val masks."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

def export(src: Path, dst: Path, keep_class_one: bool) -> int:
    dst.mkdir(parents=True, exist_ok=True); n = 0
    for p in sorted(src.glob('val_*.png')):
        a = np.asarray(Image.open(p))
        if a.ndim == 3: a = a[..., 0]
        binary = (a == 1) if keep_class_one else (a > 0)
        name = p.name[4:]
        Image.fromarray(binary.astype(np.uint8) * 255, mode='L').save(dst / name)
        n += 1
    return n

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--control-src', type=Path, required=True); p.add_argument('--seam-src', type=Path, required=True); p.add_argument('--out-root', type=Path, default=Path('outputs/predictions/r275_mature_original_val'))
    a = p.parse_args()
    if 'clean-test' in str(a).lower() or 'articular' in str(a).lower(): raise RuntimeError('Epiphysis original-val only')
    result = {'control': export(a.control_src, a.out_root/'control', False), 'seam': export(a.seam_src, a.out_root/'seam', True), 'clean_test_used': False}
    (a.out_root/'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result, indent=2))
if __name__ == '__main__': main()
