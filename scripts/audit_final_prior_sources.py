"""Read-only final-prior source/split audit; only train and val are opened."""
from pathlib import Path
import argparse
import hashlib
import json
from collections import Counter

import numpy as np
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    root = a.root
    base = root / 'outputs/nnunet/r317_image_centers_only'
    ds = 'Dataset204_TSRS_RSNAEpiphysisMaturePrior2D'
    split_file = base / 'data/nnUNet_preprocessed' / ds / 'splits_final.json'
    splits = json.loads(split_file.read_text())[0]
    raw = root / 'data/raw/TSRS_RSNA-Epiphysis'
    prepared = base / 'data/nnUNet_preprocessed' / ds
    errors = []
    tsrs = {}
    for split in ('train', 'val'):
        rows, modes = [], Counter()
        for label in sorted((raw / f'{split}_labels').glob('*.png')):
            # Match the historical prepare_r260/find_image extension precedence.
            images = [raw / split / (label.stem + ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp')]
            images = [x for x in images if x.exists() and x.stat().st_size > 0]
            key = f'{split}_{label.stem}'
            needed = [prepared / 'nnUNetPlans_2d' / f'{key}.b2nd',
                      root / 'outputs/priors/r317_image_centers_only' / split / f'{label.stem}.png',
                      prepared / 'gt_segmentations' / f'{key}.png',
                      prepared / 'nnUNetPlans_2d' / f'{key}_seg.b2nd',
                      prepared / 'nnUNetPlans_2d' / f'{key}.pkl']
            if not images or not all(x.exists() for x in needed):
                errors.append(f'missing/ambiguous TSRS source or prepared prior: {key}')
                continue
            with Image.open(label) as im:
                modes[im.mode] += 1
                labels = np.asarray(im)
            with Image.open(needed[2]) as im:
                union = np.asarray(im) > 0
            if labels.ndim != 2 or not np.array_equal(labels > 0, union):
                errors.append(f'prepared union mismatch: {key}')
            rows.append({'id': key, 'selected_image': str(images[0]), 'image_candidates': len(images), 'image_sha256': sha(images[0]), 'label_sha256': sha(label), 'prior_sha256': sha(needed[1])})
        if sorted(splits[split]) != sorted(x['id'] for x in rows):
            errors.append(f'TSRS {split} IDs differ from frozen preprocessed split')
        tsrs[split] = {'count': len(rows), 'modes': dict(modes), 'files': rows}
    ti = {x['image_sha256'] for x in tsrs['train']['files']}
    vi = {x['image_sha256'] for x in tsrs['val']['files']}
    if ti & vi:
        errors.append('TSRS train/val image content overlap')
    if {x.split('_', 1)[1] for x in splits['train']} & {x.split('_', 1)[1] for x in splits['val']}:
        errors.append('TSRS train/val case ID overlap')
    ram = {}
    rr = root / 'data/remote_variants/RAM-W600/BoneSegmentation'
    pr = root / 'outputs/priors/r323_ram_native_r317_single_seed'
    for split in ('train', 'val'):
        rows = []
        for mask in sorted((rr / 'masks' / split).glob('*.npy')):
            image = rr / 'images' / f'{mask.stem}.bmp'
            prior = pr / split / f'{mask.stem}.npy'
            if not image.exists() or not prior.exists():
                errors.append(f'RAM missing image/prior: {split}/{mask.stem}')
                continue
            value = np.load(prior)
            if value.ndim != 2 or not np.isfinite(value).all():
                errors.append(f'invalid RAM prior: {prior}')
            rows.append({'id': mask.stem, 'image_sha256': sha(image), 'prior_sha256': sha(prior)})
        ram[split] = {'count': len(rows), 'files': rows}
    if {x['id'] for x in ram['train']['files']} & {x['id'] for x in ram['val']['files']}:
        errors.append('RAM train/val case ID overlap')
    if {x['image_sha256'] for x in ram['train']['files']} & {x['image_sha256'] for x in ram['val']['files']}:
        errors.append('RAM train/val image content overlap')
    expected = {'tsrs': (875, 96), 'ram': (425, 69)}
    for name, data in [('tsrs', tsrs), ('ram', ram)]:
        if tuple(data[s]['count'] for s in ('train', 'val')) != expected[name]:
            errors.append(f'{name} count mismatch')
    result = {'passed': not errors, 'errors': errors, 'test_read': False,
              'tsrs_split_path': str(split_file), 'tsrs_split_sha256': sha(split_file),
              'tsrs': tsrs, 'ram': ram, 'missing_1833_recovered': True}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'passed': not errors, 'errors': errors[:10], 'error_count': len(errors), 'tsrs': [tsrs[s]['count'] for s in ('train', 'val')], 'ram': [ram[s]['count'] for s in ('train', 'val')], 'split_sha256': result['tsrs_split_sha256']}))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
