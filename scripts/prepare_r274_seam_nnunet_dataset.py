"""Build an isolated 3-class nnU-Net dataset: background=0, bone=1, seam=2."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image

def read_label(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    if a.ndim == 3: a = a[..., 0]
    return a.astype(np.int32)

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--raw-root', type=Path, default=Path('data/raw/TSRS_RSNA-Epiphysis'))
    p.add_argument('--source-raw', type=Path, default=Path('outputs/nnunet/r202/nnUNet_raw/Dataset202_TSRS_RSNAEpiphysis2D'))
    p.add_argument('--seam-root', type=Path, default=Path('outputs/targets/r271_instance_seam_full'))
    p.add_argument('--out-root', type=Path, default=Path('outputs/nnunet/r274_seam/data/nnUNet_raw/Dataset274_TSRS_RSNAEpiphysisSeam2D'))
    a = p.parse_args(); out = a.out_root; (out/'imagesTr').mkdir(parents=True, exist_ok=True); (out/'labelsTr').mkdir(parents=True, exist_ok=True)
    for src in sorted((a.source_raw/'imagesTr').glob('*.png')):
        shutil.copy2(src, out/'imagesTr'/src.name)
        case = src.stem.replace('_0000',''); split, stem = case.split('_', 1)
        label_path = a.raw_root / ('train_labels' if split == 'train' else 'val_labels') / f'{stem}.png'
        seam_path = a.seam_root / split / 'seam' / f'{stem}.png'
        ids = read_label(label_path); seam = np.asarray(Image.open(seam_path)) > 0
        target = np.zeros(ids.shape, dtype=np.uint8); target[ids > 0] = 1; target[(ids == 0) & seam] = 2
        Image.fromarray(target, mode='L').save(out/'labelsTr'/f'{case}.png')
    split_src = Path('outputs/nnunet/r273_local/data/nnUNet_preprocessed/Dataset202_TSRS_RSNAEpiphysis2D/splits_final.json')
    shutil.copy2(split_src, out/'splits_final.json')
    dataset = {'channel_names': {'0': 'xray'}, 'labels': {'background': 0, 'bone': 1, 'seam': 2}, 'numTraining': len(list((out/'imagesTr').glob('*.png'))), 'file_ending': '.png', 'overwrite_image_reader_writer': 'NaturalImage2DIO'}
    (out/'dataset.json').write_text(json.dumps(dataset, indent=2), encoding='utf-8')
    print(json.dumps({'dataset': str(out), 'cases': dataset['numTraining']}, indent=2))
if __name__ == '__main__': main()
