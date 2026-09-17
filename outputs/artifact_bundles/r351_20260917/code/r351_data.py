"""Aligned RAM priors; every geometric transform is shared or disabled."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset


class AlignedRamDataset(NativeWristDataset):
    def __init__(self, root: Path, prior_root: Path, split: str, augment: bool = False):
        if split not in ("train", "val"):
            raise ValueError("R351 development only permits train/val")
        if augment:
            raise ValueError("R351 pilot disables geometric augmentation to retain prior alignment")
        super().__init__(Path(root), split, False)
        self.prior_root = Path(prior_root)

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        path = self.prior_root / self.split / (item["case"] + ".npy")
        if path.exists():
            prior = np.load(path).astype(np.float32)
        else:
            prior = np.asarray(Image.open(path.with_suffix(".png")), dtype=np.float32) / 255
        # R323 maps were already built in canonical orientation, including _R.
        # They must not be flipped a second time here.
        h, w = item["original_hw"].tolist()
        value = F.interpolate(torch.from_numpy(prior)[None, None], (h, w), mode="bilinear", align_corners=False)[0]
        item["seam"] = F.pad(value, (0, item["image"].shape[-1] - w, 0, item["image"].shape[-2] - h))
        return item
