#!/usr/bin/env python3
"""Paired compact U-Net test on the reviewed TSRS keep-list.

Both arms start from the exact same two-channel weights. The control receives
a zero prior channel and native Dice/BCE; the improved arm receives the frozen
R259 prior and the fixed R260 background-only prior loss. Test and
clean-test-v2 are deliberately unsupported.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_instance_separation_segmenter import InstanceSeparationUNet, dice_loss


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    p.add_argument("--selection", type=Path, default=Path("configs/r308_tsrs_clean_panel_selection.json"))
    p.add_argument("--prior-root", type=Path, default=Path("outputs/priors/r259_frozen_relation"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/experiments/r312_clean_paired_unet_prior"))
    p.add_argument("--experiment", default="R312")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--min-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--prior-alpha", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=260)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def find_image(root: Path, split: str, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = root / split / f"{stem}{ext}"
        if p.is_file() and p.stat().st_size > 0:
            try:
                with Image.open(p) as image:
                    image.verify()
                return p
            except Exception:
                continue
    raise FileNotFoundError(f"image missing: {split}/{stem}")


class CleanPriorDataset(Dataset):
    def __init__(self, a: argparse.Namespace, split: str, stems: list[str], augment: bool, use_prior: bool):
        self.a, self.split, self.stems, self.augment, self.use_prior = a, split, stems, augment, use_prior

    def __len__(self) -> int: return len(self.stems)

    def __getitem__(self, index: int):
        stem = self.stems[index]
        xray = cv2.imread(str(find_image(self.a.raw_root, self.split, stem)), cv2.IMREAD_GRAYSCALE)
        if xray is None: raise ValueError(stem)
        ids = np.asarray(Image.open(self.a.raw_root / f"{self.split}_labels" / f"{stem}.png"))
        if ids.ndim == 3: ids = ids[..., 0]
        original_shape = ids.shape
        prior_path = self.a.prior_root / self.split / f"{stem}.png"
        if not prior_path.is_file(): raise FileNotFoundError(prior_path)
        prior = np.asarray(Image.open(prior_path).convert("L"))
        if prior.shape != ids.shape: raise RuntimeError(f"unaligned prior: {stem}")
        xray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(xray).astype(np.float32) / 255.0
        prior = prior.astype(np.float32) / 255.0
        mask = (ids > 0).astype(np.float32)
        xray = cv2.resize(xray, (self.a.img_size, self.a.img_size), interpolation=cv2.INTER_AREA)
        prior = cv2.resize(prior, (self.a.img_size, self.a.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.a.img_size, self.a.img_size), interpolation=cv2.INTER_NEAREST)
        if self.augment and random.random() < 0.5:
            xray = np.ascontiguousarray(np.fliplr(xray)); prior = np.ascontiguousarray(np.fliplr(prior)); mask = np.ascontiguousarray(np.fliplr(mask))
        if self.augment and random.random() < 0.25:
            xray = np.clip(xray ** random.uniform(0.85, 1.20), 0.0, 1.0)
        if not self.use_prior: prior = np.zeros_like(prior)
        image = np.stack([xray, prior], axis=0).astype(np.float32)
        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask[None].astype(np.float32)),
                "prior": torch.from_numpy(prior[None].astype(np.float32)), "stem": stem,
                "shape": torch.tensor(original_shape, dtype=torch.int64)}


def native_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos = target.mean().detach().clamp(1e-4, 0.5)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=(1.0 - pos) / pos)
    return bce + dice_loss(logits, target)


def make_loaders(a, stems, use_prior):
    train = CleanPriorDataset(a, "train", stems["train"], True, use_prior)
    val = CleanPriorDataset(a, "val", stems["val"], False, use_prior)
    generator = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(train, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                    pin_memory=True, generator=generator)
    vl = DataLoader(val, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True)
    return tl, vl


@torch.no_grad()
def validate(model, loader, device):
    model.eval(); inter = pred_sum = gt_sum = 0.0
    for b in loader:
        prob = torch.sigmoid(model(b["image"].to(device))["mask"])
        pred = prob >= 0.5; gt = b["mask"].to(device) > 0.5
        inter += float((pred & gt).sum()); pred_sum += float(pred.sum()); gt_sum += float(gt.sum())
    return (2.0 * inter + 1.0) / (pred_sum + gt_sum + 1.0)


def train_arm(a, stems, arm, initial_state, use_prior):
    seed_all(a.seed)
    model = InstanceSeparationUNet(a.base_channels, in_channels=2).to(a.device)
    model.load_state_dict(initial_state, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=a.device.startswith("cuda"))
    tl, vl = make_loaders(a, stems, use_prior)
    arm_dir = a.output_root / arm; arm_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0; best_epoch = 0; bad = 0; history = []
    for epoch in range(1, a.epochs + 1):
        model.train(); losses=[]; prior_losses=[]
        for b in tl:
            image=b["image"].to(a.device); target=b["mask"].to(a.device); prior=b["prior"].to(a.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=a.device.startswith("cuda")):
                logits=model(image)["mask"]; loss=native_loss(logits,target)
                aux=logits.new_zeros(())
                if use_prior:
                    valid=prior*(target < 0.5); aux=(torch.sigmoid(logits)*valid).sum()/valid.sum().clamp_min(1.0)
                    loss=loss+a.prior_alpha*aux
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu())); prior_losses.append(float(aux.detach().cpu()))
        dice=validate(model,vl,a.device); row={"epoch":epoch,"train_loss":float(np.mean(losses)),"prior_loss":float(np.mean(prior_losses)),"val_dice":dice}
        history.append(row); print(json.dumps({"arm":arm,**row}),flush=True)
        if dice > best + 1e-4:
            best,best_epoch,bad=dice,epoch,0
            torch.save({"model":model.state_dict(),"epoch":epoch,"val_dice":dice,"arm":arm},arm_dir/"best.pt")
        else: bad+=1
        if epoch >= a.min_epochs and bad >= a.patience: break
    (arm_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    return {"arm":arm,"best_epoch":best_epoch,"best_val_dice":best,"completed_epochs":len(history)}


@torch.no_grad()
def infer_arm(a, stems, arm, use_prior):
    ckpt=torch.load(a.output_root/arm/"best.pt",map_location=a.device,weights_only=False)
    model=InstanceSeparationUNet(a.base_channels,in_channels=2).to(a.device); model.load_state_dict(ckpt["model"],strict=True); model.eval()
    ds=CleanPriorDataset(a,"val",stems["val"],False,use_prior); out=a.output_root/arm/"masks"; out.mkdir(parents=True,exist_ok=True)
    for item in ds:
        prob=torch.sigmoid(model(item["image"].unsqueeze(0).to(a.device))["mask"])[0,0].cpu().numpy()
        h,w=map(int,item["shape"].tolist()); pred=cv2.resize(prob,(w,h),interpolation=cv2.INTER_LINEAR)>=a.threshold
        Image.fromarray(pred.astype(np.uint8)*255).save(out/f"{item['stem']}.png")


def main():
    a=args_parser(); joined=" ".join(map(str,vars(a).values())).lower()
    if "clean-test" in joined or "articular" in joined or a.raw_root.name!="TSRS_RSNA-Epiphysis": raise RuntimeError("TSRS Epiphysis train/original-val only")
    payload=json.loads(a.selection.read_text(encoding="utf-8")); stems={k:sorted(map(str,payload["keep"][k])) for k in ("train","val")}
    if {k:len(v) for k,v in stems.items()}!={"train":814,"val":94}: raise RuntimeError("unexpected keep-list")
    a.output_root.mkdir(parents=True,exist_ok=True); seed_all(a.seed)
    init=InstanceSeparationUNet(a.base_channels,in_channels=2).state_dict()
    # The prior channel begins neutral in both arms, matching R260's safe adaptation.
    first="enc1.net.0.weight"; init[first][:,1].zero_()
    torch.save({"model":init,"seed":a.seed},a.output_root/"shared_initialization.pt")
    results=[]
    for arm,use_prior in (("plain",False),("prior",True)):
        results.append(train_arm(a,stems,arm,copy.deepcopy(init),use_prior)); infer_arm(a,stems,arm,use_prior)
    result={"experiment":a.experiment,"dataset":"TSRS_RSNA-Epiphysis","train":814,"val":94,"shared_initialization":True,"prior":str(a.prior_root),"prior_retrained_on_clean":a.experiment!="R312","prior_alpha":a.prior_alpha,"clean_test_used":False,"arms":results}
    (a.output_root/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
