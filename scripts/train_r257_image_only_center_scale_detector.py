#!/usr/bin/env python3
"""R257 image-only epiphysis center/scale proposal detector."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset


EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant-root", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/center_detector/r257_image_only_center_scale"))
    p.add_argument("--result-json", type=Path, default=Path("outputs/analysis/r257_image_only_center_scale_result.json"))
    p.add_argument("--proposal-csv", type=Path, default=Path("outputs/analysis/r257_image_only_center_scale_val_proposals.csv"))
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--base-channels", type=int, default=24)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mask-loss-weight", type=float, default=0.35)
    p.add_argument("--scale-loss-weight", type=float, default=0.50)
    p.add_argument("--center-threshold", type=float, default=0.15)
    p.add_argument("--mask-gate", type=float, default=0.20)
    p.add_argument("--nms-kernel", type=int, default=9)
    p.add_argument("--min-center-distance", type=int, default=5)
    p.add_argument("--max-proposals", type=int, default=48)
    p.add_argument("--match-tolerance-scale", type=float, default=0.60)
    p.add_argument("--relative-close-threshold", type=float, default=0.20)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--seed", type=int, default=257)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def find_image(folder: Path, stem: str) -> Path:
    for extension in EXTENSIONS:
        path = folder / f"{stem}{extension}"
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(stem)


def read_gray(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None: raise ValueError(f"Cannot read {path}")
    return value


def read_instance(path: Path) -> np.ndarray:
    value = np.asarray(Image.open(path))
    return (value[..., 0] if value.ndim == 3 else value).astype(np.int32)


def letterbox(image: np.ndarray, instance: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    height, width = image.shape
    ratio = min(size / height, size / width)
    resized_h, resized_w = max(1, int(round(height * ratio))), max(1, int(round(width * ratio)))
    pad_y, pad_x = (size - resized_h) // 2, (size - resized_w) // 2
    out_image = np.zeros((size, size), dtype=np.uint8)
    out_instance = np.zeros((size, size), dtype=np.int32)
    out_image[pad_y:pad_y+resized_h, pad_x:pad_x+resized_w] = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    out_instance[pad_y:pad_y+resized_h, pad_x:pad_x+resized_w] = cv2.resize(instance, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    return out_image, out_instance, {"ratio": ratio, "scale_x": resized_w/width, "scale_y": resized_h/height,
                                     "pad_x": float(pad_x), "pad_y": float(pad_y),
                                     "original_width": float(width), "original_height": float(height)}


def instance_records(instance: np.ndarray, min_area: int = 4) -> list[dict[str, Any]]:
    records = []
    for value in [int(v) for v in np.unique(instance) if int(v) > 0]:
        mask = instance == value; yy, xx = np.nonzero(mask)
        if len(xx) < min_area: continue
        boundary = mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        by, bx = np.nonzero(boundary)
        records.append({"id": value, "x": float(xx.mean()), "y": float(yy.mean()), "area": float(len(xx)),
                        "scale": math.sqrt(float(len(xx))), "boundary": np.stack([by, bx], axis=1)})
    return records


def dense_targets(instance: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.zeros((size, size), dtype=np.float32)
    scale_target = np.zeros((size, size), dtype=np.float32)
    scale_weight = np.zeros((size, size), dtype=np.float32)
    yy_grid, xx_grid = np.mgrid[:size, :size]
    for record in instance_records(instance):
        sigma = max(1.5, 0.15 * record["scale"])
        heat = np.exp(-((xx_grid-record["x"])**2 + (yy_grid-record["y"])**2)/(2*sigma**2)).astype(np.float32)
        peak_y, peak_x = int(round(record["y"])), int(round(record["x"]))
        peak_y, peak_x = np.clip(peak_y, 0, size-1), np.clip(peak_x, 0, size-1)
        heat[peak_y, peak_x] = 1.0
        update = heat > center
        center = np.maximum(center, heat)
        normalized_scale = float(np.clip(record["scale"] / size, 1e-4, 1.0))
        scale_target[update] = normalized_scale
        scale_weight = np.maximum(scale_weight, heat)
    return (instance > 0).astype(np.float32), center, scale_target, scale_weight


class CenterDataset(Dataset):
    def __init__(self, root: Path, split: str, size: int, limit: int, augment: bool):
        self.root, self.split, self.size, self.augment = root, split, size, augment
        self.labels = sorted((root / f"{split}_labels").glob("*.png"))
        if limit > 0: self.labels = self.labels[:limit]

    def __len__(self) -> int: return len(self.labels)

    def load(self, index: int) -> dict[str, Any]:
        label_path = self.labels[index]; image = read_gray(find_image(self.root / self.split, label_path.stem)); original_instance = read_instance(label_path)
        image, instance, info = letterbox(image, original_instance, self.size)
        mask, center, scale_target, scale_weight = dense_targets(instance, self.size)
        if self.augment:
            if random.random() < 0.5:
                image, instance = np.fliplr(image).copy(), np.fliplr(instance).copy()
                mask, center = np.fliplr(mask).copy(), np.fliplr(center).copy()
                scale_target, scale_weight = np.fliplr(scale_target).copy(), np.fliplr(scale_weight).copy()
            if random.random() < 0.30:
                image = np.clip((image.astype(np.float32)/255.0) ** random.uniform(0.85, 1.15) * 255.0, 0, 255).astype(np.uint8)
        return {"image": image, "instance": instance, "original_instance": original_instance, "mask": mask, "center": center,
                "scale_target": scale_target, "scale_weight": scale_weight, "stem": label_path.stem, "info": info}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.load(index)
        return {"image": torch.from_numpy(item["image"][None].astype(np.float32)/255.0),
                "mask": torch.from_numpy(item["mask"][None]), "center": torch.from_numpy(item["center"][None]),
                "scale_target": torch.from_numpy(item["scale_target"][None]),
                "scale_weight": torch.from_numpy(item["scale_weight"][None]), "stem": item["stem"]}


class Block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False), nn.BatchNorm2d(output_channels), nn.SiLU(), nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False), nn.BatchNorm2d(output_channels), nn.SiLU())
    def forward(self, value: torch.Tensor) -> torch.Tensor: return self.net(value)


class CenterScaleUNet(nn.Module):
    def __init__(self, base: int):
        super().__init__(); self.pool=nn.MaxPool2d(2); self.e1=Block(1,base); self.e2=Block(base,base*2); self.e3=Block(base*2,base*4); self.e4=Block(base*4,base*8)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=Block(base*8,base*4); self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=Block(base*4,base*2); self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=Block(base*2,base)
        self.mask=nn.Conv2d(base,1,1); self.center=nn.Conv2d(base,1,1); self.scale=nn.Conv2d(base,1,1)
        nn.init.constant_(self.center.bias, -2.19)
    def forward(self, image: torch.Tensor) -> dict[str,torch.Tensor]:
        e1=self.e1(image); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3)); d3=self.d3(torch.cat([self.u3(e4),e3],1)); d2=self.d2(torch.cat([self.u2(d3),e2],1)); d1=self.d1(torch.cat([self.u1(d2),e1],1)); return {"mask":self.mask(d1),"center":self.center(d1),"scale":self.scale(d1)}


def dice_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob=torch.sigmoid(logit); inter=(prob*target).sum((1,2,3)); den=prob.sum((1,2,3))+target.sum((1,2,3)); return (1-(2*inter+1)/(den+1)).mean()


def center_focal_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Keep the focal terms in float32 even under AMP. In float16, 1 - 1e-5
    # rounds to 1, so saturated logits can make log(1 - prob) become log(0).
    target=target.float(); prob=torch.sigmoid(logit.float()).clamp(1e-5,1-1e-5); positive=target.eq(1).float(); negative=target.lt(1).float(); negative_weight=(1-target).pow(4)
    pos_loss=-(torch.log(prob)*(1-prob).pow(2)*positive).sum(); neg_loss=-(torch.log(1-prob)*prob.pow(2)*negative_weight*negative).sum(); count=positive.sum().clamp_min(1.0); return (pos_loss+neg_loss)/count


def training_loss(output: dict[str,torch.Tensor], batch: dict[str,torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    mask=batch["mask"].to(args.device); center=batch["center"].to(args.device); scale_target=batch["scale_target"].to(args.device); scale_weight=batch["scale_weight"].to(args.device)
    mask_loss=F.binary_cross_entropy_with_logits(output["mask"],mask)+dice_loss(output["mask"],mask); center_loss=center_focal_loss(output["center"],center)
    scale_error=F.smooth_l1_loss(torch.sigmoid(output["scale"]),scale_target,reduction="none"); scale_loss=(scale_error*scale_weight).sum()/scale_weight.sum().clamp_min(1.0)
    return center_loss+args.mask_loss_weight*mask_loss+args.scale_loss_weight*scale_loss


def extract_proposals(output: dict[str,torch.Tensor], args: argparse.Namespace, valid_region: np.ndarray | None = None) -> list[dict[str,float]]:
    center=torch.sigmoid(output["center"])[0,0].detach().cpu().numpy(); mask=torch.sigmoid(output["mask"])[0,0].detach().cpu().numpy(); scale=torch.sigmoid(output["scale"])[0,0].detach().cpu().numpy()*args.image_size
    kernel=np.ones((args.nms_kernel,args.nms_kernel),np.uint8); pooled=cv2.dilate(center,kernel)
    dilated_mask=cv2.dilate(mask,np.ones((5,5),np.uint8)); peaks=(center>=pooled-1e-7)&(center>=args.center_threshold)&(dilated_mask>=args.mask_gate)
    if valid_region is not None: peaks &= valid_region
    yy,xx=np.nonzero(peaks); order=np.argsort(center[yy,xx])[::-1]
    proposals=[]
    for idx in order:
        y,x=int(yy[idx]),int(xx[idx])
        if all((y-p["y"])**2+(x-p["x"])**2>=args.min_center_distance**2 for p in proposals):
            proposals.append({"x":float(x),"y":float(y),"score":float(center[y,x]),"scale":float(max(scale[y,x],1.0)),"mask_probability":float(mask[y,x])})
        if len(proposals)>=args.max_proposals: break
    return proposals


def match_centers(proposals: list[dict[str,float]], gt: list[dict[str,Any]], tolerance_scale: float) -> list[tuple[int,int,float]]:
    if not proposals or not gt: return []
    pred=np.asarray([[p["x"],p["y"]] for p in proposals]); truth=np.asarray([[g["x"],g["y"]] for g in gt]); distance=np.linalg.norm(pred[:,None]-truth[None,:],axis=-1); tolerance=np.maximum(4.0,tolerance_scale*np.asarray([g["scale"] for g in gt])); cost=distance/tolerance[None,:]; gated=np.where(cost<=1.0,cost,1e6); rows,cols=linear_sum_assignment(gated)
    return [(int(r),int(c),float(distance[r,c])) for r,c in zip(rows,cols) if gated[r,c]<1e5]


def boundary_gap(a: dict[str,Any], b: dict[str,Any]) -> float:
    distance,_=cKDTree(b["boundary"]).query(a["boundary"],k=1); return max(0.0,float(distance.min())-1.0)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: CenterDataset, args: argparse.Namespace, write_proposals: bool=False) -> tuple[dict[str,float],list[dict[str,Any]],list[dict[str,Any]]]:
    model.eval(); total_gt=total_pred=total_match=0; localization=[]; scale_errors=[]; close_localization=[]; close_scale_errors=[]; contact_localization=[]; contact_scale_errors=[]; count_errors=[]; close_total=close_covered=contact_total=contact_covered=0; per_image=[]; proposal_rows=[]
    for index in range(len(dataset)):
        item=dataset.load(index); image=torch.from_numpy(item["image"][None,None].astype(np.float32)/255.0).to(args.device); output=model(image); info=item["info"]
        valid_region=np.zeros((args.image_size,args.image_size),bool); resized_w=int(round(info["original_width"]*info["scale_x"])); resized_h=int(round(info["original_height"]*info["scale_y"])); x0,y0=int(info["pad_x"]),int(info["pad_y"]); valid_region[y0:y0+resized_h,x0:x0+resized_w]=True
        proposals=extract_proposals(output,args,valid_region); gt_original=instance_records(item["original_instance"],min_area=24)
        gt=[]
        for record in gt_original:
            transformed=dict(record); transformed["x"]=record["x"]*info["scale_x"]+info["pad_x"]; transformed["y"]=record["y"]*info["scale_y"]+info["pad_y"]; transformed["scale"]=record["scale"]*math.sqrt(info["scale_x"]*info["scale_y"]); gt.append(transformed)
        matches=match_centers(proposals,gt,args.match_tolerance_scale); mapping={g:p for p,g,_ in matches}
        total_gt+=len(gt); total_pred+=len(proposals); total_match+=len(matches); count_errors.append(abs(len(proposals)-len(gt)))
        match_errors={}
        for pred_idx,gt_idx,distance in matches:
            loc=distance/max(gt[gt_idx]["scale"],1.0); scale_error=abs(math.log(max(proposals[pred_idx]["scale"],1e-6)/max(gt[gt_idx]["scale"],1e-6))); localization.append(loc); scale_errors.append(scale_error); match_errors[gt_idx]=(loc,scale_error)
        image_close=image_close_covered=image_contact=image_contact_covered=0
        close_endpoints=set(); contact_endpoints=set()
        for i in range(len(gt_original)):
            for j in range(i+1,len(gt_original)):
                gap=boundary_gap(gt_original[i],gt_original[j]); local_scale=math.sqrt(0.5*(gt_original[i]["area"]+gt_original[j]["area"])); close=gap/max(local_scale,1.0)<=args.relative_close_threshold; contact=gap<1.0; covered=i in mapping and j in mapping and mapping[i]!=mapping[j]
                if close: close_total+=1; image_close+=1; close_covered+=int(covered); image_close_covered+=int(covered); close_endpoints.update((i,j))
                if contact: contact_total+=1; image_contact+=1; contact_covered+=int(covered); image_contact_covered+=int(covered); contact_endpoints.update((i,j))
        for gt_idx in close_endpoints & set(match_errors): close_localization.append(match_errors[gt_idx][0]); close_scale_errors.append(match_errors[gt_idx][1])
        for gt_idx in contact_endpoints & set(match_errors): contact_localization.append(match_errors[gt_idx][0]); contact_scale_errors.append(match_errors[gt_idx][1])
        per_image.append({"stem":item["stem"],"num_gt":len(gt),"num_proposals":len(proposals),"num_matched":len(matches),"close_pairs":image_close,"close_pairs_covered":image_close_covered,"contact_pairs":image_contact,"contact_pairs_covered":image_contact_covered})
        if write_proposals:
            for rank,p in enumerate(proposals):
                proposal_rows.append({"stem":item["stem"],"rank":rank,"x_letterbox":p["x"],"y_letterbox":p["y"],"scale_letterbox":p["scale"],"score":p["score"],"mask_probability":p["mask_probability"],"x_original":(p["x"]-info["pad_x"])/info["scale_x"],"y_original":(p["y"]-info["pad_y"])/info["scale_y"],"scale_original":p["scale"]/math.sqrt(info["scale_x"]*info["scale_y"])})
    metrics={"num_images":float(len(dataset)),"num_gt_centers":float(total_gt),"num_proposals":float(total_pred),"num_matched":float(total_match),"center_recall":float(total_match/max(total_gt,1)),"center_precision":float(total_match/max(total_pred,1)),"mean_normalized_localization_error":float(np.mean(localization)) if localization else None,"median_abs_log_scale_error":float(np.median(scale_errors)) if scale_errors else None,"proposal_count_mae":float(np.mean(count_errors)),"false_proposals_per_image":float((total_pred-total_match)/max(len(dataset),1)),"close_pair_count":float(close_total),"close_pair_coverage":float(close_covered/max(close_total,1)),"close_matched_endpoint_count":float(len(close_localization)),"close_endpoint_mean_localization_error":float(np.mean(close_localization)) if close_localization else None,"close_endpoint_median_abs_log_scale_error":float(np.median(close_scale_errors)) if close_scale_errors else None,"contact_pair_count":float(contact_total),"contact_pair_coverage":float(contact_covered/max(contact_total,1)),"contact_matched_endpoint_count":float(len(contact_localization)),"contact_endpoint_mean_localization_error":float(np.mean(contact_localization)) if contact_localization else None,"contact_endpoint_median_abs_log_scale_error":float(np.median(contact_scale_errors)) if contact_scale_errors else None}
    return metrics,per_image,proposal_rows


def decide(checks: dict[str,bool], limited: bool) -> tuple[bool,str]:
    if limited: return False,"sanity_only"
    passed=all(checks.values()); return passed,"allow_r258_oof_pair_prior" if passed else "no_go_image_only_proposals_insufficient"


def finite_le(value: float | None, threshold: float) -> bool:
    return value is not None and bool(np.isfinite(value)) and value <= threshold


def main() -> None:
    args=parse_args(); set_seed(args.seed); joined=str(args.variant_root).lower()
    if "clean-test" in joined or "articular" in joined or args.variant_root.name!="TSRS_RSNA-Epiphysis_contrast_v1": raise RuntimeError("R257 permits contrast-v1 Epiphysis train/val only")
    if args.image_size%8!=0: raise ValueError("--image-size must be divisible by 8")
    if args.nms_kernel<=0 or args.nms_kernel%2==0: raise ValueError("--nms-kernel must be a positive odd integer")
    train_ds=CenterDataset(args.variant_root,"train",args.image_size,args.limit_train,True); val_ds=CenterDataset(args.variant_root,"val",args.image_size,args.limit_val,False)
    if not train_ds or not val_ds: raise RuntimeError("Empty dataset")
    train_stems=[p.stem for p in train_ds.labels]; val_stems=[p.stem for p in val_ds.labels]
    if len(train_stems)!=len(set(train_stems)) or len(val_stems)!=len(set(val_stems)) or set(train_stems)&set(val_stems): raise RuntimeError("Invalid or overlapping train/val IDs")
    limited=args.limit_train>0 or args.limit_val>0
    if not limited and (len(train_stems)!=875 or len(val_stems)!=96): raise RuntimeError(f"Full R257 requires 875/96 cases, found {len(train_stems)}/{len(val_stems)}")
    generator=torch.Generator().manual_seed(args.seed); loader=DataLoader(train_ds,args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,generator=generator)
    model=CenterScaleUNet(args.base_channels).to(args.device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); amp=args.device.startswith("cuda"); scaler=torch.amp.GradScaler("cuda",enabled=amp); history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in loader:
            image=batch["image"].to(args.device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=amp): output=model(image); loss=training_loss(output,batch,args)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        metrics,_,_=evaluate(model,val_ds,args,False); row={"epoch":epoch,"train_loss":float(np.mean(losses)),**metrics}; history.append(row); print(json.dumps(row),flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True); checkpoint=args.output_dir/f"center_scale_seed{args.seed}_final.pt"; torch.save({"model":model.state_dict(),"config":vars(args)},checkpoint)
    final,per_image,proposal_rows=evaluate(model,val_ds,args,True)
    checks={"metrics_finite":all(v is not None and np.isfinite(v) for v in final.values()),"all_val_images":final["num_images"]==float(len(val_ds)) and (limited or final["num_images"]==96),"center_recall_ge_085":final["center_recall"]>=0.85,"center_precision_ge_075":final["center_precision"]>=0.75,"close_pair_coverage_ge_080":final["close_pair_coverage"]>=0.80,"contact_pair_coverage_ge_075":final["contact_pair_count"]>0 and final["contact_pair_coverage"]>=0.75,"localization_error_le_050":finite_le(final["mean_normalized_localization_error"],0.50),"scale_error_le_050":finite_le(final["median_abs_log_scale_error"],0.50),"close_endpoint_localization_le_060":finite_le(final["close_endpoint_mean_localization_error"],0.60),"contact_endpoint_localization_le_065":finite_le(final["contact_endpoint_mean_localization_error"],0.65),"close_endpoint_scale_error_le_060":finite_le(final["close_endpoint_median_abs_log_scale_error"],0.60),"contact_endpoint_scale_error_le_060":finite_le(final["contact_endpoint_median_abs_log_scale_error"],0.60),"count_mae_le_3":final["proposal_count_mae"]<=3.0,"false_proposals_le_3":final["false_proposals_per_image"]<=3.0}
    gate_pass,decision=decide(checks,limited)
    payload={"run_id":"R257_IMAGE_ONLY_CENTER_SCALE_DETECTOR","study_type":"sanity" if limited else "full_original_val_exploratory_detector_gate","scope":"TSRS_RSNA-Epiphysis_contrast_v1 original train/val only","clean_test_used":False,"config":vars(args),"checkpoint":str(checkpoint),"history":history,"final":final,"per_image":per_image,"gate_checks":checks,"gate_pass":gate_pass,"decision":decision,"claim_limit":"Passing allows only exploratory R258 OOF feasibility; it does not establish oracle-proposal replacement."}
    args.result_json.parent.mkdir(parents=True,exist_ok=True); args.result_json.write_text(json.dumps(payload,indent=2,default=str,allow_nan=False),encoding="utf-8"); args.proposal_csv.parent.mkdir(parents=True,exist_ok=True)
    with args.proposal_csv.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(proposal_rows[0]) if proposal_rows else ["stem"]); writer.writeheader(); writer.writerows(proposal_rows)
    print(json.dumps({"decision":payload["decision"],"final":final,"gate_checks":checks},indent=2),flush=True)


if __name__=="__main__": main()
