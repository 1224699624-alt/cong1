#!/usr/bin/env python3
"""Synthetic fold-isolation checks for R258 phase A."""

from __future__ import annotations

from collections import Counter

from train_r258_oof_center_basin_proposals import make_folds


def main()->None:
    stems=[str(i) for i in range(100)]; metadata={s:{"boneage":float((i%5)*40+30),"male":bool(i%2)} for i,s in enumerate(stems)}; folds,stratified=make_folds(stems,metadata,5,258)
    assert stratified and set(folds.tolist())==set(range(5)); counts=Counter(folds.tolist()); assert all(v==20 for v in counts.values())
    for fold in range(5):
        train={stems[i] for i in range(100) if folds[i]!=fold}; held={stems[i] for i in range(100) if folds[i]==fold}; assert not train&held and train|held==set(stems)
    print({"fold_counts":dict(counts),"stratified":stratified}); print("R258 synthetic checks: PASS")


if __name__=="__main__": main()
