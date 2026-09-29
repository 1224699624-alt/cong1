#!/usr/bin/env python3
"""Read-only geometry audit of RAM true-overlap member FN coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


BINS = (0.0, 1.5, 3.5, 7.5, float('inf'))
NAMES = ('boundary_1px', 'near_2to3px', 'middle_4to7px', 'deep_8pluspx')


def counts(values: np.ndarray) -> dict[str, int]:
    hist, _ = np.histogram(values, bins=BINS)
    return dict(zip(NAMES, map(int, hist)))


def summarize(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32)
    return {'n': int(arr.size), 'mean': float(arr.mean()) if arr.size else None,
            'median': float(np.median(arr)) if arr.size else None,
            'distance_bins': counts(arr)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--matches', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    matches = pd.read_csv(a.matches)
    matches = matches[matches['split'] == 'val'].copy()
    cases = sorted((a.data/'BoneSegmentation'/'masks'/'val').glob('*.npy'))
    if len(cases) != 69:
        raise ValueError(f'expected val69, got {len(cases)}')
    all_member = []
    all_overlap = []
    h0_member = []
    h0_overlap = []
    strict_member = []
    strict_overlap = []
    h0_o = []
    case_rows = []
    for path in cases:
        case = path.stem
        mask = np.load(path, mmap_mode='r') > 0
        if mask.shape[0] != 14:
            raise ValueError(f'not 14-member RAM mask: {path}')
        if case.endswith('_R'):
            mask = np.flip(mask, axis=-1)
        membership = mask.sum(axis=0)
        overlap = membership >= 2
        overlap_distance = distance_transform_edt(overlap)
        all_overlap.append(overlap_distance[overlap].astype(np.float32))
        mcase = matches[matches['case'] == case]
        one_case_h0_member = []
        one_case_h0_overlap = []
        for member in range(14):
            positive = mask[member] & overlap
            if positive.any():
                member_distance = distance_transform_edt(mask[member])
                all_member.append(member_distance[positive].astype(np.float32))
            else:
                continue
            rows = mcase[mcase['member'] == member]
            if rows.empty:
                continue
            y = rows.h0_y.to_numpy(dtype=np.int64)
            x = rows.h0_x.to_numpy(dtype=np.int64)
            if not (np.all(y >= 0) and np.all(y < mask.shape[1]) and
                    np.all(x >= 0) and np.all(x < mask.shape[2])):
                raise ValueError(f'H0 coordinate outside valid image: {case}')
            if not np.all(mask[member,y,x] & overlap[y,x]):
                raise ValueError(f'H0 coordinate not true-overlap member: {case}/{member}')
            md = member_distance[y,x].astype(np.float32)
            od = overlap_distance[y,x].astype(np.float32)
            h0_member.append(md)
            h0_overlap.append(od)
            one_case_h0_member.append(md)
            one_case_h0_overlap.append(od)
            h0_o.append(rows.o_h0.to_numpy(dtype=np.float32))
            strict = (rows.status.to_numpy() == 1) & (rows.singleton.to_numpy() == 1)
            strict_member.append(md[strict])
            strict_overlap.append(od[strict])
        c_md = np.concatenate(one_case_h0_member) if one_case_h0_member else np.empty(0)
        c_od = np.concatenate(one_case_h0_overlap) if one_case_h0_overlap else np.empty(0)
        case_rows.append({'case': case, 'h0_sample_members': int(c_md.size),
                          'h0_own_boundary_1px_fraction': float((c_md < 1.5).mean()) if c_md.size else None,
                          'h0_overlap_boundary_1px_fraction': float((c_od < 1.5).mean()) if c_od.size else None,
                          'h0_own_distance_median': float(np.median(c_md)) if c_md.size else None})
    concat = lambda x: np.concatenate(x) if x else np.empty(0,dtype=np.float32)
    h_md, h_od, h_o = concat(h0_member), concat(h0_overlap), concat(h0_o)
    if h_md.size != len(matches):
        raise RuntimeError(f'H0 row count mismatch {h_md.size} != {len(matches)}')
    o_by_own_distance = {}
    for name, lo, hi in zip(NAMES, BINS[:-1], BINS[1:]):
        selected = (h_md >= lo) & (h_md < hi)
        o_by_own_distance[name] = {'n': int(selected.sum()),
                                   'o_mean': float(h_o[selected].mean()) if selected.any() else None,
                                   'o_ge_0_5': float((h_o[selected]>=0.5).mean()) if selected.any() else None}
    output = {
        'status': 'COMPLETE', 'scope': 'RAM val69 only, original masks, frozen capped H0 matcher coordinates',
        'optimizer_updates': 0, 'test_used': False,
        'all_true_overlap_unique': summarize(concat(all_overlap)),
        'all_true_overlap_member': summarize(concat(all_member)),
        'sampled_H0_own_member': summarize(h_md),
        'sampled_H0_overlap_region': summarize(h_od),
        'strict_H0_own_member': summarize(concat(strict_member)),
        'strict_H0_overlap_region': summarize(concat(strict_overlap)),
        'o_by_sampled_H0_own_boundary_distance': o_by_own_distance,
        'limit': 'H0 coordinates are capped/matched subset (32,732 of 71,335 val member errors); distribution is descriptive and may not represent every H0 pixel.',
    }
    a.out.mkdir(parents=True)
    (a.out/'RESULT.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(case_rows).to_csv(a.out/'PER_CASE.csv',index=False)
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
