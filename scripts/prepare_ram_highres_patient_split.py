#!/usr/bin/env python3
"""Patient-grouped RAM probe split from the actual dataset metadata.

Only RAM train/val filenames and Metadata.xlsx columns A:C are read.  Rows
outside the known train/val filenames are ignored; test images/masks/labels
are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import openpyxl


SEED = 4421
SALT = 'ram-highres-readout-20260928-v2'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--actual', type=Path, required=True)
    ap.add_argument('--old-manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    actual = args.actual.resolve()
    cases = {s: sorted(x.stem for x in (actual/'BoneSegmentation'/'masks'/s).glob('*.npy'))
             for s in ('train','val')}
    if [len(cases[s]) for s in ('train','val')] != [425,69]:
        raise ValueError('RAM train425/val69 required')
    lookup = {case:s for s,names in cases.items() for case in names}
    sheet = openpyxl.load_workbook(actual/'Metadata.xlsx',read_only=True,data_only=True)['Filtered']
    by_case = {}
    for stem,study,patient in sheet.iter_rows(min_row=2,min_col=1,max_col=3,values_only=True):
        if stem is None:
            continue
        key = str(stem).strip()
        candidates = [key] if key.endswith(('_L','_R')) else [key+'_L',key+'_R']
        for case in candidates:
            if case not in lookup:
                continue
            if patient is None:
                raise ValueError(f'missing patient ID for {case}')
            raw_id = str(patient).strip()
            token = hashlib.sha256((SALT+':'+raw_id).encode()).hexdigest()[:24]
            if case in by_case and by_case[case] != token:
                raise ValueError(f'conflicting patient group: {case}')
            by_case[case] = token
    if set(by_case) != set(lookup):
        raise ValueError(f'metadata missing {len(set(lookup)-set(by_case))} train/val cases')
    train_groups = defaultdict(list)
    for case in cases['train']:
        train_groups[by_case[case]].append(case)
    val_groups = {by_case[c] for c in cases['val']}
    if set(train_groups) & val_groups:
        raise ValueError('train/val patient IDs overlap')
    old = json.loads(args.old_manifest.read_text(encoding='utf-8'))
    old_groups = old['groups']
    old_patient = {name:{by_case[c] for c in old_groups[name]}
                   for name in ('probe_fit','probe_calibration','probe_holdout')}
    old_leaks = {
        'fit_calibration':len(old_patient['probe_fit']&old_patient['probe_calibration']),
        'fit_holdout':len(old_patient['probe_fit']&old_patient['probe_holdout']),
        'calibration_holdout':len(old_patient['probe_calibration']&old_patient['probe_holdout']),
    }
    # Stable patient-wise hash order.  Stop filling each partition only at a
    # complete patient boundary; this never splits same-patient exams.
    order = sorted(train_groups, key=lambda token: hashlib.sha256(f'{SEED}:{token}'.encode()).hexdigest())
    goals = {'probe_fit':255,'probe_calibration':85}
    groups = {'probe_fit':[],'probe_calibration':[],'probe_holdout':[]}
    for patient in order:
        name = ('probe_fit' if len(groups['probe_fit']) < goals['probe_fit'] else
                'probe_calibration' if len(groups['probe_calibration']) < goals['probe_calibration'] else
                'probe_holdout')
        groups[name].extend(sorted(train_groups[patient]))
    for name in groups:
        groups[name] = sorted(groups[name])
    if set().union(*(set(x) for x in groups.values())) != set(cases['train']):
        raise ValueError('probe split does not cover train425 exactly')
    sets = {name:{by_case[c] for c in ids} for name,ids in groups.items()}
    if any(sets[x]&sets[y] for x,y in [('probe_fit','probe_calibration'),
                                       ('probe_fit','probe_holdout'),
                                       ('probe_calibration','probe_holdout')]):
        raise ValueError('patient leakage across probe subsets')
    result = {
        'status':'LOCKABLE_PATIENT_GROUPED',
        'actual_dataset_root':str(actual),
        'metadata_sha256':sha(actual/'Metadata.xlsx'),
        'patient_group_rule':'PatientID from actual RAM Metadata.xlsx column C; SHA256 salted token stored; no raw PatientID output',
        'seed':SEED,
        'split_method':'sort hashed patient tokens by SHA256(seed:token), fill fit≈255 then calibration≈85 then holdout with whole patients',
        'groups':groups,
        'val_cases':cases['val'],
        'case_patient_token':by_case,
        'counts':{name:len(ids) for name,ids in groups.items()},
        'patient_counts':{name:len(sets[name]) for name in groups},
        'val_patient_count':len(val_groups),
        'cross_patient_overlap':{'fit_calibration':len(sets['probe_fit']&sets['probe_calibration']),
                                 'fit_holdout':len(sets['probe_fit']&sets['probe_holdout']),
                                 'calibration_holdout':len(sets['probe_calibration']&sets['probe_holdout']),
                                 'train_val':len(set(train_groups)&val_groups)},
        'prior_case_only_manifest_patient_leakage':old_leaks,
        'prior_case_only_manifest_status':'SUPERSEDED_BY_VALIDATED_PATIENT_METADATA',
        'test_files_accessed':False,
        'metadata_scope':'only columns A:C; output contains train/val mapped cases only',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'counts':result['counts'],
                      'patients':result['patient_counts'],
                      'old_patient_leakage':old_leaks,
                      'new_patient_leakage':result['cross_patient_overlap']},indent=2))


if __name__=='__main__':
    main()
