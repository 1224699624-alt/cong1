#!/usr/bin/env python3
"""Test whether image-only local contrast adds to frozen C0 summaries plus O.

The H0/control pairs were fixed by an earlier audit.  This is a train425-fit,
val69-evaluate diagnostic readout; it never modifies a segmentation model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import audit_ram_o_conditional_h0 as prior_audit


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_extra(table: pd.DataFrame, split: str, existing: dict[str, np.ndarray]) -> np.ndarray:
    selected = table[(table['split'] == split) & (table['status'] == 1) & (table['singleton'] == 1)]
    pieces = []
    expected_cases = []
    expected_labels = []
    expected_o = []
    for case, group in selected.groupby('case', sort=False):
        n = len(group)
        pos = group[['h0_local_residual', 'h0_local_z']].to_numpy(dtype=np.float64)
        neg = group[['control_local_residual', 'control_local_z']].to_numpy(dtype=np.float64)
        pieces.extend((pos, neg))
        expected_cases.extend([case] * (2*n))
        expected_labels.extend([1] * n + [0] * n)
        expected_o.extend(group.o_h0.to_list() + group.o_control.to_list())
    if not pieces:
        raise ValueError(f'no strict matched rows for {split}')
    x = np.vstack(pieces)
    if not np.array_equal(np.asarray(expected_cases), existing['cases'].astype(str)):
        raise ValueError(f'case/order mismatch against frozen C0 traces: {split}')
    if not np.array_equal(np.asarray(expected_labels, dtype=np.int8), existing['labels']):
        raise ValueError(f'label/order mismatch against frozen C0 traces: {split}')
    o_error = float(np.max(np.abs(np.asarray(expected_o) - existing['o'][:, 0])))
    if o_error > 1e-6:
        raise ValueError(f'O/order mismatch against frozen C0 traces: {split}, max {o_error}')
    x[:, 1] = np.clip(x[:, 1], -5.0, 5.0)  # fixed robust cap, independent of labels/val
    if not np.isfinite(x).all():
        raise ValueError(f'nonfinite local contrast: {split}')
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--matched', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    table = pd.read_csv(args.matched)
    required = {'split','case','status','singleton','o_h0','o_control',
                'h0_local_residual','h0_local_z','control_local_residual','control_local_z'}
    if not required.issubset(table.columns):
        raise ValueError(f'missing matched columns: {required-set(table.columns)}')
    traces = {}
    for split, path in (('train', args.train), ('val', args.val)):
        with np.load(path, allow_pickle=False) as z:
            traces[split] = {key: z[key] for key in ('base', 'o', 'labels', 'cases')}
    extra = {split: build_extra(table, split, traces[split]) for split in ('train','val')}
    baseline = {}
    improved = {}
    for output, new in ((baseline, False), (improved, True)):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=500, tol=1e-6, random_state=4421))
        train_x = np.column_stack((traces['train']['base'], traces['train']['o']))
        val_x = np.column_stack((traces['val']['base'], traces['val']['o']))
        if new:
            train_x = np.column_stack((train_x, extra['train']))
            val_x = np.column_stack((val_x, extra['val']))
        model.fit(train_x, traces['train']['labels'])
        score = model.predict_proba(val_x)[:, 1]
        output.update({'auc': float(roc_auc_score(traces['val']['labels'], score)),
                       'logloss': float(log_loss(traces['val']['labels'], score)),
                       'iterations': int(model.named_steps['logisticregression'].n_iter_[0]),
                       'converged_before_max_iter': int(model.named_steps['logisticregression'].n_iter_[0]) < 500,
                       'scores': score})
    # The baseline is intentionally an exact reproduction of the earlier
    # converged base+O readout; otherwise the new feature comparison is invalid.
    if abs(baseline['auc'] - 0.9401197539437731) > 2e-6 or abs(baseline['logloss'] - 0.3569713205955205) > 2e-6:
        raise ValueError(f'base+O readout did not reproduce locked prior audit: {baseline["auc"]}, {baseline["logloss"]}')
    raw_boot = prior_audit.paired_case_bootstrap(
        traces['val']['labels'], baseline['scores'], improved['scores'],
        traces['val']['cases'].astype(str), seed=4421, repetitions=1000)
    boot = {
        'cases': raw_boot['cases'], 'repetitions': raw_boot['repetitions'],
        'seed': raw_boot['seed'],
        'delta_auc_local_minus_baseO': raw_boot['delta_auc_o_minus_base'],
        'delta_logloss_local_minus_baseO': raw_boot['delta_logloss_o_minus_base'],
    }
    result = {
        'status': 'COMPLETE', 'scope': 'read-only conditional local-contrast readout; no segmentation training',
        'fit_split': 'RAM train425', 'eval_split': 'RAM val69',
        'strict_pairs_train': len(traces['train']['labels'])//2,
        'strict_pairs_val': len(traces['val']['labels'])//2,
        'features_added': ['image-only Gaussian sigma7 local_residual', 'image-only local_z clipped[-5,5]'],
        'solver': 'train-only StandardScaler + sklearn LogisticRegression lbfgs C1 max500 tol1e-6',
        'base_plus_O': {key:value for key,value in baseline.items() if key!='scores'},
        'base_plus_O_plus_local': {key:value for key,value in improved.items() if key!='scores'},
        'delta': {'auc': improved['auc']-baseline['auc'],
                  'logloss': improved['logloss']-baseline['logloss']},
        'paired_case_bootstrap': boot,
        'input_sha256': {str(p):digest(p) for p in (args.train,args.val,args.matched)},
        'optimizer_updates': 0, 'gpu_used': False, 'test_used': False,
        'limitation': 'matched diagnostic subpopulation only; raw image and full nnU-Net features are not exhaustively conditioned on; no member recovery or segmentation metric is inferred',
    }
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','base_plus_O','base_plus_O_plus_local','delta','paired_case_bootstrap')},indent=2))


if __name__ == '__main__':
    main()
