"""Train-only converged sklearn check of the fixed RAM base-vs-base+O readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import audit_ram_o_conditional_h0 as audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with np.load(args.train, allow_pickle=False) as data:
        train_base = data['base'].astype(np.float64)
        train_o = data['o'].astype(np.float64)
        train_y = data['labels'].astype(np.int8)
    with np.load(args.val, allow_pickle=False) as data:
        val_base = data['base'].astype(np.float64)
        val_o = data['o'].astype(np.float64)
        val_y = data['labels'].astype(np.int8)
        val_cases = data['cases'].astype(str)
    if len(train_y) == 0 or len(val_y) == 0 or len(val_y) != len(val_cases):
        raise ValueError('empty or malformed train/val matched rows')
    base = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, solver='lbfgs', max_iter=500, tol=1e-6, random_state=4421
    ))
    plus = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, solver='lbfgs', max_iter=500, tol=1e-6, random_state=4421
    ))
    base.fit(train_base, train_y)
    plus.fit(np.column_stack((train_base, train_o)), train_y)
    base_scores = base.predict_proba(val_base)[:, 1]
    plus_scores = plus.predict_proba(np.column_stack((val_base, val_o)))[:, 1]
    base_iter = int(base.named_steps['logisticregression'].n_iter_[0])
    plus_iter = int(plus.named_steps['logisticregression'].n_iter_[0])
    output = {
        'status': 'COMPLETE', 'scope': 'fixed readout implementation crosscheck; no segmenter updates',
        'fit_split': 'RAM train425 only', 'eval_split': 'RAM val69 only',
        'train_rows': len(train_y), 'val_rows': len(val_y), 'val_cases': len(set(val_cases)),
        'solver': 'sklearn LogisticRegression lbfgs C=1 max_iter=500 tol=1e-6; StandardScaler train-only',
        'base': {'auc': float(roc_auc_score(val_y, base_scores)),
                 'logloss': float(log_loss(val_y, base_scores)), 'iterations': base_iter,
                 'converged_before_max_iter': base_iter < 500},
        'base_plus_O': {'auc': float(roc_auc_score(val_y, plus_scores)),
                        'logloss': float(log_loss(val_y, plus_scores)), 'iterations': plus_iter,
                        'converged_before_max_iter': plus_iter < 500},
        'paired_case_bootstrap': audit.paired_case_bootstrap(
            val_y, base_scores, plus_scores, val_cases, seed=4421, repetitions=1000
        ),
        'optimizer_updates': 0, 'test_used': False,
    }
    output['delta_O_minus_base'] = {
        'auc': output['base_plus_O']['auc'] - output['base']['auc'],
        'logloss': output['base_plus_O']['logloss'] - output['base']['logloss'],
    }
    args.out.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: output[key] for key in (
        'status', 'base', 'base_plus_O', 'delta_O_minus_base', 'paired_case_bootstrap'
    )}, indent=2))


if __name__ == '__main__':
    main()
