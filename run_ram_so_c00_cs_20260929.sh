#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/ram_so_c00_cs_20260929
cd "$ROOT"
export PYTHONPATH="$ROOT/code"
export OMP_NUM_THREADS=8
PY=/root/miniconda3/bin/python
LOCK=RUN_LOCK.json
START=$(date +%s)
for ARM in C00 CS; do
  echo "START TRAIN $ARM $(date -u +%FT%TZ)" | tee -a RUN_SEQUENCE.log
  "$PY" code/run_ram_so_c00_cs_pair.py --phase train --lock "$LOCK" --arm "$ARM" --out "results/TRAIN_$ARM" > "TRAIN_$ARM.log" 2>&1
  echo "COMPLETE TRAIN $ARM $(date -u +%FT%TZ)" | tee -a RUN_SEQUENCE.log
done
echo ALL_TRAIN_COMPLETE > TRAIN_STATUS.txt
"$PY" code/run_ram_so_c00_cs_pair.py --phase eval --lock "$LOCK" --split holdout --out results/EVAL_HOLDOUT > EVAL_HOLDOUT.log 2>&1
echo HOLDOUT_COMPLETE > HOLDOUT_STATUS.txt
"$PY" - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('/root/ram_so_c00_cs_20260929')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
data={'status':'LOCKED_BEFORE_VAL','run_lock_sha256':sha(root/'RUN_LOCK.json'),
  'holdout_sha256':sha(root/'results/EVAL_HOLDOUT/OFFICIAL.json'),
  'checkpoints':{a:sha(root/'results'/f'TRAIN_{a}'/'final.pth') for a in ('C00','CS')},
  'test_used':False,'val_used_for_training_or_choice':False}
(root/'BEFORE_VAL_LOCK.json').write_text(json.dumps(data,indent=2)+'\n')
PY
"$PY" code/run_ram_so_c00_cs_pair.py --phase eval --lock "$LOCK" --split val --out results/EVAL_VAL > EVAL_VAL.log 2>&1
echo ALL_EVALUATION_COMPLETE > EVAL_STATUS.txt
END=$(date +%s)
echo "$((END-START))" > RUN_WALL_SECONDS.txt
