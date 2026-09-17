#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
OLD_SCREEN="r317_continuous_maturity"
NEW_SCREEN="r317_image_centers_only"
OLD_MODELS="$ROOT/outputs/pair_prior/r317_continuous"
NEW_MODELS="$ROOT/outputs/pair_prior/r317_image_centers_only"
READY="$OLD_MODELS/image_centers_seed2582_selected.pt"
LOG="$ROOT/outputs/bridge_logs/r317_image_centers_only_switch.log"

cd "$ROOT"
exec >>"$LOG" 2>&1
echo "[switch] waiting for completed image_centers seed 2582 $(date -Is)"
while [[ ! -s "$READY" ]]; do
  if ! screen -ls 2>/dev/null | grep -q "[.]${OLD_SCREEN}[[:space:]]"; then
    echo "[switch] old run exited before seed 2582 completed $(date -Is)"
    exit 1
  fi
  sleep 2
done

echo "[switch] seed 2582 complete; stopping metadata-capable run $(date -Is)"
screen -S "$OLD_SCREEN" -X quit || true
for _ in $(seq 1 30); do
  if ! pgrep -f "train_r258b_prediction_relation_prior.py.*r317_continuous" >/dev/null; then break; fi
  sleep 1
done
if pgrep -f "train_r258b_prediction_relation_prior.py.*r317_continuous" >/dev/null; then
  echo "[switch] old trainer did not terminate cleanly"
  exit 1
fi

rm -rf "$NEW_MODELS"
cp -a "$OLD_MODELS" "$NEW_MODELS"
screen -dmS "$NEW_SCREEN" bash -lc "CUDA_VISIBLE_DEVICES=0 bash run_r317_image_centers_only_remote.sh"
sleep 3
screen -ls | grep -q "[.]${NEW_SCREEN}[[:space:]]"
echo "[switch] image+centers-only run launched $(date -Is)"
