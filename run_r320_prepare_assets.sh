#!/usr/bin/env bash
set -euo pipefail

EXTERNAL="/root/autodl-tmp/external"
CHECKPOINTS="$EXTERNAL/checkpoints"
PYTHON="/root/miniconda3/bin/python"
mkdir -p "$CHECKPOINTS"

# A single preparation process may write model assets or binary wheels.
exec 9>"$CHECKPOINTS/.r320_prepare.lock"
if ! flock -n 9; then
  echo "another R320 asset preparation process already owns the lock" >&2
  exit 73
fi

download_exact() {
  local name="$1"
  local expected="$2"
  local url="$3"
  local final="$CHECKPOINTS/$name"
  local partial="$final.part"
  local size=0

  [[ -f "$final" ]] && size=$(stat -c %s "$final")
  if [[ "$size" -eq "$expected" ]]; then
    echo "asset ready: $name $size"
    return
  fi

  # Preserve downloads created by the older launcher and resume them safely.
  if [[ -f "$final" && ! -f "$partial" ]]; then
    mv "$final" "$partial"
  fi
  wget -c -O "$partial" "$url"
  size=$(stat -c %s "$partial")
  if [[ "$size" -ne "$expected" ]]; then
    echo "asset size mismatch: $name $size/$expected" >&2
    exit 65
  fi
  mv "$partial" "$final"
  echo "asset ready: $name $size"
}

download_exact \
  R50_ViT-B_16.npz 461217452 \
  'https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz'
download_exact \
  vmamba_tiny_e292.pth 91649482 \
  'https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth'
download_exact \
  'causal_conv1d-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' 13531321 \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.1/causal_conv1d-1.1.1%2Bcu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
download_exact \
  'mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' 146670262 \
  'https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1%2Bcu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'

"$PYTHON" -m pip install --no-deps \
  "$CHECKPOINTS/causal_conv1d-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
  "$CHECKPOINTS/mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
"$PYTHON" -m pip install "transformers==4.39.3" "ninja==1.11.1.1"
"$PYTHON" -c 'import monai, causal_conv1d, mamba_ssm; print("Swin dependencies ready")'

echo "R320 downloadable assets are complete. MedSAM is uploaded separately."
