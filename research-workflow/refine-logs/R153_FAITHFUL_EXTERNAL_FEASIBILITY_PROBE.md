# R153 Faithful External Feasibility Probe

**Date**: 2026-07-02T01:48:02

## Decision

- Status: `blocked_or_high_risk`
- Recommendation: Do not install yet; resolve compiler/CUDA/network gaps or pivot architecture.
- Missing core modules: `['detectron2', 'fvcore', 'iopath', 'pycocotools']`
- NVCC available: `False`
- GCC available: `True`
- GitHub reachable: `True`

## Probe Outputs

### whoami

```text
shenzeyu
```

### pwd

```text
/home/shenzeyu/workspace/YOLO_SAM_generic_src
```

### python_version

```text
Python 3.11.15
```

### torch_cuda

```text
{"torch": "2.5.1+cu118", "cuda": "11.8", "cuda_available": true, "device_count": 2, "device_name_0": "NVIDIA GeForce RTX 4090", "capability_0": [8, 9]}
```

### module_presence

```text
{
  "torch": true,
  "torchvision": true,
  "detectron2": false,
  "fvcore": false,
  "iopath": false,
  "pycocotools": false,
  "timm": true,
  "transformers": false,
  "mmcv": false,
  "mmengine": false,
  "mmdet": false,
  "monai": false,
  "nnunetv2": false,
  "ninja": false
}
```

### nvidia_smi

```text
0, NVIDIA GeForce RTX 4090, 24564, 15, 595.71.05, 8.9
1, NVIDIA GeForce RTX 4090, 24564, 15, 595.71.05, 8.9
```

### nvcc

```text

```

### gcc

```text
/usr/bin/gcc
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
```

### gxx

```text
/usr/bin/g++
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
```

### conda_info

```text

```

stderr:

```text
sh: 1: conda: not found
```

### disk

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  1.9T  1.6T  167G  91% /
/dev/nvme0n1p2  1.9T  1.6T  167G  91% /
```

### git

```text
git version 2.43.0
```

### network_github

```text
https://github.com/facebookresearch/Mask2Former 200
https://github.com/IDEA-Research/MaskDINO 200
```
