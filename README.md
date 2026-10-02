# AutoCAS

**Auto-tuning Contrast Adaptive Sharpening** with a tiny learned anime setting selector. Pass `amount=None` to choose a setting automatically, including a true bypass when sharpening is unsafe.

AutoCAS is a fork of [Jamy-L/Pytorch-Contrast-Adaptive-Sharpening](https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening), an unofficial PyTorch port of AMD FidelityFX [Contrast Adaptive Sharpening (CAS)](https://github.com/GPUOpen-Effects/FidelityFX-CAS). Inference needs only the input image and PyTorch. An optional NVRTC backend fuses CAS into one CUDA kernel; CPU, unsupported layouts/dtypes, and gradient-bearing calls retain the PyTorch implementation.

## What's different from the original

- **Learned anime policy:** 775 parameters, 16 features sampled at 4,096 native-resolution locations, seven actions (bypass or CAS amounts `0, .15, .3, .5, .7, .9`). Ordinary VMAF labels are constrained by NEG, reference-envelope overshoot, flat-region noise and PSNR. Validation chooses the sharpest policy within a 3% artifact-gate budget. The model is included in `models/anime_selector.json`.
- **Fused CUDA inference:** no full-frame padding or arithmetic intermediates in the CAS core. Manual-strength outputs matched the original exactly in the tested fp16/fp32 domain. Tiled estimation also fuses weighted pooling, with output differences within `7.1e-7` fp32 and `9.8e-4` fp16 on the tested blurred inputs.
- **Measured speed:** representative 1080p fp16 automatic filtering is `2.335 -> 0.244 ms` on an RTX 3090. Extra peak PyTorch allocation is `139.69 -> 12.00 MiB`. Full reproducible results and limitations: [reports/RESULTS.md](reports/RESULTS.md).

- **Legacy auto-tuning** (`auto_mode='legacy'`): a no-reference estimator picks the sharpening amount per image (or per region) from contrast-normalized high-frequency energy. The new default uses the learned classifier described above.
- **Fidelity fixes** vs the reference `ffx_cas.h` (interior output now matches AMD CAS):
  - diagonal soft-min/max folds the cross result over all 9 taps (was diagonals-only),
  - clamp-to-edge (`replicate`) padding instead of zero padding (was a 1-px border halo),
  - reciprocal-space sharpness interpolation `-1/lerp(8,5,amount)` (was linear-in-value, off mid-range),
  - dropped the host-syncing range asserts.
- **fp16 NaN fix**: the core no longer produces `NaN` on near-black pixels (the old `reciprocal(mx+eps)` overflowed fp16 to `inf`, then `inf*0 = NaN`). fp16 now matches fp32 to ~1e-3.
- The portable implementation retains pairwise soft-min/max, fused estimator convolution, luma-only upcasting and a fused output tail.

## Illustration

![Feature Illustration](data/illustration.gif)

_A blurry image sharpened at two strengths. Notice the clouds and distant mountain stay untouched — CAS's per-pixel `amp` term attenuates sharpening in low-contrast regions._

## Requirements

PyTorch is the only requirement for the filter itself:

```
pip install torch
```

Keep `cas.py`, `_native.py`, `kernels.cu`, `anime_selector.py`, and `models/` together. CUDA acceleration uses the NVRTC library shipped with CUDA-enabled PyTorch when available; no `nvcc`, C++ compiler, Triton or additional inference package is needed. Missing NVRTC emits one warning and uses PyTorch. Set `AUTOCAS_DISABLE_CUDA=1` to force the portable path. CUDA acceleration was validated on Windows/RTX 3090; other GPUs and Linux retain a fallback but have not been benchmarked here.

Training additionally uses NumPy, Pillow, OpenCV and an FFmpeg build containing libvmaf. `calibrate.py` uses Pillow and NumPy.

## Usage

Input is a tensor of shape `[H, W]`, `[C, H, W]`, or `[B, C, H, W]`, with values in `[0, 1]`. The output has the same rank.

```python
import torch
from cas import contrast_adaptive_sharpening

img = ...  # [C, H, W] or [B, C, H, W], float in [0, 1]

# Manual amount (original behaviour, amount in [0, 1])
out = contrast_adaptive_sharpening(img, amount=0.8)

# Auto-tune: estimate the amount from the image itself (no reference needed)
out = contrast_adaptive_sharpening(img, amount=None)

# Preserve the original blur-band estimator explicitly
out = contrast_adaptive_sharpening(img, amount=None, auto_mode='legacy')

# Auto-tune, per-region map for mixed-focus content (n x n tile grid, bilinear)
out = contrast_adaptive_sharpening(img, amount=None, auto_tiles=6)
```

`better_diagonals=False` runs a faster 5-tap contrast estimate (skips the diagonal taps) at a small quality cost. A broadcastable tensor may also be passed as `amount` (e.g. a precomputed per-pixel map).

`amount=0` is AMD CAS's minimum sharpening strength, **not an identity filter**. The learned policy has a separate bypass action. Negative manual amounts continue to clamp to zero. `auto_tiles>0` retains the legacy per-region estimator; the learned policy selects one setting per image. The learned policy was trained with the default diagonal estimate.

First CUDA use compiles and loads kernels; warm steady-state benchmarks exclude that cost. Before CUDA graph capture or latency-sensitive processing, initialize using the actual device/dtype:

```python
from cas import warmup
warmup(img)  # synchronizes initialization once; ordinary calls do not host-sync
```

## How the auto-tune works

The default learned selector samples gradients, curvature, local range, clipping, color and noise-related features. A `16 -> 32 -> 7` ReLU classifier chooses a discrete strength or bypass. Sampling preserves native pixel spacing; its cost stays nearly independent of frame resolution. Feature extraction, inference and sharpening stay on the input device.

Training used 12,288 controlled degradations of 1,024 local anime/digital-art originals, split by original ID before processing. A separate confirmation set excluded all development IDs/hashes. On its 3,060 examples, fp32 setting accuracy was `13.0% -> 77.4%`, balanced accuracy `16.3% -> 24.4%`, and artifact-gate violations `69.1% -> 3.0%`. Ordinary VMAF fell `84.20 -> 81.30` and NEG fell `81.07 -> 79.79`: the policy trades aggressive enhancement for artifact control. Against unprocessed inputs, it improved VMAF `79.78 -> 81.30`.

These gates do not prove universal artifact-free output. The confirmation uses synthetic degradation of illustrations, not human ratings or a real anime video/upscaler benchmark. fp16/fp32 setting agreement was 99.8%. No temporal smoothing is applied.

The following blur-band description applies to `auto_mode='legacy'`, tiled estimation and calibration helpers:

`amount = clamp(demand(blur_score), 0, AMOUNT_MAX)`, where `blur_score` is the contrast-normalized high-frequency (Laplacian) energy of the luma. Higher score → sharper input → less demand. It is **no-reference** (no ground-truth needed) and runs on the GPU with no host syncs.

The legacy estimator has no explicit noise gate. Structured/chroma noise should still be handled upstream; neither policy denoises images. `AMOUNT_MAX` defaults to `0.9`.

## Calibration

The legacy estimator's tunable surface is the blur band `LO, HI` in `cas.py`. Fit it to **representative input** (real upscaler output, not pristine ground-truth):

```
python calibrate.py path/to/representative_frames
# prints suggested LO, HI -> paste into cas.py
```

The legacy defaults were calibrated for clean anime 2× super-resolution output. They do not control the learned model.

## Performance

RTX 3090, CUDA, batch 1 RGB, warm resident tensors, 60 randomized/interleaved samples,
representative anime images with clean/mild-blur/downsample variants. Times include
automatic estimation and filtering, and exclude decoding, encoding and initialization.

| Automatic mode | Before (ms) | Optimized legacy (ms) | Learned final (ms) | Final speedup |
|---|---:|---:|---:|---:|
| 360p fp16 | 0.717 | 0.429 | 0.116 | 6.2× |
| 720p fp16 | 1.153 | 0.503 | 0.166 | 6.9× |
| 1080p fp16 | 2.335 | 0.967 | 0.244 | 9.6× |
| 720p fp32 | 1.694 | 0.501 | 0.173 | 9.8× |

Fixed-strength CAS and tiled legacy mode also improved. [Detailed results](reports/RESULTS.md)
include all modes, p95, host wall latency, allocation counters, confidence intervals,
held-out shapes, quality results and reproduction commands.

## Tooling

- `calibrate.py` — fit the blur band `LO/HI` from a folder of representative frames.
- `tools/build_dataset.py` — generate grouped splits, degradations and VMAF/NEG labels.
- `tools/train_selector.py` — train and choose a compact classifier using validation only.
- `tools/evaluate_runtime.py` — verify the shipped fp16/fp32 runtime on separate originals.
- `tools/benchmark.py` — compare the exact original, optimized legacy and learned paths.
- `tests/test_cas.py` — differential, gradient, stream, model and CUDA graph checks.

## Credits

Original PyTorch port by [Jamy Lafenetre](https://github.com/Jamy-L) ([Pytorch-Contrast-Adaptive-Sharpening](https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening)). Algorithm: AMD FidelityFX CAS. AutoCAS keeps the original MIT license (see `LICENSE`).
