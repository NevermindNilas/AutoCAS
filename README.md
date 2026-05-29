# AutoCAS

**Auto-tuning Contrast Adaptive Sharpening** — no manual knob. The filter reads the input's own blur and sets its own sharpening strength.

AutoCAS is a fork of [Jamy-L/Pytorch-Contrast-Adaptive-Sharpening](https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening), an unofficial PyTorch port of AMD FidelityFX [Contrast Adaptive Sharpening (CAS)](https://github.com/GPUOpen-Effects/FidelityFX-CAS). It stays a lightweight, deterministic, **no-reference restoration filter** — the kind you bolt on as the final step of an upscaling pipeline to attenuate residual blur — and adds an automatic amount estimator on top, plus correctness and performance fixes.

## What's different from the original

- **Auto-tuning** (`amount=None`): a no-reference estimator picks the sharpening amount per image (or per region) from the input's contrast-normalized high-frequency energy. Blurry in → more sharpening; already-sharp in → left alone. The user never sets the slider.
- **Fidelity fixes** vs the reference `ffx_cas.h` (interior output now matches AMD CAS):
  - diagonal soft-min/max folds the cross result over all 9 taps (was diagonals-only),
  - clamp-to-edge (`replicate`) padding instead of zero padding (was a 1-px border halo),
  - reciprocal-space sharpness interpolation `-1/lerp(8,5,amount)` (was linear-in-value, off mid-range),
  - dropped the host-syncing range asserts.
- **fp16 NaN fix**: the core no longer produces `NaN` on near-black pixels (the old `reciprocal(mx+eps)` overflowed fp16 to `inf`, then `inf*0 = NaN`). fp16 now matches fp32 to ~1e-3.
- **~1.8× faster core**, bit-exact: pairwise `min`/`max` instead of `torch.stack(...).min(dim=0)` (which materialized a `[N,B,C,H,W]` tensor every call), plus cross-result reuse in the diagonal fold-in.

## Illustration

![Feature Illustration](data/illustration.gif)

_A blurry image sharpened at two strengths. Notice the clouds and distant mountain stay untouched — CAS's per-pixel `amp` term attenuates sharpening in low-contrast regions._

## Requirements

PyTorch is the only requirement for the filter itself:

```
pip install torch
```

The example/demo scripts additionally use `pillow` (and the original `example.py` uses `matplotlib`).

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

# Auto-tune, per-region map for mixed-focus content (n x n tile grid, bilinear)
out = contrast_adaptive_sharpening(img, amount=None, auto_tiles=6)
```

`better_diagonals=False` runs a faster 5-tap contrast estimate (skips the diagonal taps) at a small quality cost. A broadcastable tensor may also be passed as `amount` (e.g. a precomputed per-pixel map).

## How the auto-tune works

`amount = clamp(demand(blur_score), 0, AMOUNT_MAX)`, where `blur_score` is the contrast-normalized high-frequency (Laplacian) energy of the luma. Higher score → sharper input → less demand. It is **no-reference** (no ground-truth needed) and runs on the GPU with no host syncs.

There is **deliberately no noise gate**. CAS is a *sharpener*, not a denoiser. Broadband noise self-limits (it inflates the blur score, so the input reads as "already sharp" and the amount drops on its own); structured/chroma noise should be handled by an upstream denoise/deband stage, not inside the final sharpener. `AMOUNT_MAX` (default `0.9`) is the single safety ceiling.

## Calibration

The estimator's only tunable surface is the blur band `LO, HI` in `cas.py`. Fit it once to **representative input** (real upscaler output, not pristine ground-truth):

```
python calibrate.py path/to/representative_frames
# prints suggested LO, HI -> paste into cas.py
```

The shipped defaults are calibrated for clean anime 2× super-resolution output.

## Performance

Single image, RTX 3090, CUDA, batch 1 (numbers vary with GPU load):

| mode | 1080p fp16 | 4K fp16 | 1080p fp32 |
|------|-----------:|--------:|-----------:|
| core CAS (`amount` fixed) | ~594 fps | ~164 fps | ~344 fps |
| auto-tune (`amount=None`) | ~370 fps | ~104 fps | ~257 fps |

The core is bit-exact to the pre-optimization implementation; only `min`/`max` ordering changed, which never rounds.

## Tooling

- `calibrate.py` — fit the blur band `LO/HI` from a folder of frames.
- `demo.py` — full-resolution before/after on real frames, with the auto amount each frame picked.
- `demo_montage.py`, `ab_compare.py` — visual before/after montages.
- `make_grain_dataset.py` — synthetic grain ladder for robustness testing.

## Credits

Original PyTorch port by [Jamy Lafenetre](https://github.com/Jamy-L) ([Pytorch-Contrast-Adaptive-Sharpening](https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening)). Algorithm: AMD FidelityFX CAS. AutoCAS keeps the original MIT license (see `LICENSE`).
