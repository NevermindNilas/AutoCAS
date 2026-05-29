# -*- coding: utf-8 -*-
"""
A/B verification for the CAS fixes + auto-tune.

Runs the ORIGINAL (pre-fix) algorithm against the FIXED one and the AUTO modes
on a blurred Kodak image, saves side-by-side PNGs to data/, and prints the
numeric deltas that the audit flagged (border halo, mid-range peak shift,
auto-picked amount). Uses PIL + numpy only (no matplotlib needed).

    python ab_compare.py [path/to/image.png]
"""

import sys

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

from cas import contrast_adaptive_sharpening, estimate_amount

EPSILON = 1e-6


# --- original (pre-fix) algorithm, verbatim logic, for comparison only --------
def _min(tl):
    return th.stack(tl).min(dim=0)[0]


def _max(tl):
    return th.stack(tl).max(dim=0)[0]


def cas_original(x, amount=0.8, better_diagonals=True, pad_mode='constant'):
    x_padded = F.pad(x, pad=(1, 1, 1, 1), mode=pad_mode)  # default zero pad (the bug)
    b = x_padded[..., :-2, 1:-1]
    d = x_padded[..., 1:-1, :-2]
    e = x_padded[..., 1:-1, 1:-1]
    f = x_padded[..., 1:-1, 2:]
    h = x_padded[..., 2:, 1:-1]
    if better_diagonals:
        a = x_padded[..., :-2, :-2]
        c = x_padded[..., :-2, 2:]
        g = x_padded[..., 2:, :-2]
        i = x_padded[..., 2:, 2:]
    cross = (b, d, e, f, h)
    mn = _min(cross)
    mx = _max(cross)
    if better_diagonals:
        diag = (a, c, g, i)                               # diagonals only (the bug)
        mn = mn + _min(diag)
        mx = mx + _max(diag)
    inv_mx = th.reciprocal(mx + EPSILON)
    if better_diagonals:
        amp = inv_mx * th.minimum(mn, (2 - mx))
    else:
        amp = inv_mx * th.minimum(mn, (1 - mx))
    amp = th.sqrt(amp)
    w = -amp * (amount * (1 / 5 - 1 / 8) + 1 / 8)         # linear-in-value lerp (the bug)
    div = th.reciprocal(1 + 4 * w)
    output = ((b + d + f + h) * w + e) * div
    return output.clamp(0, 1)


# --- io helpers ---------------------------------------------------------------
def load_rgb(path):
    arr = np.asarray(Image.open(path).convert('RGB')).astype('float32') / 255.0
    return th.from_numpy(arr).permute(2, 0, 1).contiguous()        # [3, H, W]


def save_rgb(path, t):
    a = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype('uint8')
    Image.fromarray(a).save(path)


def save_gray(path, t2d):
    m = float(t2d.max()) + 1e-9
    a = ((t2d / m).clamp(0, 1).numpy() * 255.0 + 0.5).astype('uint8')
    Image.fromarray(a, mode='L').save(path)


def blur(t):                                                       # separable binomial
    ker = th.tensor([1., 2., 1.]).view(1, 1, 1, 3).repeat(3, 1, 1, 1) / 4.0
    t = t[None]
    t = F.conv2d(t, ker, padding=(0, 1), groups=3)
    t = F.conv2d(t, ker.permute(0, 1, 3, 2), padding=(1, 0), groups=3)
    return t[0]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/kodim13.png'
    im = load_rgb(path)
    sharp_in = im
    blurry = blur(im)
    _, Hh, Ww = blurry.shape

    orig = cas_original(blurry, 0.8)
    fixed = contrast_adaptive_sharpening(blurry, 0.8)
    auto_s = contrast_adaptive_sharpening(blurry, amount=None)
    auto_t = contrast_adaptive_sharpening(blurry, amount=None, auto_tiles=6)

    save_rgb('data/ab_blurry.png', blurry)
    save_rgb('data/ab_original_0.80.png', orig)
    save_rgb('data/ab_fixed_0.80.png', fixed)
    save_rgb('data/ab_auto_scalar.png', auto_s)
    save_rgb('data/ab_auto_tiled.png', auto_t)

    # --- finding 2 (isolated): only the pad mode changes (zero vs replicate),
    # everything else identical, so any diff is purely the padding bug.
    pad_diff = (cas_original(blurry, 0.8, pad_mode='constant')
                - cas_original(blurry, 0.8, pad_mode='replicate')).abs().sum(0)
    border = th.ones_like(pad_diff, dtype=th.bool)
    border[1:-1, 1:-1] = False
    pad_border_max = float(pad_diff[border].max())
    pad_interior_max = float(pad_diff[~border].max())

    # --- full orig vs fixed (all four fixes combined) for the visual diff.
    diff = (orig - fixed).abs().sum(0)                             # [H, W]
    save_gray('data/ab_diff_original_vs_fixed.png', diff)
    full_max = float(diff.max())
    full_mean = float(diff.mean())

    # --- finding 3: mid-range peak shift. Same image, amount=0.5, orig vs fixed.
    o50 = cas_original(blurry, 0.5)
    f50 = contrast_adaptive_sharpening(blurry, 0.5)
    d50 = (o50 - f50).abs()
    # exclude border so this isolates the peak/diagonal change, not the padding.
    inner = d50[..., 1:-1, 1:-1]
    peak_max = float(inner.max())
    peak_mean = float(inner.mean())

    # --- auto-picked amounts
    amt_scalar = float(estimate_amount(blurry[None]).reshape(-1)[0])
    amt_map = estimate_amount(blurry[None], tiles=6).reshape(-1)
    # auto on the already-SHARP input should pick a smaller amount than on blurry.
    amt_sharp = float(estimate_amount(sharp_in[None]).reshape(-1)[0])

    print('=' * 70)
    print(f'image            : {path}  ({Hh}x{Ww})')
    print('-' * 70)
    print('FINDING 2  zero-pad vs replicate (ISOLATED: only pad mode differs)')
    print(f'  border-ring max |diff| : {pad_border_max:.4f}  (~{pad_border_max*255:.1f}/255)')
    print(f'  interior  max |diff|   : {pad_interior_max:.4f}  (should be ~0)')
    print(f'  --> halo confined to the 1px border ring')
    print('-' * 70)
    print('COMBINED  all four fixes (orig vs fixed, amount=0.80)')
    print(f'  whole-image max |diff| : {full_max:.4f}  (~{full_max*255:.1f}/255)')
    print(f'  whole-image mean |diff|: {full_mean:.5f}')
    print('-' * 70)
    print('FINDING 1+3  diagonal fold-in + reciprocal lerp (interior, amount=0.50)')
    print(f'  interior max |diff|    : {peak_max:.4f}  (~{peak_max*255:.1f}/255)')
    print(f'  interior mean |diff|   : {peak_mean:.5f}')
    print('-' * 70)
    print('AUTO-TUNE  estimate_amount()')
    print(f'  blurry  -> scalar amount : {amt_scalar:.3f}')
    print(f'  sharp   -> scalar amount : {amt_sharp:.3f}   (should be < blurry)')
    print(f'  blurry  -> tiled  amount : min {float(amt_map.min()):.3f} '
          f'/ mean {float(amt_map.mean()):.3f} / max {float(amt_map.max()):.3f}')
    print('=' * 70)
    print('wrote: data/ab_blurry.png, ab_original_0.80.png, ab_fixed_0.80.png,')
    print('       ab_auto_scalar.png, ab_auto_tiled.png, ab_diff_original_vs_fixed.png')


if __name__ == '__main__':
    main()
