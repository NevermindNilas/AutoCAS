# -*- coding: utf-8 -*-
"""
Build a mock dataset with VARIED grain from clean source frames, so the CAS
auto-tune noise gate has real noise to react to (the real sr_out is noise-free).

For each source image: center-crop a tile (crop preserves native-resolution
noise statistics, unlike a resize), then emit several grain variants:
clean, Gaussian (4 levels), Poisson shot noise, and correlated film grain.

    python make_grain_dataset.py [--src D:\\sisr\\v3\\sr_out]
                                 [--out D:\\sisr\\mock_grain\\sr_out]
                                 [--crop 1024] [--seed 0]
"""

import argparse
import glob
import os

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image


def load(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype('float32') / 255.0
    return th.from_numpy(a).permute(2, 0, 1).contiguous()


def save(p, t):
    a = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype('uint8')
    Image.fromarray(a).save(p)


def center_crop(t, s):
    _, H, W = t.shape
    s = min(s, H, W)
    top, left = (H - s) // 2, (W - s) // 2
    return t[:, top:top + s, left:left + s]


def luma(t):
    w = t.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return (t * w).sum(0, keepdim=True)


def correlate(n, radius):
    """Blur white noise into clumps (film-grain-like), then restore its std."""
    if radius <= 0:
        return n
    k = 2 * radius + 1
    x = n[None]
    x = F.avg_pool2d(F.pad(x, (radius, radius, 0, 0), mode='reflect'), (1, k), stride=1)
    x = F.avg_pool2d(F.pad(x, (0, 0, radius, radius), mode='reflect'), (k, 1), stride=1)
    x = x[0]
    return x / (x.std() + 1e-6) * (n.std() + 1e-6)


def add_gaussian(t, sigma, gen):
    return t + th.randn(t.shape, generator=gen) * sigma


def add_poisson(t, scale, gen):
    lam = t.clamp(0, 1) * scale          # lower scale = fewer photons = noisier
    return th.poisson(lam, generator=gen) / scale


def add_film(t, sigma, radius, gen):
    n = correlate(th.randn(t.shape, generator=gen), radius)
    mod = (4 * luma(t) * (1 - luma(t))).clamp(0, 1)   # grain strongest in midtones
    return t + n * sigma * mod


# (label, fn) — fn(tile, gen) -> noisy tile
GRAINS = [
    ('clean',   lambda t, g: t),
    ('gauss04', lambda t, g: add_gaussian(t, 4 / 255, g)),
    ('gauss08', lambda t, g: add_gaussian(t, 8 / 255, g)),
    ('gauss16', lambda t, g: add_gaussian(t, 16 / 255, g)),
    ('gauss24', lambda t, g: add_gaussian(t, 24 / 255, g)),
    ('poisson', lambda t, g: add_poisson(t, 60.0, g)),
    ('film12',  lambda t, g: add_film(t, 12 / 255, 1, g)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=r'D:\sisr\v3\sr_out')
    ap.add_argument('--out', default=r'D:\sisr\mock_grain\sr_out')
    ap.add_argument('--crop', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ext', default='png,webp,jpg,jpeg')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    exts = tuple('.' + e.strip().lower() for e in args.ext.split(','))
    srcs = sorted(p for p in glob.glob(os.path.join(args.src, '*')) if p.lower().endswith(exts))
    if not srcs:
        raise SystemExit(f'no images under {args.src}')

    gen = th.Generator().manual_seed(args.seed)
    n_out = 0
    for p in srcs:
        base = os.path.splitext(os.path.basename(p))[0]
        tile = center_crop(load(p), args.crop)
        for label, fn in GRAINS:
            save(os.path.join(args.out, f'{base}__{label}.png'), fn(tile, gen))
            n_out += 1
        print(f'  {base}: {len(GRAINS)} variants  ({tile.shape[-2]}x{tile.shape[-1]})')

    print(f'wrote {n_out} images from {len(srcs)} sources -> {args.out}')


if __name__ == '__main__':
    main()
