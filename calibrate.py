# -*- coding: utf-8 -*-
"""
Fit the blur-band calibration constants (LO, HI) from a folder of REPRESENTATIVE
images for your pipeline.

The estimator turns each image into a log blur-score. The band LO..HI should span
the score percentiles you actually see (blurry .. already-sharp). This script
prints suggested values you paste into cas.py. It does not guess your taste --
feed it real upscaler output whose look you trust, not pristine ground-truth.

    python calibrate.py path/to/image_folder [--ext png,jpg,jpeg] [--max 500]
"""

import argparse
import os

import numpy as np
import torch as th
from PIL import Image

from cas import image_stats, AMOUNT_MAX


def iter_images(folder, exts):
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.rsplit('.', 1)[-1].lower() in exts:
                yield os.path.join(root, fn)


def load_rgb(path):
    arr = np.asarray(Image.open(path).convert('RGB')).astype('float32') / 255.0
    return th.from_numpy(arr).permute(2, 0, 1).contiguous()


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--ext', default='png,jpg,jpeg,bmp,webp')
    ap.add_argument('--max', type=int, default=500, help='cap number of images')
    args = ap.parse_args()

    exts = {e.strip().lower() for e in args.ext.split(',')}
    paths = list(iter_images(args.folder, exts))[:args.max]
    if not paths:
        raise SystemExit(f'no images ({sorted(exts)}) under {args.folder}')

    log_scores = []
    for p in paths:
        try:
            x = load_rgb(p)
        except Exception as ex:                              # noqa: BLE001
            print(f'  skip {p}: {ex}')
            continue
        if min(x.shape[-2:]) < 8:
            continue
        with th.no_grad():
            ls = image_stats(x)
        log_scores.append(float(ls.reshape(-1)[0]))

    ls = np.array(log_scores)
    n = len(ls)
    if n == 0:
        raise SystemExit('no usable images')

    # LO/HI: span the bulk of the blur-score distribution (clip the tails).
    lo = pct(ls, 5)
    hi = pct(ls, 95)

    print('=' * 64)
    print(f'calibrated on {n} images from {args.folder}')
    print('-' * 64)
    print(f'  log blur-score : min {ls.min():.3f}  p5 {pct(ls,5):.3f}  '
          f'p50 {pct(ls,50):.3f}  p95 {pct(ls,95):.3f}  max {ls.max():.3f}')
    print('-' * 64)
    print('suggested constants for cas.py:')
    print(f'  LO, HI       = {lo:.2f}, {hi:.2f}   # blurry .. already-sharp band')
    print(f'  AMOUNT_MAX   = {AMOUNT_MAX}   # review by eye; lower = safer')
    print('=' * 64)
    print('note: blur-only auto-tune (CAS is a sharpener). Calibrate on real upscaler')
    print('output, not pristine ground-truth. Noise belongs to an upstream stage.')


if __name__ == '__main__':
    main()
