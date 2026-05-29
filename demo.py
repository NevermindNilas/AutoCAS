# -*- coding: utf-8 -*-
"""
Real demo of blur-only auto-tuned CAS on actual upscaler output.

Picks frames spanning the auto-amount range (sharp -> soft), runs auto CAS at
full resolution (fp16 CUDA if available), saves full-res before/after PNGs to
data/demo/, and builds a 100%-crop montage (input | CAS | diff x4) labelled with
each frame's auto-picked amount. Prints a summary table.

    python demo.py [src_folder]   (default D:\\sisr\\v3\\sr_out)
"""

import glob
import os
import sys

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import cas

CROP = 448
DEV = 'cuda' if th.cuda.is_available() else 'cpu'
DT = th.float16 if DEV == 'cuda' else th.float32


def load(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype('float32') / 255.0
    return th.from_numpy(a).permute(2, 0, 1).contiguous()


def to_img(t):
    a = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype('uint8')
    return Image.fromarray(a)


def save_rgb(p, t):
    to_img(t).save(p)


def best_coords(t, s):
    """Top-left of the s x s window with the most edge energy."""
    w = t.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    y = (t * w).sum(0, keepdim=True)[None]
    gx = (y[..., :, 1:] - y[..., :, :-1]).abs()
    gy = (y[..., 1:, :] - y[..., :-1, :]).abs()
    g = F.pad(gx, (0, 1, 0, 0)) + F.pad(gy, (0, 0, 0, 1))
    energy = F.avg_pool2d(g, s, stride=32)
    idx = int(energy.reshape(-1).argmax())
    top, left = (idx // energy.shape[-1]) * 32, (idx % energy.shape[-1]) * 32
    _, H, W = t.shape
    return min(top, H - s), min(left, W - s)


def auto_cas_full(x):
    xg = x.to(DEV, DT)[None]
    with th.no_grad():
        amt = float(cas.estimate_amount(xg).reshape(-1)[0])
        out = cas.contrast_adaptive_sharpening(xg, amount=None)[0].float().cpu()
    return out, amt


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else r'D:\sisr\v3\sr_out'
    paths = sorted(glob.glob(os.path.join(src, '*.png')) + glob.glob(os.path.join(src, '*.webp')))
    if not paths:
        raise SystemExit(f'no images in {src}')

    # score every frame's auto amount, then pick 4 spanning sharp -> soft.
    amts = []
    for p in paths:
        xg = load(p).to(DEV, DT)[None]
        with th.no_grad():
            amts.append(float(cas.estimate_amount(xg).reshape(-1)[0]))
    order = np.argsort(amts)                       # low amount (already sharp) -> high (soft)
    picks = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]]

    os.makedirs('data/demo', exist_ok=True)
    try:
        font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 16)
        fontb = ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', 16)
    except Exception:                              # noqa: BLE001
        font = fontb = ImageFont.load_default()

    rows, table = [], []
    for i in picks:
        p = paths[i]
        name = os.path.splitext(os.path.basename(p))[0]
        x = load(p)
        out, amt = auto_cas_full(x)
        save_rgb(f'data/demo/{name}__in.png', x)
        save_rgb(f'data/demo/{name}__cas_auto.png', out)
        top, left = best_coords(x, CROP)
        ci = x[:, top:top + CROP, left:left + CROP]
        co = out[:, top:top + CROP, left:left + CROP]
        cd = to_img(0.5 + 4.0 * (co - ci))
        rows.append((name, amt, to_img(ci), to_img(co), cd))
        table.append((name, amt))

    S = CROP
    LBL, HDR, GAP = 230, 28, 4
    cols = ['INPUT (sr_out)', 'CAS auto', 'DIFF x4 (grey=0)']
    W = LBL + 3 * (S + GAP)
    H = HDR + len(rows) * (S + HDR)
    cv = Image.new('RGB', (W, H), (22, 22, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((6, 6), f'blur-only auto-tuned CAS  ({DEV} {str(DT).split(".")[-1]})', font=fontb, fill=(240, 240, 240))
    for j, c in enumerate(cols):
        dr.text((LBL + j * (S + GAP) + 6, 6), c, font=fontb, fill=(240, 240, 240))
    for r, (name, amt, ci, co, cd) in enumerate(rows):
        y0 = HDR + r * (S + HDR)
        tag = 'already sharp' if amt < 0.15 else ('soft -> boosted' if amt > 0.6 else 'mild')
        dr.text((6, y0 + 6), name, font=fontb, fill=(250, 230, 120))
        dr.text((6, y0 + 28), f'auto amount = {amt:.2f}', font=font, fill=(170, 230, 250))
        dr.text((6, y0 + 48), tag, font=font, fill=(180, 180, 180))
        for j, img in enumerate((ci, co, cd)):
            cv.paste(img, (LBL + j * (S + GAP), y0 + HDR))
    cv.save('data/demo_real.png')

    print('=' * 56)
    print(f'demo on {len(paths)} frames from {src}  ({DEV}/{str(DT).split(".")[-1]})')
    print(f'auto amount across set: min {min(amts):.2f}  median {np.median(amts):.2f}  max {max(amts):.2f}')
    print('-' * 56)
    for name, amt in table:
        print(f'  {name:<28} auto={amt:.2f}')
    print('-' * 56)
    print('wrote data/demo_real.png  + full-res pairs in data/demo/')


if __name__ == '__main__':
    main()
