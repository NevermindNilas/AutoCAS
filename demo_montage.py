# -*- coding: utf-8 -*-
"""
Visual demo: render INPUT | AUTO-SHARPENED | DIFF(x4) crops for a few cases that
show (a) core CAS, (b) auto-tune picking per image, (c) the noise gate backing
off on grain and leaking correlated film grain. Saves data/demo_montage.png.
"""

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import cas
from make_grain_dataset import add_gaussian, add_film

S = 256  # crop size


def load(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype('float32') / 255.0
    return th.from_numpy(a).permute(2, 0, 1).contiguous()


def blur(t):
    ker = th.tensor([1., 2., 1.]).view(1, 1, 1, 3).repeat(3, 1, 1, 1) / 4.0
    t = t[None]
    t = F.conv2d(t, ker, padding=(0, 1), groups=3)
    t = F.conv2d(t, ker.permute(0, 1, 3, 2), padding=(1, 0), groups=3)
    return t[0]


def best_crop(t, s=S):
    """Pick the s x s window with the most edge energy so the effect is visible."""
    w = t.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    y = (t * w).sum(0, keepdim=True)[None]
    gx = (y[..., :, 1:] - y[..., :, :-1]).abs()
    gy = (y[..., 1:, :] - y[..., :-1, :]).abs()
    g = F.pad(gx, (0, 1, 0, 0)) + F.pad(gy, (0, 0, 0, 1))
    energy = F.avg_pool2d(g, s, stride=16)               # coarse search grid
    idx = int(energy.reshape(-1).argmax())
    cols = energy.shape[-1]
    top, left = (idx // cols) * 16, (idx % cols) * 16
    _, H, W = t.shape
    top, left = min(top, H - s), min(left, W - s)
    return t[:, top:top + s, left:left + s]


def to_img(t):
    a = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype('uint8')
    return Image.fromarray(a)


def diff_img(inp, out):
    d = (0.5 + 4.0 * (out - inp))                        # amplify x4 around grey
    return to_img(d)


def amount_of(crop):
    return float(cas.estimate_amount(crop[None]).reshape(-1)[0])


def main():
    g = th.Generator().manual_seed(0)
    tam = best_crop(load(r'D:\sisr\v3\sr_out\tamako_x2.png'))
    lov = best_crop(load(r'D:\sisr\v3\sr_out\lovingvincent_x2.png'))
    kod = best_crop(blur(load('data/kodim13.png')))

    # (row label, input crop, output crop, amount text)
    rows = []
    rows.append(('kodim blur / manual 0.8', kod,
                 cas.contrast_adaptive_sharpening(kod, 0.8), 'amt=0.80 (manual)'))
    rows.append(('anime soft (tamako)', tam,
                 cas.contrast_adaptive_sharpening(tam, amount=None), f'auto={amount_of(tam):.2f}'))
    rows.append(('anime sharp (lovingvincent)', lov,
                 cas.contrast_adaptive_sharpening(lov, amount=None), f'auto={amount_of(lov):.2f}'))
    tg = add_gaussian(tam, 16 / 255, g).clamp(0, 1)
    rows.append(('tamako + gauss16 noise', tg,
                 cas.contrast_adaptive_sharpening(tg, amount=None), f'auto={amount_of(tg):.2f} (blur self-limits)'))
    tf = add_film(tam, 12 / 255, 1, g).clamp(0, 1)
    rows.append(('tamako + film grain', tf,
                 cas.contrast_adaptive_sharpening(tf, amount=None), f'auto={amount_of(tf):.2f} (handle upstream)'))

    try:
        font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 16)
        fontb = ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', 16)
    except Exception:                                    # noqa: BLE001
        font = fontb = ImageFont.load_default()

    LBL, HDR, GAP = 200, 26, 4
    cols = ['INPUT', 'AUTO-SHARPENED', 'DIFF x4 (grey=0)']
    W = LBL + 3 * (S + GAP)
    H = HDR + len(rows) * (S + HDR)
    canvas = Image.new('RGB', (W, H), (24, 24, 24))
    dr = ImageDraw.Draw(canvas)
    for j, c in enumerate(cols):
        dr.text((LBL + j * (S + GAP) + 6, 5), c, font=fontb, fill=(240, 240, 240))

    for i, (lab, inp, out, amt) in enumerate(rows):
        y0 = HDR + i * (S + HDR)
        dr.text((6, y0 + 6), lab, font=fontb, fill=(250, 230, 120))
        dr.text((6, y0 + 28), amt, font=font, fill=(170, 220, 250))
        for j, img in enumerate((to_img(inp), to_img(out), diff_img(inp, out))):
            canvas.paste(img, (LBL + j * (S + GAP), y0 + HDR))

    canvas.save('data/demo_montage.png')
    print('wrote data/demo_montage.png', canvas.size)


if __name__ == '__main__':
    main()
