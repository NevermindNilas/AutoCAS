"""Small learned, no-reference anime CAS setting classifier (16 -> 32 -> 7).

Samples 4096 locations at native pixel spacing, independent of frame resolution.
Negative strength means bypass. Training dependencies never enter inference.
"""
from functools import lru_cache
import json
from pathlib import Path
import threading

import torch

FEATURE_NAMES = ['luma', 'luma_squared', 'dx', 'dy', 'lap_abs', 'lap_squared',
                 'range', 'normalized_lap', 'range_squared', 'diagonal_gradient',
                 'dark_fraction', 'light_fraction', 'edge_fraction', 'flat_lap',
                 'chroma', 'highpass9']
AMOUNTS = [-1., 0., .15, .3, .5, .7, .9]
_model_lock = threading.RLock()


def extract_features(x, native=True):
    if native:
        from _native import features
        out = features(x)
        if out is not None:
            return out
    B, C, H, W = x.shape
    grid = torch.arange(64, device=x.device)
    yy = ((grid * 2 + 1) * H // 128).view(64, 1).expand(64, 64).reshape(-1)
    xx = ((grid * 2 + 1) * W // 128).view(1, 64).expand(64, 64).reshape(-1)

    def samples(dy, dx):
        z = x[:, :, (yy + dy).clamp(0, H-1), (xx + dx).clamp(0, W-1)].float()
        if C >= 3:
            return z[:, 0] * .299 + z[:, 1] * .587 + z[:, 2] * .114
        return z.mean(1)
    e, u, d, l, r = [samples(*p) for p in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]]
    a, c, g, i = [samples(*p) for p in [(-1, -1), (-1, 1), (1, -1), (1, 1)]]
    hi = torch.maximum(torch.maximum(torch.maximum(torch.maximum(e, u), d), l), r)
    lo = torch.minimum(torch.minimum(torch.minimum(torch.minimum(e, u), d), l), r)
    local_range = hi - lo
    lap = (u + d + l + r - 4 * e).abs()
    if C >= 3:
        z = x[:, :3, yy, xx].float()
        chroma = (z[:, 0] - z[:, 1]).abs() + (z[:, 1] - z[:, 2]).abs()
    else:
        chroma = torch.zeros_like(e)
    maps = [e, e*e, (r-l).abs()*.5, (d-u).abs()*.5, lap, lap*lap,
            local_range, lap/(local_range+.01), local_range*local_range,
            ((i-a).abs()+(g-c).abs())*.25, (e<.02).float(), (e>.98).float(),
            (local_range>.12).float(), torch.where(local_range<.04, lap, 0),
            chroma, (e-(a+u+c+l+e+r+g+d+i)/9).abs()]
    return torch.stack([v.mean(1) for v in maps], 1)


@lru_cache(maxsize=1)
def model_data():
    return json.loads(Path(__file__).with_name('models').joinpath('anime_selector.json').read_text())


@lru_cache(maxsize=16)
def model_tensors(device):
    if torch.device(device).type == 'cuda':
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Warm up the AutoCAS anime selector before CUDA graph capture')
    data = model_data()
    params = [data['mean'], data['inverse_std'], data['w1'], data['b1'], data['w2'], data['b2'], data['amounts']]
    tensors = [torch.tensor(p, dtype=torch.float32, device=device) for p in params]
    packed = torch.cat([p.flatten() for p in tensors]).contiguous()
    ready, producer = None, None
    if packed.is_cuda:
        event = torch.cuda.Event()
        producer = torch.cuda.current_stream(packed.device).cuda_stream
        event.record(torch.cuda.current_stream(packed.device))
        ready = {'event': event, 'complete': False}
    return tensors, packed, ready, producer


def estimate_anime_amount(x):
    # A fixed feature cost and <1k learned parameters keep classification cheap.
    features = extract_features(x)
    with _model_lock:
        tensors, packed, ready, producer = model_tensors(str(x.device))
    if ready is not None:
        stream = torch.cuda.current_stream(x.device)
        with torch.cuda.device(x.device):
            capturing = torch.cuda.is_current_stream_capturing()
        if not ready['complete']:
            if capturing and stream.cuda_stream != producer:
                raise RuntimeError('Complete AutoCAS selector warmup before CUDA graph capture')
            if not capturing:
                if ready['event'].query():
                    ready['complete'] = True
                elif stream.cuda_stream != producer:
                    stream.wait_event(ready['event'])
    from _native import eligible, classify
    if eligible(x):
        out = classify(x, features, packed, len(AMOUNTS))
        if out is not None:
            return out
    mean, invstd, w1, b1, w2, b2, amounts = tensors
    hidden = torch.relu(torch.nn.functional.linear((features-mean)*invstd, w1, b1))
    logits = torch.nn.functional.linear(hidden, w2, b2)
    return amounts[logits.argmax(1)].to(x.dtype).view(-1, 1, 1, 1)
