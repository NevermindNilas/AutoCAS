"""Locked interleaved baseline/candidate GPU benchmark. Run from repo root."""
import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import cas
from tools import baseline_cas as baseline


def percentile(a, q):
    return sorted(a)[round((len(a) - 1) * q)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='reports/benchmark.json')
    ap.add_argument('--samples', type=int, default=60)
    ap.add_argument('--baseline-only', action='store_true')
    ap.add_argument('--held-out', action='store_true')
    ap.add_argument('--representative', help='folder of source PNGs for real-image confirmation')
    args = ap.parse_args()
    torch.manual_seed(7301)
    rng = random.Random(7301)
    torch.set_num_threads(4)
    result = {'environment': {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                             'python': sys.version, 'platform': platform.platform(),
                             'baseline_sha256': hashlib.sha256(Path('tools/baseline_cas.py').read_bytes()).hexdigest()},
              'protocol': {'warmup': 20, 'samples': args.samples, 'seed': 7301,
                           'boundary': 'warm resident tensor to output; no IO; CUDA events and synchronized host wall time',
                           'acceptance': '>5% median improvement; <=5% p95/memory regression'}, 'rows': []}
    result['environment']['final_source_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                                   for p in ['cas.py','_native.py','kernels.cu','anime_selector.py','models/anime_selector.json']}
    shapes = [(360, 640, torch.float16), (720, 1280, torch.float16),
              (1080, 1920, torch.float16), (720, 1280, torch.float32)]
    if args.held_out:
        shapes = [(541, 959, torch.float16), (719, 1279, torch.float32)]
    batch = 2 if args.held_out else 1
    inputs = None
    if args.representative:
        import numpy as np
        from PIL import Image
        paths = sorted(Path(args.representative).rglob('*.png'))[:3]
        inputs = [torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).permute(2, 0, 1).float()[None]/255
                  for p in paths]
        result['inputs'] = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    for h, w, dtype in shapes:
        # Vary image content across calls while keeping both arms identical.
        if inputs:
            xs = [torch.nn.functional.interpolate(im, size=(h, w), mode='bilinear', align_corners=False)
                  .expand(batch, -1, -1, -1).contiguous().to(device='cuda', dtype=dtype) for im in inputs]
            xs[1] = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(xs[1], (1,1,1,1), mode='replicate'), 3, stride=1)
            low = torch.nn.functional.interpolate(xs[2].float(), size=(max(2,h//3), max(2,w//3)), mode='area')
            xs[2] = torch.nn.functional.interpolate(low, size=(h,w), mode='bilinear', align_corners=False).to(dtype)
            result['representative_variants'] = ['bilinear resize', '3x3 mild blur', '3x downsample and upsample']
        else:
            xs = [torch.rand(batch, 3, h, w, device='cuda', dtype=dtype) for _ in range(3)]
        for mode, kw in [('fixed', {'amount': .8}), ('legacy-auto', {'amount': None}),
                         ('tiled6', {'amount': None, 'auto_tiles': 6})]:
            arms = {'baseline': lambda x: baseline.contrast_adaptive_sharpening(x, **kw)}
            if not args.baseline_only:
                legacy_kw = dict(kw)
                if mode != 'fixed':
                    legacy_kw['auto_mode'] = 'legacy'
                arms['optimized'] = lambda x: cas.contrast_adaptive_sharpening(x, **legacy_kw)
                if mode == 'legacy-auto':
                    arms['ml'] = lambda x: cas.contrast_adaptive_sharpening(x, amount=None)
            gpu, wall, peaks, reserved = {}, {}, {}, {}
            with torch.inference_mode():
                for name, fn in arms.items():
                    for i in range(20):
                        out = fn(xs[i % 3])
                    del out
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    allocated = torch.cuda.memory_allocated()
                    out = fn(xs[0])
                    torch.cuda.synchronize()
                    peaks[name] = torch.cuda.max_memory_allocated() - allocated
                    reserved[name] = torch.cuda.max_memory_reserved()
                    del out
                    gpu[name], wall[name] = [], []
                for i in range(args.samples):
                    order = list(arms)
                    rng.shuffle(order)
                    for name in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        start.record()
                        out = arms[name](xs[i % 3])
                        end.record()
                        end.synchronize()
                        wall[name].append((time.perf_counter() - t0) * 1000)
                        gpu[name].append(start.elapsed_time(end))
                        del out
            row = {'resolution': f'{h}p', 'shape': [batch, 3, h, w], 'dtype': str(dtype), 'mode': mode, 'arms': {}}
            for name in arms:
                row['arms'][name] = {'median_ms': statistics.median(gpu[name]), 'p95_ms': percentile(gpu[name], .95),
                                     'wall_median_ms': statistics.median(wall[name]),
                                     'fps': 1000 / statistics.median(gpu[name]),
                                     'peak_extra_bytes': peaks[name], 'raw_gpu_ms': gpu[name], 'raw_wall_ms': wall[name]}
                row['arms'][name]['peak_reserved_total_bytes'] = reserved[name]
            if 'optimized' in arms:
                import numpy as np
                brng = np.random.default_rng(7301)
                for name in ['optimized', 'ml']:
                    if name not in arms:
                        continue
                    a, b = np.array(gpu['baseline']), np.array(gpu[name])
                    indices = brng.integers(args.samples, size=(4000, args.samples))
                    ratios = np.median(a[indices], axis=1)/np.median(b[indices], axis=1)
                    row['arms'][name]['speedup_ci95_paired_bootstrap'] = np.percentile(ratios, [2.5, 97.5]).tolist()
            result['rows'].append(row)
            print(h, dtype, mode, {n: round(row['arms'][n]['median_ms'], 4) for n in arms}, flush=True)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
