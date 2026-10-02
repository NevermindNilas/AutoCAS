"""Generate original-group splits, controlled degradations and constrained VMAF labels.

No images are redistributed. Retains manifests and numeric training data. FFmpeg
must include libvmaf with ordinary and NEG v0.6.1 models. Run from repo root.
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anime_selector import extract_features, AMOUNTS, FEATURE_NAMES
from tools.baseline_cas import contrast_adaptive_sharpening, estimate_amount

SEED = 94137
DEGRADATIONS = ['clean', 'blur03', 'blur06', 'blur10', 'blur16', 'down2', 'down3',
                'blur_noise003', 'blur_noise01', 'jpeg65', 'jpeg90', 'oversharpen']


def degrade(ref, name, rng):
    if name == 'clean':
        return ref.copy()
    if name.startswith('blur') and 'noise' not in name:
        return cv2.GaussianBlur(ref, (0, 0), int(name[-2:])/10)
    if name.startswith('down'):
        n = int(name[-1])
        small = cv2.resize(ref, (ref.shape[1]//n, ref.shape[0]//n), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, ref.shape[1::-1], interpolation=cv2.INTER_CUBIC).clip(0, 1)
    if 'noise' in name:
        sigma = .003 if name.endswith('003') else .01
        return (cv2.GaussianBlur(ref, (0, 0), .6) + rng.normal(0, sigma, ref.shape)).clip(0, 1).astype('float32')
    if name.startswith('jpeg'):
        bgr = np.rint(ref[..., ::-1]*255).astype('uint8')
        _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, int(name[4:])])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].astype('float32')/255
    blurred = cv2.GaussianBlur(ref, (0, 0), 1.)
    return (ref + .6*(ref-blurred)).clip(0, 1)


def luma(rgb):
    return rgb[..., 0]*.299 + rgb[..., 1]*.587 + rgb[..., 2]*.114


def artifact_metrics(rgb, ref):
    y, yr = luma(rgb), luma(ref)
    lo = cv2.erode(yr, np.ones((3, 3), 'uint8'))
    hi = cv2.dilate(yr, np.ones((3, 3), 'uint8'))
    halo = float(np.maximum(np.maximum(y-hi-.005, lo-y-.005), 0).mean())
    flat = hi-lo < .025
    residual = np.abs(y-cv2.blur(y, (3, 3)))
    noise = float(residual[flat].mean()) if flat.any() else 0.
    mse = float(np.mean((rgb-ref)**2))
    psnr = -10*np.log10(max(mse, 1e-12))
    return halo, noise, psnr


def to_yuv(rgb):
    u8 = np.rint(rgb*255).clip(0, 255).astype('uint8')
    return cv2.cvtColor(u8, cv2.COLOR_RGB2YUV_I420).tobytes()


def run_vmaf(ffmpeg, ref_file, dis_file, size, log):
    vf = (f"[1:v][0:v]libvmaf=model='version=vmaf_v0.6.1\\:name=standard\\:motion.motion_force_zero=true|"
          f"version=vmaf_v0.6.1neg\\:name=neg\\:motion.motion_force_zero=true'"
          f":n_threads=4:log_fmt=json:log_path={log.as_posix()}")
    cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
           '-s', f'{size}x{size}', '-r', '24', '-i', str(ref_file), '-f', 'rawvideo', '-pix_fmt',
           'yuv420p', '-s', f'{size}x{size}', '-r', '24', '-i', str(dis_file), '-lavfi', vf, '-f', 'null', '-']
    subprocess.run(cmd, check=True)
    data = json.loads(log.read_text())
    return np.array([[f['metrics']['standard'], f['metrics']['neg']] for f in data['frames']], dtype='float32')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--ffmpeg', required=True)
    ap.add_argument('--sources', type=int, default=320)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--output', default='artifacts/dataset')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    # Group crop suffixes before sampling/splitting. Select one crop per original.
    groups = {}
    for p in sorted(Path(args.source).rglob('*.png')):
        key = p.stem.rsplit('_', 1)[0]
        groups.setdefault(key, p)
    ids = sorted(groups)
    rng.shuffle(ids)
    ids = ids[:args.sources]
    manifest, feats, artifacts, legacy, cases = [], [], [], [], []
    ref_path, dis_path = out/'reference.yuv', out/'candidates.yuv'
    start = time.time()
    with ref_path.open('wb') as ref_file, dis_path.open('wb') as dis_file, torch.inference_mode():
        for j, key in enumerate(ids):
            p = groups[key]
            rgb = np.asarray(Image.open(p).convert('RGB')).astype('float32')/255
            if min(rgb.shape[:2]) < args.size:
                raise ValueError(f'{p}: source smaller than crop')
            top, left = rng.randrange(rgb.shape[0]-args.size+1), rng.randrange(rgb.shape[1]-args.size+1)
            ref = np.ascontiguousarray(rgb[top:top+args.size, left:left+args.size])
            split = 0 if j < int(len(ids)*.6) else (1 if j < int(len(ids)*.75) else 2)
            manifest.append({'id': key, 'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                             'split': ['train', 'validation', 'test'][split], 'crop': [top, left, args.size]})
            for name in (DEGRADATIONS[:2] if args.smoke else DEGRADATIONS):
                x = degrade(ref, name, np_rng)
                tx = torch.from_numpy(x.copy()).permute(2, 0, 1).unsqueeze(0).to('cuda')
                feats.append(extract_features(tx).cpu().numpy()[0])
                old_amount = float(estimate_amount(tx).item())
                legacy.append(old_amount)
                outputs = [x] + [contrast_adaptive_sharpening(tx, amount=a)[0].permute(1, 2, 0).cpu().numpy()
                                 for a in AMOUNTS[1:]]
                outputs += [contrast_adaptive_sharpening(tx, amount=old_amount)[0].permute(1, 2, 0).cpu().numpy()]
                artifacts.append([artifact_metrics(y, ref) for y in outputs])
                ref_bytes = to_yuv(ref)
                for y in outputs:
                    ref_file.write(ref_bytes)
                    dis_file.write(to_yuv(y))
                cases.append({'source': key, 'degradation': name, 'split': split})
            if j % 16 == 0:
                print(f'sources {j+1}/{len(ids)} cases {len(cases)} elapsed {time.time()-start:.1f}s', flush=True)
    (out/'manifest.json').write_text(json.dumps({'seed': SEED, 'source': args.source, 'size': args.size,
                                               'features': FEATURE_NAMES, 'sources': manifest, 'cases': cases}, indent=2))
    print('Scoring ordinary VMAF + NEG (motion forced zero for independent still crops)', flush=True)
    quality = run_vmaf(args.ffmpeg, ref_path, dis_path, args.size, out/'vmaf.json').reshape(len(cases), 8, 2)
    art = np.asarray(artifacts, dtype='float32')
    # Reference-relative halo/noise must not significantly exceed the input.
    safe = ((quality[:, :, 1] >= quality[:, :1, 1]-.15) &
            (art[:, :, 0] <= art[:, :1, 0]+.00025) &
            (art[:, :, 1] <= art[:, :1, 1]+np.maximum(.0005, art[:, :1, 1]*.08)) &
            (art[:, :, 2] >= art[:, :1, 2]-.3))
    safe[:, 0] = True
    utility = quality[:, :7, 0].copy()
    utility[~safe[:, :7]] = -1000
    # Strength bias: choose strongest safe setting within 0.15 VMAF of maximum.
    close = utility >= utility.max(1, keepdims=True)-.15
    labels = (close*np.arange(1, 8)).argmax(1)
    np.savez_compressed(out/'dataset.npz', features=np.asarray(feats), quality=quality, artifact=art,
                        safe=safe, labels=labels, legacy=np.asarray(legacy),
                        splits=np.array([c['split'] for c in cases]), amounts=np.array(AMOUNTS))
    print('class counts', np.bincount(labels, minlength=7).tolist(), 'cases', len(cases), flush=True)
    # Only remove generated raw streams, explicitly under this output directory.
    ref_path.unlink()
    dis_path.unlink()


if __name__ == '__main__':
    main()
