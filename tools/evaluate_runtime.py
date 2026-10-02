"""Confirm shipped fp32/fp16 runtime on separate source groups, including VMAF."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cas
from anime_selector import AMOUNTS, estimate_anime_amount
from tools.build_dataset import SEED, DEGRADATIONS, degrade, artifact_metrics, to_yuv, run_vmaf
from tools.train_selector import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--ffmpeg', required=True)
    ap.add_argument('--all', action='store_true', help='use every split for independent external confirmation')
    ap.add_argument('--exclude-manifest', help='exclude original IDs and exact source hashes used during development')
    args = ap.parse_args()
    torch.set_num_threads(4)
    folder = Path(args.dataset).parent
    manifest = json.loads((folder/'manifest.json').read_text())
    data = np.load(args.dataset)
    excluded_ids, excluded_hashes = set(), set()
    if args.exclude_manifest:
        other = json.loads(Path(args.exclude_manifest).read_text())
        excluded_ids = {s['id'] for s in other['sources']}
        excluded_hashes = {s['sha256'] for s in other['sources']}
    excluded_sources = []
    size = manifest['size']
    out = folder/'runtime'
    out.mkdir(exist_ok=True)
    artifacts, predictions, original_indices = [], [], []
    rng = np.random.default_rng(SEED)
    with (out/'reference.yuv').open('wb') as rf, (out/'runtime.yuv').open('wb') as df, torch.inference_mode():
        for s, source in enumerate(manifest['sources']):
            use = args.all or source['split']=='test'
            if source['id'] in excluded_ids or source['sha256'] in excluded_hashes:
                use = False
                excluded_sources.append(source['id'])
            if use:
                arr = np.asarray(Image.open(source['path']).convert('RGB')).astype('float32')/255
                top, left, _ = source['crop']
                ref = np.ascontiguousarray(arr[top:top+size, left:left+size])
                if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
                    raise ValueError('source file changed since dataset generation')
            for j, name in enumerate(DEGRADATIONS):
                if not use:
                    if 'noise' in name:
                        rng.normal(0, .003 if name.endswith('003') else .01, (size, size, 3))
                    continue
                x = degrade(ref, name, rng)
                original_indices.append(s*12+j)
                outputs, classes = [], []
                for dtype in [torch.float32, torch.float16]:
                    tx = torch.from_numpy(x.copy()).permute(2, 0, 1).unsqueeze(0).to(device='cuda', dtype=dtype)
                    amt = estimate_anime_amount(tx)
                    output = cas.contrast_adaptive_sharpening(tx, amount=None)[0].float().permute(1, 2, 0).cpu().numpy()
                    classes.append(int(np.abs(np.asarray(AMOUNTS)-float(amt.item())).argmin()))
                    outputs.append(output)
                    rf.write(to_yuv(ref))
                    df.write(to_yuv(output))
                predictions.append(classes)
                artifacts.append([artifact_metrics(y, ref) for y in outputs])
            if use and s % 16 == 0:
                print('runtime sources', s+1, '/', len(manifest['sources']), flush=True)
    q = run_vmaf(args.ffmpeg, out/'reference.yuv', out/'runtime.yuv', size, out/'vmaf.json').reshape(-1, 2, 2)
    art = np.asarray(artifacts)
    indices = np.asarray(original_indices)
    pred = np.asarray(predictions)
    input_q, input_art = data['quality'][indices, 0], data['artifact'][indices, 0]
    safe = ((q[:, :, 1] >= input_q[:, None, 1]-.15) &
            (art[:, :, 0] <= input_art[:, None, 0]+.00025) &
            (art[:, :, 1] <= input_art[:, None, 1]+np.maximum(.0005, input_art[:, None, 1]*.08)) &
            (art[:, :, 2] >= input_art[:, None, 2]-.3))
    # Bypass has no sharpening; half quantization alone may alter pristine PSNR.
    safe[pred==0] = True
    report = {'sources': len(np.unique(indices//12)), 'examples': len(indices),
              'model_sha256': hashlib.sha256(Path('models/anime_selector.json').read_bytes()).hexdigest(),
              'classifier_fp16_fp32_agreement': float((pred[:, 0]==pred[:, 1]).mean()),
              'excluded_development_source_ids': excluded_sources, 'arms': {}}
    mask = np.isin(np.arange(len(data['labels'])), indices)
    report['arms']['legacy'] = metrics(data, mask, np.full(len(indices), 7, dtype=int))
    report['arms']['input'] = metrics(data, mask, np.zeros(len(indices), dtype=int))
    report['arms']['oracle'] = metrics(data, mask, data['labels'][indices])
    for j, name in enumerate(['fp32', 'fp16']):
        m = metrics(data, mask, pred[:, j])
        regret = np.maximum(data['quality'][indices, data['labels'][indices], 0]-q[:, j, 0], 0)
        m.update(vmaf=float(q[:, j, 0].mean()), vmaf_neg=float(q[:, j, 1].mean()),
                 psnr=float(art[:, j, 2].mean()), halo=float(art[:, j, 0].mean()),
                 flat_noise=float(art[:, j, 1].mean()), gate_violation_rate=float((~safe[:, j]).mean()),
                 oracle_regret_mean=float(regret.mean()), oracle_regret_p95=float(np.percentile(regret,95)))
        report['arms'][name] = m
    Path('reports/runtime_quality.json').write_text(json.dumps(report, indent=2))
    Path('reports/confirmation_manifest.json').write_text(json.dumps(manifest, indent=2))
    np.savez_compressed('reports/runtime_quality_data.npz', quality=q, artifact=art, safe=safe,
                        predictions=pred, original_indices=indices)
    print(json.dumps(report, indent=2))
    (out/'reference.yuv').unlink()
    (out/'runtime.yuv').unlink()


if __name__ == '__main__':
    main()
