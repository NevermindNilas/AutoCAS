"""Generate the results report from retained measurement artifacts."""
import json
from pathlib import Path
import numpy as np


def main():
    benchmark = json.loads(Path('reports/representative_benchmark.json').read_text())
    quality = json.loads(Path('reports/runtime_quality.json').read_text())
    cold = json.loads(Path('reports/cold_start.json').read_text())
    model = json.loads(Path('reports/quality.json').read_text())
    raw = np.load('reports/runtime_quality_data.npz')
    data = np.load('artifacts/confirmation/dataset.npz')
    indices = raw['original_indices']
    # Recompute runtime regret from scored outputs, not candidate lookup scores.
    for j, dtype in enumerate(['fp32', 'fp16']):
        regret = np.maximum(data['quality'][indices, data['labels'][indices], 0]-raw['quality'][:, j, 0], 0)
        quality['arms'][dtype]['oracle_regret_mean'] = float(regret.mean())
        quality['arms'][dtype]['oracle_regret_p95'] = float(np.percentile(regret, 95))
    rng = np.random.default_rng(7301)
    ci = {}
    for j, dtype in enumerate(['fp32', 'fp16']):
        diff = (raw['quality'][:, j, 0]-data['quality'][indices, 7, 0]).reshape(-1, 12).mean(1)
        boot = diff[rng.integers(len(diff), size=(4000, len(diff)))].mean(1)
        ci[dtype] = np.percentile(boot,[2.5,97.5]).tolist()
    quality['vmaf_difference_ci95_source_bootstrap'] = ci
    Path('reports/runtime_quality.json').write_text(json.dumps(quality, indent=2))
    lines = ['# AutoCAS optimization and learned anime selector', '',
             'The automatic CUDA filter is substantially faster, and the trained policy chooses artifact-constrained settings more accurately than the old blur heuristic. It is deliberately more conservative: both ordinary VMAF and NEG are lower than the aggressive old policy. Artifact-gate failures are reduced, not eliminated.', '',
             'Baseline: clean commit `556a2c2b0be894bcbfbb94d11d8f1af3ab9380af`; exact original retained in `tools/baseline_cas.py`. No original tests or benchmark harness existed.', '',
             '## Automatic performance before and after', '',
             'Warm batch-1 RGB filtering on representative anime/digital-art images, including mild blur and downsample variants. RTX 3090, i7-13700K, Windows, PyTorch 2.13.0+cu132. CUDA-event median includes estimation, Python dispatch/launch gaps, and filtering; decoding/encoding and cold initialization are excluded. 20 warmup calls and 60 randomized, interleaved samples per arm; seed 7301.', '',
             '| Input | Original auto ms | Optimized legacy ms | Learned final ms | Final filter FPS | Speedup (95% paired CI) |',
             '|---|---:|---:|---:|---:|---:|']
    for row in benchmark['rows']:
        if row['mode'] != 'legacy-auto':
            continue
        arms = row['arms']; a,b,c = [arms[n] for n in ['baseline','optimized','ml']]
        low, high = c['speedup_ci95_paired_bootstrap']
        lines.append(f"| {row['resolution']} {row['dtype'].split('.')[-1]} | {a['median_ms']:.3f} | {b['median_ms']:.3f} | {c['median_ms']:.3f} | {c['fps']:,.0f} | {a['median_ms']/c['median_ms']:.2f}× ({low:.2f}–{high:.2f}) |")
    lines += ['', 'The optimized-legacy column preserves the original estimator. The learned column intentionally changes the setting policy and can bypass CAS. Representative workload contains varying sharpening demand; gains also pass fresh randomized tensors and held-out odd shapes/batch 2.', '',
              '## Latency and allocation guardrails', '',
              '| Input | Before p95 ms | Final p95 ms | Before host p50 ms | Final host p50 ms | Before extra peak MiB | Final extra peak MiB |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in benchmark['rows']:
        if row['mode']=='legacy-auto':
            a,c = row['arms']['baseline'],row['arms']['ml']
            lines.append(f"| {row['resolution']} {row['dtype'].split('.')[-1]} | {a['p95_ms']:.3f} | {c['p95_ms']:.3f} | {a['wall_median_ms']:.3f} | {c['wall_median_ms']:.3f} | {a['peak_extra_bytes']/2**20:.2f} | {c['peak_extra_bytes']/2**20:.2f} |")
    lines += ['', 'Allocations count transient/output PyTorch tensors above warm resident inputs and cached state. Persistent model tensors add about 7 KiB. CUDA module/driver allocations are outside PyTorch counters. Raw logs retain reserved allocator peaks too; these depend on prior allocations and are not attributed as per-call demand. No measured p50/p95/transient-allocation regression in the reported supported CUDA workloads.', '',
              f"Fresh-process kernel/model initialization after PyTorch device startup: **{cold['warmup_ms']:.2f} ms**. Driver/filesystem caches were not flushed. NVRTC recompiles per process; the application does not maintain a disk cache. Use `cas.warmup(x)` before capture or latency-sensitive processing.", '',
              '## All modes', '', '| Input | Mode | Original median ms | Optimized median ms | Speedup |', '|---|---|---:|---:|---:|']
    for row in benchmark['rows']:
        a,b = row['arms']['baseline'], row['arms']['optimized']
        lines.append(f"| {row['resolution']} {row['dtype'].split('.')[-1]} | {row['mode']} | {a['median_ms']:.3f} | {b['median_ms']:.3f} | {a['median_ms']/b['median_ms']:.2f}× |")
    lines += ['', '## Independent quality confirmation', '',
              '3,060 examples from 255 separate source originals, all development original IDs and exact file hashes excluded. Each original supplies clean, Gaussian blur 0.3/0.6/1.0/1.6, downsample 2×/3×, blur+noise 0.003/0.01, JPEG 65/90, and pre-sharpened examples. Crops are 256×256. The table measures the actual shipped fp32 runtime against the original fp32 implementation.', '',
              '| Metric | Original heuristic | Learned final fp32 |', '|---|---:|---:|']
    for title, key, fmt in [('Setting accuracy','accuracy','pct'),('Balanced accuracy','balanced_accuracy','pct'),
                            ('Artifact-gate violation rate','gate_violation_rate','pct'),('Ordinary VMAF','vmaf','num'),
                            ('VMAF-NEG','vmaf_neg','num'),('Mean reference-envelope overshoot','halo','small'),
                            ('Flat-region residual magnitude','flat_noise','small'),('Bypass fraction','bypass_rate','pct')]:
        vals = [quality['arms'][a][key] for a in ['legacy','fp32']]
        vals = [f'{v*100:.2f}%' if fmt=='pct' else (f'{v:.6f}' if fmt=='small' else f'{v:.2f}') for v in vals]
        lines.append(f'| {title} | {vals[0]} | {vals[1]} |')
    q = quality['arms']
    lines += ['', f"Always bypassing has {q['input']['accuracy']*100:.1f}% class accuracy, so overall accuracy benefits from a common bypass class. Balanced accuracy remains modest, and intermediate strengths are harder to identify. The old continuous amount is mapped to its nearest supported CAS setting solely for classification accuracy; quality uses its actual continuous output.", '',
              f"Against unprocessed input, final fp32 VMAF improves **{q['input']['vmaf']:.2f} -> {q['fp32']['vmaf']:.2f}**, and NEG improves **{q['input']['vmaf_neg']:.2f} -> {q['fp32']['vmaf_neg']:.2f}**. The constrained reference-based oracle reaches VMAF {q['oracle']['vmaf']:.2f}. Final runtime oracle regret is mean {q['fp32']['oracle_regret_mean']:.2f}, p95 {q['fp32']['oracle_regret_p95']:.2f}. Ordinary VMAF difference versus the old policy has a source-bootstrap 95% interval [{ci['fp32'][0]:.2f}, {ci['fp32'][1]:.2f}], confirming the enhancement/safety tradeoff.", '',
              f"fp16 confirmation: setting accuracy {q['fp16']['accuracy']*100:.2f}%, gate violations {q['fp16']['gate_violation_rate']*100:.2f}%, VMAF {q['fp16']['vmaf']:.2f}, NEG {q['fp16']['vmaf_neg']:.2f}. fp16 and fp32 setting agreement: {quality['classifier_fp16_fp32_agreement']*100:.2f}%.", '',
              '## Training and target construction', '',
              'Local dataset: `D:/sisr/digitalart_v5/HR`, 1,024 sampled original IDs with one crop per original, seed 94137. 614 originals/7,368 examples train, 154/1,848 validation, 256/3,072 test. Source-ID grouping occurs before crop/degradation; training-only normalization. Separate confirmation uses `valHR`; one overlapping original ID was explicitly excluded, leaving 255 originals. No source artwork is redistributed.', '',
              'Network: 16 sampled native-pixel features -> 32 ReLU units -> 7 logits, **775 learned parameters / 3,100 fp32 weight bytes**. The shipped JSON also includes normalization and metadata. Actions: bypass, minimum CAS strength 0, then .15/.3/.5/.7/.9. CAS amount 0 is not bypass.', '',
              'Label oracle selects the strongest safe setting within 0.15 ordinary VMAF of the best safe setting. A setting passes if its NEG is no worse than input minus 0.15, mean overshoot no greater than input plus 0.00025, flat-region residual no greater than input plus max(0.0005, 8% of input), and RGB PSNR loss no more than 0.3 dB. Overshoot uses the clean reference 3×3 luma envelope with 0.005 tolerance. Flat regions use reference range <0.025. Bypass always qualifies.', '',
              'FFmpeg libvmaf uses v0.6.1 ordinary and NEG models, 8-bit YUV420p, motion forced zero through each model option for independent still crops. The fixed v0 model gives reproducible sharpening comparisons; it does not model human anime preferences perfectly. Validation selection maximizes VMAF + 0.2*NEG under a <=3% gate-violation budget. Test or confirmation scores do not select weights. Selected seed 713, epoch 1600, learned-policy bypass-logit bias +1.0.', '',
              'There was a 320-original pilot and a more conservative validation policy experiment; final expanded test IDs were fresh relative to the pilot, and independent confirmation came after final weight selection. Exploratory timing during concurrent training is excluded from reported comparisons.', '',
              '## Correctness and scope', '',
              'Profiling of the original automatic path attributed about 38% of device time to separate min/max passes and about 12% to average pooling. The fused CAS core reads the clamped neighborhood directly, preserves half intermediate rounding and FMA behavior, and writes only the output. The tiled kernel reduces both weighted numerator/denominator in one pass.', '',
              '`python -m pytest tests -q -s`: **12 passed**, exit 0. Tested manual fp16/fp32 outputs were bit-exact across scalar settings, diagonal options, ranks 2/3/4, C=1/2/3/4, batches, tiny/odd shapes, black/white/binary/noisy/near-black inputs. Blurred tiled output maximum errors: 7.01e-7 fp32 and 0.0009765625 fp16. Input mutation, tensor broadcasting/dtype promotion, autograd, noncontiguous/channels-last fallback, nondefault streams, classifier parity, warmed CUDA graph replay and tiled warmup completion also passed.', '',
              'Independent review identified and corrected lazy model-upload stream races, cold graph-capture initialization, signed thread-index overflow and runtime oracle-regret accounting. Cold capture requires warmup; inference does not perform blocking host synchronization. Gradient-bearing calls and unsupported native layouts/dtypes use PyTorch.', '',
              'Limits: quality is based on synthetic degradation of anime/digital-art illustrations, not human preference ratings, real upscaler outputs, or video temporal stability. Artifact gates are proxies and have nonzero held-out failure rates; no universal artifact-free claim is made. Default learned policy is per-image; tiled mode retains the legacy estimator. CPU fallback is verified but not performance-optimized or benchmarked; native speed evidence is Windows/RTX 3090 only. No video I/O FPS claim or machine-wide tuning.', '',
              '## Reproduction and artifacts', '', 'Run from the AutoCAS root:', '', '```powershell',
              'python -m pytest tests -q -s',
              'python tools/benchmark.py --output reports/final_benchmark.json',
              'python tools/benchmark.py --held-out --output reports/heldout_benchmark.json',
              'python tools/benchmark.py --representative D:\\sisr\\digitalart_v5\\valHR --output reports/representative_benchmark.json',
              'python tools/cold_start.py',
              'python tools/build_dataset.py --source D:\\sisr\\digitalart_v5\\HR --ffmpeg D:\\test\\TheAnimeScripter\\ffmpeg_shared\\ffmpeg.exe --sources 1024 --output artifacts/dataset_v2',
              'python tools/train_selector.py --dataset artifacts/dataset_v2/dataset.npz',
              'python tools/build_dataset.py --source D:\\sisr\\digitalart_v5\\valHR --ffmpeg D:\\test\\TheAnimeScripter\\ffmpeg_shared\\ffmpeg.exe --sources 256 --output artifacts/confirmation',
              'python tools/evaluate_runtime.py --dataset artifacts/confirmation/dataset.npz --ffmpeg D:\\test\\TheAnimeScripter\\ffmpeg_shared\\ffmpeg.exe --all --exclude-manifest reports/dataset_manifest.json',
              'python tools/write_report.py', '```', '',
              'Training source folders are existing user-local data. Numeric labels, features and manifests remain available; model training needs no network. Generated raw YUV streams are removed after successful scoring.', '',
              '- `models/anime_selector.json`: shipped weights and normalization.',
              '- `reports/CONTRACT.md`: frozen equivalence and performance protocol.',
              '- `reports/baseline.json`, `final_benchmark.json`, `heldout_benchmark.json`, `representative_benchmark.json`: raw interleaved timings, allocation counters, source/model identities and paired confidence intervals.',
              '- `reports/profile_baseline.txt`, `test_results.txt`, `cold_start.json`: profile, validation and initialization evidence.',
              '- `reports/quality.json`, `runtime_quality.json`, `evaluation_data.npz`, `runtime_quality_data.npz`: selection, classification, actual quality and numeric evidence.',
              '- `reports/dataset_manifest.json`, `confirmation_manifest.json`: source hashes, crop coordinates and group identities.',
              '- `artifacts/dataset_v2/`, `artifacts/confirmation/`: local training/candidate VMAF logs and datasets (ignored by Git).', '']
    Path('reports/RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__=='__main__':
    main()
