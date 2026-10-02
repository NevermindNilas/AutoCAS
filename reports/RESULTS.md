# AutoCAS optimization and learned anime selector

The automatic CUDA filter is substantially faster, and the trained policy chooses artifact-constrained settings more accurately than the old blur heuristic. It is deliberately more conservative: both ordinary VMAF and NEG are lower than the aggressive old policy. Artifact-gate failures are reduced, not eliminated.

Baseline: clean commit `556a2c2b0be894bcbfbb94d11d8f1af3ab9380af`; exact original retained in `tools/baseline_cas.py`. No original tests or benchmark harness existed.

## Automatic performance before and after

Warm batch-1 RGB filtering on representative anime/digital-art images, including mild blur and downsample variants. RTX 3090, i7-13700K, Windows, PyTorch 2.13.0+cu132. CUDA-event median includes estimation, Python dispatch/launch gaps, and filtering; decoding/encoding and cold initialization are excluded. 20 warmup calls and 60 randomized, interleaved samples per arm; seed 7301.

| Input | Original auto ms | Optimized legacy ms | Learned final ms | Final filter FPS | Speedup (95% paired CI) |
|---|---:|---:|---:|---:|---:|
| 360p float16 | 0.717 | 0.429 | 0.116 | 8,604 | 6.17× (5.93–6.49) |
| 720p float16 | 1.153 | 0.503 | 0.166 | 6,010 | 6.93× (6.67–7.13) |
| 1080p float16 | 2.335 | 0.967 | 0.244 | 4,103 | 9.58× (9.50–9.70) |
| 720p float32 | 1.694 | 0.501 | 0.173 | 5,778 | 9.79× (9.44–9.93) |

The optimized-legacy column preserves the original estimator. The learned column intentionally changes the setting policy and can bypass CAS. Representative workload contains varying sharpening demand; gains also pass fresh randomized tensors and held-out odd shapes/batch 2.

## Latency and allocation guardrails

| Input | Before p95 ms | Final p95 ms | Before host p50 ms | Final host p50 ms | Before extra peak MiB | Final extra peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| 360p float16 | 0.980 | 0.164 | 0.733 | 0.133 | 15.39 | 1.32 |
| 720p float16 | 1.240 | 0.190 | 1.168 | 0.185 | 62.92 | 5.27 |
| 1080p float16 | 2.834 | 0.262 | 2.358 | 0.266 | 139.69 | 12.00 |
| 720p float32 | 1.800 | 0.201 | 1.712 | 0.190 | 116.06 | 10.55 |

Allocations count transient/output PyTorch tensors above warm resident inputs and cached state. Persistent model tensors add about 7 KiB. CUDA module/driver allocations are outside PyTorch counters. Raw logs retain reserved allocator peaks too; these depend on prior allocations and are not attributed as per-call demand. No measured p50/p95/transient-allocation regression in the reported supported CUDA workloads.

Fresh-process kernel/model initialization after PyTorch device startup: **24.07 ms**. Driver/filesystem caches were not flushed. NVRTC recompiles per process; the application does not maintain a disk cache. Use `cas.warmup(x)` before capture or latency-sensitive processing.

## All modes

| Input | Mode | Original median ms | Optimized median ms | Speedup |
|---|---|---:|---:|---:|
| 360p float16 | fixed | 0.346 | 0.044 | 7.78× |
| 360p float16 | legacy-auto | 0.717 | 0.429 | 1.67× |
| 360p float16 | tiled6 | 0.966 | 0.482 | 2.00× |
| 720p float16 | fixed | 0.776 | 0.082 | 9.47× |
| 720p float16 | legacy-auto | 1.153 | 0.503 | 2.29× |
| 720p float16 | tiled6 | 2.774 | 0.585 | 4.75× |
| 1080p float16 | fixed | 1.534 | 0.132 | 11.61× |
| 1080p float16 | legacy-auto | 2.335 | 0.967 | 2.41× |
| 1080p float16 | tiled6 | 7.153 | 1.004 | 7.12× |
| 720p float32 | fixed | 1.289 | 0.074 | 17.38× |
| 720p float32 | legacy-auto | 1.694 | 0.501 | 3.38× |
| 720p float32 | tiled6 | 3.320 | 0.539 | 6.16× |

## Independent quality confirmation

3,060 examples from 255 separate source originals, all development original IDs and exact file hashes excluded. Each original supplies clean, Gaussian blur 0.3/0.6/1.0/1.6, downsample 2×/3×, blur+noise 0.003/0.01, JPEG 65/90, and pre-sharpened examples. Crops are 256×256. The table measures the actual shipped fp32 runtime against the original fp32 implementation.

| Metric | Original heuristic | Learned final fp32 |
|---|---:|---:|
| Setting accuracy | 13.04% | 77.39% |
| Balanced accuracy | 16.34% | 24.37% |
| Artifact-gate violation rate | 69.05% | 3.01% |
| Ordinary VMAF | 84.20 | 81.30 |
| VMAF-NEG | 81.07 | 79.79 |
| Mean reference-envelope overshoot | 0.001887 | 0.000932 |
| Flat-region residual magnitude | 0.002956 | 0.002071 |
| Bypass fraction | 0.00% | 80.85% |

Always bypassing has 62.3% class accuracy, so overall accuracy benefits from a common bypass class. Balanced accuracy remains modest, and intermediate strengths are harder to identify. The old continuous amount is mapped to its nearest supported CAS setting solely for classification accuracy; quality uses its actual continuous output.

Against unprocessed input, final fp32 VMAF improves **79.78 -> 81.30**, and NEG improves **78.39 -> 79.79**. The constrained reference-based oracle reaches VMAF 82.63. Final runtime oracle regret is mean 1.40, p95 9.62. Ordinary VMAF difference versus the old policy has a source-bootstrap 95% interval [-3.07, -2.72], confirming the enhancement/safety tradeoff.

fp16 confirmation: setting accuracy 77.32%, gate violations 2.88%, VMAF 81.29, NEG 79.78. fp16 and fp32 setting agreement: 99.80%.

## Training and target construction

Local dataset: `D:/sisr/digitalart_v5/HR`, 1,024 sampled original IDs with one crop per original, seed 94137. 614 originals/7,368 examples train, 154/1,848 validation, 256/3,072 test. Source-ID grouping occurs before crop/degradation; training-only normalization. Separate confirmation uses `valHR`; one overlapping original ID was explicitly excluded, leaving 255 originals. No source artwork is redistributed.

Network: 16 sampled native-pixel features -> 32 ReLU units -> 7 logits, **775 learned parameters / 3,100 fp32 weight bytes**. The shipped JSON also includes normalization and metadata. Actions: bypass, minimum CAS strength 0, then .15/.3/.5/.7/.9. CAS amount 0 is not bypass.

Label oracle selects the strongest safe setting within 0.15 ordinary VMAF of the best safe setting. A setting passes if its NEG is no worse than input minus 0.15, mean overshoot no greater than input plus 0.00025, flat-region residual no greater than input plus max(0.0005, 8% of input), and RGB PSNR loss no more than 0.3 dB. Overshoot uses the clean reference 3×3 luma envelope with 0.005 tolerance. Flat regions use reference range <0.025. Bypass always qualifies.

FFmpeg libvmaf uses v0.6.1 ordinary and NEG models, 8-bit YUV420p, motion forced zero through each model option for independent still crops. The fixed v0 model gives reproducible sharpening comparisons; it does not model human anime preferences perfectly. Validation selection maximizes VMAF + 0.2*NEG under a <=3% gate-violation budget. Test or confirmation scores do not select weights. Selected seed 713, epoch 1600, learned-policy bypass-logit bias +1.0.

There was a 320-original pilot and a more conservative validation policy experiment; final expanded test IDs were fresh relative to the pilot, and independent confirmation came after final weight selection. Exploratory timing during concurrent training is excluded from reported comparisons.

## Correctness and scope

Profiling of the original automatic path attributed about 38% of device time to separate min/max passes and about 12% to average pooling. The fused CAS core reads the clamped neighborhood directly, preserves half intermediate rounding and FMA behavior, and writes only the output. The tiled kernel reduces both weighted numerator/denominator in one pass.

`python -m pytest tests -q -s`: **12 passed**, exit 0. Tested manual fp16/fp32 outputs were bit-exact across scalar settings, diagonal options, ranks 2/3/4, C=1/2/3/4, batches, tiny/odd shapes, black/white/binary/noisy/near-black inputs. Blurred tiled output maximum errors: 7.01e-7 fp32 and 0.0009765625 fp16. Input mutation, tensor broadcasting/dtype promotion, autograd, noncontiguous/channels-last fallback, nondefault streams, classifier parity, warmed CUDA graph replay and tiled warmup completion also passed.

Independent review identified and corrected lazy model-upload stream races, cold graph-capture initialization, signed thread-index overflow and runtime oracle-regret accounting. Cold capture requires warmup; inference does not perform blocking host synchronization. Gradient-bearing calls and unsupported native layouts/dtypes use PyTorch.

Limits: quality is based on synthetic degradation of anime/digital-art illustrations, not human preference ratings, real upscaler outputs, or video temporal stability. Artifact gates are proxies and have nonzero held-out failure rates; no universal artifact-free claim is made. Default learned policy is per-image; tiled mode retains the legacy estimator. CPU fallback is verified but not performance-optimized or benchmarked; native speed evidence is Windows/RTX 3090 only. No video I/O FPS claim or machine-wide tuning.

## Reproduction and artifacts

Run from the AutoCAS root:

```powershell
python -m pytest tests -q -s
python tools/benchmark.py --output reports/final_benchmark.json
python tools/benchmark.py --held-out --output reports/heldout_benchmark.json
python tools/benchmark.py --representative D:\sisr\digitalart_v5\valHR --output reports/representative_benchmark.json
python tools/cold_start.py
python tools/build_dataset.py --source D:\sisr\digitalart_v5\HR --ffmpeg D:\test\TheAnimeScripter\ffmpeg_shared\ffmpeg.exe --sources 1024 --output artifacts/dataset_v2
python tools/train_selector.py --dataset artifacts/dataset_v2/dataset.npz
python tools/build_dataset.py --source D:\sisr\digitalart_v5\valHR --ffmpeg D:\test\TheAnimeScripter\ffmpeg_shared\ffmpeg.exe --sources 256 --output artifacts/confirmation
python tools/evaluate_runtime.py --dataset artifacts/confirmation/dataset.npz --ffmpeg D:\test\TheAnimeScripter\ffmpeg_shared\ffmpeg.exe --all --exclude-manifest reports/dataset_manifest.json
python tools/write_report.py
```

Training source folders are existing user-local data. Numeric labels, features and manifests remain available; model training needs no network. Generated raw YUV streams are removed after successful scoring.

- `models/anime_selector.json`: shipped weights and normalization.
- `reports/CONTRACT.md`: frozen equivalence and performance protocol.
- `reports/baseline.json`, `final_benchmark.json`, `heldout_benchmark.json`, `representative_benchmark.json`: raw interleaved timings, allocation counters, source/model identities and paired confidence intervals.
- `reports/profile_baseline.txt`, `test_results.txt`, `cold_start.json`: profile, validation and initialization evidence.
- `reports/quality.json`, `runtime_quality.json`, `evaluation_data.npz`, `runtime_quality_data.npz`: selection, classification, actual quality and numeric evidence.
- `reports/dataset_manifest.json`, `confirmation_manifest.json`: source hashes, crop coordinates and group identities.
- `artifacts/dataset_v2/`, `artifacts/confirmation/`: local training/candidate VMAF logs and datasets (ignored by Git).
