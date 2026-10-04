# INT8 compatibility check — 2026-10-02

Checked shipped commit `026e93f5d54ec17d464e1c3c83e388eaae690c14` using PyTorch 2.13.0+cu132, RTX 3090 and CPU oneDNN. No deployed filter/model/backend changes were made. The experiment is a local diagnostic, not an enabled INT8 inference path.

## Shipped support

| Case | Result |
|---|---|
| Normalized fp16/fp32 image tensors | Passed manual, anime-auto and legacy-auto checks on CPU and CUDA |
| Raw signed int8 image tensors | Unsupported; integer arithmetic and unhandled pixel encoding produce wrong normalized-image results |
| Raw uint8 image tensors (0–255) | Unsupported; max absolute output error reached 1.0 versus correctly normalized fp32 |
| PyTorch qint8 quantized image tensor | Failed on CPU replicate padding (`QuantizedCPU` operator unavailable) |
| Native fused CUDA INT8 | Not implemented; native dispatch explicitly accepts only fp16/fp32 |
| INT8 classifier prototype | Works with CPU dynamic quantization; not connected to shipped AutoCAS |

Some binary integer images happen to yield identical outputs, which does not establish dtype support. Source images must be normalized floating point. For standard unsigned eight-bit pixels, use `pixels.to(torch.float16) / 255` before filtering. Signed/affine quantized tensors need their actual scale and zero point when dequantized; casting alone is insufficient.

## Isolated classifier quantization experiment

Reconstructed the shipped 16→32→7 network and applied PyTorch dynamic qint8 quantization to its two Linear layers. Image features, normalization, biases, strength selection, and CAS remain floating point. Dynamic quantized Linear accepts/returns floating point tensors while quantizing weights and activation computation internally. See [PyTorch API](https://docs.pytorch.org/docs/main/generated/torch.ao.quantization.quantize_dynamic.html) and [dynamic Linear documentation](https://docs.pytorch.org/docs/2.12/generated/torch.ao.nn.quantized.dynamic.modules.linear.Linear.html).

Held-out development test split: 3,072 feature vectors, batch-1 quantized inference:

| Metric | FP32 classifier | INT8 CPU prototype |
|---|---:|---:|
| Target-setting accuracy | 77.96% | 76.40% |
| Mean VMAF from precomputed selected-candidate scores | 81.49 | 81.32 |
| Artifact-gate violations from selected-candidate scores | 2.73% | 2.54% |
| CPU classifier-only median latency | 14.8 µs | 63.2 µs |

INT8/FP32 setting agreement: 96.74%. Timing uses four CPU threads, 30 warmup calls and 300 randomized/interleaved samples per arm. Feature extraction and normalization are excluded from latency. Candidate lookup scores are not a new end-to-end VMAF evaluation of an INT8 image pipeline.

The tested oneDNN dynamic INT8 operator failed with CUDA inputs (`quantized::linear_dynamic` unavailable for the CUDA backend). This describes the installed API/backend, not all possible CUDA INT8 implementations. A separate CUDA INT8 implementation would need new kernels and validation; no GPU INT8 speed claim is made.

For this 775-parameter classifier, the tested CPU quantization overhead exceeded its arithmetic savings. The current FP16 image path with FP32 classifier calculations remains the measured GPU option.

Reproduce from the AutoCAS root:

```powershell
python tools/check_precision.py
```

Raw cases and timings: `reports/precision_compatibility.json`.
