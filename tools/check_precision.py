"""Check shipped dtype compatibility and a separate CPU INT8 classifier prototype.

Does not modify the deployed filter, model weights, or backend dispatch.
"""
import copy
import json
import random
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cas
import _native
from anime_selector import model_data


def main():
    torch.set_num_threads(4)
    torch.manual_seed(802)
    result = {'torch': torch.__version__, 'quantized_engine': torch.backends.quantized.engine,
              'image_input_checks': [], 'classifier_experiment': {}}
    for device in ['cpu', 'cuda']:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        base = torch.rand(1, 3, 31, 47, device=device)
        for kind in ['float32', 'float16', 'int8_binary', 'int8_encoded', 'uint8_encoded']:
            if kind == 'float32':
                x, normalized = base, base
            elif kind == 'float16':
                x = base.half()
                normalized = x.float()
            elif kind == 'int8_binary':
                x = (base>.5).to(torch.int8)
                normalized = x.float()
            elif kind == 'int8_encoded':
                x = (base*255).round().sub(128).to(torch.int8)
                normalized = (x.float()+128)/255
            else:
                x = (base*255).round().to(torch.uint8)
                normalized = x.float()/255
            for mode in ['manual', 'anime', 'legacy']:
                kw = {'amount': .8} if mode == 'manual' else {'amount': None, 'auto_mode': mode}
                row = {'device': device, 'input': kind, 'mode': mode,
                       'native_eligible': bool(_native.eligible(x))}
                try:
                    with torch.inference_mode():
                        expected = cas.contrast_adaptive_sharpening(normalized, **kw)
                        actual = cas.contrast_adaptive_sharpening(x, **kw)
                    error = (actual.float()-expected).abs().max().item()
                    row.update(output_dtype=str(actual.dtype), finite=bool(torch.isfinite(actual).all().item()),
                               max_error_vs_normalized_fp32=error,
                               compatible=bool(error<=.003 and torch.is_floating_point(x)))
                except Exception as exc:
                    row.update(compatible=False, error=f'{type(exc).__name__}: {str(exc).splitlines()[0]}')
                result['image_input_checks'].append(row)
        if device == 'cpu':
            qx = torch.quantize_per_tensor(base.cpu(), scale=1/255, zero_point=-128, dtype=torch.qint8)
            try:
                cas.contrast_adaptive_sharpening(qx, amount=.8)
                quantized = {'success': True}
            except Exception as exc:
                quantized = {'success': False, 'error': f'{type(exc).__name__}: {str(exc).splitlines()[0]}'}
            result['quantized_image_tensor'] = quantized

    # An isolated prototype tests whether the tiny classifier can be quantized.
    # CAS, feature extraction, normalization and output strength remain floating point.
    data = model_data()
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 7)).eval()
    with torch.no_grad():
        for layer, w, b in [(model[0], 'w1', 'b1'), (model[2], 'w2', 'b2')]:
            layer.weight.copy_(torch.tensor(data[w]))
            layer.bias.copy_(torch.tensor(data[b]))
    experiment = result['classifier_experiment']
    try:
        with warnings.catch_warnings(record=True) as caught:
            quantized = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8)
        experiment['warnings'] = [str(w.message).splitlines()[0] for w in caught]
        dataset = np.load('reports/evaluation_data.npz')
        test = dataset['splits']==2
        features = torch.tensor(dataset['features'][test])
        features = (features-torch.tensor(data['mean']))*torch.tensor(data['inverse_std'])
        labels = dataset['labels'][test]
        with torch.inference_mode():
            fp_predictions = model(features).argmax(1).numpy()
            # Dynamic activation quantization can depend on batch composition;
            # use the production batch-1 lifecycle for each prediction.
            q_predictions = np.array([int(quantized(f[None]).argmax(1)) for f in features])
        experiment.update(cpu_int8_works=True, scope='classifier only; CPU onednn dynamic INT8 weights/activations; batch1; float input/output',
                          samples=len(features), setting_agreement=float((fp_predictions==q_predictions).mean()),
                          fp32_accuracy=float((fp_predictions==labels).mean()),
                          int8_accuracy=float((q_predictions==labels).mean()),
                          input_dtype=str(features.dtype), output_dtype=str(quantized(features[:1]).dtype),
                          first_weight_dtype=str(quantized[0].weight().dtype))
        for name, predictions in [('fp32',fp_predictions), ('int8',q_predictions)]:
            indices = np.flatnonzero(test)
            experiment[name+'_vmaf_candidate_lookup'] = float(dataset['quality'][indices,predictions,0].mean())
            experiment[name+'_gate_violation_rate_candidate_lookup'] = float((~dataset['safe'][indices,predictions]).mean())
        rng = random.Random(7301)
        timings = {'fp32': [], 'int8': []}
        models = {'fp32':model, 'int8':quantized}
        with torch.inference_mode():
            for m in models.values():
                for j in range(30):
                    m(features[j:j+1])
            for j in range(300):
                order = list(models)
                rng.shuffle(order)
                x = features[j%len(features):j%len(features)+1]
                for name in order:
                    start = time.perf_counter_ns()
                    models[name](x)
                    timings[name].append((time.perf_counter_ns()-start)/1000)
        experiment['latency_boundary'] = 'CPU classifier forward only; features and normalization excluded; 4 CPU threads; 30 warmup/300 interleaved samples'
        experiment['cpu_median_us'] = {k:statistics.median(v) for k,v in timings.items()}
        experiment['raw_cpu_us'] = timings
        if torch.cuda.is_available():
            try:
                copy.deepcopy(quantized).to('cuda')(features[:1].to('cuda'))
                experiment['cuda_dynamic_int8_works'] = True
            except Exception as exc:
                experiment['cuda_dynamic_int8_works'] = False
                experiment['cuda_error'] = f'{type(exc).__name__}: {str(exc).splitlines()[0]}'
    except Exception as exc:
        experiment.update(cpu_int8_works=False, error=f'{type(exc).__name__}: {str(exc).splitlines()[0]}')
    Path('reports/precision_compatibility.json').write_text(json.dumps(result, indent=2))
    for row in result['image_input_checks']:
        print(row)
    summary = {k:v for k,v in experiment.items() if k not in ('raw_cpu_us','warnings')}
    print('Classifier prototype:', json.dumps(summary, indent=2))
    print('Quantized image:',result['quantized_image_tensor'])


if __name__=='__main__':
    main()
