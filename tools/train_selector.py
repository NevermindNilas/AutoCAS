"""Train compact classifier; select seed/checkpoint/bypass bias on validation only."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anime_selector import AMOUNTS, FEATURE_NAMES


def metrics(data, mask, choices):
    indices = np.flatnonzero(mask)
    q = data['quality'][indices, choices]
    art = data['artifact'][indices, choices]
    safe = data['safe'][indices, choices]
    labels = data['labels'][indices]
    classified = choices.copy()
    old = classified == 7
    # The old policy is continuous: compare its nearest supported setting,
    # while measuring its actual continuous output for quality.
    classified[old] = np.abs(data['legacy'][indices[old], None]-np.asarray(AMOUNTS[1:])).argmin(1)+1
    oracle = data['quality'][indices, labels, 0]
    present = np.unique(labels)
    return {'samples': len(indices), 'vmaf': float(q[:, 0].mean()), 'vmaf_neg': float(q[:, 1].mean()),
            'psnr': float(art[:, 2].mean()), 'halo': float(art[:, 0].mean()), 'flat_noise': float(art[:, 1].mean()),
            'gate_violation_rate': float((~safe).mean()), 'accuracy': float((classified==labels).mean()),
            'balanced_accuracy': float(np.mean([(classified[labels==c]==c).mean() for c in present])),
            'oracle_regret_mean': float(np.maximum(oracle-q[:, 0], 0).mean()),
            'oracle_regret_p95': float(np.percentile(np.maximum(oracle-q[:, 0], 0), 95)),
            'bypass_rate': float((choices==0).mean()), 'class_counts': np.bincount(choices, minlength=8).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='artifacts/dataset/dataset.npz')
    ap.add_argument('--epochs', type=int, default=1800)
    args = ap.parse_args()
    torch.set_num_threads(4)
    data = np.load(args.dataset)
    x = torch.tensor(data['features'], dtype=torch.float32, device='cuda')
    y = torch.tensor(data['labels'], dtype=torch.long, device='cuda')
    splits = data['splits']
    train, val, test = splits==0, splits==1, splits==2
    mean = x[train].mean(0)
    invstd = x[train].std(0).clamp(min=1e-4).reciprocal()
    x = (x-mean)*invstd
    safe = torch.tensor(data['safe'][:, :7], device='cuda', dtype=torch.float32)
    quality = torch.tensor(data['quality'][:, :7, 0], device='cuda')
    regret = (quality.max(1, keepdim=True).values-quality).clamp(min=0, max=20)
    selected, best_score, history = None, -float('inf'), []
    for seed in [713, 1427, 3911]:
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 7)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.006, weight_decay=.008)
        for epoch in range(args.epochs):
            model.train()
            logits = model(x[train])
            probs = logits.softmax(1)
            loss = torch.nn.functional.cross_entropy(logits, y[train])
            loss = loss + .04*(probs*regret[train]).sum(1).mean() + 4*(probs*(1-safe[train])).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if epoch % 25 == 0 or epoch == args.epochs-1:
                model.eval()
                with torch.no_grad():
                    lv = model(x[val]).cpu().numpy()
                for bias in [0., .25, .5, .75, 1., 1.5, 2.]:
                    biased = lv.copy()
                    biased[:, 0] += bias
                    choices = biased.argmax(1)
                    m = metrics(data, val, choices)
                    # Penalize gate violations explicitly; no test-based selection.
                    # Predeclare a 3% validation gate budget, then favor sharpness
                    # within it. Infeasible candidates are ranked below feasible ones.
                    score = m['vmaf'] + .2*m['vmaf_neg']
                    if m['gate_violation_rate'] > .03:
                        score -= 1000 + 32*m['gate_violation_rate']
                    if score > best_score:
                        best_score = score
                        state = copy.deepcopy(model.state_dict())
                        state['2.bias'][0] += bias
                        selected = (seed, epoch, bias, state, m)
        print('seed', seed, 'best validation', selected[:3], selected[4], flush=True)
        history.append({'seed': seed, 'best_score_so_far': best_score})
    seed, epoch, bias, state, val_metrics = selected
    model.load_state_dict(state)
    with torch.no_grad():
        choices = model(x).argmax(1).cpu().numpy()
    params = {'schema': 1, 'features': FEATURE_NAMES, 'amounts': AMOUNTS, 'mean': mean.cpu().tolist(),
              'inverse_std': invstd.cpu().tolist(), 'w1': state['0.weight'].cpu().tolist(),
              'b1': state['0.bias'].cpu().tolist(), 'w2': state['2.weight'].cpu().tolist(),
              'b2': state['2.bias'].cpu().tolist(),
              'training': {'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                           'seed': seed, 'epoch': epoch, 'validation_bypass_bias': bias,
                           'source_groups': {'train': int(train.sum()//12), 'validation': int(val.sum()//12),
                                             'test': int(test.sum()//12)},
                           'policy': 'VMAF-maximizing labels constrained by NEG, halos, flat noise and PSNR; strongest within 0.15 VMAF'}}
    Path('models/anime_selector.json').write_text(json.dumps(params, separators=(',', ':')))
    report = {'model': {'learned_parameters': sum(p.numel() for p in model.parameters()),
                        'weight_bytes_fp32': sum(p.numel()*4 for p in model.parameters()),
                        'file_bytes': Path('models/anime_selector.json').stat().st_size,
                        'seed': seed, 'epoch': epoch, 'bypass_bias': bias},
              'selection': 'validation only; maximize VMAF + 0.2*NEG subject to <=3% artifact-gate violations', 'history': history,
              'validation': val_metrics, 'test': {}, 'by_degradation': {}}
    arms = {'input': np.zeros(test.sum(), dtype=int), 'legacy': np.full(test.sum(), 7, dtype=int),
            'ml': choices[test], 'oracle': data['labels'][test]}
    for name, picked in arms.items():
        report['test'][name] = metrics(data, test, picked)
    manifest = json.loads(Path(args.dataset).with_name('manifest.json').read_text())
    degradations = np.array([c['degradation'] for c in manifest['cases']])
    for degradation in np.unique(degradations):
        mask = test & (degradations==degradation)
        report['by_degradation'][degradation] = {n: metrics(data, mask, np.full(mask.sum(), 7, dtype=int) if n=='legacy'
                                                          else choices[mask]) for n in ['legacy', 'ml']}
    # Paired bootstrap over source groups (12 degradations each), not individual crops.
    ids = np.flatnonzero(test)
    diff = (data['quality'][ids, choices[test], 0]-data['quality'][ids, 7, 0]).reshape(-1, 12).mean(1)
    rng = np.random.default_rng(7301)
    boot = diff[rng.integers(len(diff), size=(4000, len(diff)))].mean(1)
    report['vmaf_improvement_ci95_source_bootstrap'] = np.percentile(boot, [2.5, 97.5]).tolist()
    Path('reports/quality.json').write_text(json.dumps(report, indent=2))
    Path('reports/dataset_manifest.json').write_text(json.dumps(manifest, indent=2))
    np.savez_compressed('reports/evaluation_data.npz', **{key: data[key] for key in data.files}, predictions=choices)
    print(json.dumps(report['test'], indent=2), flush=True)


if __name__ == '__main__':
    main()
