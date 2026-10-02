import torch
import pytest

import cas
import _native
from tools import baseline_cas as baseline
from anime_selector import extract_features


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('diagonals', [False, True])
def test_native_differential(dtype, diagonals):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(891)
    maximum = 0.
    with torch.inference_mode():
        for shape in [(1, 1), (2, 3), (17, 19), (1, 31, 47), (2, 17, 31),
                      (3, 32, 64), (4, 19, 33), (2, 3, 71, 119)]:
            x = torch.rand(shape, device='cuda', dtype=dtype)
            for data in [x, torch.zeros_like(x), torch.ones_like(x), x*.00001,
                         (x>.5).to(dtype), torch.full_like(x, .5)]:
                saved = data.clone()
                for amount in [-1., 0., .15, .8, 1., 2., torch.tensor(.7, device='cuda', dtype=dtype)]:
                    expected = baseline.contrast_adaptive_sharpening(data, amount, diagonals)
                    actual = cas.contrast_adaptive_sharpening(data, amount, diagonals)
                    assert actual.shape == expected.shape and actual.dtype == expected.dtype
                    assert torch.isfinite(actual).all()
                    assert actual.min() >= 0 and actual.max() <= 1
                    error = (actual-expected).abs().max().item()
                    maximum = max(maximum, error)
                    assert error <= (0.003 if dtype == torch.float16 else 2e-6)
                assert torch.equal(data, saved)
    assert not _native._unavailable
    print(f'{dtype} diagonals={diagonals}: maximum error {maximum}')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_layout_amount_and_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    x = torch.rand(2, 3, 13, 19, device=device)
    amounts = [torch.rand(2, 1, 1, 1, device=device), torch.rand(13, 19, device=device),
               torch.rand(2, 1, 13, 19, device=device)]
    for data in [x, x.transpose(-1, -2), x.contiguous(memory_format=torch.channels_last), x.double()]:
        for a in amounts[:1] + [torch.tensor(.8, device=device)]:
            torch.testing.assert_close(cas.contrast_adaptive_sharpening(data, a),
                                       baseline.contrast_adaptive_sharpening(data, a), atol=2e-6, rtol=0)
    for a in amounts:
        torch.testing.assert_close(cas.contrast_adaptive_sharpening(x, a),
                                   baseline.contrast_adaptive_sharpening(x, a), atol=2e-6, rtol=0)
    half = x.half()
    a = torch.rand(2, 1, 1, 1, device=device)
    assert cas.contrast_adaptive_sharpening(half, a).dtype == torch.float32
    for grad_x, grad_a in [(True, False), (False, True), (True, True)]:
        data = x.clone().requires_grad_(grad_x)
        amount = torch.tensor(.8, device=device, requires_grad=grad_a)
        actual = cas.contrast_adaptive_sharpening(data, amount)
        variables = [t for t in [data, amount] if t.requires_grad]
        grads = torch.autograd.grad(actual.sum(), variables)
        expected = baseline.contrast_adaptive_sharpening(data, amount)
        refs = torch.autograd.grad(expected.sum(), variables)
        for g, r in zip(grads, refs):
            torch.testing.assert_close(g, r)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_features_and_stream(dtype):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        for channels in [1, 2, 3, 4]:
            x = torch.rand(3, channels, 57, 91, device='cuda', dtype=dtype)
            torch.testing.assert_close(extract_features(x), extract_features(x, native=False), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(cas.contrast_adaptive_sharpening(x),
                                       baseline.contrast_adaptive_sharpening(x), atol=.003, rtol=0)
    stream.synchronize()


def test_legacy_auto():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    with torch.inference_mode():
        for dtype in [torch.float16, torch.float32]:
            x = torch.rand(2, 3, 129, 231, device='cuda', dtype=dtype)
            for tiles in [0, 6]:
                actual = cas.contrast_adaptive_sharpening(x, amount=None, auto_tiles=tiles, auto_mode='legacy')
                expected = baseline.contrast_adaptive_sharpening(x, amount=None, auto_tiles=tiles)
                torch.testing.assert_close(actual, expected, atol=.003 if dtype==torch.float16 else 2e-6, rtol=0)


def test_tiled_blurred_amounts():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    with torch.inference_mode():
        torch.manual_seed(984)
        x = torch.nn.functional.interpolate(torch.rand(2, 3, 97, 151, device='cuda'),
                                             size=(541, 959), mode='bilinear', align_corners=False)
        for dtype in [torch.float32, torch.float16]:
            image = x.to(dtype)
            for tiles in [1, 6, 11]:
                actual = cas.contrast_adaptive_sharpening(image, amount=None, auto_tiles=tiles, auto_mode='legacy')
                expected = baseline.contrast_adaptive_sharpening(image, amount=None, auto_tiles=tiles)
                error = (actual-expected).abs().max().item()
                print('blurred tiles', dtype, tiles, 'error', error)
                assert error <= (.003 if dtype==torch.float16 else 2e-6)


def test_learned_policy_and_capture():
    from anime_selector import estimate_anime_amount, model_tensors, AMOUNTS
    torch.manual_seed(3857)
    for device in ['cpu', 'cuda']:
        if device=='cuda' and not torch.cuda.is_available():
            continue
        with torch.inference_mode():
            x = torch.rand(3, 3, 91, 113, device=device)
            amount = estimate_anime_amount(x)
            tensors, _, _, _ = model_tensors(device if device=='cpu' else 'cuda:0')
            mean, invstd, w1, b1, w2, b2, amounts = tensors
            feat = extract_features(x)
            hidden = torch.relu(torch.nn.functional.linear((feat-mean)*invstd, w1, b1))
            expected_amount = amounts[torch.nn.functional.linear(hidden, w2, b2).argmax(1)].view(-1,1,1,1)
            torch.testing.assert_close(amount, expected_amount)
            expected = baseline.contrast_adaptive_sharpening(x, amount.clamp(0, 1))
            expected = torch.where(amount<0, x, expected)
            torch.testing.assert_close(cas.contrast_adaptive_sharpening(x, amount=None), expected, atol=2e-6, rtol=0)
    if torch.cuda.is_available():
        with torch.inference_mode():
            x = torch.rand(1, 3, 91, 113, device='cuda')
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    expected = cas.contrast_adaptive_sharpening(x, amount=None)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = cas.contrast_adaptive_sharpening(x, amount=None)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected)


def test_tiled_warmup_model_readiness():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from anime_selector import model_tensors
    model_tensors.cache_clear()
    x = torch.rand(1,3,33,57,device='cuda')
    cas.warmup(x, auto_tiles=6)
    _, _, ready, _ = model_tensors('cuda:0')
    assert ready['complete'] and ready['event'].query()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        out = cas.contrast_adaptive_sharpening(x, amount=None)
    stream.synchronize()
    assert torch.isfinite(out).all()
