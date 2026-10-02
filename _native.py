"""Optional NVRTC CUDA inference kernels; no compiler executable or extra package.

Only contiguous CUDA fp16/fp32 inference uses this path. PyTorch owns all tensors
and streams. Missing NVRTC falls back once; kernel launch errors are surfaced.
"""
import ctypes as ct
import ctypes.util
import os
from pathlib import Path
import threading
import warnings

import torch

_lock = threading.RLock()
_runtime = None
_unavailable = False
_modules = {}


class Runtime:
    def __init__(self):
        lib = Path(torch.__file__).parent / 'lib'
        if os.name == 'nt':
            self.driver = ct.WinDLL('nvcuda.dll')
            paths = list(lib.glob('nvrtc64*.dll'))
            if not paths:
                raise OSError('NVRTC DLL not found in PyTorch/lib')
            self.rtc = ct.CDLL(str(paths[0]))
        else:
            self.driver = ct.CDLL('libcuda.so.1')
            paths = list(lib.glob('libnvrtc.so*'))
            paths += list((lib.parent.parent / 'nvidia/cuda_nvrtc/lib').glob('libnvrtc.so*'))
            self.rtc = ct.CDLL(str(paths[0]) if paths else (ct.util.find_library('nvrtc') or 'libnvrtc.so'))
        signatures = {
            'cuModuleLoadData': [ct.POINTER(ct.c_void_p), ct.c_void_p],
            'cuModuleGetFunction': [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
            'cuLaunchKernel': [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p],
        }
        for name, sig in signatures.items():
            getattr(self.driver, name).argtypes = sig
            getattr(self.driver, name).restype = ct.c_int
        signatures = {
            'nvrtcCreateProgram': [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p],
            'nvrtcCompileProgram': [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)],
            'nvrtcGetPTXSize': [ct.c_void_p, ct.POINTER(ct.c_size_t)],
            'nvrtcGetPTX': [ct.c_void_p, ct.c_void_p],
            'nvrtcGetProgramLogSize': [ct.c_void_p, ct.POINTER(ct.c_size_t)],
            'nvrtcGetProgramLog': [ct.c_void_p, ct.c_void_p],
            'nvrtcDestroyProgram': [ct.POINTER(ct.c_void_p)],
        }
        for name, sig in signatures.items():
            getattr(self.rtc, name).argtypes = sig
            getattr(self.rtc, name).restype = ct.c_int

    @staticmethod
    def check(code, operation):
        if code:
            raise RuntimeError(f'{operation} failed with CUDA code {code}')

    def compile(self, half, diagonals, device):
        major, minor = torch.cuda.get_device_capability(device)
        source = Path(__file__).with_name('kernels.cu').read_bytes()
        opts = [f'--gpu-architecture=compute_{major}{minor}'.encode(), b'--std=c++11',
                b'--fmad=false', f'-DHALF={int(half)}'.encode(), f'-DDIAGONALS={int(diagonals)}'.encode()]
        prog = ct.c_void_p()
        self.check(self.rtc.nvrtcCreateProgram(ct.byref(prog), source, b'autocas.cu', 0, None, None), 'create program')
        try:
            code = self.rtc.nvrtcCompileProgram(prog, len(opts), (ct.c_char_p * len(opts))(*opts))
            if code:
                size = ct.c_size_t()
                self.rtc.nvrtcGetProgramLogSize(prog, ct.byref(size))
                log = ct.create_string_buffer(size.value)
                self.rtc.nvrtcGetProgramLog(prog, log)
                raise RuntimeError(log.value.decode(errors='replace'))
            size = ct.c_size_t()
            self.check(self.rtc.nvrtcGetPTXSize(prog, ct.byref(size)), 'PTX size')
            ptx = ct.create_string_buffer(size.value)
            self.check(self.rtc.nvrtcGetPTX(prog, ptx), 'get PTX')
            module = ct.c_void_p()
            self.check(self.driver.cuModuleLoadData(ct.byref(module), ptx), 'load module')
            functions = {}
            for name in ['cas_kernel', 'features_kernel', 'classify_kernel', 'tiled_pool_kernel']:
                fn = ct.c_void_p()
                self.check(self.driver.cuModuleGetFunction(ct.byref(fn), module, name.encode()), 'get function')
                functions[name] = fn
            # Keep modules for the process lifetime; pointers must remain valid.
            return module, functions
        finally:
            self.rtc.nvrtcDestroyProgram(ct.byref(prog))

    def launch(self, fn, grid, block, stream, values):
        params = (ct.c_void_p * len(values))(*(ct.addressof(v) for v in values))
        self.check(self.driver.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0,
                                              ct.c_void_p(stream), params, None), 'launch kernel')


def functions(x, diagonals=True):
    global _runtime, _unavailable
    if _unavailable or os.environ.get('AUTOCAS_DISABLE_CUDA') == '1':
        return None
    key = (x.device.index, x.dtype, diagonals)
    with _lock:
        if key in _modules:
            return _modules[key]
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing():
                return None  # Never compile or allocate model state during capture.
        try:
            if _runtime is None:
                _runtime = Runtime()
            with torch.cuda.device(x.device):
                _modules[key] = _runtime.compile(x.dtype == torch.float16, diagonals, x.device)
        except (OSError, RuntimeError) as exc:
            _unavailable = True
            warnings.warn(f'AutoCAS CUDA backend unavailable; using PyTorch: {exc}', RuntimeWarning, stacklevel=2)
            return None
        return _modules[key]


def eligible(x, amount=None):
    if not (torch.version.cuda and x.is_cuda and x.dtype in (torch.float16, torch.float32) and x.is_contiguous()
            and x.ndim == 4 and x.shape[0] <= 65535 and x.numel() > 0 and x.numel() <= 2**31 - 1):
        return False
    if torch.is_grad_enabled() and (x.requires_grad or (torch.is_tensor(amount) and amount.requires_grad)):
        return False
    return (not torch.is_tensor(amount) or
            (amount.device == x.device and amount.dtype == x.dtype and amount.is_contiguous() and
             (amount.ndim == 0 or tuple(amount.shape) in [(1, 1, 1, 1), (x.shape[0], 1, 1, 1),
                                                         (x.shape[0], 1, x.shape[2], x.shape[3]),
                                                         (1, 1, x.shape[2], x.shape[3])])))


def sharpen(x, amount, diagonals=True, bypass=False):
    if not eligible(x, amount):
        return None
    module = functions(x, diagonals)
    if module is None:
        return None
    out = torch.empty_like(x)
    if torch.is_tensor(amount):
        if amount.numel() == 1:
            mode = 1
        elif amount.shape[-2:] == x.shape[-2:] and amount.ndim == 4:
            mode = 3 if amount.shape[0] == x.shape[0] else 4
        else:
            mode = 2
        ptr, scalar = amount.data_ptr(), 0.
    else:
        mode, ptr, scalar = 0, 0, float(amount)
    values = [ct.c_void_p(x.data_ptr()), ct.c_void_p(out.data_ptr()), ct.c_int(x.numel()),
              ct.c_int(x.shape[-2]), ct.c_int(x.shape[-1]), ct.c_int(x.shape[1]),
              ct.c_void_p(ptr), ct.c_int(mode), ct.c_float(scalar), ct.c_int(bypass)]
    with torch.cuda.device(x.device):
        _runtime.launch(module[1]['cas_kernel'], (x.numel() + 255) // 256, 256,
                        torch.cuda.current_stream(x.device).cuda_stream, values)
    return out


def features(x):
    if not eligible(x):
        return None
    module = functions(x)
    if module is None:
        return None
    out = torch.empty((x.shape[0], 16), device=x.device, dtype=torch.float32)
    values = [ct.c_void_p(x.data_ptr()), ct.c_void_p(out.data_ptr()), ct.c_int(x.shape[1]),
              ct.c_int(x.shape[-2]), ct.c_int(x.shape[-1])]
    with torch.cuda.device(x.device):
        _runtime.launch(module[1]['features_kernel'], x.shape[0], 256,
                        torch.cuda.current_stream(x.device).cuda_stream, values)
    return out


def tiled_pool(hf, contrast, tiles):
    if not eligible(hf) or tiles > 64:
        return None
    module = functions(hf)
    if module is None:
        return None
    out = torch.empty((hf.shape[0], 1, tiles, tiles), device=hf.device, dtype=torch.float32)
    values = [ct.c_void_p(hf.data_ptr()), ct.c_void_p(contrast.data_ptr()), ct.c_void_p(out.data_ptr()),
              ct.c_int(hf.shape[-2]), ct.c_int(hf.shape[-1]), ct.c_int(tiles)]
    with torch.cuda.device(hf.device):
        _runtime.launch(module[1]['tiled_pool_kernel'], hf.shape[0]*tiles*tiles, 256,
                        torch.cuda.current_stream(hf.device).cuda_stream, values)
    return out


def classify(x, feature_tensor, weights, classes):
    module = functions(x)
    if module is None:
        return None
    out = torch.empty((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype)
    values = [ct.c_void_p(feature_tensor.data_ptr()), ct.c_void_p(weights.data_ptr()),
              ct.c_void_p(out.data_ptr()), ct.c_int(x.shape[0]), ct.c_int(classes)]
    with torch.cuda.device(x.device):
        _runtime.launch(module[1]['classify_kernel'], (x.shape[0] + 31) // 32, 32,
                        torch.cuda.current_stream(x.device).cuda_stream, values)
    return out
