"""Fresh-process cold backend/model initialization after PyTorch device init."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import cas
torch.cuda.init()
x = torch.rand(1, 3, 720, 1280, device='cuda', dtype=torch.float16)
torch.cuda.synchronize()
start = time.perf_counter()
cas.warmup(x)
elapsed = (time.perf_counter()-start)*1000
result = {'warmup_ms': elapsed, 'boundary': 'first NVRTC compile, CUDA module load, model upload and filtering; PyTorch import/device init excluded',
          'persistent_disk_cache': False}
Path('reports/cold_start.json').write_text(json.dumps(result, indent=2))
print(result)
