import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.baseline_cas import contrast_adaptive_sharpening
torch.set_num_threads(4)
x = torch.rand(1, 3, 720, 1280, device='cuda', dtype=torch.float16)
with torch.inference_mode():
    for _ in range(20):
        contrast_adaptive_sharpening(x, amount=None)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(10):
            contrast_adaptive_sharpening(x, amount=None)
        torch.cuda.synchronize()
Path('reports/profile_baseline.txt').write_text(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=30))
print(Path('reports/profile_baseline.txt').read_text())
