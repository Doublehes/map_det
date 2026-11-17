"""设备相关工具: NPU/CUDA 兼容的同步、AMP、线程控制"""
from contextlib import nullcontext

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def is_npu(device) -> bool:
    return getattr(device, 'type', str(device)) == 'npu'


def synchronize(device):
    """同步设备 (仅 NPU/CUDA 有效), 用于准确计时"""
    dev_type = getattr(device, 'type', None)
    if dev_type == 'npu':
        torch.npu.synchronize()
    elif dev_type == 'cuda':
        torch.cuda.synchronize()


def setup_worker_threads(num_threads: int = 1):
    """DataLoader worker 内限制线程数, 避免多 worker × 多线程超订阅"""
    try:
        torch.set_num_threads(max(1, int(num_threads)))
    except Exception:
        pass
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass


def amp_context(device, enabled: bool, dtype: str = 'float16'):
    """返回 autocast 上下文; 未启用时为空上下文"""
    if not enabled:
        return nullcontext()
    torch_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
    dev_type = getattr(device, 'type', None)
    if dev_type == 'npu':
        return torch.npu.amp.autocast(dtype=torch_dtype)
    if dev_type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch_dtype)
    return nullcontext()


def build_grad_scaler(device, enabled: bool, dtype: str = 'float16'):
    """构建 GradScaler; bf16 或未启用时返回 None"""
    if not enabled or dtype == 'bfloat16':
        return None
    dev_type = getattr(device, 'type', None)
    if dev_type == 'npu':
        return torch.npu.amp.GradScaler()
    if dev_type == 'cuda':
        return torch.cuda.amp.GradScaler()
    return None
