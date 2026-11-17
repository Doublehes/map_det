"""NPU 冒烟测试: 基础算子 + 真实数据单 batch 前向/反向 + 计时

用法:
    python tools/npu_smoke_test.py --config configs_npu/stage1.py
    python tools/npu_smoke_test.py --ops-only
"""
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_ops(device):
    print(f'\n===== 基础算子测试 (device={device}) =====')
    torch.manual_seed(0)

    def check(name, fn):
        try:
            out = fn()
            if isinstance(out, torch.Tensor):
                torch.npu.synchronize() if device.type == 'npu' else None
                print(f'  [OK]   {name}  shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()}')
            else:
                print(f'  [OK]   {name}')
            return True
        except Exception as e:
            print(f'  [FAIL] {name}: {type(e).__name__}: {e}')
            return False

    ok = True
    x = torch.randn(4, 8, device=device, requires_grad=True)
    ok &= check('matmul', lambda: x @ x.t())

    conv = nn.Conv2d(8, 16, 3, padding=1).to(device)
    img = torch.randn(2, 8, 32, 32, device=device)
    ok &= check('conv2d', lambda: conv(img))

    mha = nn.MultiheadAttention(64, 8, batch_first=True).to(device)
    q = torch.randn(2, 16, 64, device=device)
    ok &= check('multihead_attention', lambda: mha(q, q, q)[0])

    feat = torch.randn(2, 8, 16, 16, device=device)
    grid = torch.rand(2, 20, 20, 2, device=device) * 2 - 1
    ok &= check('grid_sample', lambda: F.grid_sample(feat, grid, align_corners=False))

    theta = torch.tensor([[1.0, 0.0, 0.1], [0.0, 1.0, 0.0]], device=device).unsqueeze(0)
    ok &= check('affine_grid', lambda: F.affine_grid(theta, (1, 3, 16, 16), align_corners=False))

    interpolate = lambda: F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
    ok &= check('interpolate', interpolate)

    lin = nn.Linear(8, 8).to(device)
    y = lin(x)
    ok &= check('backward', lambda: (y.sum().backward() or y.sum()))
    return ok


def test_model(config_path, device):
    from configs.loader import load_config
    from data.dataset import MapTRDataset, collate_fn
    from models.maptr import MapTR

    print(f'\n===== 模型单 batch 测试 (config={config_path}) =====')
    cfg = load_config(config_path)
    cfg.device = device
    print(f'  device={cfg.device}')
    print(f'  data_root={cfg.data.data_root}')

    ds = MapTRDataset(cfg.data.train_ann_file, cfg.data.data_root, cfg.data, is_train=True)
    sample = ds[0]
    batch = collate_fn([sample, sample])
    print(f'  imgs={tuple(batch["imgs"].shape)} vectors={[list(v.keys()) for v in batch["vectors"]]}')

    model = MapTR(cfg.model).to(device)
    model.train()
    total = sum(p.numel() for p in model.parameters())
    print(f'  参数量: {total/1e6:.2f}M')

    imgs = batch['imgs'].to(device)
    intrinsics = batch['intrinsics'].to(device)
    extrinsics = batch['extrinsics'].to(device)
    batch_cpu = {k: v for k, v in batch.items() if k not in ['imgs', 'intrinsics', 'extrinsics']}

    # warmup
    for _ in range(2):
        out = model(imgs, intrinsics, extrinsics, batch=batch_cpu, return_all_layers=True)
        loss_dict = model.compute_loss(out[0], out[1], out[2], batch_cpu, heatmap_pred=out[3])
        loss = sum(loss_dict.values())
        loss.backward()
        model.zero_grad()

    sync = (lambda: torch.npu.synchronize()) if device.type == 'npu' else (lambda: None)
    sync()
    t0 = time.time()
    out = model(imgs, intrinsics, extrinsics, batch=batch_cpu, return_all_layers=True)
    loss_dict = model.compute_loss(out[0], out[1], out[2], batch_cpu, heatmap_pred=out[3])
    loss = sum(loss_dict.values())
    loss.backward()
    sync()
    dt = time.time() - t0
    print(f'  losses: { {k: float(v) for k, v in loss_dict.items()} }')
    print(f'  total loss={float(loss):.4f}  finite={torch.isfinite(loss).item()}')
    print(f'  单 step 前向+反向耗时: {dt:.3f}s')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs_npu/stage1.py')
    parser.add_argument('--ops-only', action='store_true')
    args = parser.parse_args()

    print(f'torch={torch.__version__}')
    try:
        import torch_npu
        print(f'torch_npu={torch_npu.__version__}')
        print(f'npu.is_available={torch.npu.is_available()}  device_count={torch.npu.device_count()}')
        device = torch.device('npu:0')
        torch.npu.set_device(device)
    except Exception as e:
        print(f'torch_npu 不可用: {e}')
        device = torch.device('cpu')

    ok = test_ops(device)
    if ok and not args.ops_only:
        ok = test_model(args.config, device)
    print('\n===== 冒烟测试结果:', 'PASS' if ok else 'FAIL', '=====')


if __name__ == '__main__':
    main()
