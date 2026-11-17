"""单步性能分解: 统计各模块 forward/backward 耗时 (默认 NPU)

用法:
    python tools/profile_step.py --config configs_npu/stage1.py
    python tools/profile_step.py --config configs_npu/stage1.py --device npu:0 --iters 10
"""
import sys
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from data.dataset import MapTRDataset, collate_fn
from models.maptr import MapTR
from utils.device import is_npu, synchronize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs_npu/stage1.py')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else cfg.device
    if is_npu(device):
        torch.npu.set_device(device)
    print(f'device={device}, config={args.config}')

    def sync():
        synchronize(device)

    ds = MapTRDataset(cfg.data.train_ann_file, cfg.data.data_root, cfg.data, is_train=True)
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn, drop_last=True)
    batch = next(iter(loader))
    imgs = batch['imgs'].to(device)
    intrinsics = batch['intrinsics'].to(device)
    extrinsics = batch['extrinsics'].to(device)
    batch_cpu = {k: v for k, v in batch.items() if k not in ['imgs', 'intrinsics', 'extrinsics']}

    model = MapTR(cfg.model).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    times = {}
    counts = {}

    def wrap(module, name):
        if module is None:
            return
        orig = module.forward

        def timed(*a, **kw):
            sync()
            t0 = time.time()
            out = orig(*a, **kw)
            sync()
            times[name] = times.get(name, 0.0) + (time.time() - t0)
            counts[name] = counts.get(name, 0) + 1
            return out
        module.forward = timed

    enc = model.bev_encoder
    wrap(model.backbone, 'backbone')
    wrap(model.grid_mask, 'grid_mask')
    wrap(enc.input_proj, 'bev.input_proj')
    wrap(enc.layers[0], 'layer0.total')
    wrap(enc.layers[0].self_attn, 'layer0.self_attn')
    wrap(enc.layers[0].cross_attn, 'layer0.cross_attn')
    wrap(enc.layers[0].ffn, 'layer0.ffn')
    wrap(enc, 'bev_encoder')
    wrap(model.seg_head, 'seg_head')

    def one_step(record=False):
        if record:
            times.clear(); counts.clear()
        sync()
        t0 = time.time()
        out = model(imgs, intrinsics, extrinsics, batch=batch_cpu, return_all_layers=True)
        sync()
        t_fwd = time.time() - t0

        t0 = time.time()
        loss_dict = model.compute_loss(out[0], out[1], out[2], batch_cpu, heatmap_pred=out[3])
        loss = sum(loss_dict.values())
        sync()
        t_loss = time.time() - t0

        t0 = time.time()
        optimizer.zero_grad()
        loss.backward()
        sync()
        t_bwd = time.time() - t0

        t0 = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
        optimizer.step()
        sync()
        t_step = time.time() - t0
        return t_fwd, t_loss, t_bwd, t_step, float(loss)

    for _ in range(args.warmup):
        one_step(record=False)

    totals = {'forward': 0.0, 'loss': 0.0, 'backward': 0.0, 'optim': 0.0}
    mod_acc = {}
    mod_cnt = {}
    for _ in range(args.iters):
        t_fwd, t_loss, t_bwd, t_step, loss = one_step(record=True)
        totals['forward'] += t_fwd
        totals['loss'] += t_loss
        totals['backward'] += t_bwd
        totals['optim'] += t_step
        for k, v in times.items():
            mod_acc[k] = mod_acc.get(k, 0.0) + v
            mod_cnt[k] = mod_cnt.get(k, 0) + counts[k]

    n = args.iters
    print(f'\n===== 平均单步 (n={n}, batch={cfg.data.batch_size}) =====')
    print(f'  forward  : {totals["forward"]/n:.3f}s')
    print(f'  loss     : {totals["loss"]/n:.3f}s')
    print(f'  backward : {totals["backward"]/n:.3f}s')
    print(f'  optim    : {totals["optim"]/n:.3f}s')
    print(f'  sum      : {sum(totals.values())/n:.3f}s')
    print('\n----- 模块 forward 耗时 -----')
    for k in sorted(mod_acc, key=lambda x: -mod_acc[x]):
        print(f'  {k:22s} {mod_acc[k]/n:.3f}s  (calls/iter={mod_cnt[k]/n:.1f})')


if __name__ == '__main__':
    main()
