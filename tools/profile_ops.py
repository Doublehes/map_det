"""NPU 算子级 profiling: 用 torch_npu.profiler 采集 op_summary

用法:
    python tools/profile_ops.py --config configs_npu/stage1.py --out /tmp/npu_prof
"""
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from data.dataset import MapTRDataset, collate_fn
from models.maptr import MapTR
from utils.device import is_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs_npu/stage1.py')
    parser.add_argument('--out', type=str, default='/tmp/npu_prof')
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.device
    if is_npu(device):
        torch.npu.set_device(device)
    print(f'device={device}')

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

    def one_step():
        out = model(imgs, intrinsics, extrinsics, batch=batch_cpu, return_all_layers=True)
        loss_dict = model.compute_loss(out[0], out[1], out[2], batch_cpu, heatmap_pred=out[3])
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    import torch_npu
    prof = torch_npu.profiler
    Path(args.out).mkdir(parents=True, exist_ok=True)
    acts = [prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU]
    schedule = prof.schedule(wait=0, warmup=1, active=args.steps - 1, repeat=1)
    experimental_config = prof._ExperimentalConfig(
        profiler_level=prof.ProfilerLevel.Level1)

    with prof.profile(activities=acts, schedule=schedule,
                      on_trace_ready=prof.tensorboard_trace_handler(args.out),
                      experimental_config=experimental_config) as p:
        for _ in range(args.steps):
            one_step()
            p.step()

    print(f'PROFILER_OUT={args.out}')


if __name__ == '__main__':
    main()
