"""model_slim2 训练: 2D 分割图 → 线检测 (自监督, 无线 GT)

运行: conda run -n workspace python model_slim2/train.py [--config default]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_slim2.config as cfg_module
from model_slim2.dataset import Slim2Dataset, collate_fn
from model_slim2.losses import sample_map, loss_shares
from model_slim2.model import Slim2Model


def load_pretrained(model, checkpoint_path, device):
    print(f'[预训练] 加载: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  [警告] 缺少的key ({len(missing)}): {missing[:5]}...')
    if unexpected:
        print(f'  [警告] 多余的key ({len(unexpected)}): {unexpected[:5]}...')
    print('  [完成] 预训练权重加载')


@torch.no_grad()
def run_val(model, loader, device):
    """无 GT 验证 (自监督, 无人工标注)。

    把所有预测线 (B, N, P, 2) 的归一化 (u,v) 坐标汇总, 分为两类指标:

    一、与监督信号相关的"正确性"指标
        val_cost      : 预测点采样代价图的 Lp 均值。点落在低代价(可通行)区 -> 越低越好
        val_interior  : 预测点采样距离场的平均值。值越大越靠可通行区几何中心
                        (距离场 = 到边界的距离, 脊线处最大)

    二、与监督信号无关的"几何诊断"指标 (仅看形状是否合理, 无线 GT)
        记一条线的 P 个点为 p0..p15, 相邻段长 s_i = ||p_{i+1} - p_i||:

        val_arc       : 平均弧长 = mean(Σ s_i)。即折线真实路径长度(归一化坐标)
        val_span      : 平均净跨 = mean(||p15 - p0||)。首尾端点的直线距离
        val_arc_span  : 缠绕度 = mean(arc / span)。1=笔直, 越大越卷曲
                        (圆滑弯道 ~1~2; 十几以上说明打结/绕圈/塌成小团)
        val_turn      : 平均转角 = mean(mean(1 - cosθ_i))。θ 为相邻段夹角
                        0=全程笔直, 2=每步掉头。越小越顺
        val_spacing_cv: 等距度 = mean(std(s) / mean(s))。段长的变异系数(尺度无关)
                        0=各段完全等长(等距), 越大越不均匀

    注: arc/span/turn/spacing_cv 均在归一化坐标上计算, 用于训练期快速判断
    线是在"敞开、顺畅、等距", 还是打结/塌缩/输入无关(跨样本方差为 0)。
    """
    model.eval()
    tot = {'cost': 0.0, 'interior': 0.0, 'arc': 0.0, 'span': 0.0,
           'arc_span': 0.0, 'turn': 0.0, 'spacing_cv': 0.0,
           'count': 0.0, 'n': 0, 'lines': 0}
    for batch in loader:
        inp = batch['input'].to(device)
        cost = batch['cost'].to(device)
        dist = batch['dist_field'].to(device)
        _, points = model(inp)
        B, N, P, _ = points.shape

        samp_cost = sample_map(cost, points)
        samp_dist = sample_map(dist, points)
        valid = (samp_dist > 0).float().clamp_min(1e-6)

        # ── 几何量 (每条线一个值, 均为归一化坐标) ──
        seg = (points[:, :, 1:] - points[:, :, :-1]).norm(dim=-1)   # (B,N,P-1) 相邻段长 s_i
        arc = seg.sum(dim=-1)                                       # (B,N) 弧长 Σs_i (真实长度)
        span = (points[:, :, -1] - points[:, :, 0]).norm(dim=-1)    # (B,N) 净跨 ‖p15-p0‖

        v = points[:, :, 1:] - points[:, :, :-1]
        u = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)        # 单位方向
        cos = (u[:, :, :-1] * u[:, :, 1:]).sum(dim=-1)              # 相邻段夹角余弦
        turn = (1.0 - cos).mean(dim=-1)                            # (B,N) 平均转角, 0=直
        spacing_cv = seg.std(dim=-1) / seg.mean(dim=-1).clamp_min(1e-6)  # (B,N) 等距变异系数

        tot['cost'] += (samp_cost * valid).sum().item()
        tot['interior'] += (samp_dist * valid).sum().item()
        n_lines = B * N
        tot['arc'] += arc.mean().item() * n_lines
        tot['span'] += span.mean().item() * n_lines
        tot['arc_span'] += (arc / span.clamp_min(1e-6)).mean().item() * n_lines
        tot['turn'] += turn.mean().item() * n_lines
        tot['spacing_cv'] += spacing_cv.mean().item() * n_lines
        tot['lines'] += n_lines
        tot['count'] += float(valid.sum().item()) or B * N
        tot['n'] += 1

    model.train()
    c = max(tot['count'], 1.0)
    nl = max(tot['lines'], 1)
    return {
        # 正确性 (依赖监督信号)
        'val_cost': tot['cost'] / c,          # 采样代价, 越低越好
        'val_interior': tot['interior'] / c,  # 采样距离场均值, 越高越好 (越居中)
        # 几何诊断 (只看形状, 无线 GT)
        'val_arc': tot['arc'] / nl,           # 弧长 Σs_i, 应≈区域中心线长度
        'val_span': tot['span'] / nl,         # 净跨 ‖p15-p0‖, 应与 arc 相近
        'val_arc_span': tot['arc_span'] / nl, # 缠绕度 arc/span: 1=笔直, ~1-2=圆滑弯道, 十几=打结/绕圈
        'val_turn': tot['turn'] / nl,         # 平均转角: 0=笔直, 越大越折
        'val_spacing_cv': tot['spacing_cv'] / nl,  # 等距变异系数: 0=完全等距
    }


def main():
    parser = argparse.ArgumentParser(description='Slim2 训练')
    parser.add_argument('--config', default='default', choices=list(cfg_module.CONFIGS.keys()))
    args = parser.parse_args()
    cfg = cfg_module.CONFIGS[args.config]

    if os.path.exists(cfg.work_dir):
        print(f'[警告] 工作目录已存在: {cfg.work_dir}')
        import datetime
        cfg.work_dir += f'_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(cfg.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'logs'))
    print(f'[设备] {cfg.device}')
    print(f'[工作目录] {cfg.work_dir}')

    train_dataset = Slim2Dataset(cfg.data.seg_dir, cfg.data, is_train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn, drop_last=True)

    val_loader = None
    if cfg.get('val_interval', 0) > 0:
        val_dataset = Slim2Dataset(cfg.data.seg_dir, cfg.data, is_train=False)
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers, collate_fn=collate_fn, drop_last=False)

    model = Slim2Model(cfg).to(cfg.device)
    if cfg.pretrained and os.path.exists(cfg.pretrained):
        load_pretrained(model, cfg.pretrained, cfg.device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_iters = cfg.num_epochs * len(train_loader)
    if cfg.scheduler == 'cosine':
        main_lr = CosineAnnealingLR(optimizer, T_max=max(total_iters - cfg.warmup_iters, 1),
                                    eta_min=cfg.lr * 0.01)
        if cfg.warmup_iters > 0:
            warmup = LinearLR(optimizer, start_factor=1.0 / 3, total_iters=cfg.warmup_iters)
            scheduler = SequentialLR(optimizer, [warmup, main_lr], milestones=[cfg.warmup_iters])
        else:
            scheduler = main_lr
    else:
        scheduler = None

    print(f'[模型] 总参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    global_step = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            inp = batch['input'].to(cfg.device)
            cls_scores, reg_preds = model(inp)
            loss_dict = model.compute_loss(cls_scores, reg_preds, batch)
            loss = loss_dict['total']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
            optimizer.step()

            total_loss += loss.item()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            for k, v in loss_dict.items():
                writer.add_scalar(f'loss/{k}', v.item(), global_step)
            shares = loss_shares(loss_dict)
            for k, v in shares.items():
                writer.add_scalar(f'share/{k}', v, global_step)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

            if batch_idx % cfg.log_interval == 0:
                items = ' '.join(f'{k}={v.item():.3f}' for k, v in loss_dict.items())
                share_str = ' '.join(f'{k}={100 * v:.0f}%' for k, v in shares.items())
                print(f'[E {epoch + 1}/{cfg.num_epochs}] [{batch_idx}/{len(train_loader)}] {items}')
                print(f'    share: {share_str}')

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f'[Epoch {epoch + 1}] 平均 loss={avg_loss:.4f} 耗时={time.time() - t0:.0f}s')
        writer.add_scalar('epoch/avg_loss', avg_loss, epoch)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        # torch.save(ckpt, os.path.join(cfg.work_dir, 'latest.pth'))
        if (epoch + 1) % cfg.checkpoint_interval == 0:
            torch.save(ckpt, os.path.join(cfg.work_dir, f'epoch_{epoch + 1}.pth'))
        print(f'[保存] {cfg.work_dir}')

        if val_loader is not None and (epoch + 1) % cfg.val_interval == 0:
            metrics = run_val(model, val_loader, cfg.device)
            print(f'[Val Epoch {epoch + 1}] ' +
                  ' '.join(f'{k}={v:.4f}' for k, v in metrics.items()))
            for k, v in metrics.items():
                writer.add_scalar(f'metrics/{k}', v, epoch)

    writer.close()
    print('[训练完成]')


if __name__ == '__main__':
    main()
