import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, MultiStepLR, LinearLR, SequentialLR,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.default import cfg
from data.dataset import MapTRDataset, collate_fn
from models.maptr import MapTR
from eval import run_eval
from utils.timer import Timer

def _get_lr_str(optimizer):
    lrs = sorted(set(round(g['lr'], 8) for g in optimizer.param_groups))
    return 'lr=' + ', '.join(f'{lr:.2e}' for lr in lrs)


def build_optimizer(model, cfg):
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_mult = cfg.backbone_lr_mult if 'backbone' in name else 1.0
        param_groups.append({
            'params': [param],
            'lr': cfg.lr * lr_mult,
            'weight_decay': cfg.weight_decay,
        })
    return AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def train_one_epoch(model, loader, optimizer, scheduler, epoch, cfg, seg_only=False):
    model.train()
    total_loss = total_cls_loss = total_reg_loss = total_seg_loss = 0.0

    data_times, model_times = [], []
    epoch_start = time.time()

    t_data = time.time()
    for batch_idx, batch in enumerate(loader):
        
        imgs = batch['imgs'].to(cfg.device)
        intrinsics = batch['intrinsics'].to(cfg.device)
        extrinsics = batch['extrinsics'].to(cfg.device)

        t_model = time.time()
        cls_scores, reg_preds, seg_preds = model(imgs, intrinsics, extrinsics, seg_only=seg_only)

        batch_cpu = {k: v for k, v in batch.items() if k not in ['imgs', 'intrinsics', 'extrinsics']}

        loss_dict = model.compute_loss(cls_scores, reg_preds, seg_preds, batch_cpu, seg_only=seg_only)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
        optimizer.step()
        t_end = time.time()

        data_times.append(t_model - t_data)
        model_times.append(t_end - t_model)

        total_loss += loss.item()
        total_cls_loss += loss_dict.get('cls_loss', torch.tensor(0.0)).item()
        total_reg_loss += loss_dict.get('reg_loss', torch.tensor(0.0)).item()
        total_seg_loss += loss_dict.get('seg_loss', torch.tensor(0.0)).item() + loss_dict.get('dice_loss', torch.tensor(0.0)).item()

        if batch_idx % 50 == 0:
            avg_data = sum(data_times[-50:]) / min(len(data_times), 50)
            avg_model = sum(model_times[-50:]) / min(len(model_times), 50)
            iter_time = avg_data + avg_model
            iters_done = epoch * len(loader) + batch_idx + 1
            iters_total = cfg.num_epochs * len(loader)
            eta = (iters_total - iters_done) * iter_time
            line_loss = '' if seg_only else f'cls={loss_dict.get("cls_loss",0):.4f} reg={loss_dict.get("reg_loss",0):.4f} '
            log = (
                f'[E {epoch+1}/{cfg.num_epochs}] [{batch_idx}/{len(loader)}] '
                f'ETA={eta/60:.0f}min '
                f'{_get_lr_str(optimizer)} '
                f'data_t={avg_data:.3f}s '
                f'model_t={avg_model:.3f}s '
                f'loss_total={loss.item():.4f} '
                f'{line_loss}'
                f'seg={loss_dict.get("seg_loss",0):.4f}+{loss_dict.get("dice_loss",0):.4f} '
            )
            print(log)

        if scheduler is not None:
            scheduler.step()
        
        t_data = time.time()

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / len(loader)
    print(f'[Epoch {epoch+1}] 平均 loss={avg_loss:.4f} {_get_lr_str(optimizer)} 耗时={epoch_time:.0f}s')
    return avg_loss


def save_checkpoint(model, optimizer, epoch, cfg, save_dir, filename=None):
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f'epoch_{epoch+1}.pth'
    path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f'[保存] {path}')


def load_pretrained(model, checkpoint_path, cfg, freeze_backbone=False):
    print(f'[预训练] 加载: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  [警告] 缺少的key ({len(missing)}): {missing[:5]}...')
    if unexpected:
        print(f'  [警告] 多余的key ({len(unexpected)}): {unexpected[:5]}...')
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'  [冻结] backbone {frozen/1e6:.1f}M/{total/1e6:.1f}M 参数已冻结')
    print('  [完成] 预训练权重加载')


def main():
    parser = argparse.ArgumentParser(description='MapTR 道路结构线检测训练')
    parser.add_argument('--work-dir', type=str, default='./work_dirs/maptr', help='工作目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的 checkpoint')
    parser.add_argument('--pretrained', type=str, default=None, help='预训练权重 (仅加载模型, 从epoch0开始)')
    parser.add_argument('--freeze-backbone', action='store_true', help='冻结backbone只训练其余部分')
    parser.add_argument('--seg-only', action='store_true', help='仅训练分割头, 跳过线分类和回归')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖 cfg.num_epochs')
    parser.add_argument('--eval-interval', type=int, default=1, help='每 N 个 epoch 执行一次评测 (0=禁用)')
    args = parser.parse_args()

    print(f'[设备] {cfg.device}')
    print(f'[配置] num_epochs={cfg.num_epochs}, batch_size={cfg.batch_size}, lr={cfg.lr}')
    print(f'[数据] train={cfg.train_ann_file}')

    train_dataset = MapTRDataset(
        ann_file=cfg.train_ann_file,
        data_root=cfg.data_root,
        cfg=cfg,
        is_train=True,
    )

    val_dataset = MapTRDataset(
        ann_file=cfg.val_ann_file,
        data_root=cfg.data_root,
        cfg=cfg,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = MapTR(cfg).to(cfg.device)
    # model.bev_encoder.debug_dir = "./debug_bev_porj"
    # print(model)
    start_epoch = 0

    if args.epochs is not None:
        cfg.num_epochs = args.epochs

    if args.pretrained and os.path.exists(args.pretrained):
        load_pretrained(model, args.pretrained, cfg, args.freeze_backbone)

    if args.seg_only:
        for name, param in model.named_parameters():
            if 'decoder' in name or name.startswith('head.'):
                param.requires_grad = False
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'[分割模式] decoder + head 已冻结, {frozen/1e6:.1f}M/{total/1e6:.1f}M 参数冻结')

    optimizer = build_optimizer(model, cfg)

    total_iters = cfg.num_epochs * len(train_loader)
    if cfg.scheduler == 'none':
        scheduler = None
    else:
        if cfg.scheduler == 'step':
            milestones = [m * len(train_loader) for m in cfg.lr_milestones]
            main_lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=cfg.lr_gamma)
        else:
            main_lr_scheduler = CosineAnnealingLR(
                optimizer, T_max=total_iters - cfg.warmup_iters,
                eta_min=cfg.lr * cfg.min_lr_ratio)

        if cfg.warmup_iters > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=cfg.warmup_ratio, total_iters=cfg.warmup_iters)
            scheduler = SequentialLR(
                optimizer, [warmup_scheduler, main_lr_scheduler],
                milestones=[cfg.warmup_iters])
        else:
            scheduler = main_lr_scheduler

    if args.resume and os.path.exists(args.resume):
        print(f'[恢复] 从 {args.resume} 恢复训练')
        checkpoint = torch.load(args.resume, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[模型] 总参数: {total_params/1e6:.2f}M, 可训练: {trainable_params/1e6:.2f}M')

    timer = Timer()
    for epoch in range(start_epoch, cfg.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg, seg_only=args.seg_only)
        save_checkpoint(model, optimizer, epoch, cfg, args.work_dir)

        if not args.seg_only and args.eval_interval > 0 and (epoch + 1) % args.eval_interval == 0:
            print(f'\n{"="*50}\n[评测] Epoch {epoch+1}')
            model.eval()
            with timer('评测整体'):
                results, evaluator = run_eval(model, val_loader, cfg, n_workers=4)
                evaluator.print_results(results)
            model.train()

    print('[训练完成]')


if __name__ == '__main__':
    main()
