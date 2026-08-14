import os
import sys
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_slim.config as cfg_module
from model_slim.dataset import SlimDataset, collate_fn
from model_slim.model import SlimModel
from model_slim.eval import run_eval


def build_optimizer(model, cfg):
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_groups.append({'params': [param], 'lr': cfg.lr, 'weight_decay': cfg.weight_decay})
    return AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def load_pretrained(model, checkpoint_path, device):
    print(f'[预训练] 加载: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)   # 兼容 完整ckpt / 裸 state_dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  [警告] 缺少的key ({len(missing)}): {missing[:5]}...')
    if unexpected:
        print(f'  [警告] 多余的key ({len(unexpected)}): {unexpected[:5]}...')
    print('  [完成] 预训练权重加载')


def main():
    parser = argparse.ArgumentParser(description='Slim 模型训练')
    parser.add_argument('--config', default='default', choices=list(cfg_module.CONFIGS.keys()),
                        help='配置变体: default / resnet34 / decode_layer3')
    args = parser.parse_args()
    cfg = cfg_module.CONFIGS[args.config]

    os.makedirs(cfg.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'logs'))
    print(f'[设备] {cfg.device}')
    print(f'[数据] train={cfg.data.train_ann_file}, val={cfg.data.val_ann_file}, batch_size={cfg.data.batch_size}')
    print(f'[工作目录] {cfg.work_dir}')

    train_dataset = SlimDataset(cfg.data.train_ann_file, cfg.data, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = None
    if getattr(cfg, 'val_interval', 0) > 0:
        val_dataset = SlimDataset(cfg.data.val_ann_file, cfg.data, is_train=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )

    model = SlimModel(cfg).to(cfg.device)

    if cfg.pretrained and os.path.exists(cfg.pretrained):
        load_pretrained(model, cfg.pretrained, cfg.device)

    optimizer = build_optimizer(model, cfg)

    total_iters = cfg.num_epochs * len(train_loader)
    if cfg.scheduler == 'cosine':
        main_lr = CosineAnnealingLR(
            optimizer, T_max=total_iters - cfg.warmup_iters, eta_min=cfg.lr * 0.01)
        if cfg.warmup_iters > 0:
            warmup = LinearLR(optimizer, start_factor=1.0 / 3, total_iters=cfg.warmup_iters)
            scheduler = SequentialLR(optimizer, [warmup, main_lr], milestones=[cfg.warmup_iters])
        else:
            scheduler = main_lr
    else:
        scheduler = None

    print(f'[模型] 总参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    global_step = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            raster = batch['raster'].to(cfg.device)
            cls_scores, reg_preds, seg_pred, _ = model(raster, return_all_layers=True)
            loss_dict = model.compute_loss(cls_scores, reg_preds, seg_pred, batch)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
            optimizer.step()

            total_loss += loss.item()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            cls_l = loss_dict.get('cls_loss', torch.tensor(0.0)).item()
            reg_l = loss_dict.get('reg_loss', torch.tensor(0.0)).item()
            seg_l = loss_dict.get('seg_loss', torch.tensor(0.0)).item()
            dice_l = loss_dict.get('dice_loss', torch.tensor(0.0)).item()
            writer.add_scalar('loss/total', loss.item(), global_step)
            writer.add_scalar('loss/cls', cls_l, global_step)
            writer.add_scalar('loss/reg', reg_l, global_step)
            writer.add_scalar('loss/seg', seg_l, global_step)
            writer.add_scalar('loss/dice', dice_l, global_step)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

            if batch_idx % cfg.log_interval == 0:
                cls_list = model.head.last_layer_cls_losses or []
                reg_list = model.head.last_layer_reg_losses or []
                layer_str = '  '.join(
                    f'L{l}: cls={c.item():.4f} reg={r.item():.4f}'
                    for l, (c, r) in enumerate(zip(cls_list, reg_list))
                    if c is not None)
                print(f'[E {epoch+1}/{cfg.num_epochs}] [{batch_idx}/{len(train_loader)}] '
                      f'loss={loss.item():.4f} cls={cls_l:.4f} reg={reg_l:.4f} '
                      f'seg={seg_l:.4f} dice={dice_l:.4f} | {layer_str}')

        avg_loss = total_loss / len(train_loader)
        print(f'[Epoch {epoch+1}] 平均 loss={avg_loss:.4f} 耗时={time.time()-t0:.0f}s')
        writer.add_scalar('epoch/avg_loss', avg_loss, epoch)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(cfg.work_dir, 'latest.pth'))
        if (epoch + 1) % cfg.checkpoint_interval == 0:
            torch.save(ckpt, os.path.join(cfg.work_dir, f'epoch_{epoch+1}.pth'))
        print(f'[保存] {cfg.work_dir}')

        if val_loader is not None and (epoch + 1) % cfg.val_interval == 0:
            print(f'\n[评测] Epoch {epoch+1}')
            model.eval()
            results = run_eval(model, val_loader, cfg, getattr(cfg, 'score_thr', 0.3),
                               cfg.device, getattr(cfg, 'eval_workers', 4))
            model.train()
            writer.add_scalar('metrics/mAP', results.get('mAP', 0.0), epoch)
            for cat_name, r in results.items():
                if isinstance(r, dict):
                    writer.add_scalar(f'metrics/AP/{cat_name}', r.get('AP', 0.0), epoch)

    writer.close()
    print('[训练完成]')


if __name__ == '__main__':
    main()
