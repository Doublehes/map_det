"""Slim 模型评测: 复用主项目 VectorEvaluate, 输出 per-class AP@[0.5,1.0,1.5] + mAP

运行: python model_slim/eval.py [--config resnet34] [--checkpoint path] [--score-thr 0.3]
checkpoint 默认 cfg.work_dir/latest.pth
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_slim.config as cfg_module
from model_slim.dataset import SlimDataset, collate_fn
from model_slim.model import SlimModel
from metrics.vector_eval import VectorEvaluate


@torch.no_grad()
def run_eval(model, val_loader, cfg, score_thr=0.3, device=None, n_workers=4):
    """推理全量 val 集并评测 (复用 VectorEvaluate: chamfer AP@阈值 + mAP)"""
    if device is None:
        device = cfg.device
    roi_w, roi_h = cfg.data.roi_size
    pc0, pc1 = cfg.data.pc_range[0], cfg.data.pc_range[1]

    predictions = {}
    for batch in tqdm(val_loader, desc='推理', unit='batch'):
        raster = batch['raster'].to(device)
        tokens = batch['token']

        cls_scores, reg_preds, _, _ = model(raster)

        B = cls_scores.shape[0]
        for bi in range(B):
            token = tokens[bi]
            if token in predictions:
                continue

            scores = cls_scores[bi].sigmoid().cpu().numpy()
            lines = reg_preds[bi].cpu().numpy()

            pred_vectors, pred_scores, pred_labels = [], [], []
            for qi in range(len(scores)):
                max_score = scores[qi].max()
                if max_score < score_thr:
                    continue
                cls_id = int(scores[qi].argmax())
                line = lines[qi].copy()
                line[:, 0] = line[:, 0] * roi_w + pc0
                line[:, 1] = line[:, 1] * roi_h + pc1
                pred_vectors.append(line)
                pred_scores.append(float(max_score))
                pred_labels.append(cls_id)

            predictions[token] = {
                'vectors': pred_vectors,
                'scores': pred_scores,
                'labels': pred_labels,
            }

    num_total_preds = sum(len(p['scores']) for p in predictions.values())
    print(f'\n评测 {len(predictions)} 个样本, {num_total_preds} 个预测')

    evaluator = VectorEvaluate(cfg, n_workers=n_workers)
    evaluator.prepare_gts(val_loader.dataset)
    results, _ = evaluator.evaluate(predictions)
    evaluator.print_results(results)
    return results


def main():
    parser = argparse.ArgumentParser(description='Slim 模型评测')
    parser.add_argument('--config', default='default', choices=list(cfg_module.CONFIGS.keys()),
                        help='配置变体: default / resnet34 / decode_layer3')
    parser.add_argument('--checkpoint', type=str, default=None, help='默认 cfg.work_dir/latest.pth')
    parser.add_argument('--score-thr', type=float, default=0.3, help='置信度阈值')
    args = parser.parse_args()
    cfg = cfg_module.CONFIGS[args.config]

    device = cfg.device
    ckpt_path = args.checkpoint or os.path.join(cfg.work_dir, 'latest.pth')
    print(f'[加载] {ckpt_path}')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'checkpoint 不存在: {ckpt_path}')

    model = SlimModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    print(f'[加载] epoch={ckpt.get("epoch", "?")}')

    val_dataset = SlimDataset(cfg.data.val_ann_file, cfg.data, is_train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    run_eval(model, val_loader, cfg, args.score_thr, device)


if __name__ == '__main__':
    main()
