import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MapTRDataset, collate_fn
from models.maptr import MapTR
from metrics.vector_eval import VectorEvaluate
from utils.timer import Timer


def plot_pr_curves(pr_data, id2cat, thresholds, save_dir=None, writer=None, epoch=0):
    from matplotlib import pyplot as plt
    for label in id2cat.values():
        for thr in thresholds:
            d = pr_data.get(label, {}).get(thr, {})
            if 'recalls' not in d:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(d['recalls'], d['precisions'], 'b-', linewidth=2)
            axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
            axes[0].set_title(f'PR Curve  {label} @{thr}  AP={d["ap"]:.4f}')
            axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1); axes[0].grid(True)
            axes[1].plot(d['scores'], d['precisions'], 'g-', linewidth=2)
            axes[1].set_xlabel('Score Threshold'); axes[1].set_ylabel('Precision')
            axes[1].set_title('Precision vs Score')
            axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1); axes[1].grid(True)
            axes[2].plot(d['scores'], d['recalls'], 'r-', linewidth=2)
            axes[2].set_xlabel('Score Threshold'); axes[2].set_ylabel('Recall')
            axes[2].set_title('Recall vs Score')
            axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1); axes[2].grid(True)
            plt.tight_layout()
            if writer is not None:
                writer.add_figure(f'pr_curve/{label}/thr_{thr}', fig, epoch)
            if save_dir is not None:
                fig.savefig(str(save_dir / f'{label}_thr{thr}.png'))
            plt.close(fig)


@torch.no_grad()
def run_eval(model, val_loader, cfg, score_thr=None, device=None, n_workers=4, writer=None, epoch=0):
    """推理全量 val 集并评测

    Args:
        model: 已加载权重的模型 (调用前外部切换 eval 模式)
        val_loader: val 集 DataLoader
        cfg: 配置对象
        score_thr: 置信度阈值, 默认 cfg.score_thr
        device: 推理设备, 默认 cfg.device
        n_workers: 评测匹配多进程数
        writer: TensorBoard SummaryWriter
        epoch: 当前 epoch

    Returns:
        (results, evaluator) — results 包含 mAP 等指标, evaluator 可调用 print_results
    """
    if score_thr is None:
        score_thr = getattr(cfg, 'score_thr', 0.3)
    if device is None:
        device = cfg.device

    roi_w, roi_h = cfg.data.roi_size
    pc0, pc1 = cfg.data.pc_range[0], cfg.data.pc_range[1]

    predictions = {}
    for batch in tqdm(val_loader, desc='推理', unit='batch'):
        imgs = batch['imgs'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        tokens = batch['token']

        cls_scores, reg_preds, _, _, _ = model(imgs, intrinsics, extrinsics)

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

    timer = Timer()
    evaluator = VectorEvaluate(cfg, n_workers=n_workers)
    with timer('评测GT加载'):
        evaluator.prepare_gts(val_loader.dataset)
    with timer('评测'):
        results, pr_data = evaluator.evaluate(predictions)
    evaluator.print_results(results)

    if writer is not None and pr_data:
        precisions, recalls = [], []
        for label in evaluator.id2cat.values():
            for thr in evaluator.thresholds:
                d = pr_data.get(label, {}).get(thr, {})
                if 'scores' not in d:
                    continue
                mask = d['scores'] >= score_thr
                if mask.any():
                    idx = np.where(mask)[0][-1]
                    precisions.append(d['precisions'][idx])
                    recalls.append(d['recalls'][idx])
        avg_precision = float(np.mean(precisions)) if precisions else 0.0
        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        writer.add_scalar('metrics/Precision', avg_precision, epoch)
        writer.add_scalar('metrics/Recall', avg_recall, epoch)
        writer.add_scalar('metrics/mAP', results.get('mAP', 0.0), epoch)

    return results, evaluator


@torch.no_grad()
def main():
    timer = Timer()

    parser = argparse.ArgumentParser(description='MapTR 评测')
    parser.add_argument('config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--score-thr', type=float, default=None, help='置信度阈值 (默认 cfg.score_thr)')
    parser.add_argument('--batch-size', type=int, default=None, help='推理 batch size')
    parser.add_argument('--num-workers', type=int, default=None, help='dataloader workers')
    parser.add_argument('--device', type=str, default=None, help='推理设备')
    parser.add_argument('--output-dir', type=str, default="./work_dirs/eval", help='PR 曲线保存目录')
    args = parser.parse_args()

    from configs.loader import load_config
    cfg = load_config(args.config)

    device = torch.device(args.device) if args.device else cfg.device
    score_thr = args.score_thr if args.score_thr is not None else cfg.score_thr
    batch_size = args.batch_size if args.batch_size else cfg.data.batch_size

    print(f'[设备] {device}')
    print(f'[score_thr] {score_thr}')
    print(f'[数据] val={cfg.data.val_ann_file}')

    with timer('数据集加载'):
        val_dataset = MapTRDataset(
            ann_file=cfg.data.val_ann_file,
            data_root=cfg.data.data_root,
            cfg=cfg.data,
            is_train=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers or cfg.data.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )

    with timer('模型加载'):
        model = MapTR(cfg.model).to(device)
        print(f'[模型] 加载 checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'  [警告] 缺少 {len(missing)} 个 key')
        if unexpected:
            print(f'  [警告] 多余 {len(unexpected)} 个 key')
        model.eval()


    results, evaluator = run_eval(model, val_loader, cfg, score_thr, device)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_pr_curves(evaluator._pr_data, evaluator.id2cat, evaluator.thresholds,
                       save_dir=out_dir)
        print(f'[PR曲线] 已保存至 {out_dir}')

    return results


if __name__ == '__main__':
    main()
