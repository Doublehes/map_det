"""推理+可视化: BEV预测结果 + 相机投影"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.default import cfg
from data.dataset import MapTRDataset, collate_fn
from models.maptr import MapTR

CAT_NAMES = {0: 'guide_line', 1: 'boundary'}
GT_COLORS = {0: 'lime', 1: 'red'}
PRED_COLORS = {0: 'cyan', 1: 'orange'}


def denormalize_lines(lines, roi_size):
    cx, cy = roi_size[0] / 2, roi_size[1] / 2
    return lines * np.array([roi_size[0] / 2, roi_size[1] / 2]) + np.array([cx, cy])


def project_to_image(pts_3d, K, extr):
    if isinstance(K, torch.Tensor):
        K = K.numpy()
    if isinstance(extr, torch.Tensor):
        extr = extr.numpy()
    if pts_3d.shape[1] == 2:
        z = np.zeros((pts_3d.shape[0], 1), dtype=np.float32)
        pts_3d = np.concatenate([pts_3d, z], axis=-1)
    N = pts_3d.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    cam_pts = (extr @ np.concatenate([pts_3d, ones], axis=-1).T).T
    valid = cam_pts[:, 2] > 0.001
    z = np.clip(cam_pts[:, 2:3], 1e-6, None)
    uv = (K @ np.concatenate([cam_pts[:, :2] / z, np.ones((N, 1), dtype=np.float32)], axis=-1).T).T
    return uv[:, :2], valid


def decode_predictions(cls_scores, reg_preds, roi_size, pc_range, score_thresh=0.3):
    cls_scores = cls_scores.sigmoid().cpu().numpy()
    reg_preds = reg_preds.cpu().numpy()

    lines_by_cls = {0: [], 1: []}
    scores_by_cls = {0: [], 1: []}
    for qi in range(len(cls_scores)):
        max_score, cls_id = cls_scores[qi].max(), cls_scores[qi].argmax()
        if max_score < score_thresh:
            continue
        line = denormalize_lines(reg_preds[qi], roi_size)
        line[:, 0] += pc_range[0]
        line[:, 1] += pc_range[1]
        lines_by_cls[cls_id].append(line)
        scores_by_cls[cls_id].append(max_score)
    return lines_by_cls, scores_by_cls


def get_cam_names(sample):
    available = sorted(k for k, v in sample['cams'].items() if v['img_fpath'] is not None)
    cam_names = list(available) + ['dummy'] * max(0, cfg.num_cams - len(available))
    return cam_names[:cfg.num_cams]


def draw_bev(ax, gt_lines, pred_lines, title):
    pc = cfg.pc_range
    ax.set_xlim(pc[0], pc[3])
    ax.set_ylim(pc[1], pc[4])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)

    ax.add_patch(Rectangle((pc[0], pc[1]), pc[3]-pc[0], pc[4]-pc[1],
                            fill=False, edgecolor='gray', linestyle='--', alpha=0.5))
    ax.plot(0, 0, 'k^', markersize=10)
    ax.arrow(0, 0, 2, 0, head_width=0.3, head_length=0.5, fc='black', ec='black', alpha=0.5)

    for cls_id, lines in gt_lines.items():
        for pts in lines:
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], color=GT_COLORS[cls_id], linewidth=1.5)
            ax.scatter(pts[0, 0], pts[0, 1], color=GT_COLORS[cls_id], s=12, marker='o', zorder=3)
            ax.scatter(pts[-1, 0], pts[-1, 1], color=GT_COLORS[cls_id], s=12, marker='s', zorder=3)

    for cls_id, lines in pred_lines.items():
        for pts in lines:
            ax.plot(pts[:, 0], pts[:, 1], color=PRED_COLORS[cls_id], linewidth=2, linestyle='--')
            ax.scatter(pts[0, 0], pts[0, 1], color=PRED_COLORS[cls_id], s=20, marker='o', zorder=4)
            ax.scatter(pts[-1, 0], pts[-1, 1], color=PRED_COLORS[cls_id], s=20, marker='s', zorder=4)


def draw_cams(axes, imgs, intrinsics, extrinsics, gt_lines, pred_lines, cam_names):
    H, W = cfg.img_h, cfg.img_w
    mean = np.array(cfg.img_norm['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm['std'], dtype=np.float32)

    for ci in range(len(cam_names)):
        ax = axes[ci]
        img = imgs[ci].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * std + mean, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f'{ci}: {cam_names[ci]}', fontsize=7)
        ax.axis('off')

        Ki = intrinsics[ci]
        extri = extrinsics[ci]

        for cls_id, lines in gt_lines.items():
            for pts in lines:
                pts = np.array(pts)
                uv, valid = project_to_image(pts, Ki, extri)
                in_img = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & valid
                if in_img.sum() < 2:
                    continue
                ax.plot(uv[in_img, 0], uv[in_img, 1], color=GT_COLORS[cls_id], linewidth=1.5)

        for cls_id, lines in pred_lines.items():
            for pts in lines:
                pts = np.array(pts)
                uv, valid = project_to_image(pts, Ki, extri)
                in_img = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & valid
                if in_img.sum() < 2:
                    continue
                ax.plot(uv[in_img, 0], uv[in_img, 1], color=PRED_COLORS[cls_id], linewidth=2, linestyle='--')


@torch.no_grad()
def infer():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='work_dirs/maptr/epoch_4.pth')
    parser.add_argument('--save-dir', type=str, default='work_dirs/infer')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--score-thresh', type=float, default=0.3)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = MapTRDataset(cfg.val_ann_file, cfg.data_root, cfg, is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = MapTR(cfg).to(cfg.device)
    ckpt = torch.load(args.checkpoint, map_location=cfg.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'[加载] {args.checkpoint} epoch={ckpt.get("epoch", "?")}')

    rendered = 0
    for batch_idx, batch in enumerate(loader):
        if rendered >= args.num_samples:
            break

        imgs = batch['imgs'].to(cfg.device)
        intrinsics = batch['intrinsics'].to(cfg.device)
        extrinsics = batch['extrinsics'].to(cfg.device)

        cls_scores, reg_preds, _ = model(imgs, intrinsics, extrinsics)

        pred_lines, pred_scores = decode_predictions(
            cls_scores[0], reg_preds[0], cfg.roi_size, cfg.pc_range, args.score_thresh)

        sample = ds.samples[batch_idx]
        gt_raw = sample['map_geom']

        cam_names = get_cam_names(sample)

        fig = plt.figure(figsize=(20, 12))
        ax_bev = fig.add_subplot(3, 3, (1, 3))
        draw_bev(ax_bev, gt_raw, pred_lines,
                 f'BEV — GT(lime/red)  Pred(cyan/orange)\n{sample["token"][:16]}... | score>{args.score_thresh}')

        cam_axes = [fig.add_subplot(3, 3, 4 + ci) for ci in range(cfg.num_cams)]
        draw_cams(cam_axes, imgs[0], intrinsics[0].cpu(), extrinsics[0].cpu(), gt_raw, pred_lines, cam_names)

        plt.tight_layout()
        out = save_dir / f'infer_{batch_idx:04d}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[保存] {out}')
        rendered += 1

    print(f'[完成] 共 {rendered} 张, 保存至 {save_dir}')


if __name__ == '__main__':
    infer()
