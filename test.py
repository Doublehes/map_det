import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

from configs.default import config_default as cfg
from data.dataset import MapTRDataset, collate_fn
from models.maptr import MapTR


def denormalize_lines(lines, roi_size):
    return lines * np.array([roi_size[0], roi_size[1]], dtype=np.float32)


def visualize_bev(bev_feat, save_path):
    feat = bev_feat[0].mean(dim=0).cpu().numpy()
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6) * 255
    feat = feat.astype(np.uint8)
    cv2.imwrite(save_path, feat)
    print(f'[保存] BEV特征图: {save_path}')


def visualize_prediction(imgs, cls_scores, reg_preds, idx, save_dir, roi_size):
    B = imgs.shape[0]
    for bi in range(min(B, 4)):
        img = imgs[bi, 0].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array(cfg.img_norm['std']) + np.array(cfg.img_norm['mean']))
        img = np.clip(img, 0, 255).astype(np.uint8)

        scores = cls_scores[bi].sigmoid().cpu().numpy()
        lines = reg_preds[bi].cpu().numpy()

        h, w = 800, 400
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

        for qi in range(len(scores)):
            max_score = scores[qi].max()
            if max_score < 0.3:
                continue
            cls_id = scores[qi].argmax()
            line = denormalize_lines(lines[qi], roi_size)

            color = (0, 0, 255) if cls_id == 0 else (255, 0, 0)
            pts = []
            for p in line:
                px = int(p[0] / roi_size[0] * w)
                py = int(p[1] / roi_size[1] * h)
                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, color, thickness=2)

        save_path = Path(save_dir) / f'pred_{idx}_{bi}.png'
        cv2.imwrite(str(save_path), canvas)
        print(f'[保存] 预测结果: {save_path}')


@torch.no_grad()
def test():
    cfg.num_epochs = 1
    save_dir = './work_dirs/maptr/test_viz'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    ds = MapTRDataset(cfg.val_ann_file, cfg.data_root, cfg, is_train=False)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = MapTR(cfg).to(cfg.device)
    model.eval()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 2:
            break

        imgs = batch['imgs'].to(cfg.device)
        intrinsics = batch['intrinsics'].to(cfg.device)
        extrinsics = batch['extrinsics'].to(cfg.device)

        cls_scores, reg_preds, seg_preds = model(imgs, intrinsics, extrinsics)

        visualize_prediction(imgs, cls_scores, reg_preds, batch_idx, save_dir, cfg.roi_size)
        print(f'[批次 {batch_idx}] 完成')

    print(f'[测试完成] 结果保存在 {save_dir}')


if __name__ == '__main__':
    test()
