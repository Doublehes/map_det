import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

from data.dataset import MapTRDataset, collate_fn

cfg = None
from models.maptr import MapTR

@torch.no_grad()
def test():
    import argparse
    parser = argparse.ArgumentParser(description='MapTR 推理测试')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args, _ = parser.parse_known_args()

    global cfg
    from configs.loader import load_config
    cfg = load_config(args.config)

    cfg.num_epochs = 1
    save_dir = './work_dirs/maptr/test_viz'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    ds = MapTRDataset(cfg.data.val_ann_file, cfg.data.data_root, cfg.data, is_train=False)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = MapTR(cfg.model).to(cfg.device)
    model.eval()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 2:
            break

        imgs = batch['imgs'].to(cfg.device)
        intrinsics = batch['intrinsics'].to(cfg.device)
        extrinsics = batch['extrinsics'].to(cfg.device)

        cls_scores, reg_preds, seg_preds, _ = model(imgs, intrinsics, extrinsics)

        visualize_prediction(imgs, cls_scores, reg_preds, batch_idx, save_dir, cfg.data.roi_size)
        print(f'[批次 {batch_idx}] 完成')

    print(f'[测试完成] 结果保存在 {save_dir}')


if __name__ == '__main__':
    test()
