"""可视化数据集: BEV + BEV掩码 + 多相机图像上的GT线投影 (纯OpenCV)"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.default import cfg
from data.dataset import MapTRDataset

CAT_NAMES = {0: 'guide_line', 1: 'boundary'}
CAT_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}  # BGR


def project_to_image(pts_3d, K, extr):
    """3D点(车辆坐标系) → 图像像素坐标"""
    if pts_3d.ndim == 2 and pts_3d.shape[1] == 2:
        z = np.zeros((pts_3d.shape[0], 1), dtype=np.float32)
        pts_3d = np.concatenate([pts_3d, z], axis=-1)
    N = pts_3d.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    homo = np.concatenate([pts_3d, ones], axis=-1)
    cam_pts = (extr @ homo.T).T
    cam_pts = cam_pts[:, :3]
    valid = cam_pts[:, 2] > 0.001
    z = np.clip(cam_pts[:, 2:3], 1e-6, None)
    uv = (K @ np.concatenate([cam_pts[:, :2] / z, np.ones((N, 1))], axis=-1).T).T
    return uv[:, :2], valid


def visualize_sample(sample_idx, save_dir, is_train=False):
    dataset = MapTRDataset(
        ann_file=cfg.train_ann_file,
        data_root=cfg.data_root,
        cfg=cfg,
        is_train=is_train,
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    pc_min_x, pc_min_y, _, pc_max_x, pc_max_y, _ = cfg.pc_range
    x_range = pc_max_x - pc_min_x  # 40
    y_range = pc_max_y - pc_min_y  # 20

    BEV_W, BEV_H = 800, 400       # BEV/掩码面板尺寸
    GAP = 20
    CANVAS_W = BEV_W * 2 + GAP     # 1620
    CAM_W, CAM_H = 530, 260
    CAM_GAP = 15

    for idx in sample_idx:
        item = dataset[idx]
        sample = dataset.samples[idx]

        imgs = item['imgs']
        intrinsics = item['intrinsics']
        extrinsics = item['extrinsics']
        sem_mask = item.get('semantic_mask')
        vectors_raw = sample['map_geom']

        available_cams = sorted(k for k, v in sample['cams'].items() if v['img_fpath'] is not None)

        total_h = BEV_H + GAP + CAM_H * 2 + CAM_GAP + 20
        canvas = np.zeros((total_h, CANVAS_W, 3), dtype=np.uint8)

        # ========== BEV 俯视图 (左) ==========
        bev = canvas[:BEV_H, :BEV_W]

        # 网格
        for x in range(-10, 31, 5):
            px = int((x - pc_min_x) / x_range * BEV_W)
            cv2.line(bev, (px, 0), (px, BEV_H), (50, 50, 50), 1)
        for y in range(-10, 11, 5):
            py = int((pc_max_y - y) / y_range * BEV_H)
            cv2.line(bev, (0, py), (BEV_W, py), (50, 50, 50), 1)

        # 自车
        ex = int((0 - pc_min_x) / x_range * BEV_W)
        ey = int((pc_max_y - 0) / y_range * BEV_H)
        tri = np.array([
            [ex, ey - 10], [ex - 7, ey + 6], [ex + 7, ey + 6]
        ], dtype=np.int32)
        cv2.fillPoly(bev, [tri], (180, 180, 180))
        cv2.arrowedLine(bev, (ex, ey), (ex + 40, ey), (100, 100, 100), 2, tipLength=0.3)

        # GT 线
        for cls_id, lines in vectors_raw.items():
            color = CAT_COLORS.get(cls_id, (200, 200, 200))
            for pts_list in lines:
                pts = np.array(pts_list, dtype=np.float32)
                if pts.shape[1] < 2:
                    continue
                xy = pts[:, :2]
                col = ((xy[:, 0] - pc_min_x) / x_range * BEV_W).astype(np.int32)
                row = ((pc_max_y - xy[:, 1]) / y_range * BEV_H).astype(np.int32)
                pix = np.stack([col, row], axis=1)
                cv2.polylines(bev, [pix], False, color, 2, lineType=cv2.LINE_AA)
                cv2.circle(bev, tuple(pix[0]), 4, color, -1)
                cv2.circle(bev, tuple(pix[-1]), 4, color, -1)

        cv2.putText(bev, f'BEV  idx={idx}  {sample["scene_name"]}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(bev, 'X(前)', (BEV_W - 50, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        cv2.putText(bev, 'Y(左)', (4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # ========== BEV 掩码 (右) ==========
        mask_panel = canvas[:BEV_H, BEV_W + GAP:BEV_W * 2 + GAP]

        if sem_mask is not None:
            mask_raw = sem_mask[0].numpy()  # (80, 160)
            mask_disp = (mask_raw * 255).astype(np.uint8)
            mask_disp = cv2.resize(mask_disp, (BEV_W, BEV_H),
                                   interpolation=cv2.INTER_NEAREST)
            mask_bgr = cv2.cvtColor(mask_disp, cv2.COLOR_GRAY2BGR)
            green = np.full_like(mask_bgr, (0, 255, 0), dtype=np.uint8)
            mask_panel[:] = np.where(mask_bgr > 0, green, (25, 25, 25))
            cv2.putText(mask_panel, f'BEV Mask (5x nearest)  shape={list(mask_raw.shape)}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        else:
            mask_panel[:] = (25, 25, 25)
            cv2.putText(mask_panel, 'No mask (eval mode)', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # ========== 相机视图 ==========
        mean = np.array(cfg.img_norm['mean'], dtype=np.float32)
        std = np.array(cfg.img_norm['std'], dtype=np.float32)
        cam_start_y = BEV_H + GAP
        num_cams = min(len(available_cams), cfg.num_cams)

        for ci in range(num_cams):
            row = ci // 3
            col = ci % 3
            cx = col * (CAM_W + CAM_GAP)
            cy = cam_start_y + row * (CAM_H + CAM_GAP)
            panel = canvas[cy:cy + CAM_H, cx:cx + CAM_W]

            # 反归一化
            img = imgs[ci].numpy().transpose(1, 2, 0)
            img = img * std + mean
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 投影 GT 线
            Ki = intrinsics[ci]
            extri = extrinsics[ci]
            for cls_id, lines in vectors_raw.items():
                color = CAT_COLORS.get(cls_id, (200, 200, 200))
                for pts_list in lines:
                    pts = np.array(pts_list, dtype=np.float32)
                    uv, valid = project_to_image(pts, Ki, extri)
                    if valid.sum() < 2:
                        continue
                    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < cfg.img_w) & \
                             (uv[:, 1] >= 0) & (uv[:, 1] < cfg.img_h) & valid
                    if in_img.sum() < 2:
                        continue
                    uv_proj = uv[in_img].astype(np.int32)
                    for j in range(len(uv_proj) - 1):
                        cv2.line(img, tuple(uv_proj[j]), tuple(uv_proj[j + 1]), color, 2)
                    cv2.circle(img, tuple(uv_proj[0]), 3, color, -1)
                    cv2.circle(img, tuple(uv_proj[-1]), 3, color, -1)

            img_resized = cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
            panel[:] = img_resized
            cv2.putText(panel, f'{ci}: {available_cams[ci]}', (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # ========== 保存 ==========
        out_path = Path(save_dir) / f'vis_{idx:04d}.png'
        cv2.imwrite(str(out_path), canvas)
        print(f'[可视化] 已保存 {out_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MapTR数据可视化 (OpenCV)')
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 2],
                        help='要可视化的样本索引')
    parser.add_argument('--save-dir', type=str, default='work_dirs/vis',
                        help='保存目录')
    parser.add_argument('--is-train', action='store_true', default=False,
                        help='使用训练模式 (包含分割掩码)')
    args = parser.parse_args()
    visualize_sample(args.indices, args.save_dir, args.is_train)
