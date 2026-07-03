"""可视化数据集: BEV + 多相机图像上的GT线投影"""
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.default import cfg
from data.dataset import MapTRDataset

CAT_NAMES = {0: 'guide_line', 1: 'boundary'}
CAT_COLORS = {0: 'lime', 1: 'red'}


def denormalize_vectors(norm_pts, roi_size):
    """将归一化[-1,1]的点还原到BEV坐标"""
    cx, cy = roi_size[0] / 2, roi_size[1] / 2
    pts = norm_pts * np.array([roi_size[0] / 2, roi_size[1] / 2]) + np.array([cx, cy])
    return pts


def to_vehicle_frame(bev_pts, pc_range):
    """从BEV局部坐标转换到车辆坐标系"""
    x = bev_pts[:, 0] + pc_range[0]
    y = bev_pts[:, 1] + pc_range[1]
    return np.stack([x, y], axis=-1)


def project_to_image(pts_3d, K, extr):
    """3D点(车辆坐标系) → 图像像素坐标
    
    pts_3d: (N, 3)  or (N, 2)  (z=0 地面)
    K: (3, 3)  内参
    extr: (4, 4)  外参  vehicle→camera
    """
    if pts_3d.ndim == 2 and pts_3d.shape[1] == 2:
        z = np.zeros((pts_3d.shape[0], 1), dtype=np.float32)
        pts_3d = np.concatenate([pts_3d, z], axis=-1)

    N = pts_3d.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    homo = np.concatenate([pts_3d, ones], axis=-1)  # (N, 4)

    cam_pts = (extr @ homo.T).T  # (N, 4)
    cam_pts = cam_pts[:, :3]
    valid = cam_pts[:, 2] > 0.001  # 在相机前方

    z = np.clip(cam_pts[:, 2:3], 1e-6, None)
    uv = (K @ np.concatenate([cam_pts[:, :2] / z, np.ones((N, 1), dtype=np.float32)], axis=-1).T).T
    uv = uv[:, :2]
    return uv, valid


def draw_bev(ax, vectors_raw, vectors_norm, title='BEV', sem_mask=None):
    pc_range = cfg.pc_range
    roi_size = cfg.roi_size
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    # 分割掩码热力图 (BEV背景)
    if sem_mask is not None:
        mask = sem_mask[0].numpy()  # (160, 80)
        extent = [pc_range[0], pc_range[3], pc_range[1], pc_range[4]]
        ax.imshow(mask, extent=extent, origin='lower', cmap='hot', alpha=0.3, vmin=0, vmax=1)

    # 画ROI边界
    from matplotlib.patches import Rectangle
    roi_rect = Rectangle(
        (pc_range[0], pc_range[1]),
        roi_size[0], roi_size[1],
        fill=False, edgecolor='gray', linestyle='--', alpha=0.5
    )
    ax.add_patch(roi_rect)

    # 画车辆位置
    ax.plot(0, 0, 'k^', markersize=10, label='ego')
    ax.arrow(0, 0, 2, 0, head_width=0.3, head_length=0.5, fc='black', ec='black', alpha=0.5)

    # 画原始GT线(来自raw数据, 车辆坐标系)
    for cls_id, lines in vectors_raw.items():
        color = CAT_COLORS.get(cls_id, 'white')
        label = CAT_NAMES.get(cls_id, f'cls{cls_id}')
        for pts in lines:
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5,
                    label=label if f'_{cls_id}' not in str(ax.get_legend_handles_labels()) else "")
            ax.scatter(pts[0, 0], pts[0, 1], color=color, s=15, marker='o', zorder=3)
            ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=15, marker='s', zorder=3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')


def draw_camera_view(axes, imgs, intrinsics, extrinsics, vectors_raw, cam_names):
    """画6个相机视图, 叠加上投影的GT线"""
    H, W = cfg.img_h, cfg.img_w
    mean = np.array(cfg.img_norm['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm['std'], dtype=np.float32)

    for ci in range(len(cam_names)):
        ax = axes[ci]
        # 反归一化图像 (mean, std)
        img = imgs[ci].numpy().transpose(1, 2, 0)  # (H,W,3)
        img = img * std + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f'{ci}: {cam_names[ci]}', fontsize=8)
        ax.axis('off')

        Ki = intrinsics[ci]
        extri = extrinsics[ci]

        # 投影每条GT线到图像
        for cls_id, lines in vectors_raw.items():
            color = CAT_COLORS.get(cls_id, 'white')
            for pts in lines:
                pts = np.array(pts)
                uv, valid = project_to_image(pts, Ki, extri)
                if valid.sum() < 2:
                    continue
                # 只保留在图像内的点
                in_img = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & valid
                if in_img.sum() < 2:
                    continue
                uv_plot = uv[in_img]
                ax.plot(uv_plot[:, 0], uv_plot[:, 1], color=color, linewidth=2)
                ax.scatter(uv_plot[0, 0], uv_plot[0, 1], color=color, s=20, marker='o', zorder=3)
                ax.scatter(uv_plot[-1, 0], uv_plot[-1, 1], color=color, s=20, marker='s', zorder=3)


def visualize_sample(sample_idx, save_dir, is_train=False):
    dataset = MapTRDataset(
        ann_file=cfg.train_ann_file,
        data_root=cfg.data_root,
        cfg=cfg,
        is_train=is_train,
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for idx in sample_idx:
        item = dataset[idx]
        sample = dataset.samples[idx]

        imgs = item['imgs']
        intrinsics = item['intrinsics']
        extrinsics = item['extrinsics']
        token = item['token']
        sem_mask = item.get('semantic_mask')

        # 相机名 (与 _load_images 排序逻辑一致)
        available = sorted(k for k, v in sample['cams'].items() if v['img_fpath'] is not None)
        cam_names = list(available) + ['dummy'] * max(0, cfg.num_cams - len(available))
        cam_names = cam_names[:cfg.num_cams]

        # 原始GT线 (车辆坐标系, 地面z≈0)
        vectors_raw = sample['map_geom']  # {cls_id: list of (N,3) arrays}

        fig = plt.figure(figsize=(20, 12))

        # BEV (含分割掩码热力图)
        ax_bev = fig.add_subplot(3, 3, (1, 3))
        draw_bev(ax_bev, vectors_raw, item['vectors'],
                 title=f'BEV GT Lines\n{token[:16]}... scene={sample["scene_name"]} idx={sample["sample_idx"]}',
                 sem_mask=sem_mask)

        # 相机视图 (数量与 imgs 一致)
        n_cams = imgs.shape[0]
        cam_axes = []
        for ci in range(n_cams):
            ax = fig.add_subplot(3, 3, 4 + ci)
            cam_axes.append(ax)
        draw_camera_view(cam_axes, imgs, intrinsics, extrinsics, vectors_raw, cam_names)

        plt.tight_layout()
        out_path = Path(save_dir) / f'vis_{idx:04d}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[可视化] 已保存 {out_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MapTR数据可视化')
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 2],
                        help='要可视化的样本索引')
    parser.add_argument('--save-dir', type=str, default='work_dirs/vis',
                        help='保存目录')
    parser.add_argument('--is-train', action='store_true', default=False,
                        help='使用训练模式 (包含分割掩码)')
    args = parser.parse_args()
    visualize_sample(args.indices, args.save_dir, args.is_train)
