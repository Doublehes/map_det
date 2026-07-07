"""推理+可视化: BEV预测结果 + 相机投影 (纯OpenCV)"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.default import cfg
from data.dataset import MapTRDataset, collate_fn
from models.maptr import MapTR

CAT_NAMES = {0: 'guide_line', 1: 'boundary'}
GT_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}       # BGR: 中心线绿, 边界线红
PRED_COLORS = {0: (255, 0, 0), 1: (255, 0, 0)}    # BGR: 所有预测线蓝色


def denormalize_lines(lines, roi_size):
    return lines * np.array([roi_size[0], roi_size[1]], dtype=np.float32)


def project_to_image(pts_3d, K, extr):
    if isinstance(K, torch.Tensor):
        K = K.numpy()
    if isinstance(extr, torch.Tensor):
        extr = extr.numpy()
    if pts_3d.shape[1] == 2:
        z = -0.1 * np.ones((pts_3d.shape[0], 1), dtype=np.float32)
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
    # available = sorted(k for k, v in sample['cams'].items() if v['img_fpath'] is not None)
    available = [k for k, v in sample['cams'].items() if v['img_fpath'] is not None]
    cam_names = list(available) + ['dummy'] * max(0, cfg.num_cams - len(available))
    return cam_names[:cfg.num_cams]


def world_to_panel(pts, pc_min_x, pc_min_y, pc_max_x, pc_max_y, pw, ph):
    """车辆坐标 → 面板像素(col,row)"""
    col = ((pts[:, 0] - pc_min_x) / (pc_max_x - pc_min_x) * pw).astype(np.int32)
    row = ((pc_max_y - pts[:, 1]) / (pc_max_y - pc_min_y) * ph).astype(np.int32)
    return np.stack([col, row], axis=1)


def draw_lines(canvas, lines_dict, color_map, pix_fn, marker_size=4):
    """绘制一组线: cls=0 箭头线, cls=1 实线
    
    pix_fn: callable(pts) → (N,2) int32 像素坐标, 或 None(跳过)
    """
    for cls_id, lines in lines_dict.items():
        color = color_map.get(cls_id, (200, 200, 200))
        for pts_list in lines:
            pts = np.array(pts_list, dtype=np.float32)
            if pts.shape[1] < 2:
                continue
            pix = pix_fn(pts)
            if pix is None or len(pix) < 2:
                continue
            if cls_id == 0:
                for i in range(len(pix) - 1):
                    cv2.arrowedLine(canvas, tuple(pix[i]), tuple(pix[i + 1]),
                                    color, 2, tipLength=0.25)
            else:
                cv2.polylines(canvas, [pix], False, color, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, tuple(pix[0]), marker_size, color, -1)
            cv2.circle(canvas, tuple(pix[-1]), marker_size, color, -1)


def draw_bev_panel(panel, pc_range, gt_lines, pred_lines):
    pw, ph = panel.shape[1], panel.shape[0]
    x_min, y_min, _, x_max, y_max, _ = pc_range

    for x in range(-10, 31, 5):
        px = int((x - x_min) / (x_max - x_min) * pw)
        cv2.line(panel, (px, 0), (px, ph), (50, 50, 50), 1)
    for y in range(-10, 11, 5):
        py = int((y_max - y) / (y_max - y_min) * ph)
        cv2.line(panel, (0, py), (pw, py), (50, 50, 50), 1)

    ex = int((0 - x_min) / (x_max - x_min) * pw)
    ey = int((y_max - 0) / (y_max - y_min) * ph)
    tri = np.array([[ex, ey - 10], [ex - 7, ey + 6], [ex + 7, ey + 6]], dtype=np.int32)
    cv2.fillPoly(panel, [tri], (180, 180, 180))
    cv2.arrowedLine(panel, (ex, ey), (ex + 40, ey), (100, 100, 100), 2, tipLength=0.3)

    bev_pix = lambda pts: world_to_panel(pts[:, :2], x_min, y_min, x_max, y_max, pw, ph)
    draw_lines(panel, gt_lines, GT_COLORS, bev_pix, marker_size=4)
    draw_lines(panel, pred_lines, PRED_COLORS, bev_pix, marker_size=4)

    cv2.putText(panel, 'BEV  GT(green/red)  Pred(blue)', (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(panel, 'X(前)', (pw - 45, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    cv2.putText(panel, 'Y(左)', (4, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)


def draw_mask_panel(panel, mask, title, flip_v=False):
    """绘制分割掩码 (原始像素, INTER_NEAREST 缩放)"""
    h, w = mask.shape[:2]  # (80, 160)
    if flip_v:
        mask = np.flip(mask, axis=0)
    disp = (mask * 255).astype(np.uint8)
    disp = cv2.resize(disp, (panel.shape[1], panel.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    green = np.full_like(bgr, (0, 255, 0), dtype=np.uint8)
    panel[:] = np.where(bgr > 0, green, (25, 25, 25))
    flip_tag = ' (flipped)' if flip_v else ''
    cv2.putText(panel, f'{title}{flip_tag}  shape=({h},{w})', (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def draw_cam_panels(canvas, imgs, intrinsics, extrinsics, gt_lines, pred_lines,
                    cam_names, start_y, cam_w, cam_h, gap):
    mean = np.array(cfg.img_norm['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm['std'], dtype=np.float32)
    img_w, img_h = cfg.img_w, cfg.img_h

    for ci in range(min(len(cam_names), cfg.num_cams)):
        row = ci // 3
        col = ci % 3
        cx = col * (cam_w + gap)
        cy = start_y + row * (cam_h + gap)
        panel = canvas[cy:cy + cam_h, cx:cx + cam_w]

        img = imgs[ci].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * std + mean, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)

        Ki = intrinsics[ci].cpu().numpy() if isinstance(intrinsics[ci], torch.Tensor) else intrinsics[ci]
        extri = extrinsics[ci].cpu().numpy() if isinstance(extrinsics[ci], torch.Tensor) else extrinsics[ci]

        # GT 线（实线）
        for cls_id, lines in gt_lines.items():
            color = GT_COLORS.get(cls_id, (200, 200, 200))
            for pts_list in lines:
                pts = np.array(pts_list, dtype=np.float32)
                uv, valid = project_to_image(pts, Ki, extri)
                in_img = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
                         (uv[:, 1] >= 0) & (uv[:, 1] < img_h) & valid
                if in_img.sum() < 2:
                    continue
                uv_proj = uv[in_img].astype(np.int32)
                cv2.polylines(img, [uv_proj], False, color, 2, lineType=cv2.LINE_AA)
                cv2.circle(img, tuple(uv_proj[0]), 3, color, -1)
                cv2.circle(img, tuple(uv_proj[-1]), 3, color, -1)

        # 预测线: guide_line 箭头线, boundary 实线
        for cls_id, lines in pred_lines.items():
            color = PRED_COLORS.get(cls_id, (200, 200, 200))
            for pts_list in lines:
                pts = np.array(pts_list, dtype=np.float32)
                uv, valid = project_to_image(pts, Ki, extri)
                in_img = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
                         (uv[:, 1] >= 0) & (uv[:, 1] < img_h) & valid
                if in_img.sum() < 2:
                    continue
                uv_proj = uv[in_img].astype(np.int32)
                if cls_id == 0:
                    for i in range(len(uv_proj) - 1):
                        cv2.arrowedLine(img, tuple(uv_proj[i]), tuple(uv_proj[i + 1]),
                                        color, 2, tipLength=0.25)
                else:
                    cv2.polylines(img, [uv_proj], False, color, 2, lineType=cv2.LINE_AA)
                cv2.circle(img, tuple(uv_proj[0]), 3, color, -1)
                cv2.circle(img, tuple(uv_proj[-1]), 3, color, -1)

        img_resized = cv2.resize(img, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
        panel[:] = img_resized
        cv2.putText(panel, f'{ci}: {cam_names[ci]}', (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


@torch.no_grad()
def infer():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='work_dirs/maptr/epoch_4.pth')
    parser.add_argument('--save-dir', type=str, default='work_dirs/infer')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--score-thresh', type=float, default=0.7)
    parser.add_argument('--seg-thresh', type=float, default=0.4,
                        help='预测分割掩码二值化阈值')
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

    pw, ph = 520, 260
    gap = 15
    cw = pw
    ch = ph
    canvas_w = pw * 3 + gap * 2
    total_h = ph + gap + ch * 2 + 20
    scale_tag = f'{pw/40:.0f}px/m'

    rendered = 0
    for batch_idx, batch in enumerate(loader):
        # if batch_idx % 10 != 0:
        #     continue
        if rendered >= args.num_samples:
            break

        imgs = batch['imgs'].to(cfg.device)
        intrinsics = batch['intrinsics'].to(cfg.device)
        extrinsics = batch['extrinsics'].to(cfg.device)

        cls_scores, reg_preds, seg_preds = model(imgs, intrinsics, extrinsics)

        gt_seg_mask = batch['semantic_mask'][0].numpy()  # (1, 80, 160)
        pred_seg_mask = seg_preds[0].sigmoid().cpu().numpy()  # (1, 80, 160)

        pred_lines, pred_scores = decode_predictions(
            cls_scores[0], reg_preds[0], cfg.roi_size, cfg.pc_range, args.score_thresh)

        sample = ds.samples[batch_idx]
        gt_raw = sample['map_geom']
        cam_names = get_cam_names(sample)

        canvas = np.zeros((total_h, canvas_w, 3), dtype=np.uint8)

        # BEV 面板
        draw_bev_panel(canvas[:ph, :pw], cfg.pc_range, gt_raw, pred_lines)

        # GT 掩码面板 (需要 flip 对齐 BEV 坐标)
        draw_mask_panel(canvas[:ph, pw + gap:pw * 2 + gap],
                        gt_seg_mask[0], 'GT Mask', flip_v=True)

        # Pred 掩码面板 (已经是 BEV 坐标，不 flip, 按阈值二值化)
        pred_binary = (pred_seg_mask[0] > args.seg_thresh).astype(np.float32)
        draw_mask_panel(canvas[:ph, pw * 2 + gap * 2:pw * 3 + gap * 2],
                        pred_binary, f'Pred Mask  >{args.seg_thresh}', flip_v=False)

        # 标题行
        title = f'idx={batch_idx}  token={sample["token"][:16]}  score>{args.score_thresh}  seg>{args.seg_thresh}  {scale_tag}'
        cv2.putText(canvas, title, (8, ph - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        # 相机视图
        draw_cam_panels(canvas, imgs[0], intrinsics[0], extrinsics[0],
                        gt_raw, pred_lines, cam_names,
                        start_y=ph + gap, cam_w=cw, cam_h=ch, gap=gap)

        out = save_dir / f'infer_{batch_idx:04d}.png'
        cv2.imwrite(str(out), canvas)
        print(f'[保存] {out}')
        rendered += 1

    print(f'[完成] 共 {rendered} 张, 保存至 {save_dir}')


if __name__ == '__main__':
    infer()
