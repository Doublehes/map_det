"""推理+可视化: BEV预测结果 + 相机投影 (纯OpenCV)"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset import MapTRDataset, collate_fn

cfg = None
from models.maptr import MapTR

CAT_NAMES = {0: 'guide_line', 1: 'boundary'}
GT_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}       # BGR: 中心线绿, 边界线红
PRED_COLORS = {0: (255, 0, 0), 1: (255, 0, 0)}    # BGR: 所有预测线蓝色
SEG_COLORS = [(0, 255, 0), (0, 0, 255)]             # BGR: seg mask 每类颜色


def denormalize_lines(lines, roi_size):
    return lines * np.array([roi_size[0], roi_size[1]], dtype=np.float32)


def vectors_to_world(vec_dict, roi_size, pc_min_x, pc_min_y):
    """归一化 vector → 世界坐标 dict-of-lists"""
    out = {}
    for cls_id, arr in vec_dict.items():
        lines = []
        for i in range(arr.shape[0]):
            pts = arr[i, 0].cpu().numpy().copy()
            pts[:, 0] = pts[:, 0] * roi_size[0] + pc_min_x
            pts[:, 1] = pts[:, 1] * roi_size[1] + pc_min_y
            lines.append(pts)
        out[cls_id] = lines
    return out


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
    cam_names = list(available) + ['dummy'] * max(0, cfg.data.num_cams - len(available))
    return cam_names[:cfg.data.num_cams]


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


def draw_mask_panel(panel, mask, title, flip_v=False, colors=None):
    """绘制分割掩码 (支持多通道彩色叠加, INTER_NEAREST 缩放)"""
    if mask.ndim == 3 and mask.shape[0] > 1:
        if colors is None:
            colors = SEG_COLORS
        h, w = mask.shape[1:]
        panel[:] = (25, 25, 25)
        occupied = np.zeros((panel.shape[0], panel.shape[1]), dtype=bool)
        for c in range(mask.shape[0]):
            ch = mask[c]
            if flip_v:
                ch = np.flip(ch, axis=0)
            disp = cv2.resize(ch, (panel.shape[1], panel.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            new = (disp > 0) & ~occupied
            for i in range(3):
                panel[..., i][new] = colors[c][i]
            occupied |= (disp > 0)
        flip_tag = ' (flipped)' if flip_v else ''
        cv2.putText(panel, f'{title}{flip_tag}  shape=({mask.shape[0]},{h},{w})', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    else:
        h, w = mask.shape[:2]
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
    mean = np.array(cfg.data.img_norm['mean'], dtype=np.float32)
    std = np.array(cfg.data.img_norm['std'], dtype=np.float32)
    img_w, img_h = cfg.data.img_w, cfg.data.img_h

    for ci in range(min(len(cam_names), cfg.data.num_cams)):
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

        # GT 线 (guide_line 用箭头显示方向)
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
                if cls_id == 0:
                    for i in range(len(uv_proj) - 1):
                        cv2.arrowedLine(img, tuple(uv_proj[i]), tuple(uv_proj[i + 1]),
                                        color, 2, tipLength=0.25)
                else:
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


def render_heatmap_pair(gt_heatmap, pred_heatmap, gt_lines, pred_lines,
                        pc_range, save_path, idx, score_thresh,
                        max_sigma=5.0):
    """保存 GT vs Pred 热力图对比 + 剖面曲线 (双栏, 每栏结构与 visualize.py 一致)

    Layout per panel:
        Top: heatmap 顺时针旋转90° (前向朝下, 左侧在左)
             横线标注 1/4(x=0m,青), 1/2(x=10m,黄), 3/4(x=20m,品红)
        Bottom: 3条横截面曲线, X轴与热力图左右对齐
        Info bar
    """
    pc_min_x, pc_min_y, _, pc_max_x, pc_max_y, _ = pc_range
    x_range = pc_max_x - pc_min_x
    y_range = pc_max_y - pc_min_y

    slice_dists = [0.0, 10.0, 20.0]
    slice_colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
    slice_rows = [int((d - pc_min_x) / x_range * 160) for d in slice_dists]

    Y_LABEL_W = 70
    mr = 20
    HM_W, HM_H = 360, 720
    CV_H = 220
    INFO_H = 30
    MARGIN = 10
    GAP = 20
    PANEL_GAP = 40
    panel_W = MARGIN + Y_LABEL_W + HM_W + mr + MARGIN
    total_W = panel_W * 2 + PANEL_GAP
    total_H = MARGIN + HM_H + GAP + CV_H + GAP + INFO_H

    canvas = np.zeros((total_H, total_W, 3), dtype=np.uint8)

    for panel_idx, (heat, lines_dict, title_prefix, color_map) in enumerate([
        (gt_heatmap, gt_lines, 'GT', GT_COLORS),
        (pred_heatmap, pred_lines, 'Pred', PRED_COLORS),
    ]):
        panel_off = panel_idx * (panel_W + PANEL_GAP)
        hm_left = panel_off + MARGIN + Y_LABEL_W
        hm_top = MARGIN

        heat_rot = heat.T[::-1, ::-1]
        hm_rows, hm_cols = heat_rot.shape

        # ===== Top: rotated heatmap =====
        hm_raw = (heat_rot * 255).astype(np.uint8)
        hm_big = cv2.resize(hm_raw, (HM_W, HM_H), interpolation=cv2.INTER_LINEAR)
        hm_color = cv2.applyColorMap(hm_big, cv2.COLORMAP_JET)
        canvas[hm_top:hm_top + HM_H, hm_left:hm_left + HM_W] = hm_color

        for cls_id, lines in lines_dict.items():
            color = color_map.get(cls_id, (200, 200, 200))
            for pts_list in lines:
                pts = np.array(pts_list, dtype=np.float32)
                if pts.shape[1] < 2:
                    continue
                xy = pts[:, :2]
                dx = ((pc_max_x - xy[:, 0]) / x_range * HM_H).astype(np.int32)
                dy = ((pc_max_y - xy[:, 1]) / y_range * HM_W).astype(np.int32)
                pix = np.stack([dy, dx], axis=1) + np.array([hm_left, hm_top])
                if cls_id == 0:
                    for i in range(len(pix) - 1):
                        cv2.arrowedLine(canvas, tuple(pix[i]), tuple(pix[i + 1]),
                                        color, 1, tipLength=0.25)
                else:
                    cv2.polylines(canvas, [pix], False, color, 1, lineType=cv2.LINE_AA)

        for i, (d, row_i, sc) in enumerate(zip(slice_dists, slice_rows, slice_colors)):
            line_y = hm_top + int((160 - row_i) / 160 * HM_H)
            for x in range(hm_left, hm_left + HM_W, 8):
                cv2.line(canvas, (x, line_y), (min(x + 4, hm_left + HM_W - 1), line_y), sc, 2)
            label = f'x={d:.0f}m'
            cv2.putText(canvas, label, (hm_left + HM_W - 70, line_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, sc, 1)

        cv2.putText(canvas, f'{title_prefix} Heatmap (rotated)  idx={idx}',
                    (hm_left + 5, hm_top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, 'front (x=30m)',
                    (hm_left + HM_W // 2 - 45, hm_top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, 'rear (x=-10m)',
                    (hm_left + HM_W // 2 - 45, hm_top + HM_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ===== Bottom: cross-section curves =====
        cv_top = MARGIN + HM_H + GAP
        mt, mb = 30, 45
        ph = CV_H - mt - mb
        pw = HM_W

        bg_left = hm_left
        bg_right = hm_left + HM_W
        cv2.rectangle(canvas, (bg_left, cv_top), (bg_right, cv_top + CV_H), (25, 25, 25), -1)

        for val, label in [(0.0, '0.0'), (0.25, ''), (0.5, '0.5'), (0.75, ''), (1.0, '1.0')]:
            gy = cv_top + mt + int((1.0 - val) * ph)
            cv2.line(canvas, (bg_left, gy), (bg_right, gy), (40, 40, 40), 1)
            if label:
                cv2.putText(canvas, label, (panel_off + MARGIN + 2, gy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        for label, frac in [('left', 0), ('-5', 0.25), ('0', 0.5), ('5', 0.75), ('right', 1.0)]:
            gx = bg_left + int(frac * pw)
            cv2.line(canvas, (gx, cv_top + mt + ph), (gx, cv_top + mt + ph + 6), (80, 80, 80), 1)
            cv2.putText(canvas, label, (gx - 12, cv_top + mt + ph + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        cv2.putText(canvas, f'{title_prefix} Cross-Section Profiles',
                    (bg_left + 5, cv_top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        for i, (d, row_i, sc) in enumerate(zip(slice_dists, slice_rows, slice_colors)):
            flip_row = np.clip(159 - row_i, 0, 159)
            vals = heat_rot[flip_row, :]
            pts = []
            for ci in range(hm_cols):
                v = vals[ci]
                cx = bg_left + int(ci / (hm_cols - 1) * pw)
                cy = cv_top + mt + int((1.0 - v) * ph)
                cy = int(np.clip(cy, cv_top + mt, cv_top + mt + ph))
                pts.append((cx, cy))

            for j in range(len(pts) - 1):
                cv2.line(canvas, pts[j], pts[j + 1], sc, 2, lineType=cv2.LINE_AA)

            mid_ci = hm_cols // 2
            cv2.circle(canvas, pts[mid_ci], 4, sc, -1)

        center_vals = [heat_rot[np.clip(159 - ri, 0, 159), hm_cols // 2] for ri in slice_rows]
        legend_y = cv_top + mt + ph + 22
        for i, (d, sc, cv_mid) in enumerate(zip(slice_dists, slice_colors, center_vals)):
            lx = bg_left + i * (pw // 3)
            cv2.line(canvas, (lx, legend_y), (lx + 25, legend_y), sc, 2)
            cv2.putText(canvas, f'x={d:.0f}m  c={cv_mid:.3f}',
                        (lx + 30, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, sc, 1)

    # ===== Info bar =====
    info_y = MARGIN + HM_H + GAP + CV_H + GAP
    cv2.putText(canvas[info_y:],
                f'params: adaptive Gaussian  max_sigma={max_sigma:.1f}m  score>{score_thresh}',
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imwrite(str(save_path), canvas)


@torch.no_grad()
def infer():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='work_dirs/maptr/epoch_4.pth')
    parser.add_argument('--save-dir', type=str, default='work_dirs/infer')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--score-thresh', type=float, default=0.7)
    parser.add_argument('--seg-thresh', type=float, default=0.4,
                        help='预测分割掩码二值化阈值')
    args = parser.parse_args()

    global cfg
    from configs.loader import load_config
    cfg = load_config(args.config)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = MapTRDataset(cfg.data.val_ann_file, cfg.data.data_root, cfg.data, is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = MapTR(cfg.model).to(cfg.device)
    ckpt = torch.load(args.checkpoint, map_location=cfg.device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
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

        cls_scores, reg_preds, seg_preds, heatmap_pred = model(imgs, intrinsics, extrinsics)

        gt_seg_mask = batch['semantic_mask'][0].numpy()  # (num_classes, 80, 160)
        pred_seg_mask = seg_preds[0].sigmoid().cpu().numpy()  # (num_classes, 80, 160)
        gt_heatmap = batch['soft_heatmap'][0].numpy()            # (1, 80, 160)
        pred_heatmap = heatmap_pred[0].sigmoid().cpu().numpy()    # (1, 80, 160)

        pred_lines, pred_scores = decode_predictions(
            cls_scores[0], reg_preds[0], cfg.data.roi_size, cfg.data.pc_range, args.score_thresh)

        sample = ds.samples[batch_idx]
        gt_raw = vectors_to_world(
            batch['vectors'][0], cfg.data.roi_size, cfg.data.pc_range[0], cfg.data.pc_range[1])
        cam_names = get_cam_names(sample)

        canvas = np.zeros((total_h, canvas_w, 3), dtype=np.uint8)

        # BEV 面板
        draw_bev_panel(canvas[:ph, :pw], cfg.data.pc_range, gt_raw, pred_lines)

        # GT 掩码面板 (需要 flip 对齐 BEV 坐标)
        draw_mask_panel(canvas[:ph, pw + gap:pw * 2 + gap],
                        gt_seg_mask, 'GT Mask', flip_v=True)

        # Pred 掩码面板 (已经是 BEV 坐标，不 flip)
        pred_binary = (pred_seg_mask > args.seg_thresh).astype(np.float32)
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

        heat_out = save_dir / f'infer_{batch_idx:04d}_heatmap.png'
        render_heatmap_pair(
            gt_heatmap=gt_heatmap[0],
            pred_heatmap=pred_heatmap[0],
            gt_lines=gt_raw,
            pred_lines=pred_lines,
            pc_range=cfg.data.pc_range,
            save_path=heat_out,
            idx=batch_idx,
            score_thresh=args.score_thresh,
        )
        print(f'[保存] {heat_out}')

        rendered += 1

    print(f'[完成] 共 {rendered} 张, 保存至 {save_dir}')


if __name__ == '__main__':
    infer()
