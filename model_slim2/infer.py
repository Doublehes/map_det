"""model_slim2 推理可视化: 分割上色 | 代价图 | 预测线叠加

运行: conda run -n workspace python model_slim2/infer.py [--config default] [--save DIR]
checkpoint: cfg.work_dir/latest.pth
按键: A/<- 上一张, D/->/Space 下一张, ESC 退出 (--save 时不弹窗)
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_slim2.config as cfg_module
from model_slim2.dataset import Slim2Dataset
from model_slim2.losses import sample_map, loss_shares
from model_slim2.model import Slim2Model

CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (200, 200, 200), 3: (0, 255, 255)}
IGNORE_COLOR = (60, 60, 60)
LINE_COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
               (0, 165, 255), (255, 255, 0), (128, 0, 255), (0, 255, 128)]


def colorize_mask(mask: np.ndarray, ignore_index: int) -> np.ndarray:
    """标签图 → 上色 BGR"""
    out = np.full((*mask.shape, 3), IGNORE_COLOR, dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        out[mask == cls_id] = color
    out[mask == ignore_index] = IGNORE_COLOR
    return out


def draw_pred_lines(canvas: np.ndarray, points: np.ndarray) -> np.ndarray:
    """预测线 (N, P, 2) 归一化 (u,v) → 画到 BGR 画布, 带方向箭头"""
    h, w = canvas.shape[:2]
    img = canvas.copy()
    for qi, line in enumerate(points):
        color = LINE_COLORS[qi % len(LINE_COLORS)]
        pts = np.array([[int(np.clip(u, 0, 1) * (w - 1)), int(np.clip(v, 0, 1) * (h - 1))]
                        for u, v in line], dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.arrowedLine(img, tuple(pts[i]), tuple(pts[i + 1]), color, 2, tipLength=0.3)
        cv2.circle(img, tuple(pts[0]), 4, color, -1)
        cv2.putText(img, f'q{qi}', tuple(pts[0] + [6, -6]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def render(sample, points_np, cost_np, dist_np, ignore_index, kept, total):
    """四联面板: 分割上色 | 代价图(JET) | 距离场(VIRIDIS) | 预测线叠加距离场"""
    mask = sample['mask'].numpy()
    cost8 = (np.clip(cost_np, 0, 1) * 255).astype(np.uint8)
    cost_vis = cv2.applyColorMap(cost8, cv2.COLORMAP_JET)
    dist8 = (np.clip(dist_np, 0, 1) * 255).astype(np.uint8)
    dist_vis = cv2.applyColorMap(dist8, cv2.COLORMAP_VIRIDIS)
    pred_vis = draw_pred_lines(dist_vis, points_np)

    panels = [
        ('seg', colorize_mask(mask, ignore_index)),
        ('cost', cost_vis),
        ('dist', dist_vis),
        ('pred@dist', pred_vis),
    ]
    for name, p in panels:
        cv2.putText(p, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    sep = np.full((mask.shape[0], 5, 3), 255, dtype=np.uint8)
    parts = []
    for i, (_, p) in enumerate(panels):
        if i > 0:
            parts.append(sep)
        parts.append(p)
    body = np.concatenate(parts, axis=1)

    header = np.full((30, body.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(header, f'token={sample["token"]}  kept={kept}/{total}',
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return np.concatenate([header, body], axis=0)


PER_QUERY_KEYS = ['cost', 'interior', 'smooth', 'spacing', 'arc', 'span']


def geometry_str(points_np):
    """预测线 (N,P,2) → 几何诊断字符串: 每条线的 弧长/净跨(缠绕度)/平均转角"""
    parts = []
    for qi, line in enumerate(points_np):
        seg = np.linalg.norm(np.diff(line, axis=0), axis=-1)          # (P-1,)
        arc = float(seg.sum())
        span = float(np.linalg.norm(line[-1] - line[0]))
        v = np.diff(line, axis=0)
        v = v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6, None)
        cos = (v[:-1] * v[1:]).sum(-1) if len(v) > 1 else np.array([0.0])
        turn = float((1.0 - cos).mean())
        parts.append(f'q{qi}[arc={arc:.2f} span={span:.2f} '
                     f'arc/span={arc / max(span, 1e-6):.2f} turn={turn:.2f}]')
    return ' '.join(parts)


def print_breakdown(token, bd_np):
    """控制台打印 per-query 损失明细表 (每行一条线, 每列一项损失)"""
    n = bd_np['cost'].shape[0]
    print(f'[loss breakdown] token={token}')
    print('  q | ' + ' '.join(f'{k:>8}' for k in PER_QUERY_KEYS))
    for q in range(n):
        row = ' '.join(f'{bd_np[k][q]:8.3f}' for k in PER_QUERY_KEYS)
        print(f'{q:3d} | {row}')


def render_pair_heatmap(matrix, title, vmax, cell=44):
    """(N,N) 线对矩阵 → JET 热力图, 单元格标注数值 + q 轴标签"""
    n = matrix.shape[0]
    lab, top = 30, 30
    canvas = np.full((top + lab + n * cell, lab + n * cell, 3), 30, np.uint8)
    m8 = (np.clip(matrix / max(vmax, 1e-6), 0, 1) * 255).astype(np.uint8)
    big = cv2.applyColorMap(
        cv2.resize(m8, (n * cell, n * cell), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_JET)
    y0, x0 = top + lab, lab
    canvas[y0:y0 + n * cell, x0:x0 + n * cell] = big
    for i in range(n):
        for j in range(n):
            cv2.putText(canvas, f'{matrix[i, j]:.2f}',
                        (x0 + j * cell + 3, y0 + i * cell + cell // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    for j in range(n):
        cv2.putText(canvas, f'q{j}', (x0 + j * cell + cell // 2 - 8, top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    for i in range(n):
        cv2.putText(canvas, f'q{i}', (4, y0 + i * cell + cell // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_loss_panel(bd_np, margin, thr):
    """per-pair 热力图面板: repel | spread (N×N)"""
    repel = render_pair_heatmap(bd_np['repel'], f'repel (margin={margin})', margin)
    spread = render_pair_heatmap(bd_np['spread'], f'spread (thr={thr})', 1.0 - thr)
    sep = np.full((repel.shape[0], 5, 3), 255, np.uint8)
    return np.concatenate([repel, sep, spread], axis=1)


def main():
    parser = argparse.ArgumentParser(description='Slim2 推理可视化')
    parser.add_argument('--config', default='default', choices=list(cfg_module.CONFIGS.keys()))
    parser.add_argument('--ckpt', default=None, help='checkpoint 路径')
    parser.add_argument('--save', default='', help='保存目录 (不弹窗)')
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()
    cfg = cfg_module.CONFIGS[args.config]

    ckpt_path = os.path.join(cfg.work_dir, 'latest.pth') if args.ckpt is None else args.ckpt
    print(f'[加载] {ckpt_path}')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'checkpoint 不存在: {ckpt_path}')

    model = Slim2Model(cfg).to(cfg.device)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f'[加载] 缺失 {len(missing)} 个参数, 未使用 {len(unexpected)} 个参数 {missing}')
    print(f'[加载] 未使用 {len(unexpected)} 个参数 {unexpected}')

    model.eval()
    print(f'[加载] epoch={ckpt.get("epoch", "?")}')

    ds = Slim2Dataset(cfg.data.seg_dir, cfg.data, is_train=False)
    thr = cfg.cost_filter_thr

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    LEFT_KEYS = {81, 2424832, 65361}
    RIGHT_KEYS = {83, 2555904, 65363}
    win = 'slim2'
    win_loss = 'slim2_loss'

    idx = args.start
    with torch.no_grad():
        while 0 <= idx < len(ds):
            sample = ds[idx]
            inp = sample['input'].unsqueeze(0).to(cfg.device)
            cls_scores, reg_preds = model(inp)
            # 对全部 query 计算自监督 loss (与训练一致), 仅打印到控制台
            batch = {
                'cost': sample['cost'].unsqueeze(0).to(cfg.device),
                'dist_field': sample['dist_field'].unsqueeze(0).to(cfg.device),
            }
            loss_dict = model.compute_loss(cls_scores, reg_preds, batch)
            loss_str = ' '.join(f'{k}={v.item():.3f}' for k, v in loss_dict.items())
            shares = loss_shares(loss_dict)
            share_str = ' '.join(f'{k}={100 * v:.0f}%' for k, v in shares.items())

            # 逐项 loss 明细: 5×per-query + 2×per-pair
            bd = model.head.loss_breakdown(cls_scores, reg_preds,
                                           batch['cost'], batch['dist_field'])
            bd_np = {k: v[0].cpu().numpy() for k, v in bd.items()}
            print_breakdown(sample['token'], bd_np)
            loss_panel = render_loss_panel(bd_np, cfg.map_det_head.repel_margin,
                                           cfg.map_det_head.spread_thr)

            points = reg_preds[0].cpu().numpy()                     # (N, P, 2)
            geom = geometry_str(points)
            cost_np = sample['cost'][0].numpy()                    # (H, W)
            dist_np = sample['dist_field'][0].numpy()              # (H, W)

            # import matplotlib.pyplot as plt
            # plt.plot(points[0, :, 0], points[0, :, 1])
            # plt.show()

            # 按平均采样代价过滤
            cost_t = torch.from_numpy(cost_np)[None, None]
            samp = sample_map(cost_t, torch.from_numpy(points)[None]).squeeze(0).numpy()
            mean_cost = samp.mean(axis=-1)                         # (N,)
            keep = mean_cost < thr
            kept_points = points[keep] if keep.any() else points[:0]

            panel = render(sample, kept_points, cost_np, dist_np,
                           cfg.data.ignore_index, int(keep.sum()), len(points))

            if save_dir:
                cv2.imwrite(str(save_dir / f'{sample["token"]}.png'), panel)
                cv2.imwrite(str(save_dir / f'{sample["token"]}_loss.png'), loss_panel)
                print(f'[{idx + 1}/{len(ds)}] token={sample["token"]} '
                      f'kept={int(keep.sum())}/{len(points)}  {loss_str}')
                print(f'    share: {share_str}')
                print(f'    geom : {geom}')
                idx += 1
                if idx >= len(ds):
                    break
            else:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.imshow(win, panel)
                cv2.namedWindow(win_loss, cv2.WINDOW_NORMAL)
                cv2.imshow(win_loss, loss_panel)
                print(f'[样本 {idx}] kept={int(keep.sum())}/{len(points)} '
                      f'mean_cost={mean_cost.round(3).tolist()}  {loss_str}')
                print(f'    share: {share_str}')
                print(f'    geom : {geom}')
                key = cv2.waitKey(0)
                if key in (27, ord('q')):
                    break
                elif key in LEFT_KEYS or key == ord('a'):
                    idx = (idx - 1) % len(ds)
                else:
                    idx += 1

    if save_dir:
        print(f'[完成] 保存到 {save_dir}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
