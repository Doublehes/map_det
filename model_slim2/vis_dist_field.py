"""model_slim2 距离场调试脚本 (独立, 不接入模型)

距离场 = 局部厚度归一化:
    field = dist / local_thickness             # 中心=1, 到边界衰减到 0

骨架 = 最长路径的前向段 + 前向可达支路 (累计回退 <= V_BACK_TOL)
骨架坐标再做拉普拉斯平滑 (端点/岔口固定), 画成抗锯齿折线。
骨架图只用于显示和关键点, 不参与距离场定义。

运行:
    conda run -n deeptouchpro python model_slim2/vis_dist_field.py
    conda run -n deeptouchpro python model_slim2/vis_dist_field.py --selftest
按键: ↑/w 扫描行上移, ↓/s 扫描行下移
      A/← 上一张, D/→/Space 下一张, ESC/Q 退出
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_slim2.config as cfg_module
from model_slim2.field_pipeline import (
    CLASS_AWARE_SMOOTH, MIN_ROAD_WIDTH_PX, SMOOTH_SIGMA, FILL_MAX_AREA,
    KEEP_MAIN, CC_CONNECTIVITY, LOW_COST_CLASSES, S_LOW, S_OTHER,
    THICKNESS_MAX_R, PRUNE_LEN, V_BACK_TOL, SKEL_SMOOTH_ITERS,
    SKEL_SMOOTH_LAMBDA, KP_SPACING, KP_VFRONT, KP_MAX,
    drivable_binary, preclean_mask, build_field, build_skeleton,
    skeleton_guided_field, extract_keypoints_from_skeleton,
    selftest as pipeline_selftest,
)

CLASS_COLORS = cfg_module.CLASS_COLORS
IGNORE_COLOR = cfg_module.IGNORE_COLOR

def load_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f'无法读取: {path}')
    if m.ndim == 3:
        m = m[..., 0]
    return np.ascontiguousarray(m)


def colorize_mask(mask: np.ndarray, ignore_index: int) -> np.ndarray:
    out = np.full((*mask.shape, 3), IGNORE_COLOR, dtype=np.uint8)
    for c, col in CLASS_COLORS.items():
        out[mask == c] = col
    out[mask == ignore_index] = IGNORE_COLOR
    return out


def field_stats(field: np.ndarray, clean_mask: np.ndarray) -> dict:
    """距离场诊断指标 (仅用于终端打印)"""
    clean = clean_mask.astype(bool)
    f = np.clip(field, 0.0, 1.0).astype(np.float32)
    if not clean.any():
        return dict(plateau=0.0, max=0.0)
    return dict(plateau=float((f[clean] >= 0.99).mean()),
                max=float(f[clean].max()))


def to_vis(field: np.ndarray, cmap) -> np.ndarray:
    u8 = (np.clip(field, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cmap)


def render_profile(base_row: np.ndarray, corrected_row, scan_row: int,
                   show_corrected: bool = True,
                   w: int = 560, h: int = 280) -> np.ndarray:
    """把某一行的 base / 修正后距离场画成曲线图 (cv2 手绘)

    x 轴: 图像列 u; y 轴: 场值 0~1。
    绿=base(局部厚度), 青=修正后(骨架引导)。
    """
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)
    x0, x1 = 64, w - 16
    y0, y1 = h - 40, 64               # y0=场值0, y1=场值1

    def ypx(v):
        return int(y0 - np.clip(v, 0, 1) * (y0 - y1))

    # 纵轴网格线 + 刻度数字
    for gv in (0.0, 0.5, 1.0):
        yy = ypx(gv)
        cv2.line(canvas, (x0, yy), (x1, yy), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'{gv:.1f}', (x0 - 36, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)
    cv2.line(canvas, (x0, y1), (x0, y0), (120, 120, 120), 1, cv2.LINE_AA)   # 纵轴
    cv2.line(canvas, (x0, y0), (x1, y0), (120, 120, 120), 1, cv2.LINE_AA)   # 横轴

    def draw(vals, color, thickness=2):
        n = len(vals)
        if n < 2:
            return
        xs = np.linspace(x0, x1, n).astype(int)
        ys = np.array([ypx(v) for v in vals])
        for i in range(n - 1):
            cv2.line(canvas, (int(xs[i]), int(ys[i])), (int(xs[i + 1]), int(ys[i + 1])),
                     color, thickness, cv2.LINE_AA)

    draw(base_row, (0, 255, 0), 1)          # base BGR 绿
    if show_corrected and corrected_row is not None:
        draw(corrected_row, (255, 255, 0), 2)   # corrected BGR 青

    # 标题 + 轴标签
    cv2.putText(canvas, f'scanline profile  row = {scan_row}',
                (x0, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'field value', (6, (y0 + y1) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'image col  u  ->', (x0, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)

    # 图例 (右上: 色块 + 文字)
    lx, ly = x1 - 108, 46
    legend = [('base', (0, 255, 0))]
    if show_corrected and corrected_row is not None:
        legend.insert(0, ('corrected', (255, 255, 0)))
    for i, (name, color) in enumerate(legend):
        yy = ly + i * 18
        cv2.line(canvas, (lx, yy), (lx + 26, yy), color, 2, cv2.LINE_AA)
        cv2.putText(canvas, name, (lx + 32, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas




def render(mask, road_mask, base_f, win_f, skeleton, token, min_w, thick_max,
           scan_row, show_corrected=True, skel_sigma=3.0,
           kp_pts=None, kp_cands=None, v_front=KP_VFRONT):
    seg = colorize_mask(mask, 255)
    base_vis = to_vis(base_f, cv2.COLORMAP_VIRIDIS)
    win_vis = to_vis(win_f, cv2.COLORMAP_VIRIDIS)

    # 道路区域(喂给距离场/骨架的 pre['mask']): 绿=最终道路区域, 其余黑
    road_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    road_vis[road_mask > 0] = (0, 255, 0)

    # 骨架面板: 分割压暗为底 + 绿=平滑支路 + 红=平滑主骨架
    skel_vis = (seg * 0.45).astype(np.uint8)
    pos = skeleton.get('pos', {})
    for a, b in skeleton.get('edges', []):
        if a not in pos or b not in pos:
            continue
        ya, xa = pos[a]
        yb, xb = pos[b]
        cv2.line(skel_vis,
                 (int(round(xa)), int(round(ya))),
                 (int(round(xb)), int(round(yb))),
                 (0, 255, 0), 1, cv2.LINE_AA)
    main_pts = skeleton.get('main_pts', [])
    if len(main_pts) >= 2:
        pts = np.array([[int(round(x)), int(round(y))] for y, x in main_pts], np.int32)
        cv2.polylines(skel_vis, [pts], False, (0, 0, 255), 2, cv2.LINE_AA)

    # 车前关键点面板: 分割压暗为底 + 暗点=候选局部极大 + 亮点=入选关键点
    kp_vis = (seg * 0.5).astype(np.uint8)
    vf_row = int(round(v_front * (kp_vis.shape[0] - 1)))
    cv2.line(kp_vis, (0, vf_row), (kp_vis.shape[1] - 1, vf_row), (0, 200, 200), 1, cv2.LINE_AA)
    for (u, v, dv) in (kp_cands or []):
        cx = int(round(u * (kp_vis.shape[1] - 1)))
        cy = int(round(v * (kp_vis.shape[0] - 1)))
        cv2.circle(kp_vis, (cx, cy), 1, (90, 160, 90), -1, cv2.LINE_AA)
    for i, (u, v, dv) in enumerate(kp_pts or []):
        cx = int(round(u * (kp_vis.shape[1] - 1)))
        cy = int(round(v * (kp_vis.shape[0] - 1)))
        rad = 3
        cv2.circle(kp_vis, (cx, cy), rad, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(kp_vis, str(i), (cx + rad + 1, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    panels = [('seg', seg),
              ('road region', road_vis),
              ('base field (thickness)', base_vis)]
    if show_corrected:
        panels.append(('corrected field (skel-guided)', win_vis))
    panels += [('skeleton  main=red  branch=green', skel_vis),
               ('keypoints  front local-max + NMS', kp_vis)]
    # 扫描行标记: base + (可选) corrected
    for idx in ((2, 3) if show_corrected else (2,)):
        cv2.line(panels[idx][1], (0, scan_row), (panels[idx][1].shape[1] - 1, scan_row),
                 (255, 255, 255), 1, cv2.LINE_AA)
    for name, p in panels:
        cv2.putText(p, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    sep = np.full((mask.shape[0], 5, 3), 255, np.uint8)
    parts = []
    for i, (_, p) in enumerate(panels):
        if i > 0:
            parts.append(sep)
        parts.append(p)
    body = np.concatenate(parts, axis=1)

    header = np.full((30, body.shape[1], 3), 30, np.uint8)
    field_desc = 'thickness+skel_guided' if show_corrected else 'thickness'
    cv2.putText(header, f'token={token}  field={field_desc}  sigma={skel_sigma:.1f}  '
                        f'smooth={skeleton["stats"].get("smooth_iters", 0)}x'
                        f'{SKEL_SMOOTH_LAMBDA}  min_width={min_w}px  '
                        f'thickness_max_r={thick_max}(0=auto)  scan_row={scan_row}',
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return np.concatenate([header, body], axis=0)


def profile_row(new_f: np.ndarray, fg: np.ndarray) -> int:
    """选择剖面所在行: 取可通行区内距离最大处所在的行(最宽处)"""
    if fg.sum() == 0:
        return new_f.shape[0] // 2
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    return int(np.unravel_index(np.argmax(dist), dist.shape)[0])


def main():
    parser = argparse.ArgumentParser(description='距离场 / 骨架调试 (共享预处理)')
    parser.add_argument('--config', default='default', choices=list(cfg_module.CONFIGS.keys()))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--split', default='train', choices=('train', 'val'),
                        help='读取 train_dir 还是 val_dir')
    parser.add_argument('--min-width', type=float, default=MIN_ROAD_WIDTH_PX)
    parser.add_argument('--thickness-max-r', type=int, default=THICKNESS_MAX_R,
                        help='局部厚度最大半径上限, 0=自动(dist.max)')
    parser.add_argument('--smooth-iters', type=int, default=SKEL_SMOOTH_ITERS,
                        help='骨架几何平滑迭代次数, 0=关闭')
    parser.add_argument('--skel-field-sigma', type=float, default=3.0,
                        help='骨架引导距离场的门控宽度 sigma (px)')
    parser.add_argument('--no-skel-field', dest='skel_field', action='store_false',
                        default=True, help='关闭骨架引导距离场修正')
    parser.add_argument('--selftest', action='store_true',
                        help='运行合成自测后退出 (不读数据/不弹窗)')
    parser.add_argument('--no-keep-main', dest='keep_main', action='store_false',
                        default=KEEP_MAIN, help='关闭"只保留最大可通行连通域"')
    parser.add_argument('--cc-connectivity', type=int, default=CC_CONNECTIVITY,
                        choices=(4, 8), help='最大连通域连通性')
    parser.add_argument('--smooth', type=float, default=SMOOTH_SIGMA)
    parser.add_argument('--prune', type=int, default=PRUNE_LEN)
    parser.add_argument('--fill-max-area', type=int, default=FILL_MAX_AREA)
    parser.add_argument('--s-low', type=float, default=S_LOW,
                        help='class3 软值 (>0.5 向低代价区扩张; 越大越激进)')
    parser.add_argument('--s-other', type=float, default=S_OTHER,
                        help='其它类软值 (0=不扩张; <0 轻微回缩)')
    parser.add_argument('--no-class-aware', dest='class_aware', action='store_false',
                        default=CLASS_AWARE_SMOOTH, help='关闭类别感知平滑')
    parser.add_argument('--kp-spacing', type=float, default=KP_SPACING)
    parser.add_argument('--kp-max', type=int, default=KP_MAX)
    parser.add_argument('--kp-vfront', type=float, default=KP_VFRONT)
    args = parser.parse_args()
    if args.selftest:
        pipeline_selftest()
        return

    min_w, thick_max = args.min_width, args.thickness_max_r
    keep_main, cc_conn = args.keep_main, args.cc_connectivity
    smooth_sigma, prune_len, fill_max_area = args.smooth, args.prune, args.fill_max_area
    s_low, s_other, class_aware = args.s_low, args.s_other, args.class_aware
    smooth_iters = args.smooth_iters
    skel_field_on = args.skel_field
    skel_field_sigma = args.skel_field_sigma
    kp = dict(spacing=args.kp_spacing, max_pts=args.kp_max,
              v_front=args.kp_vfront)

    cfg = cfg_module.CONFIGS[args.config]
    data_dir = cfg.data.train_dir if args.split == 'train' else cfg.data.val_dir
    dirs = [Path(d) for d in data_dir] if isinstance(data_dir, (list, tuple)) else [Path(data_dir)]
    ignore_index = cfg.data.ignore_index
    drivable = tuple(cfg.data.drivable_classes)

    files = []
    for d in dirs:
        files.extend(sorted([p for p in d.iterdir() if p.suffix.lower() == '.png'],
                           key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem))
    print(f'[vis_dist_field] {len(files)} 张, drivable={drivable}')
    print(f'  参数: min_width={min_w}px, smooth={smooth_sigma}, fill_max_area={fill_max_area}, '
          f'keep_main={keep_main}(conn={cc_conn})  |  thickness_max_r={thick_max}(0=auto), prune={prune_len}px, '
          f'skel_smooth={smooth_iters}x{SKEL_SMOOTH_LAMBDA}')
    print(f'  类别感知平滑: {class_aware} (low_cost={LOW_COST_CLASSES}, s_low={s_low}, s_other={s_other})')
    print(f'  距离场: 局部厚度归一化  thickness_max_r={thick_max}(0=auto)  |  '
          f'skel_guided={skel_field_on} sigma={skel_field_sigma}')

    win, win_prof = 'dist_field', 'profile'
    LEFT_KEYS = {81, 2424832, 65361}
    RIGHT_KEYS = {83, 2555904, 65363}
    UP_KEYS = {82, 2490368, 65362}
    DOWN_KEYS = {84, 2621440, 65364}

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_prof, cv2.WINDOW_NORMAL)

    def load(i):
        path = files[i]
        mask = load_mask(path)
        fg = drivable_binary(mask, drivable, ignore_index)
        pre = preclean_mask(fg, min_w, smooth_sigma, fill_max_area, keep_main, cc_conn,
                            mask=mask, low_cost_classes=LOW_COST_CLASSES,
                            ignore_index=ignore_index, s_low=s_low, s_other=s_other,
                            class_aware=class_aware)
        skel_result = build_skeleton(pre['mask'], prune_len, keep_main, smooth_iters)
        sk = skel_result['stats']
        dist = cv2.distanceTransform(pre['mask'], cv2.DIST_L2, 5)
        base_f = build_field(pre['mask'], thick_max, dist=dist)
        base_f[pre['mask'] == 0] = 0.0
        if skel_field_on:
            win_f, finfo = skeleton_guided_field(
                pre['mask'], dist, base_f, skel_result, skel_field_sigma)
        else:
            win_f, finfo = base_f.copy(), None
        kp_pts = extract_keypoints_from_skeleton(
            skel_result, pre['mask'].shape, **kp)
        kp_cands = []
        st = field_stats(win_f, pre['mask'])
        st_base = field_stats(base_f, pre['mask'])
        print(f'[样本 {i}] token={path.stem}  normal={int(pre["mask"].sum())}px  '
              f'merged_low={int(pre["expanded_low"].sum())}px  '
              f'filled={int(pre["filled"].sum())}px  removed={int(pre["removed"].sum())}px  '
              f'cc_removed={int(pre["removed_cc"].sum())}px  keep_main={keep_main} conn={cc_conn}  '
              f'field base: plateau={st_base["plateau"]*100:.1f}% max={st_base["max"]:.2f}  |  '
              f'corrected: plateau={st["plateau"]*100:.1f}% max={st["max"]:.2f}')
        if finfo is not None:
            print(f'    skel_guided: sigma={skel_field_sigma} skel={finfo["skel_field"]:.3f} '
                  f'low_base={finfo["low_base_field"]:.3f}(n={finfo["low_base_n"]}) '
                  f'far_diff={finfo["far_diff"]:.3f} region={finfo["region_px"]}px')
        print(f'  skel: pix={sk["pix"]}->{sk["pix_pruned"]}->{sk["pix_kept"]} '
              f'smooth={sk["smooth_iters"]}x{SKEL_SMOOTH_LAMBDA} '
              f'main={sk["main_len"]} (v{sk["main_v0"]}->{sk["main_v1"]}, back={sk["main_back"]:.2f}, '
              f'border_trim={sk["border_trim"]}) '
              f'branch={sk["branch_pix"]} (max_back={sk["branch_max_back"]:.2f}) '
              f'removed_backward={sk["removed_backward"]} '
              f'end={sk["endpoints"]} junc={sk["junctions"]} comps={sk["comps"]}')
        if sk['main_back'] > V_BACK_TOL or sk['branch_max_back'] > V_BACK_TOL:
            print(f'    [warn] smooth v_back main={sk["main_back"]:.2f} '
                  f'branch={sk["branch_max_back"]:.2f} > tol={V_BACK_TOL}')
        print(f'    骨架关键点: 入选={len(kp_pts)}  spacing={kp["spacing"]}  '
               + ' '.join(f'({u:.2f},{v:.2f})' for (u, v, _) in kp_pts))
        return dict(mask=mask, pre=pre, base_f=base_f, win_f=win_f,
                    skel_field_on=skel_field_on, skel_field_sigma=skel_field_sigma,
                    skeleton=skel_result, kp_pts=kp_pts, kp_cands=kp_cands,
                    token=path.stem, h=win_f.shape[0])

    def show(s):
        cv2.imshow(win, render(
            s['mask'], s['pre']['mask'], s['base_f'], s['win_f'], s['skeleton'],
            s['token'], min_w, thick_max, scan_row,
            show_corrected=s['skel_field_on'], skel_sigma=s['skel_field_sigma'],
            kp_pts=s['kp_pts'], kp_cands=s['kp_cands'], v_front=kp['v_front']))
        cv2.imshow(win_prof, render_profile(
            s['base_f'][scan_row], s['win_f'][scan_row],
            scan_row, show_corrected=s['skel_field_on']))

    idx = args.start % len(files)
    s = load(idx)
    scan_row = profile_row(s['win_f'], s['pre']['mask'])

    while True:
        show(s)
        key = cv2.waitKey(0)
        if key in (27, ord('q')):
            break
        elif key in UP_KEYS or key == ord('w'):
            scan_row = max(0, scan_row - 1)
        elif key in DOWN_KEYS or key == ord('s'):
            scan_row = min(s['h'] - 1, scan_row + 1)
        elif key in LEFT_KEYS or key == ord('a'):
            idx = (idx - 1) % len(files)
            s = load(idx)
            scan_row = profile_row(s['win_f'], s['pre']['mask'])
        elif key in RIGHT_KEYS or key in (ord('d'), ord(' ')):
            idx = (idx + 1) % len(files)
            s = load(idx)
            scan_row = profile_row(s['win_f'], s['pre']['mask'])
        # 其它键: 忽略, 重绘当前帧

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
