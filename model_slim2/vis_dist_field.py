"""model_slim2 距离场调试脚本 (独立, 不接入模型)

目标: 对比"现有距离场"与"新距离场(方案A)"的效果, 交互式查看, 调好再进模型。

新距离场 (方案 A: 按局部路半宽归一化):
    1) 形态学开运算剔除 < min_road_width_px 的窄结构(含附着细支)
    2) dist = 到边界距离变换
    3) local_max = 局部最大(灰度膨胀) ≈ 该处脊线高度(= 局部路半宽)
    4) field = clip(dist / local_max, 0, 1)  -> 中心恒为 1, 宽路衰减慢/窄路衰减快, 无平台

现有距离场 (对照): field = clip(dist / dist_max_px, 0, 1), 非可通行区 = 0

运行:
    conda run -n deeptouchpro python model_slim2/vis_dist_field.py
按键: ↑/w 扫描行上移, ↓/s 扫描行下移
      A/← 上一张, D/→/Space 下一张, ESC/Q 退出
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_slim2.config as cfg_module

CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (200, 200, 200), 3: (0, 255, 255)}
IGNORE_COLOR = (60, 60, 60)

# ── 调试参数 (改这里即可, 或加命令行) ──
MIN_ROAD_WIDTH_PX = 5      # 距离场: 路宽 < 该值 -> 不算路 (剔除)
RIDGE_WINDOW_PX = 30        # 局部最大窗口半径, 应 >= 数据最大路半宽

# 骨架专用参数 (与距离场解耦)
SKEL_MIN_WIDTH_PX = 6      # 骨架: 剔除宽度 < 该值的结构 (开运算, 去毛刺/细缝)
SMOOTH_SIGMA = 2.0          # 骨架: 边界高斯平滑 sigma (px), 0=不平滑
PRUNE_LEN = 50              # 毛刺枝长度阈值 (像素): 端点枝 < 该值则剪除
FILL_MAX_AREA = 8           # 只填面积 <= 该值的封闭孔洞 (px); 更大的视为真口袋, 不填


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


def drivable_binary(mask: np.ndarray, drivable_classes, ignore_index: int) -> np.ndarray:
    fg = np.zeros(mask.shape, dtype=np.uint8)
    for c in drivable_classes:
        fg[mask == c] = 1
    fg[mask == ignore_index] = 0
    return fg


def old_dist_field(fg: np.ndarray, max_dist_px: float) -> np.ndarray:
    """现有实现 (对照): clip(dist/max_dist_px, 0, 1), 非可通行区 = 0"""
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    out = np.clip(dist / max(max_dist_px, 1e-6), 0.0, 1.0).astype(np.float32)
    out[fg == 0] = 0.0
    return out


def build_field(fg: np.ndarray, min_road_width_px: float = MIN_ROAD_WIDTH_PX,
                ridge_window_px: float = RIDGE_WINDOW_PX):
    """方案 A 新距离场。返回 (field, fg_opened, removed_mask)

    field      : (H,W) float32, 中心=1 两边衰减, 非前景=0
    fg_opened  : 开运算后的可通行 mask (去窄路)
    removed    : 被剔除的像素 (原可通行 但 开运算后不属于前景)
    """
    # 1) 开运算剔除窄结构
    r = max(1, int(round(min_road_width_px / 2.0)))
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fg_open = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ell)
    removed = ((fg == 1) & (fg_open == 0))

    # 2) 到边界距离
    dist = cv2.distanceTransform(fg_open, cv2.DIST_L2, 5)

    # 3) 局部最大 = 脊线高度 (灰度膨胀 = 最大滤波)
    R = max(1, int(round(ridge_window_px)))
    ell2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1))
    local_max = cv2.dilate(dist, ell2)
    local_max = np.maximum(local_max, 1e-6)

    # 4) 按局部半宽归一化: 中心=1, 宽慢窄快, 无平台
    field = np.clip(dist / local_max, 0.0, 1.0).astype(np.float32)
    field[fg_open == 0] = 0.0
    return field, fg_open, removed


def to_vis(field: np.ndarray, cmap) -> np.ndarray:
    u8 = (np.clip(field, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cmap)


def _largest_cc(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """取最大连通域 (bool)"""
    m = mask.astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=connectivity)
    if n <= 2:
        return mask.astype(bool)
    sizes = [int((lab == k).sum()) for k in range(1, n)]
    return lab == (1 + int(np.argmax(sizes)))


def _fill_small_holes(mask: np.ndarray, max_area: int = FILL_MAX_AREA) -> np.ndarray:
    """只填"被可动区完全包围"且面积 <= max_area 的孔洞 (bool -> bool)

    更大的封闭区域视为真正的不可通行"口袋/小岛", 保持不可通行, 避免骨架穿过类别 0。
    与图像边界相连的背景属于"外部", 不填。
    """
    mask = mask.astype(bool)
    inv = ~mask
    lab, n = ndi.label(inv)
    if n == 0:
        return mask
    border = (set(lab[0, :].tolist()) | set(lab[-1, :].tolist())
              | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist()))
    border.discard(0)
    out = mask.copy()
    for k in range(1, n + 1):
        if k in border:
            continue
        comp = (lab == k)
        if int(comp.sum()) <= max_area:
            out |= comp
    return out


def _preclean_mask(fg: np.ndarray, skel_min_width: float = SKEL_MIN_WIDTH_PX,
                   smooth_sigma: float = SMOOTH_SIGMA, fill_holes: bool = True,
                   fill_max_area: int = FILL_MAX_AREA, keep_main: bool = True) -> np.ndarray:
    """骨架前处理: 开运算去窄结构 -> 高斯平滑边界 -> 填小洞 -> 最大连通域 -> bool mask

    注意: 不做闭运算 (会把非可通行缝隙糊上导致骨架穿过类别0)。
    孔洞只填面积 <= fill_max_area 的小洞 (噪声), 大的封闭口袋保持不可通行。
    """
    r = max(1, int(round(skel_min_width / 2.0)))
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, ell)
    if smooth_sigma and smooth_sigma > 0:          # 平滑锯齿边界, 减少毛刺
        f = cv2.GaussianBlur(m.astype(np.float32), (0, 0), float(smooth_sigma))
        m = (f > 0.5).astype(np.uint8)
    if fill_holes:                                 # 只填小孔洞, 消除绕洞的环/枝
        m = _fill_small_holes(m > 0, fill_max_area).astype(np.uint8)
    if keep_main:
        m = _largest_cc(m > 0).astype(np.uint8)
    return m > 0


def _nbrs(y, x, H, W, s):
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and s[ny, nx]:
                out.append((ny, nx))
    return out


def prune_spurs(skel: np.ndarray, prune_len: int = 10) -> np.ndarray:
    """剪除长度 < prune_len 的毛刺(端点分支): 只删"从端点走到第一个交叉点"的短枝。"""
    skel = skel.astype(bool).copy()
    if prune_len <= 0 or not skel.any():
        return skel
    H, W = skel.shape
    for _ in range(8):
        ys, xs = np.where(skel)
        endpoints = [(y, x) for y, x in zip(ys.tolist(), xs.tolist())
                     if len(_nbrs(y, x, H, W, skel)) == 1]
        removed = False
        for (y, x) in endpoints:
            if not skel[y, x] or len(_nbrs(y, x, H, W, skel)) != 1:
                continue
            path = [(y, x)]
            prev, cy, cx = None, y, x
            hit_junction = False
            while True:
                deg = len(_nbrs(cy, cx, H, W, skel))
                if deg >= 3:
                    hit_junction = True
                    break
                nb = [p for p in _nbrs(cy, cx, H, W, skel) if p != prev]
                if not nb:
                    break                      # 走到了孤立短段
                if len(path) > prune_len:      # 已超长 -> 真枝, 停止
                    break
                nxt = nb[0]
                path.append(nxt)
                prev, cy, cx = (cy, cx), nxt[0], nxt[1]
            short = len(path) <= prune_len
            dead_end = len(_nbrs(cy, cx, H, W, skel)) <= 1
            if short and (hit_junction or dead_end):
                for (py, px) in path[:-1] if hit_junction else path:
                    skel[py, px] = False
                removed = True
        if not removed:
            break
    return skel


def build_skeleton(fg: np.ndarray, skel_min_width: float = SKEL_MIN_WIDTH_PX,
                   smooth_sigma: float = SMOOTH_SIGMA, prune_len: int = PRUNE_LEN,
                   fill_holes: bool = True, fill_max_area: int = FILL_MAX_AREA,
                   keep_main: bool = True):
    """提取道路骨架(完整路网) -> 返回 (skel_bool, longest_bool, stats_dict)

    流程: 开运算(skel_min_width) -> 高斯平滑(smooth_sigma) -> 只填小洞(fill_max_area)
          -> 最大连通域 -> skeletonize -> 剪毛刺(prune_len) -> 最大骨架连通分量
    """
    clean = _preclean_mask(fg, skel_min_width, smooth_sigma, fill_holes, fill_max_area, keep_main)
    skel = skeletonize(clean) if clean.any() else np.zeros_like(clean, bool)
    n_raw = int(skel.sum())
    skel = prune_spurs(skel, prune_len)
    if keep_main:
        skel = _largest_cc(skel)
    longest = longest_path(skel)
    # 统计: 端点 / 交叉点 / 连通分量
    H, W = skel.shape
    ys, xs = np.where(skel)
    n_end = n_junc = 0
    for y, x in zip(ys.tolist(), xs.tolist()):
        d = len(_nbrs(y, x, H, W, skel))
        n_end += int(d == 1)
        n_junc += int(d >= 3)
    ncc, _ = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    stats = dict(pix=n_raw, pix_pruned=int(skel.sum()), endpoints=n_end,
                 junctions=n_junc, comps=int(ncc) - 1, longest=int(longest.sum()),
                 removed_spur=n_raw - int(skel.sum()))
    return skel, longest, stats


def longest_path(skel: np.ndarray) -> np.ndarray:
    """骨架图直径(最长路径) -> bool mask。两次 BFS(无权, 8 邻接)。"""
    from collections import deque
    H, W = skel.shape
    ys, xs = np.where(skel)
    if len(ys) == 0:
        return np.zeros_like(skel, bool)
    coords = set(zip(ys.tolist(), xs.tolist()))

    def bfs(start):
        dist = {start: 0}
        par = {start: None}
        q = deque([start])
        far = start
        while q:
            cy, cx = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nb = (cy + dy, cx + dx)
                    if nb in coords and nb not in dist:
                        dist[nb] = dist[(cy, cx)] + 1
                        par[nb] = (cy, cx)
                        q.append(nb)
                        if dist[nb] > dist[far]:
                            far = nb
        return far, par

    a = next(iter(coords))
    b, _ = bfs(a)
    c, par = bfs(b)
    mask = np.zeros_like(skel, bool)
    cur = c
    while cur is not None:
        mask[cur] = True
        cur = par[cur]
    return mask


def render_profile(old_row: np.ndarray, win_row: np.ndarray, scan_row: int,
                   w: int = 560, h: int = 280) -> np.ndarray:
    """把某一行的 旧场 / 窗口法 字段值画成曲线图 (cv2 手绘, 不依赖 matplotlib)

    x 轴: 图像列 u; y 轴: 场值 0~1。蓝=旧场(old dist), 绿=窗口法新场(window)。
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

    def draw(vals, color):
        n = len(vals)
        if n < 2:
            return
        xs = np.linspace(x0, x1, n).astype(int)
        ys = np.array([ypx(v) for v in vals])
        for i in range(n - 1):
            cv2.line(canvas, (int(xs[i]), int(ys[i])), (int(xs[i + 1]), int(ys[i + 1])), color, 2, cv2.LINE_AA)

    draw(old_row, (255, 128, 0))    # 旧场   BGR 蓝
    draw(win_row, (0, 255, 0))      # 窗口法 BGR 绿

    # 标题 + 轴标签
    cv2.putText(canvas, f'scanline profile  row = {scan_row}',
                (x0, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'field value', (6, (y0 + y1) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'image col  u  ->', (x0, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)

    # 图例 (右上: 色块 + 文字)
    lx, ly = x1 - 150, 46
    cv2.line(canvas, (lx, ly), (lx + 26, ly), (255, 128, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'old dist', (lx + 32, ly + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.line(canvas, (lx, ly + 20), (lx + 26, ly + 20), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'window', (lx + 32, ly + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def render(mask, old_f, win_f, skel, longest, removed, fg, token, min_w, win_r, scan_row):
    seg = colorize_mask(mask, 255)
    old_vis = to_vis(old_f, cv2.COLORMAP_VIRIDIS)
    win_vis = to_vis(win_f, cv2.COLORMAP_VIRIDIS)

    # 窗口法场 + 移除区域高亮(红)
    win_hl = win_vis.copy()
    win_hl[removed] = (0, 0, 255)

    # 骨架面板: 分割压暗为底 + 绿=完整骨架 + 红=最长路径
    k3 = np.ones((3, 3), np.uint8)
    skel_d = cv2.dilate(skel.astype(np.uint8), k3) > 0
    long_d = cv2.dilate(longest.astype(np.uint8), k3) > 0
    skel_vis = (seg * 0.45).astype(np.uint8)
    skel_vis[skel_d] = (0, 255, 0)
    skel_vis[long_d] = (0, 0, 255)

    panels = [('seg', seg), ('old dist', old_vis), ('window field', win_vis),
              ('window + removed', win_hl), ('skeleton  all=green  longest=red', skel_vis)]
    # 扫描行标记 (只画在3个场面板上, 不含 seg 与 skeleton)
    for _, p in panels[1:4]:
        cv2.line(p, (0, scan_row), (p.shape[1] - 1, scan_row), (255, 255, 255), 1, cv2.LINE_AA)
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
    cv2.putText(header, f'token={token}  min_width={min_w}px  window={win_r}px  '
                        f'removed={100.0 * removed.mean():.1f}%  scan_row={scan_row}',
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return np.concatenate([header, body], axis=0)


def profile_row(new_f: np.ndarray, fg: np.ndarray) -> int:
    """选择剖面所在行: 取可通行区内距离最大处所在的行(最宽处)"""
    if fg.sum() == 0:
        return new_f.shape[0] // 2
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    return int(np.unravel_index(np.argmax(dist), dist.shape)[0])


def main():
    parser = argparse.ArgumentParser(description='距离场调试 (方案A vs 现有)')
    parser.add_argument('--config', default='default', choices=list(cfg_module.CONFIGS.keys()))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--min-width', type=float, default=MIN_ROAD_WIDTH_PX)
    parser.add_argument('--window', type=float, default=RIDGE_WINDOW_PX)
    parser.add_argument('--skel-min-width', type=float, default=SKEL_MIN_WIDTH_PX)
    parser.add_argument('--smooth', type=float, default=SMOOTH_SIGMA)
    parser.add_argument('--prune', type=int, default=PRUNE_LEN)
    parser.add_argument('--fill-max-area', type=int, default=FILL_MAX_AREA)
    args = parser.parse_args()
    min_w, win_r = args.min_width, args.window
    skel_w, smooth_sigma, prune_len = args.skel_min_width, args.smooth, args.prune
    fill_max_area = args.fill_max_area

    cfg = cfg_module.CONFIGS[args.config]
    seg_dir = Path(cfg.data.seg_dir)
    ignore_index = cfg.data.ignore_index
    drivable = tuple(cfg.data.drivable_classes)
    max_dist_px = cfg.data.get('dist_max_px', 10.0)

    files = sorted([p for p in seg_dir.iterdir() if p.suffix.lower() == '.png'],
                   key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    print(f'[vis_dist_field] {len(files)} 张, drivable={drivable}, max_dist_px={max_dist_px}')
    print(f'  参数: min_road_width={min_w}px, ridge_window={win_r}px  |  '
          f'skel_min_width={skel_w}px, smooth={smooth_sigma}, prune={prune_len}px')

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
        old_f = old_dist_field(fg, max_dist_px)
        win_f, fg_open, removed = build_field(fg, min_w, win_r)
        skel, longest, sk = build_skeleton(fg, skel_w, smooth_sigma, prune_len, True, fill_max_area)
        print(f'[样本 {i}] token={path.stem}  removed={100.0 * removed.mean():.1f}%  '
              f'old: max={old_f.max():.2f} plateau={100.0 * (old_f >= 0.999).mean():.1f}%  '
              f'win: max={win_f.max():.2f}  |  skel: pix={sk["pix"]}->{sk["pix_pruned"]} '
              f'end={sk["endpoints"]} junc={sk["junctions"]} comps={sk["comps"]} '
              f'longest={sk["longest"]}px')
        return dict(mask=mask, fg=fg, old_f=old_f, win_f=win_f, skel=skel, longest=longest,
                    removed=removed, token=path.stem, h=win_f.shape[0])

    def show(s):
        cv2.imshow(win, render(s['mask'], s['old_f'], s['win_f'], s['skel'], s['longest'],
                               s['removed'], s['fg'], s['token'], min_w, win_r, scan_row))
        cv2.imshow(win_prof, render_profile(s['old_f'][scan_row], s['win_f'][scan_row], scan_row))

    idx = args.start % len(files)
    s = load(idx)
    scan_row = profile_row(s['win_f'], s['fg'])

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
            scan_row = profile_row(s['win_f'], s['fg'])
        elif key in RIGHT_KEYS or key in (ord('d'), ord(' ')):
            idx = (idx + 1) % len(files)
            s = load(idx)
            scan_row = profile_row(s['win_f'], s['fg'])
        # 其它键: 忽略, 重绘当前帧

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
