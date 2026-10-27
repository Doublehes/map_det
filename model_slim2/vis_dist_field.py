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
import heapq
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_slim2.config as cfg_module

CLASS_COLORS = cfg_module.CLASS_COLORS
IGNORE_COLOR = cfg_module.IGNORE_COLOR

# ── 调试参数 (改这里即可, 或加命令行) ──
# 预处理 (距离场 / 骨架共用, 每帧只跑一次)
MIN_ROAD_WIDTH_PX = 6      # 开运算: 路宽 < 该值 -> 不算路 (剔除窄结构)
SMOOTH_SIGMA = 2.0          # 边界高斯平滑 sigma (px), 0=不平滑
FILL_MAX_AREA = 12           # 只填面积 <= 该值的封闭孔洞 (px); 更大的视为真口袋, 不填
KEEP_MAIN = True           # 只保留最大可通行连通域 (删掉其它小块可通行区)
CC_CONNECTIVITY = 8        # 最大连通域用的连通性: 4 或 8

# 类别感知平滑 (与 config TARGET_CLASSES_6 对应: 4=road, 3=low_cost)
LOW_COST_CLASSES = (3, 255)     # 低代价区: 与之接壤时允许"更大胆"的平滑(并入)
S_LOW = 0.8                 # class3 软值 (>0.5 => 边界向3推进; 越大越激进)
S_OTHER = 0.6               # 其它类软值 (0=不扩张; <0=轻微回缩)
CLASS_AWARE_SMOOTH = True   # False=退化为二值平滑

# 距离场: 局部厚度归一化
THICKNESS_MAX_R = 0         # 逐级开运算最大半径上限; 0=自动(=round(dist.max()))
THICKNESS_MIN_R = 1         # 最小探测半径
PRUNE_LEN = 30              # 骨架: 毛刺枝长度阈值 (像素): 端点枝 < 该值则剪除
V_BACK_TOL = 2              # 骨架前向约束: 整条路径允许累计回退(v 增大) <= 该值 (px)
BORDER_MARGIN = 15          # 主骨架端点距图像边界 <= 该值才做边界收尾 (px)
BORDER_CENTER_ALPHA = 0.8   # 收到 dist >= alpha*局部最大 的位置 (0~1)
BORDER_MAX_TRIM = 60        # 边界收尾最多回看/删除的像素数
SKEL_SMOOTH_ITERS =10       # 骨架几何平滑迭代次数; 0=关闭
SKEL_SMOOTH_LAMBDA = 0.5    # 拉普拉斯平滑权重

# 车前关键点 (从距离场 win_f 局部极大提取, 仅车前 v<=VFRONT)
KP_WIN = 2                  # 局部极大窗口半径 (px): field >= dilate(field, 2*win+1)
KP_EPS = 0.02               # 平台容差 (字段单位[0,1]): 处理均匀宽度走廊的 field 平台
KP_FIELD_FLOOR = 0.3        # 字段强度下限[0,1]: 丢弃太弱的枝
KP_VFRONT = 0.7             # 只看车前: 保留 v <= 该值 (v=0 顶部=车前)
KP_DMIN = 0.05              # 关键点最小间隔 (归一化): NMS 保证离散
KP_MAX = 100                  # 最多关键点个数


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


def local_thickness(dist: np.ndarray, mask: np.ndarray,
                    max_r: int = THICKNESS_MAX_R,
                    min_r: int = THICKNESS_MIN_R) -> np.ndarray:
    """局部厚度 T(x) = 包含 x 的最大内切圆半径 (逐级开运算实现, 无需骨架)

    T(x) = max{ r : x ∈ opening(mask, disk_r) }
         = max{ r : dist(x, {dist>=r}) <= r }
    从大到小枚举 r, 记录每个点首次被覆盖的 r 即为最大半径。
    """
    m = mask > 0
    Rmax = int(np.floor(dist.max())) if max_r <= 0 else int(max_r)
    Rmax = max(Rmax, min_r)
    T = np.zeros_like(dist, dtype=np.float32)
    for r in range(Rmax, min_r - 1, -1):
        E = (dist >= r).astype(np.uint8)                    # 半径 r 的圆心集合
        d = cv2.distanceTransform((1 - E).astype(np.uint8), cv2.DIST_L2, 5)
        cover = (d <= r) & m                                # opening_r 覆盖区域
        new = cover & (T == 0)
        T[new] = r                                          # 首次覆盖 => 最大 r
        if not (T[m] == 0).any():                           # 全道路已覆盖, 提前停
            break
    T[T == 0] = float(min_r)                                # 安全兜底
    return np.maximum(T, 1e-6)


def build_field(clean_mask: np.ndarray, thickness_max_r: int = THICKNESS_MAX_R,
                dist: np.ndarray = None) -> np.ndarray:
    """由已预处理的可通行掩膜计算"局部厚度归一化"基础场

    clean_mask: preclean_mask()['mask'] (uint8 0/1)
    dist: 可选, 已算好的边界距离 (避免重复 distanceTransform)
    返回 field: (H,W) float32, 中心(含颈部)=1, 到边界衰减到 0, 非前景=0
    """
    clean_mask = clean_mask.astype(np.uint8)
    if clean_mask.sum() == 0:
        return np.zeros(clean_mask.shape, np.float32)
    if dist is None:
        dist = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    T = local_thickness(dist, clean_mask, max_r=thickness_max_r)
    field = np.clip(dist / T, 0.0, 1.0).astype(np.float32)
    field[clean_mask == 0] = 0.0
    return field


def field_stats(field: np.ndarray, clean_mask: np.ndarray) -> dict:
    """距离场诊断指标 (仅用于终端打印)"""
    clean = clean_mask.astype(bool)
    f = np.clip(field, 0.0, 1.0).astype(np.float32)
    if not clean.any():
        return dict(plateau=0.0, max=0.0)
    return dict(plateau=float((f[clean] >= 0.99).mean()),
                max=float(f[clean].max()))


def _selftest_mask() -> np.ndarray:
    """合成等宽直道"""
    m = np.zeros((120, 61), np.uint8)
    c = 30
    m[10:110, c - 10:c + 11] = 1
    return m


def selftest() -> None:
    """不读数据、不弹窗的合成自测"""
    def run(m):
        clean = m.astype(np.uint8)
        dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
        field = build_field(clean, THICKNESS_MAX_R, dist=dist)
        field[clean == 0] = 0.0
        return field

    field = run(_selftest_mask())
    print(f'[selftest] 直道: center(60,30)={field[60, 30]:.3f} '
          f'boundary(5,30)={field[5, 30]:.3f}')
    assert field[60, 30] > 0.95, '直道中心应接近 1'
    assert field[5, 30] == 0.0, '道路外应为 0'

    field = run(np.zeros((20, 20), np.uint8))
    assert float(field.sum()) == 0.0
    print('[selftest] 空 mask: OK')
    print('[selftest] PASS')


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


def preclean_mask(fg: np.ndarray, min_width_px: float = MIN_ROAD_WIDTH_PX,
                  smooth_sigma: float = SMOOTH_SIGMA, fill_max_area: int = FILL_MAX_AREA,
                  keep_main: bool = KEEP_MAIN, connectivity: int = CC_CONNECTIVITY,
                  mask: np.ndarray = None, low_cost_classes: tuple = LOW_COST_CLASSES,
                  ignore_index: int = 255, s_low: float = S_LOW,
                  s_other: float = S_OTHER,
                  class_aware: bool = CLASS_AWARE_SMOOTH) -> dict:
    """距离场/骨架共用的预处理: 开运算 -> 类别感知高斯平滑 -> 填小洞 -> 最大连通域

    类别感知平滑 (road 与 class3 接壤时更"大胆", 与其它类接壤时收敛):
      对"软值图"做高斯再阈值 0.5:
        road=1.0, low_cost(3)=s_low(>0.5 => 向3推进), 其它类=s_other(0=不扩张),
        非 road/非 low 的其它类 s_other; ignore=0
    mask 为 None 或 class_aware=False 时退化为二值平滑。

    返回 dict:
      mask        : 清理后的可通行 mask (uint8 0/1)
      removed     : 被开运算剔除
      filled      : 被填的小孔
      removed_cc  : 被"只保留最大连通域"剔除
      expanded_low: 因贴近 class3 而被平滑并入的像素 (bool)
    """
    r = max(1, int(round(min_width_px / 2.0)))
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, ell)
    removed = (fg == 1) & (m == 0)
    expanded_low = np.zeros(m.shape, dtype=bool)

    if smooth_sigma and smooth_sigma > 0:          # 平滑锯齿边界 (类别感知)
        if class_aware and mask is not None:
            road = m > 0
            # 只让 class3 在"紧邻道路的窄带"内起作用: 按到最近道路像素的距离衰减
            dist = cv2.distanceTransform((~road).astype(np.uint8), cv2.DIST_L2, 5)
            R = max(1.0, 3.0 * float(smooth_sigma))
            gate = np.clip(1.0 - (dist - 1.0) / R, 0.0, 1.0).astype(np.float32)
            soft = np.zeros(m.shape, dtype=np.float32)
            soft[road] = 1.0
            low = np.isin(mask, tuple(low_cost_classes)) & (~road)
            soft[low] = float(s_low) * gate[low]
            other = (~road) & (~low) & (mask != ignore_index)
            soft[other] = float(s_other) * gate[other]
            f = cv2.GaussianBlur(soft, (0, 0), float(smooth_sigma))
            smoothed = f > 0.5
            expanded_low = smoothed & low
            m = smoothed.astype(np.uint8)
        else:
            f = cv2.GaussianBlur(m.astype(np.float32), (0, 0), float(smooth_sigma))
            m = (f > 0.5).astype(np.uint8)

    before_fill = m > 0
    if fill_max_area and fill_max_area > 0:        # 只填小孔洞
        after_fill = _fill_small_holes(before_fill, int(fill_max_area))
    else:
        after_fill = before_fill
    filled = after_fill & (~before_fill)

    if keep_main:
        main = _largest_cc(after_fill, connectivity)
    else:
        main = after_fill
    removed_cc = after_fill & (~main)
    return dict(mask=main.astype(np.uint8), removed=removed,
                filled=filled, removed_cc=removed_cc, expanded_low=expanded_low)


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


def _path_mask(shape, path) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for (y, x) in path:
        mask[y, x] = True
    return mask


def ordered_longest_path(skel: np.ndarray):
    """骨架图直径(最长路径) -> 有序点列 [(y,x), ...]。两次 BFS(无权, 8 邻接)。"""
    H, W = skel.shape
    ys, xs = np.where(skel)
    if len(ys) == 0:
        return []
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
    path = []
    cur = c
    while cur is not None:
        path.append(cur)
        cur = par[cur]
    path.reverse()
    return path


def _best_forward_window(v, v_tol: int):
    """最长连续窗口, 满足累计正向 v 增量 <= v_tol"""
    best_s = best_e = 0
    left = 0
    cost = 0
    for right in range(1, len(v)):
        cost += max(0, int(v[right] - v[right - 1]))
        while cost > v_tol and left < right:
            cost -= max(0, int(v[left + 1] - v[left]))
            left += 1
        if right - left > best_e - best_s:
            best_s, best_e = left, right
    return best_s, best_e


def forward_main_path(path, v_tol: int = V_BACK_TOL):
    """把最长路径裁成累计回退 <= v_tol 的最长前向段"""
    if len(path) <= 1:
        return list(path)
    best = (path, 0, 0)
    best_key = (-1, -1)
    for cand in (path, path[::-1]):
        v = [p[0] for p in cand]
        s, e = _best_forward_window(v, v_tol)
        key = (e - s + 1, int(v[s]))
        if key > best_key:
            best_key = key
            best = (cand, s, e)
    cand, s, e = best
    return list(cand[s:e + 1])


def _near_image_border(p, shape, margin: int) -> bool:
    y, x = p
    H, W = shape
    return min(y, x, H - 1 - y, W - 1 - x) <= margin


def _trim_border_end(path, dist, at_front: bool, alpha: float, max_trim: int):
    """把贴边端点收到 dist >= alpha*局部最大 的位置, 去掉朝角落的尾巴"""
    if len(path) < 3:
        return list(path)
    m = min(int(max_trim), len(path) // 2)
    if m < 2:
        return list(path)
    if at_front:
        seg = path[:m]
        ds = [float(dist[p]) for p in seg]
        mx = max(ds)
        if ds[0] >= alpha * mx:
            return list(path)
        k = next((i for i, d in enumerate(ds) if d >= alpha * mx), m - 1)
        return list(path[k:])
    seg = list(reversed(path[-m:]))
    ds = [float(dist[p]) for p in seg]
    mx = max(ds)
    if ds[0] >= alpha * mx:
        return list(path)
    k = next((i for i, d in enumerate(ds) if d >= alpha * mx), m - 1)
    return list(path[:len(path) - k])


def trim_main_border(main, dist, shape,
                     margin: int = BORDER_MARGIN,
                     alpha: float = BORDER_CENTER_ALPHA,
                     max_trim: int = BORDER_MAX_TRIM):
    """只修正主骨架两端: 贴边时收到道路断口中心, 避免朝图像角落生长"""
    if not main:
        return []
    main = list(main)
    if _near_image_border(main[0], shape, margin):
        main = _trim_border_end(main, dist, True, alpha, max_trim)
    if main and _near_image_border(main[-1], shape, margin):
        main = _trim_border_end(main, dist, False, alpha, max_trim)
    return main


def forward_filter(skel: np.ndarray, main, v_tol: int = V_BACK_TOL):
    """从主骨架出发, 只保留前向可达的支路。

    支路第一步必须 v 不增 (禁止从主路直接后退), 之后允许累计回退 <= v_tol。
    返回 (kept_mask, removed_mask, stats)
    """
    H, W = skel.shape
    coords = set(zip(*np.where(skel)))
    empty = (np.zeros(shape=(H, W), dtype=bool),
             np.zeros(shape=(H, W), dtype=bool),
             dict(branch_pix=0, branch_max_back=0))
    if not coords:
        return empty
    main_set = set(main) & coords
    if not main_set:
        return (np.zeros(shape=(H, W), dtype=bool), skel.astype(bool).copy(),
                dict(branch_pix=0, branch_max_back=0))

    best = {p: 0 for p in main_set}
    dq = deque(main_set)
    while dq:
        y, x = dq.popleft()
        c = best[(y, x)]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (y + dy, x + dx)
                if nb not in coords or nb in main_set:
                    continue
                if (y, x) in main_set and nb[0] > y:
                    continue                       # 支路第一步必须前进/水平
                nc = c + max(0, nb[0] - y)
                if nc <= v_tol and nc < best.get(nb, 10 ** 9):
                    best[nb] = nc
                    if nb[0] > y:
                        dq.append(nb)
                    else:
                        dq.appendleft(nb)

    kept = set(best)
    kept_mask = _path_mask((H, W), kept)
    removed_mask = skel.astype(bool) & ~kept_mask
    branch_costs = [c for p, c in best.items() if p not in main_set]
    return kept_mask, removed_mask, dict(
        branch_pix=len(branch_costs),
        branch_max_back=int(max(branch_costs)) if branch_costs else 0)


def _skeleton_graph(mask: np.ndarray):
    """骨架 mask -> (nodes, adj, edges), 8 邻接, 边去重"""
    nodes = set(zip(*np.where(mask)))
    adj = {p: [] for p in nodes}
    for (y, x) in nodes:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (y + dy, x + dx)
                if nb in nodes:
                    adj[(y, x)].append(nb)
    edges = []
    seen = set()
    for p, nbs in adj.items():
        for q in nbs:
            e = (p, q) if p < q else (q, p)
            if e not in seen:
                seen.add(e)
                edges.append(e)
    return nodes, adj, edges


def smooth_skeleton_graph(mask: np.ndarray, iters: int = SKEL_SMOOTH_ITERS,
                          lam: float = SKEL_SMOOTH_LAMBDA):
    """对骨架图做几何平滑: 端点/岔口固定, 度=2 节点做拉普拉斯平滑"""
    nodes, adj, edges = _skeleton_graph(mask)
    pos = {p: np.array([float(p[0]), float(p[1])]) for p in nodes}
    if iters <= 0 or not nodes:
        return pos, edges

    # 只有含端点/岔口的连通块才平滑; 纯环保持原坐标, 避免整体收缩
    comp = {}
    smoothable = set()
    cid = 0
    for start in nodes:
        if start in comp:
            continue
        stack = [start]
        comp[start] = cid
        members = []
        while stack:
            p = stack.pop()
            members.append(p)
            for q in adj[p]:
                if q not in comp:
                    comp[q] = cid
                    stack.append(q)
        if any(len(adj[p]) != 2 for p in members):
            smoothable.update(p for p in members if len(adj[p]) == 2)
        cid += 1
    if not smoothable:
        return pos, edges

    lam = float(np.clip(lam, 0.0, 1.0))
    for _ in range(int(iters)):
        new = {p: v.copy() for p, v in pos.items()}
        for p in smoothable:
            m = 0.5 * (pos[adj[p][0]] + pos[adj[p][1]])
            new[p] = (1.0 - lam) * pos[p] + lam * m
        pos = new
    return pos, edges


def _smoothed_branch_max_back(adj, pos, main_set) -> float:
    """平滑坐标下, 从主路出发的支路最大累计 v 回退 (Dijkstra)"""
    if not main_set:
        return 0.0
    best = {p: 0.0 for p in main_set}
    heap = [(0.0, p) for p in main_set]
    heapq.heapify(heap)
    while heap:
        c, p = heapq.heappop(heap)
        if c != best.get(p):
            continue
        y = pos[p][0]
        for q in adj[p]:
            if q in main_set:
                continue
            nc = c + max(0.0, pos[q][0] - y)
            if nc < best.get(q, 1e18):
                best[q] = nc
                heapq.heappush(heap, (nc, q))
    vals = [c for p, c in best.items() if p not in main_set]
    return max(vals) if vals else 0.0


def build_skeleton(clean_mask: np.ndarray, prune_len: int = PRUNE_LEN,
                   keep_main: bool = True, smooth_iters: int = SKEL_SMOOTH_ITERS) -> dict:
    """由已预处理的可通行掩膜提取道路骨架

    流程: skeletonize -> 剪毛刺 -> [最大骨架连通分量]
          -> 最长路径裁剪成前向主段 -> 主骨架边界收尾 -> 前向支路过滤
    返回 (kept, main, removed_backward, stats)
    """
    clean = clean_mask.astype(bool)
    skel = skeletonize(clean) if clean.any() else np.zeros_like(clean, bool)
    n_raw = int(skel.sum())
    skel = prune_spurs(skel, prune_len)
    if keep_main:
        skel = _largest_cc(skel)
    n_pruned = int(skel.sum())
    dist = cv2.distanceTransform(clean.astype(np.uint8), cv2.DIST_L2, 5)

    path = ordered_longest_path(skel)
    main_full = forward_main_path(path)
    main = trim_main_border(main_full, dist, skel.shape)
    border_trim = len(set(main_full) - set(main))
    if border_trim:
        for p in (set(main_full) - set(main)):
            skel[p] = False

    main_mask = _path_mask(skel.shape, main)
    kept, removed, fstats = forward_filter(skel, main)
    kept = kept.astype(bool)
    removed = removed.astype(bool)

    H, W = kept.shape
    ys, xs = np.where(kept)
    n_end = n_junc = 0
    for y, x in zip(ys.tolist(), xs.tolist()):
        d = len(_nbrs(y, x, H, W, kept))
        n_end += int(d == 1)
        n_junc += int(d >= 3)
    ncc, _ = cv2.connectedComponents(kept.astype(np.uint8), connectivity=8)

    _, adj, _ = _skeleton_graph(kept)
    pos, edges = smooth_skeleton_graph(kept, smooth_iters, SKEL_SMOOTH_LAMBDA)
    main_pts = [(float(pos[p][0]), float(pos[p][1])) for p in main if p in pos]
    main_v = [p[0] for p in main_pts]
    main_back = float(sum(max(0.0, main_v[i] - main_v[i - 1])
                          for i in range(1, len(main_v))))
    main_set = set(main) & set(pos)
    branch_max_back = _smoothed_branch_max_back(adj, pos, main_set)

    stats = dict(
        pix=n_raw, pix_pruned=n_pruned, pix_kept=int(kept.sum()),
        removed_spur=n_raw - n_pruned, removed_backward=int(removed.sum()),
        main_len=len(main),
        main_v0=int(round(main_v[0])) if main_v else 0,
        main_v1=int(round(main_v[-1])) if main_v else 0,
        main_back=main_back,
        border_trim=border_trim,
        branch_pix=fstats['branch_pix'],
        branch_max_back=branch_max_back,
        endpoints=n_end, junctions=n_junc, comps=int(ncc) - 1,
        longest=len(main),
        smooth_iters=int(smooth_iters),
    )
    return dict(kept=kept, main=main_mask, removed=removed,
                pos=pos, edges=edges, main_pts=main_pts, stats=stats)


def render_profile(win_row: np.ndarray, scan_row: int,
                   w: int = 560, h: int = 280) -> np.ndarray:
    """把某一行的距离场值画成曲线图 (cv2 手绘, 不依赖 matplotlib)

    x 轴: 图像列 u; y 轴: 场值 0~1。绿=局部厚度场。
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

    draw(win_row, (0, 255, 0), 2)           # BGR 绿

    # 标题 + 轴标签
    cv2.putText(canvas, f'scanline profile  row = {scan_row}',
                (x0, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'field value', (6, (y0 + y1) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'image col  u  ->', (x0, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)

    # 图例 (右上: 色块 + 文字)
    lx, ly = x1 - 100, 46
    cv2.line(canvas, (lx, ly), (lx + 26, ly), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'thickness', (lx + 32, ly + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def extract_keypoints_lm(field: np.ndarray, win: float = KP_WIN, eps: float = KP_EPS,
                         field_floor: float = KP_FIELD_FLOOR, v_front: float = KP_VFRONT,
                         d_min: float = KP_DMIN, max_pts: int = KP_MAX):
    """从距离场提取"车前关键点": 局部极大 + 强度下限 + 只看车前 + 间隔NMS

    field: 归一化距离场 (局部厚度 = dist/T, 值域[0,1]; 前景 = field > 0)
    返回 (keypoints, candidates):
      keypoints  : [(u, v, field), ...] 按 u 排序(左->右), 最多 max_pts 个
      candidates : [(u, v, field), ...] 车前全部局部极大候选(未 NMS)
    """
    H, W = field.shape
    if field.sum() == 0:
        return [], []
    r = max(1, int(win))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    mx = cv2.dilate(field, kern)
    cand_mask = (field >= mx - eps) & (field > 0)       # 局部极大(含平台)
    ys, xs = np.where(cand_mask)
    cands = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        dv = float(field[y, x])
        if dv < field_floor:
            continue
        u = x / (W - 1)
        v = y / (H - 1)
        if v > v_front:                                 # 只看车前 (v=0 顶部=车前)
            continue
        cands.append((u, v, dv))

    cands.sort(key=lambda t: -t[2])                     # 按脊线强度降序
    kept = []
    for (u, v, dv) in cands:                            # 间隔 NMS
        if all((u - ku) ** 2 + (v - kv) ** 2 >= d_min * d_min for (ku, kv, _) in kept):
            kept.append((u, v, dv))
            if len(kept) >= max_pts:
                break
    kept.sort(key=lambda t: t[0])                       # 按 u 左->右
    return kept, cands


def render(mask, road_mask, win_f, skeleton, token, min_w, thick_max,
           scan_row, kp_pts=None, kp_cands=None, v_front=KP_VFRONT):
    seg = colorize_mask(mask, 255)
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
              ('dist field (thickness)', win_vis),
              ('skeleton  main=red  branch=green', skel_vis),
              ('keypoints  front local-max + NMS', kp_vis)]
    # 扫描行标记
    cv2.line(panels[2][1], (0, scan_row), (panels[2][1].shape[1] - 1, scan_row),
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
    cv2.putText(header, f'token={token}  field=thickness  smooth={skeleton["stats"].get("smooth_iters", 0)}x'
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
    parser.add_argument('--kp-win', type=float, default=KP_WIN)
    parser.add_argument('--kp-dmin', type=float, default=KP_DMIN)
    parser.add_argument('--kp-max', type=int, default=KP_MAX)
    parser.add_argument('--kp-field-floor', type=float, default=KP_FIELD_FLOOR)
    parser.add_argument('--kp-vfront', type=float, default=KP_VFRONT)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    min_w, thick_max = args.min_width, args.thickness_max_r
    keep_main, cc_conn = args.keep_main, args.cc_connectivity
    smooth_sigma, prune_len, fill_max_area = args.smooth, args.prune, args.fill_max_area
    s_low, s_other, class_aware = args.s_low, args.s_other, args.class_aware
    smooth_iters = args.smooth_iters
    kp = dict(win=args.kp_win, d_min=args.kp_dmin, max_pts=args.kp_max,
              field_floor=args.kp_field_floor, v_front=args.kp_vfront)

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
    print(f'  距离场: 局部厚度归一化  thickness_max_r={thick_max}(0=auto)')

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
        win_f = build_field(pre['mask'], thick_max, dist=dist)
        win_f[pre['mask'] == 0] = 0.0
        kp_pts, kp_cands = extract_keypoints_lm(win_f, **kp)
        st = field_stats(win_f, pre['mask'])
        print(f'[样本 {i}] token={path.stem}  normal={int(pre["mask"].sum())}px  '
              f'merged_low={int(pre["expanded_low"].sum())}px  '
              f'filled={int(pre["filled"].sum())}px  removed={int(pre["removed"].sum())}px  '
              f'cc_removed={int(pre["removed_cc"].sum())}px  keep_main={keep_main} conn={cc_conn}  '
              f'field: plateau={st["plateau"]*100:.1f}% max={st["max"]:.2f}  |  '
              f'skel: pix={sk["pix"]}->{sk["pix_pruned"]}->{sk["pix_kept"]} '
              f'smooth={sk["smooth_iters"]}x{SKEL_SMOOTH_LAMBDA} '
              f'main={sk["main_len"]} (v{sk["main_v0"]}->{sk["main_v1"]}, back={sk["main_back"]:.2f}, '
              f'border_trim={sk["border_trim"]}) '
              f'branch={sk["branch_pix"]} (max_back={sk["branch_max_back"]:.2f}) '
              f'removed_backward={sk["removed_backward"]} '
              f'end={sk["endpoints"]} junc={sk["junctions"]} comps={sk["comps"]}')
        if sk['main_back'] > V_BACK_TOL or sk['branch_max_back'] > V_BACK_TOL:
            print(f'    [warn] smooth v_back main={sk["main_back"]:.2f} '
                  f'branch={sk["branch_max_back"]:.2f} > tol={V_BACK_TOL}')
        print(f'    车前关键点: 候选={len(kp_cands)} 入选={len(kp_pts)}  '
               + ' '.join(f'({u:.2f},{v:.2f},f={d:.2f})' for (u, v, d) in kp_pts))
        return dict(mask=mask, pre=pre, win_f=win_f, skeleton=skel_result,
                    kp_pts=kp_pts, kp_cands=kp_cands,
                    token=path.stem, h=win_f.shape[0])

    def show(s):
        cv2.imshow(win, render(s['mask'], s['pre']['mask'], s['win_f'],
                               s['skeleton'], s['token'], min_w, thick_max,
                               scan_row, s['kp_pts'], s['kp_cands'], kp['v_front']))
        cv2.imshow(win_prof, render_profile(s['win_f'][scan_row], scan_row))

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
