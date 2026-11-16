"""标签图 -> 预处理 / 骨架 / 距离场 / 关键点 (vis 与 dataset 共用)

纯几何逻辑, 不依赖 config / GUI / CLI。
"""
import heapq
from collections import deque

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

# ── 默认参数 ──
MIN_ROAD_WIDTH_PX = 6      # 开运算圆盘直径阈值(px): 路宽 < 该值 -> 剔除
SMOOTH_SIGMA = 2.0         # 边界高斯平滑 sigma(px); 0=不平滑
FILL_MAX_AREA = 12         # 只填“完全被道路包围且面积<=该值”的小孔(px)
KEEP_MAIN = True           # True=只保留最大可通行连通域
CC_CONNECTIVITY = 8        # 连通域连通性: 4 或 8

LOW_COST_CLASSES = (3, 255)  # 低代价类 id; 255 是 ignore, 当前也按低代价参与平滑
S_LOW = 0.8                # low-cost 类软值 [0,1]; >0.5 向该类扩张
S_OTHER = 0.6              # 其它类软值 [0,1]; 0=不扩张; <0=回缩
CLASS_AWARE_SMOOTH = True  # True=类别感知平滑; False=二值平滑

THICKNESS_MAX_R = 0        # 局部厚度最大半径; 0=自动=floor(dist.max())
THICKNESS_MIN_R = 1        # 局部厚度最小探测半径
PRUNE_LEN = 30             # 端点毛刺长度阈值(px); 短于该值剪除
V_BACK_TOL = 2             # 前向约束: 允许累计回退(v增大)<=该值(px)
BORDER_MARGIN = 15         # 主骨架端点距图像边界<=该值才做边界收尾(px)
BORDER_CENTER_ALPHA = 0.8  # 边界收尾目标: dist >= alpha*局部最大
BORDER_MAX_TRIM = 60       # 边界收尾最多回看/删除的像素数
SKEL_SMOOTH_ITERS = 10     # 拉普拉斯平滑迭代次数; 0=关闭
SKEL_SMOOTH_LAMBDA = 0.5   # 每次平滑步长 [0,1]; 越大越强

KP_SPACING = 0.1           # 骨架 FPS 覆盖半径/关键点间距 (归一化 uv)
KP_VFRONT = 0.9            # 只看车前: v<=该值; v=0 为顶部
KP_MAX = 100               # 最多关键点数; 也是 padding 长度


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
    nodes, adj, _ = _skeleton_graph(skel)
    if not nodes:
        return []
    node_list = list(nodes)
    index = {p: i for i, p in enumerate(node_list)}
    n = len(node_list)

    def bfs(start: int):
        dist = np.full(n, -1, np.int32)
        par = np.full(n, -1, np.int32)
        dist[start] = 0
        q = deque([start])
        far = start
        while q:
            u = q.popleft()
            for nb in adj[node_list[u]]:
                v = index[nb]
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
                    if dist[v] > dist[far]:
                        far = v
        return far, par

    b, _ = bfs(0)
    c, par = bfs(b)
    path = []
    cur = c
    while cur >= 0:
        path.append(node_list[cur])
        cur = int(par[cur])
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
    if not nodes:
        return {}, edges
    node_list = list(nodes)
    index = {p: i for i, p in enumerate(node_list)}
    n = len(node_list)
    pos = np.array([[float(p[0]), float(p[1])] for p in node_list], dtype=np.float32)
    if iters <= 0:
        return {p: pos[index[p]] for p in node_list}, edges

    # 只有含端点/岔口的连通块才平滑; 纯环保持原坐标, 避免整体收缩
    comp = {}
    smooth_idx = []
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
            smooth_idx.extend(index[p] for p in members if len(adj[p]) == 2)
        cid += 1
    if not smooth_idx:
        return {p: pos[index[p]] for p in node_list}, edges

    smooth_idx = np.asarray(smooth_idx, dtype=np.int32)
    n1 = np.zeros(n, dtype=np.int32)
    n2 = np.zeros(n, dtype=np.int32)
    for p in node_list:
        i = index[p]
        if len(adj[p]) == 2:
            n1[i] = index[adj[p][0]]
            n2[i] = index[adj[p][1]]

    lam = float(np.clip(lam, 0.0, 1.0))
    for _ in range(int(iters)):
        pos[smooth_idx] = ((1.0 - lam) * pos[smooth_idx]
                           + lam * 0.5 * (pos[n1[smooth_idx]] + pos[n2[smooth_idx]]))
    return {p: pos[index[p]] for p in node_list}, edges


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
                   keep_main: bool = True, smooth_iters: int = SKEL_SMOOTH_ITERS,
                   with_stats: bool = True) -> dict:
    """由已预处理的可通行掩膜提取道路骨架

    流程: skeletonize -> 剪毛刺 -> [最大骨架连通分量]
          -> 最长路径裁剪成前向主段 -> 主骨架边界收尾 -> 前向支路过滤
    with_stats=False 时只返回 pos/edges, 供 dataset 省掉统计开销。
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

    kept, removed, fstats = forward_filter(skel, main)
    kept = kept.astype(bool)
    removed = removed.astype(bool)
    pos, edges = smooth_skeleton_graph(kept, smooth_iters, SKEL_SMOOTH_LAMBDA)

    if not with_stats:
        return dict(pos=pos, edges=edges)

    main_mask = _path_mask(skel.shape, main)
    H, W = kept.shape
    ys, xs = np.where(kept)
    n_end = n_junc = 0
    for y, x in zip(ys.tolist(), xs.tolist()):
        d = len(_nbrs(y, x, H, W, kept))
        n_end += int(d == 1)
        n_junc += int(d >= 3)
    ncc, _ = cv2.connectedComponents(kept.astype(np.uint8), connectivity=8)

    _, adj, _ = _skeleton_graph(kept)
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


def skeleton_guided_field(clean_mask: np.ndarray, dist: np.ndarray,
                          base_field: np.ndarray, skeleton_result: dict,
                          sigma: float = 3.0):
    """整条骨架引导的距离场修正

    W(p)  = 原始骨架节点处的 dist
    f_sk  = clip(dist / W, 0, 1)
    gate  = exp(-(d_skel / sigma)^2)
    field = (1-gate)*base + gate*f_sk

    W 用原始节点 (窄颈处平滑坐标可能偏 1px); gate 用平滑后的骨架折线。
    返回 (field, info)
    """
    clean = clean_mask.astype(bool)
    base = base_field.astype(np.float32)
    if not clean.any():
        return np.zeros(clean.shape, np.float32), dict(
            skel_field=0.0, low_base_field=0.0, low_base_n=0, far_diff=0.0,
            skel_px=0, region_px=0)

    pos = skeleton_result.get('pos', {})
    edges = skeleton_result.get('edges', [])
    nodes = list(pos.keys())
    if not nodes:
        return base.copy(), dict(
            skel_field=0.0, low_base_field=0.0, low_base_n=0, far_diff=0.0,
            skel_px=0, region_px=0)

    # W: 原始骨架节点处的局部半宽
    # 用 EDT 最近节点索引代替 cKDTree 查询 (40k 像素上快约 6 倍)
    H, W = clean.shape
    nodes_arr = np.asarray(nodes, dtype=np.int32)
    node_mask = np.zeros((H, W), dtype=bool)
    W_img = np.zeros((H, W), dtype=np.float32)
    yy = nodes_arr[:, 0]
    xx = nodes_arr[:, 1]
    node_mask[yy, xx] = True
    W_img[yy, xx] = np.maximum(dist[yy, xx], 1e-6)
    _, near = ndi.distance_transform_edt(~node_mask, return_indices=True)
    Wpx = W_img[near[0], near[1]]
    f_sk = np.clip(dist / np.maximum(Wpx, 1e-6), 0.0, 1.0).astype(np.float32)

    # 平滑骨架折线 -> 门控距离
    sk = np.zeros((H, W), np.uint8)
    for a, b in edges:
        ya, xa = pos[a]
        yb, xb = pos[b]
        cv2.line(sk,
                 (int(round(xa)), int(round(ya))),
                 (int(round(xb)), int(round(yb))),
                 1, 1)
    if not sk.any():
        for p in nodes:
            y = int(np.clip(round(pos[p][0]), 0, H - 1))
            x = int(np.clip(round(pos[p][1]), 0, W - 1))
            sk[y, x] = 1
    if not sk.any():
        return base.copy(), dict(
            skel_field=0.0, low_base_field=0.0, low_base_n=0, far_diff=0.0,
            skel_px=0, region_px=0)

    sk_bool = sk.astype(bool)
    d_skel = cv2.distanceTransform((~sk_bool).astype(np.uint8), cv2.DIST_L2, 5)
    gate = np.exp(-(d_skel / max(float(sigma), 1e-6)) ** 2).astype(np.float32)
    field = (1.0 - gate) * base + gate * f_sk
    field[~clean] = 0.0
    field = field.astype(np.float32)

    low_base = sk_bool & (base < 0.9)
    low_base_n = int(low_base.sum())
    low_base_field = float(field[low_base].mean()) if low_base_n else 1.0
    far = (d_skel > 5.0) & clean
    far_diff = float(np.abs(field[far] - base[far]).mean()) if far.any() else 0.0
    info = dict(
        skel_field=float(field[sk_bool].mean()),
        low_base_field=low_base_field,
        low_base_n=low_base_n,
        far_diff=far_diff,
        skel_px=int(sk_bool.sum()),
        region_px=int((gate > 0.01).sum()),
    )
    return field, info






def extract_keypoints_from_skeleton(skeleton_result: dict, shape,
                                    spacing: float = KP_SPACING,
                                    v_front: float = KP_VFRONT,
                                    max_pts: int = KP_MAX):
    """骨架节点全局 NMS: 任意两个关键点距离 >= spacing.

    - 取所有平滑骨架节点
    - 过滤 v > v_front
    - 按 (v, u) 顺序贪心 NMS
    - 最多 max_pts 个
    """
    pos = skeleton_result.get('pos', {})
    max_pts = int(max_pts)
    spacing = float(spacing)
    if not pos or max_pts <= 0 or spacing <= 0:
        return []
    H, W = shape

    uv = []
    for node, (cy, cx) in pos.items():
        v = float(cy) / (H - 1)
        u = float(cx) / (W - 1)
        if v <= v_front:
            uv.append((u, v, 1.0))
    if not uv:
        return []

    pts = np.asarray(uv, dtype=np.float32)
    order = np.lexsort((pts[:, 0], pts[:, 1]))   # 主键 v, 次键 u
    kept = []
    d2 = spacing * spacing
    for i in order:
        u, v = float(pts[i, 0]), float(pts[i, 1])
        if all((u - ku) ** 2 + (v - kv) ** 2 >= d2 for ku, kv, _ in kept):
            kept.append((u, v, 1.0))
            if len(kept) >= max_pts:
                break
    kept.sort(key=lambda t: t[0])
    return kept


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

    # 骨架引导场: 合成窄颈
    neck = _selftest_mask()
    for y in range(50, 60):
        neck[y, :] = 0
        neck[y, 28:33] = 1
    skel_r = build_skeleton(neck, PRUNE_LEN, True, SKEL_SMOOTH_ITERS)
    dist = cv2.distanceTransform(neck.astype(np.uint8), cv2.DIST_L2, 5)
    base = build_field(neck, THICKNESS_MAX_R, dist=dist)
    base[neck == 0] = 0.0
    guided, _ = skeleton_guided_field(neck, dist, base, skel_r, 3.0)
    print(f'[selftest] 窄颈: base(50,30)={base[50, 30]:.3f} '
          f'guided(50,30)={guided[50, 30]:.3f} '
          f'guided(55,30)={guided[55, 30]:.3f} '
          f'guided(60,30)={guided[60, 30]:.3f}')
    assert base[50, 30] < 0.5, '合成颈入口 base 应偏低'
    assert guided[50, 30] > 0.95, '修正后窄颈入口应接近 1'
    assert guided[55, 30] > 0.95, '修正后窄颈中心应接近 1'
    assert guided[60, 30] > 0.95, '修正后窄颈出口应接近 1'
    assert guided[5, 30] == 0.0, '道路外应为 0'
    print('[selftest] PASS')
