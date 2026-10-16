"""model_slim2 自监督损失辅助函数: 采样 + 光滑/排斥/长度/等距/展布

所有点均为 (u,v) 归一化坐标 [0,1] (u→宽, v→高), 与 MapTRHead 输出约定一致。

每个损失都拆成"未聚合值"与"聚合函数"两部分:
    - per-query (B, N): cost/interior/smooth/spacing/arc/span  每条线一个值
    - per-pair  (B, N, N): repel/spread                    每两条线一个值
    - *_loss(...) 聚合函数供训练使用; *_per_* 供调试/可视化逐项查看
"""
import torch
import torch.nn.functional as F


def lp_reduce(x: torch.Tensor, dim: int = -1, p: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    """非负张量的 Lp 范数聚合: (mean(x^p) + eps)^(1/p)

    p>1 放大该维度上的大值(如最尖折角/最贴边界的点), p=1 退化为均值。
    根号内加 eps 避免 x 全 0 时 ^(1/p) 在 0 处梯度无穷(否则线一旦变直/变均匀就出 nan)。
    仅用于线内逐点/逐段维度, 避免均值把"难点"平均掉。
    """
    if p == 1.0:
        return x.mean(dim=dim)
    return (x.clamp_min(0.0).pow(p).mean(dim=dim) + eps).pow(1.0 / p)


def sample_map(map_2d: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """在 (B, 1, H, W) 地图上采样归一化坐标点

    用于采样代价图 (越低越好) 和距离场 (越大越好, interior 用 1 - dist 转成惩罚)。

    points: (B, N, P, 2), 归一化 [0,1], [..., 0]=u(宽), [..., 1]=v(高)
    Returns: (B, N, P) 采样值
    """
    B = points.shape[0]
    grid = points * 2.0 - 1.0                     # [0,1] → [-1,1] (grid_sample 约定)
    grid = grid.view(B, -1, 1, 2)                 # (B, N*P, 1, 2)
    out = F.grid_sample(map_2d, grid, mode='bilinear',
                        padding_mode='border', align_corners=False)
    return out.view(B, points.shape[1], points.shape[2])


def cosine_sim(flat: torch.Tensor) -> torch.Tensor:
    """batch 余弦相似度矩阵: (B, N, D) → (B, N, N)"""
    norm = flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x = flat / norm
    return x @ x.transpose(1, 2)


def _off_diag_mask(n: int, device) -> torch.Tensor:
    """(N, N) 布尔掩码, 对角线为 True"""
    return torch.eye(n, dtype=torch.bool, device=device)


# ─────────────────────────── per-query 未聚合值 (B, N) ───────────────────────────

def cost_per_query(cost: torch.Tensor, points: torch.Tensor, p: float = 4.0) -> torch.Tensor:
    """每条线的采样代价 Lp 范数 (B, N), 越低越好

    Lp(线内点): p>1 放大高代价点, 强调"越过障碍的坏点"; p=1 退化为均值。
    - p=2: L2, 比均值更看重高点
    - p=4~8: 强烈放大高代价点, 同时保留其余点信息
    """
    score = sample_map(cost, points)                  # (B, N, T)
    return lp_reduce(score, dim=-1, p=p)


def interior_per_query(dist_field: torch.Tensor, points: torch.Tensor, p: float = 4.0) -> torch.Tensor:
    """每条线的居中惩罚 (B, N) = Lp(1 - 距离场), 越低越好(越居中)

    Lp 放大"最贴边界"的点, 促使整条线都远离边界, 而非只在平均意义上居中。
    """
    return lp_reduce(1.0 - sample_map(dist_field, points), dim=-1, p=p)


def smooth_per_query(points: torch.Tensor, img_hw=None, min_step_px: float = 2.0,
                     p: float = 4.0, norm_eps: float = 0.1) -> torch.Tensor:
    """形状正则 (B, N): 转向(防打结) + 最小步长(防点挤堆/稳定方向), 越低越平滑

    points: (B, N, P, 2) 归一化 (u, v), u→宽, v→高
    img_hw: (H, W); 提供时换算到各向同性像素坐标再算夹角(避免宽高比扭曲角度)
    min_step_px: 相邻段最小步长(像素), 段更短时惩罚
    p: 线内 Lp 聚合指数 (>1 强调最尖折角 / 最短段)
    norm_eps: 段长平滑下界(像素), 用 sqrt(|v|^2+norm_eps^2) 代替 |v|, 使 |v|→0 时梯度有界
    """
    pts = points
    if img_hw is not None:
        h, w = img_hw
        pts = points * points.new_tensor([w, h])       # (u,v) -> [w,h]

    v = pts[:, :, 1:] - pts[:, :, :-1]                 # (B,N,P-1,2) 段向量
    s = torch.sqrt(v.square().sum(dim=-1) + norm_eps * norm_eps)  # 平滑段长, |v|→0 梯度有界
    u = v / s.unsqueeze(-1)                            # 软单位方向, |u|<=1
    cos = (u[:, :, :-1] * u[:, :, 1:]).sum(dim=-1)     # (B,N,P-2) 相邻段夹角余弦

    turn = lp_reduce(1.0 - cos, dim=-1, p=p)           # 转向: 折角越大越罚(掉头=2)
    short = lp_reduce(torch.relu(min_step_px - s), dim=-1, p=p)  # 短边: 段过短越罚
    return turn + short


def spacing_per_query(points: torch.Tensor, img_hw=None, p: float = 4.0,
                      norm_eps: float = 0.1, eps: float = 1e-6) -> torch.Tensor:
    """等距惩罚 (B, N): 段长偏离均值的程度, 尺度无关, 越低越等距

    对每段取 |s_i - mean(s)| / mean(s), 再做线内 Lp 聚合(强调最不均匀的一段)。
    仅约束相对均匀; 绝对尺度由 arc 的弧长下界负责。
    mean(s) 用 detach 的尺度做分母, 使梯度有界; s 用平滑段长防 0 处异常。
    """
    pts = points
    if img_hw is not None:
        h, w = img_hw
        pts = points * points.new_tensor([w, h])
    v = pts[:, :, 1:] - pts[:, :, :-1]
    s = torch.sqrt(v.square().sum(dim=-1) + norm_eps * norm_eps)  # (B,N,P-1) 平滑段长
    s_mean = s.mean(dim=-1, keepdim=True).clamp_min(eps).detach()
    dev = (s - s_mean).abs() / s_mean                  # (B,N,P-1) 相对偏差
    return lp_reduce(dev, dim=-1, p=p)


def arc_per_query(points: torch.Tensor, min_arc: float) -> torch.Tensor:
    """每条线弧长(总路径长度)不足 min_arc 的惩罚 (B, N)"""
    d = (points[:, :, 1:] - points[:, :, :-1]).norm(dim=-1).sum(dim=-1)
    return torch.relu(min_arc - d)


def span_per_query(points: torch.Tensor, min_span: float) -> torch.Tensor:
    """每条线首尾端点距不足 min_span 的惩罚 (B, N)

    端点距 = ‖p_{P-1} - p_0‖ 是"净展布", 无法被原地振荡伪造, 与弧长(arc_per_query)互补:
    弧长管真实长度, span 管张开(防紧凑团/环)。梯度=单位向量, 无除法, 数值安全。
    """
    d = (points[:, :, -1] - points[:, :, 0]).norm(dim=-1)
    return torch.relu(min_span - d)


# ─────────────────────────── per-pair 未聚合值 (B, N, N) ───────────────────────────

def point_to_polyline_dist(points: torch.Tensor) -> torch.Tensor:
    """点到折线距离: 第 i 条线每个点 到 第 j 条线折线(线段) 的最近距离

    points: (B, N, P, 2) 归一化 (u,v)
    Returns: (B, N, N, P)  [b, i, j, k] = 第 i 条线第 k 个点 到 第 j 条线折线的距离

    相比"点到离散点", 点到线段与采样密度无关, 一长一短重合时也能稳定测出重合。
    """
    B, N, P, _ = points.shape
    p = points.unsqueeze(2).unsqueeze(4)              # (B, N, 1, P, 1, 2)   点
    a = points[:, None, :, None, :-1, :]              # (B, 1, N, 1, P-1, 2) 线段起点
    b = points[:, None, :, None, 1:, :]               # (B, 1, N, 1, P-1, 2) 线段终点
    ab = b - a                                        # (B, 1, N, 1, P-1, 2)
    ap = p - a                                        # (B, N, N, P, P-1, 2)
    t = (ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-6)   # (B, N, N, P, P-1)
    t = t.clamp(0.0, 1.0)                             # 投影参数裁剪到线段内
    closest = a + t.unsqueeze(-1) * ab                # 线段上最近点
    d = (p - closest).norm(dim=-1)                    # (B, N, N, P, P-1)
    return d.min(dim=-1)[0]                           # (B, N, N, P)


def repel_per_pair(points: torch.Tensor, margin: float) -> torch.Tensor:
    """线对排斥明细 (B, N, N): 第 i 条线各点落到第 j 条线折线 margin 内的平均惩罚

    对角 (i==j) 置 0。
    """
    B, N, P, _ = points.shape
    if N < 2:
        return points.new_zeros(B, N, N)
    d = point_to_polyline_dist(points)                # (B, N, N, P)
    penalty = torch.relu(margin - d).mean(dim=-1)     # (B, N, N) 对点取均值
    return penalty.masked_fill(_off_diag_mask(N, points.device)[None], 0.0)


def spread_per_pair(points: torch.Tensor, thr: float) -> torch.Tensor:
    """线对形状雷同明细 (B, N, N): 余弦相似度超过 thr 的部分, 对角置 0"""
    B, N, P, _ = points.shape
    if N < 2:
        return points.new_zeros(B, N, N)
    flat = points.reshape(B, N, -1)
    flat = flat - flat.mean(dim=-1, keepdim=True)
    sim = cosine_sim(flat)                            # (B, N, N)
    sim = sim.masked_fill(_off_diag_mask(N, points.device)[None], 0.0)
    return torch.relu(sim - thr)


def start_gather_per_pair(points: torch.Tensor) -> torch.Tensor:
    """低端聚合明细 (B, N, N): 第 i / j 条线"起点 p0"之间的距离, 对角置 0

    越小越聚合 -> 多条线共享"车后"共同起点 (下端可重合)。
    """
    B, N, P, _ = points.shape
    if N < 2:
        return points.new_zeros(B, N, N)
    p0 = points[:, :, 0, :]                               # (B, N, 2)
    d = (p0[:, :, None, :] - p0[:, None, :, :]).norm(dim=-1)   # (B, N, N)
    return d.masked_fill(_off_diag_mask(N, points.device)[None], 0.0)


def diverge_per_pair(points: torch.Tensor, margin: float) -> torch.Tensor:
    """顶端发散明细 (B, N, N): 第 i 条线"终点" 到 第 j 条线"所有点"的最小距离 < margin 的 hinge, 对角置 0

    只取终点(顶端), 与其它线的全部点(含中段采样点)取最小距离 -> 防止终点压到别的线上/两线顶端重合。
    底端不参与 -> 仍允许低端聚合(共享车后起点)。越小越分散(不同岔路)。
    """
    B, N, P, _ = points.shape
    if N < 2:
        return points.new_zeros(B, N, N)
    pl = points[:, :, -1, :]                                  # (B, N, 2) 每条线终点
    diff = pl[:, :, None, None, :] - points[:, None, :, :, :]  # (B, N, N, P, 2)
    d_min = diff.norm(dim=-1).min(dim=-1)[0]                  # (B, N, N) 终点 i 到线 j 所有点的最小距离
    return torch.relu(margin - d_min).masked_fill(_off_diag_mask(N, points.device)[None], 0.0)


# ─────────────────────────── 聚合函数 (训练使用, 数值与原实现一致) ───────────────────────────

def cost_loss(cost: torch.Tensor, points: torch.Tensor, p: float = 4.0) -> torch.Tensor:
    """采样代价 Lp 范数均值, 点落在低代价(可通行)区"""
    return cost_per_query(cost, points, p).mean()


def interior_loss(dist_field: torch.Tensor, points: torch.Tensor, p: float = 4.0) -> torch.Tensor:
    """居中惩罚: Lp(1 - 距离场) 均值, 越小线越靠区域中心"""
    return interior_per_query(dist_field, points, p).mean()


def smooth_loss(points: torch.Tensor, img_hw=None, min_step_px: float = 2.0,
                p: float = 4.0) -> torch.Tensor:
    """形状正则: 转向(防打结) + 最小步长(防挤堆), 线内 Lp 聚合"""
    return smooth_per_query(points, img_hw, min_step_px, p).mean()


def spacing_loss(points: torch.Tensor, img_hw=None, p: float = 4.0) -> torch.Tensor:
    """等距正则: 段长相对偏差的线内 Lp 聚合, 尺度无关"""
    return spacing_per_query(points, img_hw, p).mean()


def repel_loss(points: torch.Tensor, margin: float) -> torch.Tensor:
    """线间排斥, 防塌缩: 每个点到另一条线折线的距离 < margin 时 hinge (归一化)

    距离用"点到折线"(而非点到离散点), 聚合用所有越界点的均值(而非 min),
    因此无论线多长/采样多稀, 只要两线重合就产生稠密梯度。
    """
    B, N, _, _ = points.shape
    if N < 2:
        return points.new_zeros(())
    return repel_per_pair(points, margin).sum() / (B * N * (N - 1))


def arc_loss(points: torch.Tensor, min_arc: float) -> torch.Tensor:
    """弧长下界, 防线缩点: 每条线弧长 < min_arc 时惩罚 (归一化)"""
    return arc_per_query(points, min_arc).mean()


def span_loss(points: torch.Tensor, min_span: float) -> torch.Tensor:
    """端点距下界, 逼线张开: 每条线首尾端点距 < min_span 时惩罚 (归一化)"""
    return span_per_query(points, min_span).mean()


def spread_loss(points: torch.Tensor, thr: float) -> torch.Tensor:
    """输出多样性, 防雷同: query 两两输出余弦相似度 > thr 时惩罚

    与 repel 互补: repel 约束空间距离, spread 约束输出形状差异。
    """
    B, N, _, _ = points.shape
    if N < 2:
        return points.new_zeros(())
    return spread_per_pair(points, thr).mean()


def start_gather_loss(points: torch.Tensor) -> torch.Tensor:
    """低端聚合: 各线起点 p0 两两距离 (趋 0 -> 共享车后起点)"""
    B, N, _, _ = points.shape
    if N < 2:
        return points.new_zeros(())
    return start_gather_per_pair(points).sum() / (B * N * (N - 1))


def diverge_loss(points: torch.Tensor, margin: float) -> torch.Tensor:
    """顶端发散: 各线终点 p_last 距离 < margin 的 hinge (逼岔路终点分开)"""
    B, N, _, _ = points.shape
    if N < 2:
        return points.new_zeros(())
    return diverge_per_pair(points, margin).sum() / (B * N * (N - 1))


# ─────────────────────────── 关键点覆盖率 ───────────────────────────

def keypoint_dist_per_kp(points: torch.Tensor, kps: torch.Tensor) -> torch.Tensor:
    """每个关键点到"全部预测点"的最近点距 (纯点到点)

    points: (B, N, P, 2) 归一化 (u,v) 预测点
    kps:    (B, K, 2) 归一化关键点 (补 0 占位由 valid 掩码过滤)
    Returns: (B, K)  [b, k] = min_{n,p} ‖kp_k - p_{n,p}‖
    """
    B, N, P, _ = points.shape
    pf = points.reshape(B, N * P, 2)                       # (B, N*P, 2)
    d = (kps.unsqueeze(2) - pf.unsqueeze(1)).norm(dim=-1)  # (B, K, N*P)
    return d.min(dim=-1)[0]                                # (B, K)


def keypoint_coverage_per_kp(points: torch.Tensor, kps: torch.Tensor,
                             thr: float = 0.05) -> torch.Tensor:
    """未覆盖惩罚明细 (B, K) = relu(d - thr)

    关键点到最近预测点距离 d <= thr 视为已覆盖 -> 0; 超出部分线性惩罚。
    最小化它等价于把所有关键点拉进阈值内 (最大化覆盖率)。
    """
    return torch.relu(keypoint_dist_per_kp(points, kps) - thr)


def keypoint_coverage_loss(points: torch.Tensor, kps: torch.Tensor, thr: float = 0.05,
                           valid: torch.Tensor = None) -> torch.Tensor:
    """关键点覆盖率损失 (标量): 有效关键点上 relu(d - thr) 的均值

    valid: (B, K) 1=真实关键点, 0=补 0 占位; 仅对真实关键点求均值。
    """
    pen = keypoint_coverage_per_kp(points, kps, thr)       # (B, K)
    if valid is None:
        return pen.mean()
    v = valid.float()
    return (pen * v).sum() / v.sum().clamp_min(1.0)


def keypoint_coverage_rate(points: torch.Tensor, kps: torch.Tensor, thr: float = 0.05,
                           valid: torch.Tensor = None) -> torch.Tensor:
    """覆盖率指标 (标量): 有效关键点中 d <= thr 的比例 (仅供监控, 不可导)"""
    d = keypoint_dist_per_kp(points, kps)                  # (B, K)
    covered = (d <= thr).float()
    if valid is None:
        return covered.mean()
    v = valid.float()
    return (covered * v).sum() / v.sum().clamp_min(1.0)


def loss_shares(loss_dict: dict) -> dict:
    """各损失项占总损失的百分比 (所有项非负, 占比之和=1)

    用于诊断"哪一项在主导", 解决不同项量纲/正负不可比的问题。
    """
    total = sum(v for k, v in loss_dict.items() if k != 'total')
    if total <= 0:
        return {k: 0.0 for k in loss_dict if k != 'total'}
    return {k: float(v / total) for k, v in loss_dict.items() if k != 'total'}
