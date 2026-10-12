"""model_slim2 自监督损失辅助函数: 采样 + 光滑/排斥/长度/多样性/单调

所有点均为 (u,v) 归一化坐标 [0,1] (u→宽, v→高), 与 MapTRHead 输出约定一致。

每个损失都拆成"未聚合值"与"聚合函数"两部分:
    - per-query (B, N): cost/interior/smooth/length/order/grow  每条线一个值
    - per-pair  (B, N, N): repel/spread                    每两条线一个值
    - *_loss(...) 聚合函数供训练使用; *_per_* 供调试/可视化逐项查看
"""
import torch
import torch.nn.functional as F

SPAN_MAX = 2.0 ** 0.5     # 单位正方形内两点最大距离(对角线), 用于归一化端点距


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

def cost_per_query(cost: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """每条线的平均采样代价 (B, N), 越低越好"""
    return sample_map(cost, points).mean(dim=-1)


def interior_per_query(dist_field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """每条线的居中惩罚 (B, N) = 1 - 平均距离场, 越低越好(越居中)"""
    return 1.0 - sample_map(dist_field, points).mean(dim=-1)


def smooth_per_query(points: torch.Tensor) -> torch.Tensor:
    """每条线的二阶差分模长 (B, N), 越小越平滑"""
    second = points[:, :, 2:] - 2.0 * points[:, :, 1:-1] + points[:, :, :-2]
    return second.norm(dim=-1).mean(dim=-1)


def length_per_query(points: torch.Tensor, min_length: float) -> torch.Tensor:
    """每条线端点距不足 min_length 的惩罚 (B, N)"""
    d = (points[:, :, -1] - points[:, :, 0]).norm(dim=-1)
    return torch.relu(min_length - d)


def order_per_query(points: torch.Tensor) -> torch.Tensor:
    """每条线 u 增量为负的惩罚 (B, N)"""
    du = points[:, :, 1:, 0] - points[:, :, :-1, 0]
    return torch.relu(-du).mean(dim=-1)


def grow_per_query(cost: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """可通行区加权的端点距惩罚 (B, N) = 1 - span×weight/SPAN_MAX, 越低越好(越长)

    用端点距(span)而非弧长: 直线横跨走廊时 span 最大, 来回摆动反而减小 span,
    因此没有"靠摆动骗奖励"的退化空间。权重 = 两端点处 (1 - cost) 的均值。
    归一化到 [0,1] 使该项与其它惩罚同号同量级。
    """
    span = (points[:, :, -1] - points[:, :, 0]).norm(dim=-1)          # (B, N)
    ends = torch.stack([points[:, :, 0], points[:, :, -1]], dim=2)    # (B, N, 2, 2)
    weight = (1.0 - sample_map(cost, ends)).clamp(0.0, 1.0).mean(dim=-1)  # (B, N) 两端点权重
    return 1.0 - span * weight / SPAN_MAX



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


# ─────────────────────────── 聚合函数 (训练使用, 数值与原实现一致) ───────────────────────────

def cost_loss(cost: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """采样代价均值, 点落在低代价(可通行)区"""
    return cost_per_query(cost, points).mean()


def interior_loss(dist_field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """居中惩罚: 1 - 距离场均值, 越小线越靠区域中心"""
    return interior_per_query(dist_field, points).mean()


def smooth_loss(points: torch.Tensor) -> torch.Tensor:
    """二阶差分 (归一化), 防锯齿: 惩罚折线抖动, 让 16 个点构成的线更平滑"""
    return smooth_per_query(points).mean()


def repel_loss(points: torch.Tensor, margin: float) -> torch.Tensor:
    """线间排斥, 防塌缩: 每个点到另一条线折线的距离 < margin 时 hinge (归一化)

    距离用"点到折线"(而非点到离散点), 聚合用所有越界点的均值(而非 min),
    因此无论线多长/采样多稀, 只要两线重合就产生稠密梯度。
    """
    B, N, _, _ = points.shape
    if N < 2:
        return points.new_zeros(())
    return repel_per_pair(points, margin).sum() / (B * N * (N - 1))


def length_loss(points: torch.Tensor, min_length: float) -> torch.Tensor:
    """端点长度, 防线缩点: 每条线首尾两点距离 < min_length 时惩罚 (归一化)"""
    return length_per_query(points, min_length).mean()


def spread_loss(points: torch.Tensor, thr: float) -> torch.Tensor:
    """输出多样性, 防雷同: query 两两输出余弦相似度 > thr 时惩罚

    与 repel 互补: repel 约束空间距离, spread 约束输出形状差异。
    """
    B, N, _, _ = points.shape
    if N < 2:
        return points.new_zeros(())
    return spread_per_pair(points, thr).mean()


def order_loss(points: torch.Tensor) -> torch.Tensor:
    """点序单调 (弱), 防回折: 相邻点 u 增量为负时 hinge"""
    return order_per_query(points).mean()


def grow_loss(cost: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """增长激励 (非负惩罚): 1 - 加权端点距/SPAN_MAX, 最小化即最大化线在可通行区内的跨度"""
    return grow_per_query(cost, points).mean()


def loss_shares(loss_dict: dict) -> dict:
    """各损失项占总损失的百分比 (所有项非负, 占比之和=1)

    用于诊断"哪一项在主导", 解决不同项量纲/正负不可比的问题。
    """
    total = sum(v for k, v in loss_dict.items() if k != 'total')
    if total <= 0:
        return {k: 0.0 for k in loss_dict if k != 'total'}
    return {k: float(v / total) for k, v in loss_dict.items() if k != 'total'}
