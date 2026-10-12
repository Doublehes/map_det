"""model_slim2 检测头: 继承 MapTRHead, 仅重写 loss 为自监督

forward / query / transformer decoder / cls+reg 分支全部复用 MapTRHead,
只把基于 GT 的匈牙利匹配 loss 换成基于代价图/距离场的自监督损失。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.default import AttrDict
from models.head import MapTRHead

from .losses import (
    cost_loss, interior_loss, smooth_loss, repel_loss, length_loss, spread_loss,
    order_loss, grow_loss,
    cost_per_query, interior_per_query, smooth_per_query, length_per_query, order_per_query,
    grow_per_query, repel_per_pair, spread_per_pair,
)

# MapTRHead.__init__ 需要但自监督用不到的字段 (占位默认值)
_MAPTR_PLACEHOLDER = {
    'matcher_cls_weight': 5.0,
    'matcher_reg_weight': 50.0,
    'loss_cls_weight': 5.0,
    'loss_reg_weight': 50.0,
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,
    'l1_beta': 0.01,
    'aux_loss': False,
    'aux_weight': 1.0,
}


class Slim2Head(MapTRHead):
    """自监督线检测头: 复用 MapTRHead 结构, 重写 loss"""

    def __init__(self, cfg):
        cfg = AttrDict(dict(cfg))
        for k, v in _MAPTR_PLACEHOLDER.items():
            cfg.setdefault(k, v)
        super().__init__(cfg)

        # 自监督损失权重
        self.w_cost = cfg.get('cost_weight', 1.0)
        self.w_interior = cfg.get('interior_weight', 0.5)
        self.w_smooth = cfg.get('smooth_weight', 2.0)
        self.w_repel = cfg.get('repel_weight', 1.0)
        self.repel_margin = cfg.get('repel_margin', 0.075)
        self.w_length = cfg.get('length_weight', 1.0)
        self.min_length = cfg.get('min_length', 0.4)
        self.w_spread = cfg.get('spread_weight', 0.5)
        self.spread_thr = cfg.get('spread_thr', 0.9)
        self.w_order = cfg.get('order_weight', 0.1)
        self.w_grow = cfg.get('grow_weight', 0.2)

    def loss(self, cls_scores, reg_preds, cost, dist_field):
        """自监督损失: 采样代价图/距离场 + 正则

        cls_scores: (B, N, num_classes) logits (自监督下不参与损失)
        reg_preds:  (B, N, num_points, 2) 归一化 (u,v)
        cost:       (B, 1, H, W) 代价图 [0,1]
        dist_field: (B, 1, H, W) 可通行区距离场 [0,1]
        Returns: loss dict (含 'total')
        """
        device = reg_preds.device
        cost = cost.to(device)
        dist_field = dist_field.to(device)

        # ───────────────────────── 自监督损失说明 ─────────────────────────
        # 核心思路: 无线 GT, 用"代价图 + 距离场"作为信号, 把预测线放到可通行区域中心。
        # 前 2 项是"正确信号", 后 5 项是"防退化的正则" (单独用前 2 项会塌缩到平凡解)。
        #
        # 1) l_cost     : 预测点采样代价图取均值 ↓
        #                 → 点落在低代价(可通行)区, 防"线进障碍区"
        # 2) l_interior : 1 - 预测点采样距离场均值 ↓ (距离场越大越靠区域中心)
        #                 → 线居中而非贴边, 这是让线成为"中心线"的关键
        # 3) l_smooth   : 每条线 16 点二阶差分模长均值 ↓
        #                 → 线平滑, 防锯齿/乱序
        # 4) l_repel    : 每个点到另一条线折线的距离 < margin 时 hinge ↓
        #                 → 防 8 条线塌缩重叠成一条
        # 5) l_length   : 每条线端点距 < min_length 时 hinge ↓
        #                 → 防线缩成一个点/极短
        # 6) l_spread   : query 两两输出余弦相似度 > thr 时惩罚 ↓
        #                 → 防多条 query 输出雷同 (与 repel 互补: repel 管空间, spread 管形状)
        # 7) l_order    : 相邻点 u 增量为负时 hinge ↓ (弱权重)
        #                 → 防折线自交叉/回折
        # 8) l_grow     : 1 - 加权端点距/SPAN_MAX ↓ (最小化=最大化线跨度)
        #                 → 促线在可通行区内尽量长(横跨走廊), 到边界自动停 (cost 兜底越界)
        # 注: 以上 8 项均 >=0, 越大越差; 用 loss_shares 看各项占比(解决量纲/正负不可比)
        # ──────────────────────────────────────────────────────────────────
        loss_dict = {
            'cost': self.w_cost * cost_loss(cost, reg_preds),
            'interior': self.w_interior * interior_loss(dist_field, reg_preds),
            'smooth': self.w_smooth * smooth_loss(reg_preds),
            'repel': self.w_repel * repel_loss(reg_preds, self.repel_margin),
            'length': self.w_length * length_loss(reg_preds, self.min_length),
            'spread': self.w_spread * spread_loss(reg_preds, self.spread_thr),
            'order': self.w_order * order_loss(reg_preds),
            'grow': self.w_grow * grow_loss(cost, reg_preds),
        }
        loss_dict['total'] = sum(loss_dict.values())
        return loss_dict

    def loss_breakdown(self, cls_scores, reg_preds, cost, dist_field):
        """返回未聚合的 7 项损失明细 (供调试/可视化逐项查看)

        Returns: dict
            per-query (B, N): cost / interior / smooth / length / order / grow
            per-pair  (B, N, N): repel / spread (对角为 0)
        """
        device = reg_preds.device
        cost = cost.to(device)
        dist_field = dist_field.to(device)
        return {
            'cost': cost_per_query(cost, reg_preds),
            'interior': interior_per_query(dist_field, reg_preds),
            'smooth': smooth_per_query(reg_preds),
            'length': length_per_query(reg_preds, self.min_length),
            'order': order_per_query(reg_preds),
            'grow': grow_per_query(cost, reg_preds),
            'repel': repel_per_pair(reg_preds, self.repel_margin),
            'spread': spread_per_pair(reg_preds, self.spread_thr),
        }
