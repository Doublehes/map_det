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
    cost_loss, interior_loss, smooth_loss, repel_loss, arc_loss, spread_loss,
    spacing_loss, span_loss,
    cost_per_query, interior_per_query, smooth_per_query, arc_per_query,
    repel_per_pair, spread_per_pair, spacing_per_query, span_per_query,
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
        self.w_arc = cfg.get('arc_weight', 1.0)
        self.min_arc = cfg.get('min_arc', 0.4)
        self.w_span = cfg.get('span_weight', 1.0)
        self.min_span = cfg.get('min_span', 0.4)
        self.w_spread = cfg.get('spread_weight', 0.5)
        self.spread_thr = cfg.get('spread_thr', 0.9)
        self.w_spacing = cfg.get('spacing_weight', 1.0)
        self.smooth_min_step_px = cfg.get('smooth_min_step_px', 2.0)
        self.lp_p = cfg.get('lp_p', 4.0)

    def init_weights(self):
        """在基类 xavier 初始化基础上, 把初始预测/采样点设为一条先验直线。

        默认初始下 reference_points_embed 与 reg_branch 末层(xavier)输出≈0,
        经 sigmoid ≈ 0.5 -> 16 个点叠在图像中心成"一团", 导致训练对初值敏感、
        容易塌进局部极小。这里改为初始即为一条竖向直线, 起步就满足
        arc/span/spacing/smooth, 绕开塌缩盆地。

        先验直线(归一化 u,v; u→宽, v→高): u 居中 0.5, v 从 0.9 铺到 0.1 (首点在下, 自下而上)。
        """
        super().init_weights()
        P = self.num_points
        u = torch.full((P,), 0.5)
        v = torch.linspace(0.9, 0.1, P)
        line = torch.stack([u, v], dim=-1).reshape(-1)      # (2P,) [u0,v0,u1,v1,...]
        line = line.clamp(1e-4, 1 - 1e-4)
        logit = torch.log(line / (1 - line))                # 反 sigmoid
        with torch.no_grad():
            # (1) 采样参考点层: 初始采样位置 = 先验直线
            self.reference_points_embed.weight.zero_()
            self.reference_points_embed.bias.copy_(logit)
            # (2) 回归分支末层: 初始预测 = 先验直线 (各层共享同一对象)
            for branch in self.reg_branches:
                branch[-1].weight.zero_()
                branch[-1].bias.copy_(logit)

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
        # 核心思路: 无线 GT, 用"代价图 + 距离场"作为信号, 把预测线放到可通行区域中心
        # 并保持合理几何形状。所有项均 >=0, 越大越差; 线内逐点/逐段聚合用 Lp(放大难点)。
        #
        # 正确信号 (贴可通行区/脊线):
        #   l_cost    : Lp(采样代价图) ↓          → 点落在低代价(可通行)区
        #   l_interior: Lp(1 - 采样距离场) ↓      → 线居中于可通行区(脊线), 关键"中心线"信号
        # 形状正则 (无线 GT, 只看几何):
        #   l_smooth  : Lp(1-cos相邻段夹角) + Lp(relu(最小步长-段长)) ↓
        #               → 转向惩罚(防打结/回折) + 最小步长(防点挤堆)
        #   l_spacing : Lp(段长相对偏差 |s-mean(s)|/mean(s)) ↓ → 各段等距(尺度无关)
        #   l_arc     : relu(min_arc - 弧长Σ‖Δp‖) ↓ → 弧长(真实长度)下界
        #   l_span    : relu(min_span - ‖p_last - p_first‖) ↓ → 端点距下界, 逼线张开成条
        #               (弧长可被原地振荡伪造, 端点距不能; 二者互补)
        # 已关闭 (单线场景下多为 no-op, 权重置 0):
        #   l_repel / l_spread
        # 注: 用 loss_shares 看各项占总损失比例(解决量纲/正负不可比), 辅助权重平衡。
        # ──────────────────────────────────────────────────────────────────
        img_hw = cost.shape[-2:]
        loss_dict = {
            'cost': self.w_cost * cost_loss(cost, reg_preds, self.lp_p),
            'interior': self.w_interior * interior_loss(dist_field, reg_preds, self.lp_p),
            'smooth': self.w_smooth * smooth_loss(reg_preds, img_hw, self.smooth_min_step_px, self.lp_p),
            'spacing': self.w_spacing * spacing_loss(reg_preds, img_hw, self.lp_p),
            'arc': self.w_arc * arc_loss(reg_preds, self.min_arc),
            'span': self.w_span * span_loss(reg_preds, self.min_span),
            'repel': self.w_repel * repel_loss(reg_preds, self.repel_margin),
            'spread': self.w_spread * spread_loss(reg_preds, self.spread_thr),
        }
        loss_dict['total'] = sum(loss_dict.values())
        return loss_dict

    def loss_breakdown(self, cls_scores, reg_preds, cost, dist_field):
        """返回未聚合的逐项损失明细 (供调试/可视化逐项查看)

        Returns: dict
            per-query (B, N): cost / interior / smooth / spacing / arc / span
            per-pair  (B, N, N): repel / spread (对角为 0)
        """
        device = reg_preds.device
        cost = cost.to(device)
        dist_field = dist_field.to(device)
        img_hw = cost.shape[-2:]
        return {
            'cost': cost_per_query(cost, reg_preds, self.lp_p),
            'interior': interior_per_query(dist_field, reg_preds, self.lp_p),
            'smooth': smooth_per_query(reg_preds, img_hw, self.smooth_min_step_px, self.lp_p),
            'spacing': spacing_per_query(reg_preds, img_hw, self.lp_p),
            'arc': arc_per_query(reg_preds, self.min_arc),
            'span': span_per_query(reg_preds, self.min_span),
            'repel': repel_per_pair(reg_preds, self.repel_margin),
            'spread': spread_per_pair(reg_preds, self.spread_thr),
        }
