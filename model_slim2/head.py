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
    spacing_loss, span_loss, start_gather_loss, diverge_loss,
    keypoint_coverage_loss, keypoint_coverage_per_kp,
    cost_per_query, interior_per_query, smooth_per_query, arc_per_query,
    repel_per_pair, spread_per_pair, spacing_per_query, span_per_query,
    start_gather_per_pair, diverge_per_pair,
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
        self.w_start_gather = cfg.get('start_gather_weight', 0.0)
        self.w_diverge = cfg.get('diverge_weight', 0.0)
        self.diverge_margin = cfg.get('diverge_margin', 0.2)
        self.w_kp_coverage = cfg.get('keypoint_coverage_weight', 0.0)
        self.kp_coverage_thr = cfg.get('keypoint_coverage_thr', 0.05)
        self.decoder_layer_loss_weights = cfg.get('decoder_layer_loss_weights', None)
        self.last_layer_losses = None
        self.lp_p = cfg.get('lp_p', 4.0)

    def init_weights(self):
        """在基类 xavier 初始化基础上, 把初始采样/预测点设为一组先验竖直线。

        默认初始下 reference_points_embed 与 reg_branch 末层(xavier)输出≈0,
        经 sigmoid ≈ 0.5 -> 16 个点叠在图像中心成"一团", 导致训练对初值敏感、
        容易塌进局部极小。这里改为初始即为竖直线, 起步就满足
        arc/span/spacing/smooth, 绕开塌缩盆地。

        多条 query: 第 n 条 query 的参考线 = (u_n, v_p), 其中
          u_n = linspace(0.2, 0.8, N)  (N=1 时取 0.5) —— 各 query 的 u 均匀铺开
          v_p = linspace(0.9, 0.1, P)  (所有 query 共享; 首点在下, 自下而上)
        reference_points_embed 是逐 query 调用的 Linear, 这里用 pinv 精确拟合
        (保留 query_embedding 的随机初始化), 使 ref_points = sigmoid(Q @ Wᵀ) 恰好等于目标线。
        reg_branches 末层为共享 Linear(bias 无法 per-query), 保持中心先验 (u=0.5)。
        """
        super().init_weights()
        N, P, D = self.num_queries, self.num_points, self.embed_dims
        assert N <= D, f'num_queries({N}) 不能超过 embed_dims({D})'
        u = torch.full((N,), 0.5) if N == 1 else torch.linspace(0.2, 0.8, N)
        v = torch.linspace(0.9, 0.1, P)
        U = u[:, None].expand(N, P)
        V = v[None, :].expand(N, P)
        coords = torch.stack([U, V], dim=-1).clamp(1e-4, 1 - 1e-4)   # (N, P, 2)
        L = torch.log(coords / (1 - coords)).reshape(N, 2 * P)        # (N, 2P) 目标 logits
        with torch.no_grad():
            # (1) 采样参考点层: 精确 per-query (Q @ Wᵀ = L, Q=query_embedding.weight 随机保留)
            Q = self.query_embedding.weight                              # (N, D)
            W = (torch.linalg.pinv(Q) @ L).t()                           # (2P, D)
            self.reference_points_embed.weight.copy_(W)
            self.reference_points_embed.bias.zero_()
            # (2) 回归分支末层: 共享中心先验 (u=0.5, v=0.9->0.1; 各层共享同一对象)
            ctr = torch.stack([torch.full((P,), 0.5), v], dim=-1).reshape(-1)
            ctr = ctr.clamp(1e-4, 1 - 1e-4)
            ctr_logit = torch.log(ctr / (1 - ctr))
            for branch in self.reg_branches:
                branch[-1].weight.zero_()
                branch[-1].bias.copy_(ctr_logit)

    def _single_layer_terms(self, reg_preds, cost, dist_field,
                            keypoints, kp_valid, img_hw):
        """单层自监督损失各项 (不含 total)。全部 >=0, 越大越差。"""
        if keypoints is not None:
            l_kp_coverage = self.w_kp_coverage * keypoint_coverage_loss(
                reg_preds, keypoints, self.kp_coverage_thr, kp_valid)
        else:
            l_kp_coverage = reg_preds.new_zeros(())
        return {
            'cost': self.w_cost * cost_loss(cost, reg_preds, self.lp_p),
            'interior': self.w_interior * interior_loss(dist_field, reg_preds, self.lp_p),
            'smooth': self.w_smooth * smooth_loss(reg_preds, img_hw, self.smooth_min_step_px, self.lp_p),
            'spacing': self.w_spacing * spacing_loss(reg_preds, img_hw, self.lp_p),
            'arc': self.w_arc * arc_loss(reg_preds, self.min_arc),
            'span': self.w_span * span_loss(reg_preds, self.min_span),
            'repel': self.w_repel * repel_loss(reg_preds, self.repel_margin),
            'spread': self.w_spread * spread_loss(reg_preds, self.spread_thr),
            'start_gather': self.w_start_gather * start_gather_loss(reg_preds),
            'diverge': self.w_diverge * diverge_loss(reg_preds, self.diverge_margin),
            'keypoint_coverage': l_kp_coverage,
        }

    def loss(self, cls_scores, reg_preds, cost, dist_field,
             keypoints=None, keypoints_valid=None):
        """自监督损失: 采样代价图/距离场 + 正则 (支持多层 decoder)

        cls_scores: 自监督下不参与损失 (单层或多层 list 均可)
        reg_preds:  (B, N, num_points, 2) 或 list[每层一个 (B, N, num_points, 2)]
        cost:       (B, 1, H, W) 代价图 [0,1]
        dist_field: (B, 1, H, W) 可通行区距离场 [0,1]
        keypoints:  (B, K, 2) 归一化关键点 (可为 None; 补 0 占位由 keypoints_valid 过滤)
        keypoints_valid: (B, K) 1=真实关键点, 0=占位
        Returns: loss dict (含 'total')

        多层时按 `decoder_layer_loss_weights` (缺省全 1) 加权: 各项聚合到同名键,
        因此 loss_shares / 训练日志键名不变; 各层加权损失存于 self.last_layer_losses。
        """
        is_multi = isinstance(reg_preds, (list, tuple))
        device = (reg_preds[0] if is_multi else reg_preds).device
        cost = cost.to(device)
        dist_field = dist_field.to(device)
        kp = None if keypoints is None else keypoints.to(device)
        kp_valid = None if keypoints_valid is None else keypoints_valid.to(device)
        img_hw = cost.shape[-2:]

        # ───────────────────────── 自监督损失说明 ─────────────────────────
        # 核心思路: 无线 GT, 用"代价图 + 距离场"作为信号, 把预测线放到可通行区域中心
        # 并保持合理几何形状。所有项均 >=0, 越大越差; 线内逐点/逐段聚合用 Lp(放大难点)。
        #
        # 正确信号 (贴可通行区/脊线):
        #   l_cost    : Lp(采样代价图) ↓          → 点落在低代价(可通行)区
        #   l_interior: Lp(1 - 采样距离场) ↓      → 线居中于可通行区(脊线), 关键"中心线"信号
        # 形状正则 (无线 GT, 只看几何):
        #   l_smooth  : Lp(1-cos相邻段夹角) + Lp(relu(最小步长-段长)) ↓
        #   l_spacing : Lp(段长相对偏差 |s-mean(s)|/mean(s)) ↓ → 各段等距(尺度无关)
        #   l_arc     : relu(min_arc - 弧长Σ‖Δp‖) ↓ → 弧长(真实长度)下界
        #   l_span    : relu(min_span - ‖p_last - p_first‖) ↓ → 端点距下界, 逼线张开成条
        # 多线结构 (N>1, 默认权重 0, 见 exp3):
        #   l_start_gather / l_diverge
        # 关键点覆盖率 (默认权重 0, 见 exp4):
        #   l_keypoint_coverage: mean_k relu(d_k - thr) ↓, d_k=关键点 k 到所有预测点最近点距
        # 已关闭 (默认权重置 0): l_repel / l_spread
        # 多层: loss_dict[k] = Σ_l w_l·term_l[k]; total = Σ_l w_l·L_l (见 exp7)
        # 注: 用 loss_shares 看各项占总损失比例, 辅助权重平衡。
        # ──────────────────────────────────────────────────────────────────
        if is_multi:
            L = len(reg_preds)
            weights = self.decoder_layer_loss_weights
            if weights is None:
                weights = [1.0] * L
            assert len(weights) == L, \
                f'decoder_layer_loss_weights 长度 {len(weights)} != decoder 层数 {L}'
            per_layer = [self._single_layer_terms(rp, cost, dist_field, kp, kp_valid, img_hw)
                         for rp in reg_preds]
            keys = list(per_layer[0].keys())
            loss_dict = {
                k: sum(w * terms[k] for w, terms in zip(weights, per_layer))
                for k in keys
            }
            self.last_layer_losses = [
                w * sum(terms.values()) for w, terms in zip(weights, per_layer)
            ]
        else:
            loss_dict = self._single_layer_terms(reg_preds, cost, dist_field, kp, kp_valid, img_hw)
            self.last_layer_losses = None

        loss_dict['total'] = sum(loss_dict.values())
        return loss_dict

    def loss_breakdown(self, cls_scores, reg_preds, cost, dist_field, keypoints=None):
        """返回未聚合的逐项损失明细 (供调试/可视化逐项查看)

        Returns: dict
            per-query (B, N): cost / interior / smooth / spacing / arc / span
            per-pair  (B, N, N): repel / spread (对角为 0)
            per-kp    (B, K): keypoint_coverage (仅当提供 keypoints 时)
        """
        if isinstance(reg_preds, (list, tuple)):     # 多层时只看最后一层
            reg_preds = reg_preds[-1]
        device = reg_preds.device
        cost = cost.to(device)
        dist_field = dist_field.to(device)
        img_hw = cost.shape[-2:]
        out = {
            'cost': cost_per_query(cost, reg_preds, self.lp_p),
            'interior': interior_per_query(dist_field, reg_preds, self.lp_p),
            'smooth': smooth_per_query(reg_preds, img_hw, self.smooth_min_step_px, self.lp_p),
            'spacing': spacing_per_query(reg_preds, img_hw, self.lp_p),
            'arc': arc_per_query(reg_preds, self.min_arc),
            'span': span_per_query(reg_preds, self.min_span),
            'repel': repel_per_pair(reg_preds, self.repel_margin),
            'spread': spread_per_pair(reg_preds, self.spread_thr),
            'start_gather': start_gather_per_pair(reg_preds),
            'diverge': diverge_per_pair(reg_preds, self.diverge_margin),
        }
        if keypoints is not None:
            out['keypoint_coverage'] = keypoint_coverage_per_kp(
                reg_preds, keypoints.to(device), self.kp_coverage_thr)
        return out
