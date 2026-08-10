import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import MapTransformerDecoder
from .losses import (
    HungarianMatcher, focal_loss, l1_loss,
    MaskFocalLoss, MaskDiceLoss, HeatmapLoss,
)


class SinePositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, num_feats, normalize=True):
        super().__init__()
        self.num_feats = num_feats
        self.normalize = normalize

    def forward(self, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps)
            x_embed = x_embed / (x_embed[:, :, -1:] + eps)

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = 2. ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MapTRHead(nn.Module):
    """检测头: BEV position encoding + query + reference points + transformer decoder + cls/reg"""

    def __init__(self, cfg):
        super().__init__()
        self.num_queries = cfg.num_queries
        self.num_classes = cfg.num_classes
        self.num_points = cfg.num_points
        self.embed_dims = cfg.embed_dims
        self.bev_embed_dims = cfg.bev_embed_dims

        # BEV position encoding
        self.bev_pos_embed = SinePositionalEncoding(self.embed_dims // 2, normalize=True)
        self.input_proj = nn.Conv2d(self.bev_embed_dims, self.embed_dims, kernel_size=1)

        # query
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        self.reference_points_embed = nn.Linear(self.embed_dims, self.num_points * 2)

        # decoder
        self.transformer = MapTransformerDecoder(cfg)

        # prediction heads (one per decoder layer, shared since different_heads=False)
        cls_branch = nn.Linear(self.embed_dims, self.num_classes)
        reg_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.LayerNorm(self.embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
            nn.LayerNorm(self.embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims * 2, self.num_points * 2),
        )
        num_layers = cfg.num_decoder_layers
        self.cls_branches = nn.ModuleList([cls_branch for _ in range(num_layers)])
        self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_layers)])

        # 损失: Hungarian 匹配 + 分类/回归损失参数
        self.matcher = HungarianMatcher(
            cls_weight=cfg.matcher_cls_weight,
            reg_weight=cfg.matcher_reg_weight,
        )
        self.loss_cls_weight = cfg.loss_cls_weight
        self.loss_reg_weight = cfg.loss_reg_weight
        self.focal_gamma = cfg.focal_gamma
        self.focal_alpha = cfg.focal_alpha
        self.l1_beta = cfg.l1_beta

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # focal loss bias init: 初始预测概率 ≈ 0.01
        bias_init = math.log(0.01 / 0.99)
        for m in self.cls_branches:
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, bias_init)

    def _prepare_context(self, bev_features):
        """Add positional encoding to BEV features."""
        B, C, H, W = bev_features.shape
        bev_mask = bev_features.new_zeros(B, H, W, dtype=torch.bool)
        pos_embed = self.bev_pos_embed(bev_mask)
        bev_embed = self.input_proj(bev_features) + pos_embed
        return bev_embed

    def forward(self, bev_feat, return_all_layers=False):
        """
        bev_feat: (B, bev_embed_dims, H, W)
        Returns:
            cls_scores: (B, num_queries, num_classes)
            reg_preds: (B, num_queries, num_points, 2)
            若 return_all_layers=True: cls_scores/reg_preds 为 list, 每层一个
        """
        # 1. Position encoding
        bev_embed = self._prepare_context(bev_feat)  # (B, embed_dims, H, W)

        # 2. BEV feature flatten for decoder
        bs, c, h, w = bev_embed.shape
        feat_flatten = bev_embed.flatten(2).transpose(1, 2)  # (B, H*W, embed_dims)
        feat_flatten = feat_flatten.permute(1, 0, 2)         # (H*W, B, embed_dims)

        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long, device=bev_feat.device)
        level_start_index = torch.cat([spatial_shapes.new_zeros(1),
                                        spatial_shapes.prod(1).cumsum(0)[:-1]])

        # 3. Query + reference points
        query = self.query_embedding.weight.unsqueeze(1).repeat(1, bs, 1)  # (num_q, B, embed_dims)
        ref_points = self.reference_points_embed(query.permute(1, 0, 2)).sigmoid()  # (B, num_q, num_pts*2)
        ref_points = ref_points.view(bs, self.num_queries, self.num_points, 2)

        # 4. Decoder
        inter_queries, inter_refs = self.transformer(
            query, feat_flatten, feat_flatten,
            reference_points=ref_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches=self.reg_branches,
        )

        # 5. Classification + regression
        if return_all_layers:
            cls_scores_all = [self.cls_branches[l](inter_queries[l]) for l in range(len(inter_queries))]
            reg_preds_all = inter_refs
            return cls_scores_all, reg_preds_all

        cls_scores = self.cls_branches[-1](inter_queries[-1])   # (B, num_q, num_classes)
        reg_preds = inter_refs[-1]                                # (B, num_q, num_pts, 2)

        return cls_scores, reg_preds

    def loss(self, cls_scores, reg_preds, gt_vectors):
        """Hungarian 匹配后计算分类 FocalLoss + 回归 SmoothL1Loss

        gt_vectors: list of dict {cls_id: (N, 1|2, num_points, 2)}, 长度 = batch_size
        Returns:
            {'cls_loss': Tensor, 'reg_loss': Tensor}
        """
        loss_dict = {}
        bs = len(gt_vectors)
        device = cls_scores.device

        # 1. 展平GT: 将各个类别的线合并为统一的列表
        gt_labels_list, gt_lines_list = [], []
        for i in range(bs):
            vec = gt_vectors[i]
            labels, lines = [], []
            for cls_id in sorted(vec.keys()):
                cls_lines = vec[cls_id]
                for j in range(cls_lines.shape[0]):
                    labels.append(cls_id)
                    line = cls_lines[j].to(device).float()
                    if line.shape[0] == 1:
                        line = line.expand(2, -1, -1)
                    lines.append(line)
            if len(labels) == 0:
                labels.append(0)
                lines.append(torch.zeros((2, self.num_points, 2), device=device, dtype=torch.float))
            gt_labels_list.append(torch.tensor(labels, device=device, dtype=torch.long))
            gt_lines_list.append(torch.stack(lines, dim=0))

        # 2. Hungarian匹配: 每个query匹配到最优GT线
        indices = self.matcher(cls_scores, reg_preds, gt_labels_list, gt_lines_list)

        # 3. 计算匹配后的损失
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        num_matched = 0

        for i in range(bs):
            pred_idx, tgt_idx = indices[i][:2]
            best_perm = indices[i][2] if len(indices[i]) > 2 else None
            num_q, num_cls = cls_scores.shape[1], cls_scores.shape[2]

            # 所有 query 的 cls target: 负样本全0, 正样本 one-hot
            cls_target = cls_scores[i].new_zeros(num_q, num_cls)

            if len(pred_idx) > 0:
                tgt_cls = gt_labels_list[i][tgt_idx]
                cls_target[pred_idx, tgt_cls] = 1.0

                matched_reg = reg_preds[i, pred_idx]
                tgt_lines = gt_lines_list[i][tgt_idx]
                if best_perm is not None and tgt_lines.dim() == 4:
                    M = len(pred_idx)
                    tgt_lines = tgt_lines[torch.arange(M, device=tgt_lines.device), best_perm]
                total_reg_loss += l1_loss(matched_reg, tgt_lines, self.l1_beta)
                num_matched += len(pred_idx)

            # cls loss: 所有 query 都计算, 负样本被推向全0
            total_cls_loss += focal_loss(
                cls_scores[i], cls_target, self.focal_gamma, self.focal_alpha, reduction='sum')

        if num_matched > 0:
            total_cls_loss = total_cls_loss / num_matched
            total_reg_loss = total_reg_loss / num_matched
        else:
            total_cls_loss = cls_scores.mean() * 0.0
            total_reg_loss = reg_preds.mean() * 0.0

        loss_dict['cls_loss'] = self.loss_cls_weight * total_cls_loss
        loss_dict['reg_loss'] = self.loss_reg_weight * total_reg_loss
        return loss_dict


class MapSegHead(nn.Module):
    """分割头: 上采样 BEV 特征到分割图, 4x4px/m 均一分辨率"""

    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.bev_embed_dims
        self.embed_dims = cfg.bev_embed_dims
        self.num_classes = cfg.num_classes

        self.conv_in = nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.embed_dims, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_classes, kernel_size=1),
        )
        self.mask_focal_loss = MaskFocalLoss(loss_weight=1.0, gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)
        self.mask_dice_loss = MaskDiceLoss(loss_weight=1.0)
        self.loss_seg_weight = cfg.loss_seg_weight
        self.loss_dice_weight = cfg.loss_dice_weight
        self._init_bias()

    def _init_bias(self):
        nn.init.constant_(self.upsample[-1].bias, math.log(0.01 / 0.99))

    def forward(self, bev_feat):
        x = self.relu(self.conv_in(bev_feat))
        return self.upsample(x)

    def loss(self, seg_preds, gt_semantic_mask):
        """分割损失: per-class focal + dice"""
        gt_semantic_mask = gt_semantic_mask.to(seg_preds.device)
        return {
            'seg_loss': self.loss_seg_weight * self.mask_focal_loss(seg_preds, gt_semantic_mask),
            'dice_loss': self.loss_dice_weight * self.mask_dice_loss(seg_preds, gt_semantic_mask),
        }


class BEVHeatMapHead(nn.Module):
    """热力图预测头: 从BEV特征回归连续热力图 (B, 1, 80, 160)"""

    def __init__(self, cfg):
        super().__init__()
        self.conv_in = nn.Conv2d(cfg.bev_embed_dims, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_out = nn.Conv2d(128, 1, kernel_size=1)
        self.heatmap_loss_fn = HeatmapLoss(
            loss_weight=cfg.loss_heatmap_weight,
            threshold=cfg.loss_threshold,
            beta=cfg.loss_beta,
        )

    def forward(self, bev_feat):
        x = self.relu(self.conv_in(bev_feat))
        x = self.upsample(x)
        return self.conv_out(x)

    def loss(self, heatmap_pred, gt_heatmap):
        """热力图损失: masked smooth l1 (权重在 loss_fn 内部)"""
        gt_heatmap = gt_heatmap.to(heatmap_pred.device)
        return {'heatmap_loss': self.heatmap_loss_fn(heatmap_pred, gt_heatmap)}
