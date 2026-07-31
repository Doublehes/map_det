import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .bev_encoder import BEVFormerEncoder, GridMask
from .head import MapTRHead, MapSegHead, BEVHeatMapHead
from .losses import MapTRCriterion


class BEVTransform:
    """BEV 特征变换: 翻转 + 旋转 + 平移, 统一接口

    归一化坐标 [-1,1] 到世界坐标:
        x_world = x_ndc * half_x + x_center
        y_world = -y_ndc * half_y + y_center

    theta 矩阵由三步推导: (x_ndc_out, y_ndc_out) → 世界 → 反向变换 → (x_ndc_in, y_ndc_in)

        x_in = c*x_out + (-s*yr)*y_out + ((c-1)*x_center + s*y_center) / half_x - dx/half_x
        y_in = (s*xr)*x_out + c*y_out + (s*x_center + (1-c)*y_center) / half_y + dy/half_y
    """

    def __init__(self, x_range=40.0, y_range=20.0, x_center=10.0, y_center=0.0):
        self.half_x = x_range / 2.0
        self.half_y = y_range / 2.0

        self.ratio_xy = x_range / y_range    # theta[1,0]: s * ratio_xy
        self.ratio_yx = y_range / x_range    # theta[0,1]: -s * ratio_yx

        self.x_center = x_center
        self.y_center = y_center

    def flip_bev(self, bev_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is not None and mask.any():
            mask = mask.to(bev_feat.device)
            bev_feat[mask] = bev_feat[mask].flip(dims=[-2])
        return bev_feat

    def _affine_transform(self, bev_feat: torch.Tensor,
                          rot_angle: torch.Tensor = None,
                          dx: torch.Tensor = None,
                          dy: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = bev_feat.shape
        device, dtype = bev_feat.device, bev_feat.dtype

        if rot_angle is not None:
            if rot_angle.dim() == 0:
                rot_angle = rot_angle.expand(B)
            assert rot_angle.shape == (B,)
            c = torch.cos(rot_angle)
            s = torch.sin(rot_angle)
        else:
            c, s = 1.0, 0.0

        tx = dx / self.half_x if dx is not None else 0.0
        ty = dy / self.half_y if dy is not None else 0.0

        theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = c
        theta[:, 0, 1] = -s * self.ratio_yx
        theta[:, 0, 2] = ((c - 1) * self.x_center + s * self.y_center) / self.half_x - tx
        theta[:, 1, 0] = s * self.ratio_xy
        theta[:, 1, 1] = c
        theta[:, 1, 2] = (s * self.x_center + (1 - c) * self.y_center) / self.half_y + ty

        grid = F.affine_grid(theta, bev_feat.shape, align_corners=False)
        return F.grid_sample(bev_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    def transform_bev(self, bev_feat: torch.Tensor, flip: torch.Tensor = None,
                       rot_angle: torch.Tensor = None,
                       dx: torch.Tensor = None,
                       dy: torch.Tensor = None) -> torch.Tensor:
        bev_feat = self.flip_bev(bev_feat, flip)
        need_affine = (
            (rot_angle is not None and rot_angle.abs().max() > 1e-6) or
            (dx is not None and dx.abs().max() > 1e-6) or
            (dy is not None and dy.abs().max() > 1e-6)
        )
        if need_affine:
            bev_feat = self._affine_transform(bev_feat, rot_angle, dx, dy)
        return bev_feat


class MapTR(nn.Module):
    """MapTR: 多相机 → BEV → Transformer解码 → 道路结构线检测"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = Backbone(cfg.img_backbone)
        self.grid_mask = GridMask(use_grid_mask=True)
        self.bev_encoder = BEVFormerEncoder(cfg.bev_encoder)

        assert cfg.map_det_head.type == 'maptr'
        self.head = MapTRHead(cfg.map_det_head)
        self.seg_head = MapSegHead(cfg.map_seg_head) if cfg.map_seg_head.get('enabled', True) else None
        self.heatmap_head = BEVHeatMapHead(cfg.heatmap_head) if cfg.heatmap_head.get('enabled', True) else None
        self.criterion = MapTRCriterion(cfg.loss)

        pc = cfg.bev_encoder.pc_range
        self.bev_transform = BEVTransform(
            x_range=float(pc[3] - pc[0]),
            y_range=float(pc[4] - pc[1]),
            x_center=float((pc[3] + pc[0]) / 2),
            y_center=float((pc[4] + pc[1]) / 2),
        )

    def forward(self, imgs, intrinsics, extrinsics, seg_only=False, batch=None,
                return_all_layers=False):
        batch_size, num_cams, C, H, W = imgs.shape

        imgs_flat = imgs.view(batch_size * num_cams, C, H, W)
        img_feats = self.backbone(imgs_flat)
        img_feats = [self.grid_mask(f) for f in img_feats]

        bev_feat = self.bev_encoder(img_feats, intrinsics, extrinsics, imgs=imgs)

        if batch is not None:
            bev_feat = self.bev_transform.transform_bev(
                bev_feat,
                flip=batch.get('flip'),
                rot_angle=batch.get('rot_angle'),
                dx=batch.get('dx'),
                dy=batch.get('dy'),
            )

        if not seg_only:
            cls_scores, reg_preds = self.head(bev_feat, return_all_layers=return_all_layers)
        else:
            cls_scores = None
            reg_preds = None

        seg_pred = self.seg_head(bev_feat) if self.seg_head else None
        heatmap_pred = self.heatmap_head(bev_feat) if self.heatmap_head else None

        return cls_scores, reg_preds, seg_pred, heatmap_pred, bev_feat

    def compute_loss(self, cls_scores, reg_preds, seg_preds, batch, seg_only=False, heatmap_pred=None):
        sem_mask = batch.get('semantic_mask')
        if sem_mask is not None and seg_preds is not None:
            sem_mask = sem_mask.to(seg_preds.device)
        soft_heatmap = batch.get('soft_heatmap')
        if soft_heatmap is not None and heatmap_pred is not None:
            soft_heatmap = soft_heatmap.to(heatmap_pred.device)
        return self.criterion(
            cls_scores, reg_preds,
            batch['vectors'],
            sem_mask,
            seg_preds,
            gt_heatmap=soft_heatmap,
            heatmap_pred=heatmap_pred,
            seg_only=seg_only,
        )
