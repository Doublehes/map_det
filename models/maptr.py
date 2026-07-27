import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .bev_encoder import BEVFormerEncoder, GridMask
from .head import MapTRHead, MapSegHead, BEVHeatMapHead
from .losses import MapTRCriterion


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

    def forward(self, imgs, intrinsics, extrinsics, seg_only=False):
        batch_size, num_cams, C, H, W = imgs.shape

        imgs_flat = imgs.view(batch_size * num_cams, C, H, W)
        img_feats = self.backbone(imgs_flat)
        img_feats = [self.grid_mask(f) for f in img_feats]

        bev_feat = self.bev_encoder(img_feats, intrinsics, extrinsics, imgs=imgs)

        if not seg_only:
            cls_scores, reg_preds = self.head(bev_feat)
        else:
            cls_scores = None
            reg_preds = None

        seg_pred = self.seg_head(bev_feat) if self.seg_head else None
        heatmap_pred = self.heatmap_head(bev_feat) if self.heatmap_head else None

        return cls_scores, reg_preds, seg_pred, heatmap_pred

    def compute_loss(self, cls_scores, reg_preds, seg_preds, batch, seg_only=False, heatmap_pred=None):
        sem_mask = batch.get('semantic_mask')
        if sem_mask is not None and seg_preds is not None:
            sem_mask = torch.flip(sem_mask, [2,]).to(seg_preds.device)
        else:
            sem_mask = None
        soft_heatmap = batch.get('soft_heatmap')
        return self.criterion(
            cls_scores, reg_preds,
            batch['vectors'],
            sem_mask,
            seg_preds,
            gt_heatmap=soft_heatmap,
            heatmap_pred=heatmap_pred,
            seg_only=seg_only,
        )
