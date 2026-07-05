import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .bev_encoder import BEVFormerEncoder, GridMask
from .head import MapTRHead, MapSegHead
from .losses import MapTRCriterion


class MapTR(nn.Module):
    """MapTR: 多相机 → BEV → Transformer解码 → 道路结构线检测"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = build_backbone(cfg)
        self.grid_mask = GridMask(use_grid_mask=True)
        self.bev_encoder = BEVFormerEncoder(cfg)

        self.input_proj = nn.Sequential(
            nn.Conv2d(cfg.fpn_out_channels, cfg.bev_embed_dims, kernel_size=1),
            nn.BatchNorm2d(cfg.bev_embed_dims),
            nn.ReLU(inplace=True),
        )

        self.head = MapTRHead(cfg)
        self.seg_head = MapSegHead(cfg)
        self.criterion = MapTRCriterion(cfg)

    def forward(self, imgs, intrinsics, extrinsics, seg_only=False):
        batch_size, num_cams, C, H, W = imgs.shape

        imgs_flat = imgs.view(batch_size * num_cams, C, H, W)
        img_feats = self.backbone(imgs_flat)
        img_feats = [self.grid_mask(f) for f in img_feats]
        img_feats_proj = [self.input_proj(f) for f in img_feats]

        bev_feat = self.bev_encoder(img_feats_proj, intrinsics, extrinsics, imgs=imgs)

        if not seg_only:
            cls_scores, reg_preds = self.head(bev_feat)
        else:
            cls_scores = None
            reg_preds = None

        seg_pred = self.seg_head(bev_feat)

        return cls_scores, reg_preds, seg_pred

    def compute_loss(self, cls_scores, reg_preds, seg_preds, batch, seg_only=False):
        sem_mask = batch.get('semantic_mask')
        if sem_mask is not None:
            sem_mask = sem_mask.to(seg_preds.device)
        return self.criterion(
            cls_scores, reg_preds,
            batch['vectors'],
            sem_mask,
            seg_preds,
            seg_only=seg_only,
        )
