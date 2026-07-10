import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .bev_encoder import BEVFormerEncoder, GridMask
from .head import MapTRHead, MapSegHead
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
        self.seg_head = MapSegHead(cfg.map_seg_head)
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

        seg_pred = self.seg_head(bev_feat)

        return cls_scores, reg_preds, seg_pred

    def compute_loss(self, cls_scores, reg_preds, seg_preds, batch, seg_only=False):
        sem_mask = batch.get('semantic_mask')
        if sem_mask is not None:
            sem_mask = torch.flip(sem_mask, [2,]).to(seg_preds.device)
        return self.criterion(
            cls_scores, reg_preds,
            batch['vectors'],
            sem_mask,
            seg_preds,
            seg_only=seg_only,
        )
