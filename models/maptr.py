import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .bev_encoder import BEVFormerEncoder, GridMask
from .decoder import MapTransformerDecoder
from .head import MapTRHead, MapSegHead
from .losses import MapTRCriterion


class MapTR(nn.Module):
    """MapTR: 多相机 → BEV → Transformer解码 → 道路结构线检测

    流程: 多视角图像 → Backbone+FPN → BEVEncoder → Transformer解码器 → Head
    输出: 分类分数(32×2) + 点序列回归(32×16×2) + 分割图(1×160×80)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 图像编码: ResNet50 + FPN, 输出多尺度特征
        self.backbone = build_backbone(cfg)
        self.grid_mask = GridMask(use_grid_mask=True)

        # BEV编码: 多相机特征 → 统一的BEV特征图 (256×40×80)
        self.bev_encoder = BEVFormerEncoder(cfg)

        # 特征投影: FPN输出(256) → BEV维度(256)
        self.input_proj = nn.Sequential(
            nn.Conv2d(cfg.fpn_out_channels, cfg.bev_embed_dims, kernel_size=1),
            nn.BatchNorm2d(cfg.bev_embed_dims),
            nn.ReLU(inplace=True),
        )

        # Transformer解码器: 从BEV特征中解码出线query
        self.decoder = MapTransformerDecoder(cfg)

        # 预测头: 分类(2类) + 回归(16点×2) + 分割辅助头
        self.head = MapTRHead(cfg)
        self.seg_head = MapSegHead(cfg)
        self.criterion = MapTRCriterion(cfg)

    def forward(self, imgs, intrinsics, extrinsics):
        """
        imgs: (B, 6, 3, 176, 320)  多视角图像
        intrinsics: (B, 6, 3, 3)    相机内参
        extrinsics: (B, 6, 4, 4)    相机外参
        """
        batch_size, num_cams, C, H, W = imgs.shape

        # 1. 合并batch和相机维度, 统一编码
        imgs_flat = imgs.view(batch_size * num_cams, C, H, W)

        # 2. Backbone + FPN 提取多尺度特征
        img_feats = self.backbone(imgs_flat)
        img_feats = [self.grid_mask(f) for f in img_feats]
        img_feats_proj = [self.input_proj(f) for f in img_feats]

        # 3. BEV编码: 将多相机特征转换到BEV空间
        bev_feat = self.bev_encoder(img_feats_proj, intrinsics, extrinsics)

        # 4. Transformer解码器: query与BEV特征交互, 输出32个线query
        # 使用 deformable cross-attention, 直接传入2D BEV特征图保持空间结构
        query = self.decoder(bev_feat)

        # 6. Head: 分类 + 点序列回归 + 分割
        cls_scores, reg_preds = self.head(query)
        seg_pred = self.seg_head(bev_feat)

        return cls_scores, reg_preds, seg_pred

    def compute_loss(self, cls_scores, reg_preds, seg_preds, batch):
        sem_mask = batch.get('semantic_mask')
        if sem_mask is not None:
            sem_mask = sem_mask.to(cls_scores.device)
        return self.criterion(
            cls_scores, reg_preds,
            batch['vectors'],
            sem_mask,
            seg_preds,
        )
