import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.head import MapTRHead, MapSegHead

from configs.default import AttrDict

# 各 depth 的 c2/c3/c4 输出通道 (layer1/2/3)
_RESNET_CHANNELS = {
    18: [64, 128, 256], 34: [64, 128, 256],
    50: [256, 512, 1024], 101: [256, 512, 1024], 152: [256, 512, 1024],
}


class ResNetBackbone(nn.Module):
    """ResNet backbone: 取 stem + layer1/2/3, 输出 [c2, c3, c4] (stride 4/8/16)

    输入 160x320 → c2=40x80, c3=20x40, c4=10x20; layer4/fc 不挂载, 不被训练
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.depth in _RESNET_CHANNELS, f'不支持的 depth: {cfg.depth}'
        weights = 'IMAGENET1K_V1' if cfg.get('pretrained', False) else None
        resnet = getattr(torchvision.models, f'resnet{cfg.depth}')(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.out_channels = _RESNET_CHANNELS[cfg.depth]

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)   # 40×80 (stride4)
        c3 = self.layer2(c2)  # 20×40
        c4 = self.layer3(c3)  # 10×20
        return [c2, c3, c4]


def build_backbone(cfg):
    return ResNetBackbone(cfg)


class FPN(nn.Module):
    """FPN neck: 多级 top-down 融合, 返回最细层 (默认 P2 = stride4)

    in_channels: 各级输入通道 (c2/c3/c4)
    out_channels: 输出通道 (默认 256 = bev_embed_dims)
    use_level: 返回哪一层 (0=P2 最细)
    """

    def __init__(self, cfg):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, cfg.out_channels, kernel_size=1) for c in cfg.in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cfg.out_channels, cfg.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(cfg.out_channels),
                nn.ReLU(inplace=True),
            ) for _ in cfg.in_channels
        ])
        self.use_level = cfg.use_level

    def forward(self, feats):
        laterals = [conv(f) for f, conv in zip(feats, self.lateral)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='bilinear', align_corners=False)
        outs = [conv(l) for l, conv in zip(laterals, self.fpn_convs)]
        return outs[self.use_level]


def build_neck(cfg):
    return FPN(cfg)


class SlimModel(nn.Module):
    """精简模型: RGB栅格 → ResNet backbone → FPN neck → MapTRHead 解码"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg.backbone)
        neck_cfg = dict(cfg.neck)
        if not neck_cfg.get('in_channels'):
            neck_cfg['in_channels'] = self.backbone.out_channels
        self.neck = build_neck(AttrDict(neck_cfg))
        self.head = MapTRHead(cfg.map_det_head)
        self.seg_head = MapSegHead(cfg.map_seg_head) if cfg.map_seg_head.get('enabled', True) else None

    def forward(self, raster, return_all_layers=False):
        """
        raster: (B, 3, H, W) 归一化 RGB 栅格图
        Returns:
            cls_scores: (B, num_queries, num_classes) 或 list(每层)
            reg_preds: (B, num_queries, num_points, 2) 或 list(每层)
            seg_pred: (B, num_classes, seg_h, seg_w) 或 None
            bev_feat: (B, bev_embed_dims, bev_h, bev_w)
        """
        feats = self.backbone(raster)
        bev_feat = self.neck(feats)
        cls_scores, reg_preds = self.head(bev_feat, return_all_layers=return_all_layers)
        seg_pred = self.seg_head(bev_feat) if self.seg_head else None
        return cls_scores, reg_preds, seg_pred, bev_feat

    def compute_loss(self, cls_scores, reg_preds, seg_pred, batch):
        """Hungarian 匹配 + 分类/回归损失 + 分割损失"""
        loss_dict = self.head.loss(cls_scores, reg_preds, batch['vectors'])
        if self.seg_head is not None and seg_pred is not None and batch.get('semantic_mask') is not None:
            loss_dict.update(self.seg_head.loss(seg_pred, batch['semantic_mask']))
        return loss_dict
