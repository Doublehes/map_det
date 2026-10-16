"""model_slim2 模型: ResNet18(5入通道) + FPN + Slim2Head(MapTRHead 子类)

纯 2D 任务: 输入分割图 (one-hot), 输出每条线 16 个 (u,v) 归一化坐标点。
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.default import AttrDict

from .head import Slim2Head

# 各 depth 的 c2/c3/c4 输出通道 (layer1/2/3)
_RESNET_CHANNELS = {
    18: [64, 128, 256], 34: [64, 128, 256],
    50: [256, 512, 1024], 101: [256, 512, 1024], 152: [256, 512, 1024],
}


class ResNetBackbone(nn.Module):
    """ResNet backbone: 输入通道可配, 取 stem + layer1/2/3, 输出 [c2, c3, c4]"""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.depth in _RESNET_CHANNELS, f'不支持的 depth: {cfg.depth}'
        weights = 'IMAGENET1K_V1' if cfg.get('pretrained', False) else None
        resnet = getattr(torchvision.models, f'resnet{cfg.depth}')(weights=weights)

        in_ch = cfg.get('in_channels', 3)
        if in_ch != 3:
            old = resnet.conv1.weight.data                            # (64, 3, 7, 7)
            new = old.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1)
            resnet.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
            resnet.conv1.weight.data = new

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.out_channels = _RESNET_CHANNELS[cfg.depth]

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)    # stride4
        c3 = self.layer2(c2)   # stride8
        c4 = self.layer3(c3)   # stride16
        return [c2, c3, c4]


class FPN(nn.Module):
    """FPN neck: 多级 top-down 融合, 返回最细层 (P2 stride4)"""

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
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode='bilinear', align_corners=False)
        outs = [conv(l) for l, conv in zip(laterals, self.fpn_convs)]
        return outs[self.use_level]


class Slim2Model(nn.Module):
    """分割图 → ResNet → FPN → Slim2Head"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = ResNetBackbone(cfg.backbone)
        neck_cfg = dict(cfg.neck)
        if not neck_cfg.get('in_channels'):
            neck_cfg['in_channels'] = self.backbone.out_channels
        self.neck = FPN(AttrDict(neck_cfg))
        head_cfg = AttrDict(dict(cfg.map_det_head))
        head_cfg['bev_embed_dims'] = cfg.neck.out_channels
        self.head = Slim2Head(head_cfg)

    def forward(self, x, return_all_layers=False):
        """x: (B, in_channels, H, W) one-hot 分割输入
        Returns:
            cls_scores: (B, N, num_classes) 或 list(每层)
            reg_preds:  (B, N, num_points, 2) 归一化 (u,v) 或 list(每层)
        """
        bev_feat = self.neck(self.backbone(x))
        cls_scores, reg_preds, _, _ = self.head(bev_feat, return_all_layers=return_all_layers)
        return cls_scores, reg_preds

    def compute_loss(self, cls_scores, reg_preds, batch):
        """自监督损失: 由 head 重写的 loss 计算"""
        return self.head.loss(cls_scores, reg_preds, batch['cost'], batch['dist_field'],
                              batch.get('keypoints'), batch.get('keypoints_valid'))
