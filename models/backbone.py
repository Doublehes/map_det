import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    def __init__(self, depth=50, pretrained=True):
        super().__init__()
        if depth == 50:
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet50(weights=weights)
        else:
            raise ValueError(f'Unsupported backbone depth: {depth}')

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # output channels for FPN
        self.out_channels = [512, 1024, 2048]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i, ch in enumerate(in_channels):
            self.lateral_convs.append(
                nn.Conv2d(ch, out_channels, kernel_size=1)
            )

        for i in range(len(in_channels)):
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        extra_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn_convs.append(extra_out)

    def forward(self, feats):
        laterals = [conv(f) for f, conv in zip(feats, self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='bilinear', align_corners=False
            )

        outs = [conv(l) for l, conv in zip(laterals, self.fpn_convs[:-1])]

        outs.append(self.fpn_convs[-1](outs[-1]))

        return outs


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.type == 'resnet':
            self.img_backbone = ResNetBackbone(depth=cfg.depth, pretrained=True)
        else:
            raise ValueError(f'Unsupported backbone type: {cfg.type}')
        
        self.fpn = FPN(
            in_channels=cfg.fpn.in_channels,
            out_channels=cfg.fpn.out_channels,
        )
        self.out_channels = cfg.fpn.out_channels
        self.num_feat_levels = cfg.num_feat_levels

    def forward(self, x):
        feats = self.img_backbone(x)
        feats = self.fpn(feats)
        return feats[:self.num_feat_levels]


