import torch
import torch.nn as nn
import torch.nn.functional as F


class MapTRHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_queries = cfg.num_queries
        self.num_classes = cfg.num_classes
        self.num_points = cfg.num_points
        self.embed_dims = cfg.embed_dims

        self.classifier = nn.Linear(self.embed_dims, self.num_classes)
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims * 2, self.num_points * 2),
        )

    def forward(self, query):
        cls_scores = self.classifier(query)

        reg_outputs = self.regressor(query)
        reg_outputs = reg_outputs.view(-1, self.num_queries, self.num_points, 2)

        return cls_scores, reg_outputs


class MapSegHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.bev_embed_dims
        self.num_classes = cfg.num_classes

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, 128, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, bev_feat):
        return self.upsample(bev_feat)
