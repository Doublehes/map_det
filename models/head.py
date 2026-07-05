import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import MapTransformerDecoderNew


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
        self.transformer = MapTransformerDecoderNew(cfg)

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
        num_layers = cfg.decoder_num_layers
        self.cls_branches = nn.ModuleList([cls_branch for _ in range(num_layers)])
        self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_layers)])

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

    def forward(self, bev_feat):
        """
        bev_feat: (B, bev_embed_dims, H, W)
        Returns:
            cls_scores: (B, num_queries, num_classes)
            reg_preds: (B, num_queries, num_points, 2)
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

        # 5. Classification + regression from last layer
        cls_scores = self.cls_branches[-1](inter_queries[-1])   # (B, num_q, num_classes)
        reg_preds = inter_refs[-1]                                # (B, num_q, num_pts, 2)

        return cls_scores, reg_preds


class MapSegHead(nn.Module):
    """分割头: 上采样 BEV 特征到分割图"""

    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.bev_embed_dims
        self.embed_dims = cfg.bev_embed_dims
        self.canvas_size = cfg.canvas_size
        self.bev_h = cfg.bev_h
        self.bev_w = cfg.bev_w

        self.conv_in = nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dims, 128, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self._init_bias()

    def _init_bias(self):
        nn.init.constant_(self.upsample[-1].bias, math.log(0.01 / 0.99))

    def forward(self, bev_feat):
        x = self.relu(self.conv_in(bev_feat))
        return self.upsample(x)
