import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dims, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)

    def forward(self, query, key=None, value=None, key_padding_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
        return self.attn(query, key, value, key_padding_mask=key_padding_mask)[0]


class FFN(nn.Module):
    def __init__(self, embed_dims, ffn_channels, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dims, ffn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_channels, embed_dims),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DeformableCrossAttention(nn.Module):
    """Deformable cross-attention: query → sampling offsets → grid_sample

    代替标准 MHA, 每个 query 只采样 num_points 个位置 (而非全部 BEV token).
    计算量: O(num_queries × num_points), 与 BEV 分辨率无关.
    """
    def __init__(self, embed_dims=512, num_points=16, num_queries=32):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_points = num_points

        # 每个 query 的参考点 (归一化坐标 [0,1])
        self.ref_points = nn.Embedding(num_queries, 2)
        nn.init.uniform_(self.ref_points.weight, 0.0, 1.0)

        # 采样偏移 + 注意力权重
        self.offset_proj = nn.Linear(embed_dims, num_points * 2)
        self.weight_proj = nn.Linear(embed_dims, num_points)
        self.value_proj = nn.Conv2d(embed_dims, embed_dims, 1)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, bev_feat_2d):
        """
        query: (B, N_q, C)        decoder query
        bev_feat_2d: (B, C, H, W)  BEV特征图 (保持2D空间结构)
        """
        B, N_q, C = query.shape
        _, _, H, W = bev_feat_2d.shape

        # 1. Value 投影
        value = self.value_proj(bev_feat_2d)

        # 2. 参考点 + 偏移 → 采样位置
        ref = self.ref_points.weight.view(1, N_q, 1, 2)
        offset = self.offset_proj(query).view(B, N_q, self.num_points, 2)
        offset_norm = offset / torch.tensor([W, H], device=query.device).view(1, 1, 1, 2)
        sampling_loc = ref + offset_norm

        # 3. 注意力权重 (softmax over points)
        weights = self.weight_proj(query).softmax(dim=-1)

        # 4. grid_sample 双线性插值采样
        grid = sampling_loc.reshape(B, N_q * self.num_points, 1, 2) * 2 - 1
        sampled = F.grid_sample(
            value, grid,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        sampled = sampled.squeeze(-1)
        sampled = sampled.view(B, C, N_q, self.num_points)
        sampled = sampled.permute(0, 2, 3, 1)

        # 5. 加权求和
        out = (sampled * weights.unsqueeze(-1)).sum(dim=2)
        return self.output_proj(out)


class MapTransformerLayer(nn.Module):
    def __init__(self, embed_dims, num_heads=8, ffn_channels=1024, dropout=0.1, num_queries=32, num_points=16):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dims, num_heads, dropout)
        self.cross_attn = DeformableCrossAttention(embed_dims, num_points, num_queries)
        self.ffn = FFN(embed_dims, ffn_channels, dropout)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, bev_feat_2d, query_pos=None):
        if query_pos is not None:
            q = query + query_pos
        else:
            q = query

        q = self.norm1(query + self.dropout(self.self_attn(q)))
        q = self.norm2(q + self.dropout(self.cross_attn(q, bev_feat_2d)))
        q = self.norm3(q + self.dropout(self.ffn(q)))
        return q


class MapTransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dims = cfg.embed_dims
        self.bev_embed_dims = cfg.bev_embed_dims
        self.num_layers = cfg.decoder_num_layers
        self.num_queries = cfg.num_queries

        self.query_embed = nn.Embedding(self.num_queries, self.embed_dims)
        self.query_pos = nn.Parameter(torch.randn(1, self.num_queries, self.embed_dims) * 0.02)

        self.input_proj = nn.Conv2d(self.bev_embed_dims, self.embed_dims, kernel_size=1)

        self.layers = nn.ModuleList([
            MapTransformerLayer(
                embed_dims=self.embed_dims,
                num_heads=cfg.num_heads,
                ffn_channels=cfg.ffn_channels,
                dropout=cfg.dropout,
                num_queries=cfg.num_queries,
                num_points=cfg.num_points,
            ) for _ in range(self.num_layers)
        ])

    def forward(self, bev_feat_2d):
        """
        bev_feat_2d: (B, 256, 40, 80)  BEV特征图 (保持2D)
        """
        B = bev_feat_2d.shape[0]
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos.expand(B, -1, -1)

        memory = self.input_proj(bev_feat_2d)

        for layer in self.layers:
            query = layer(query, memory, query_pos)

        return query
