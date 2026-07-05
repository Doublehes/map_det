import torch
import torch.nn as nn
import torch.nn.functional as F

from .deformable_attn import CustomMSDeformableAttention


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


class MapTransformerLayer(nn.Module):
    """Decoder layer: self_attn → norm → cross_attn → norm → ffn → norm"""

    def __init__(self, cfg):
        super().__init__()
        self.embed_dims = cfg.embed_dims
        self.num_heads = cfg.num_heads

        self.self_attn = nn.MultiheadAttention(
            self.embed_dims, self.num_heads, dropout=cfg.dropout, batch_first=False)

        self.cross_attn = CustomMSDeformableAttention(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            num_levels=1,
            num_points=cfg.num_points,
            dropout=cfg.dropout,
            batch_first=False,
        )

        self.ffn = FFN(self.embed_dims, cfg.ffn_channels, cfg.dropout)

        self.norm1 = nn.LayerNorm(self.embed_dims)
        self.norm2 = nn.LayerNorm(self.embed_dims)
        self.norm3 = nn.LayerNorm(self.embed_dims)

    def forward(self, query, key, value, reference_points=None,
                spatial_shapes=None, level_start_index=None,
                query_key_padding_mask=None, key_padding_mask=None):
        # self-attention
        identity = query
        q = self.norm1(query)
        q = self.self_attn(q, q, q, key_padding_mask=query_key_padding_mask)[0]
        query = identity + q

        # cross-attention
        identity = query
        q = self.norm2(query)
        q = self.cross_attn(
            q, key, value, identity=None,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = identity + q

        # FFN
        identity = query
        q = self.norm3(query)
        q = self.ffn(q)
        query = identity + q

        return query


class MapTransformerDecoderNew(nn.Module):
    """Decoder with reference point iteration (1 layer)."""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([
            MapTransformerLayer(cfg) for _ in range(cfg.decoder_num_layers)
        ])

    def forward(self, query, key, value, reference_points,
                spatial_shapes, level_start_index,
                reg_branches=None,
                query_key_padding_mask=None,
                key_padding_mask=None):
        """
        query: (num_q, bs, embed_dims)
        key/value: (H*W, bs, embed_dims)
        reference_points: (bs, num_q, num_pts, 2)  normalized [0,1]
        reg_branches: list of nn.Module, one per layer
        """
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            # y-axis reversal
            tmp = reference_points.clone()
            tmp[..., 1:2] = 1.0 - reference_points[..., 1:2]

            output = layer(
                output, key, value,
                reference_points=tmp,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
            )

            if reg_branches is not None:
                reg_points = reg_branches[lid](output.permute(1, 0, 2))
                bs, num_q, np2 = reg_points.shape
                reg_points = reg_points.view(bs, num_q, np2 // 2, 2)
                new_reference_points = reg_points.sigmoid()
                reference_points = new_reference_points.clone().detach()

            intermediate.append(output.permute(1, 0, 2))
            intermediate_reference_points.append(new_reference_points)

        return intermediate, intermediate_reference_points
