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


class PointPositionalEncoding(nn.Module):
    """从参考点坐标生成正弦位置编码

    输入: reference_points (B, num_q, num_points, 2), normalized [0,1]
    输出: (B, num_q, num_points, embed_dims)
    """

    def __init__(self, num_feats, embed_dims, normalize=True):
        super().__init__()
        self.num_feats = num_feats
        self.embed_dims = embed_dims
        self.normalize = normalize
        self.pos_proj = nn.Linear(num_feats * 2, embed_dims)

    def forward(self, reference_points):
        x_embed = reference_points[..., 0]
        y_embed = reference_points[..., 1]
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=x_embed.device)
        dim_t = 2. ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)  # (B, N, P, num_feats*2)
        return self.pos_proj(pos)  # (B, N, P, embed_dims)


class Gather(nn.Module):
    """聚合点粒度查询回实例查询: [B, num_queries, num_points, embed_dims] → [B, num_queries, embed_dims]

    通过 concat + MLP + LayerNorm 实现 Gather 聚合。
    """

    def __init__(self, embed_dims, num_points, gather_mlp_dims=None, residual='none'):
        super().__init__()
        input_dims = num_points * embed_dims
        hidden_dims = gather_mlp_dims if gather_mlp_dims is not None else embed_dims * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, embed_dims),
            nn.LayerNorm(embed_dims),
        )
        # 残差模式: 'none'=无残差 / 'mean'=点均值残差(A) / 'identity'=实例query残差(B, 对齐原版MapQR)
        self.residual = residual

    def forward(self, x, identity=None):
        # x: (B, num_queries, num_points, embed_dims)
        B, N, P, D = x.shape
        out = self.mlp(x.flatten(2))  # (B, num_queries, embed_dims)
        if self.residual == 'mean':
            out = out + x.mean(dim=2)
        elif self.residual == 'identity':
            assert identity is not None, 'residual=identity 需要传入 scatter 前的实例 query'
            out = out + identity
        return out


class MapTransformerLayer(nn.Module):
    """Decoder layer, 支持原版MapTR和SGQ两种模式。

    原版 (use_sgq=False): self_attn → cross_attn → ffn
    SGQ (use_sgq=True): self_attn → scatter → cross_attn(point-level) → ffn → gather

    batch_first=True 统一内部格式，入口/出口保持 batch_first=False 兼容上层接口。
    """

    def __init__(self, cfg):
        super().__init__()
        self.embed_dims = cfg.embed_dims
        self.num_heads = cfg.num_heads
        self.num_points = cfg.num_points
        self.use_sgq = getattr(cfg, 'use_sgq', False)

        # Self-attention (实例维度, batch_first=True)
        self.self_attn = nn.MultiheadAttention(
            self.embed_dims, self.num_heads, dropout=cfg.dropout, batch_first=True)

        # Deformable cross-attention (batch_first=True)
        self.cross_attn = CustomMSDeformableAttention(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            num_levels=1,
            num_points=self.num_points,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.ffn = FFN(self.embed_dims, cfg.ffn_channels, cfg.dropout)

        self.norm1 = nn.LayerNorm(self.embed_dims)
        self.norm2 = nn.LayerNorm(self.embed_dims)
        self.norm3 = nn.LayerNorm(self.embed_dims)

        # SGQ 特有模块 (仅 use_sgq=True 时创建)
        if self.use_sgq:
            num_feats = self.embed_dims // 2
            self.point_pos_embed = PointPositionalEncoding(num_feats, self.embed_dims)
            gather_mlp_dims = getattr(cfg, 'gather_mlp_dims', self.embed_dims * 2)
            residual = getattr(cfg, 'gather_residual', 'none')
            self.gather = Gather(self.embed_dims, self.num_points, gather_mlp_dims, residual)

    def _convert_kv(self, key, value, bs):
        """将 key/value 从 (H*W, B, D) 转为 (B, H*W, D) 以适配 batch_first=True"""
        if key.dim() == 3 and key.shape[0] != bs:
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        return key, value

    def _forward_original(self, query_bf, key, value, reference_points,
                           spatial_shapes, level_start_index,
                           q_k_mask, k_mask):
        """原版路径: self_attn → cross_attn → ffn (无scatter/gather)"""
        # Self-attention (实例维度)
        identity = query_bf
        q = self.norm1(query_bf)
        q = self.self_attn(q, q, q, key_padding_mask=q_k_mask)[0]
        query_bf = identity + q

        # Cross-attention
        identity = query_bf
        q = self.norm2(query_bf)
        q = self.cross_attn(
            q, key, value, identity=None,
            key_padding_mask=k_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query_bf = identity + q

        # FFN
        identity = query_bf
        q = self.norm3(query_bf)
        q = self.ffn(q)
        query_bf = identity + q

        return query_bf

    def _forward_sgq(self, query_bf, key, value, reference_points,
                      spatial_shapes, level_start_index,
                      q_k_mask, k_mask):
        """SGQ 路径: self_attn → scatter → cross_attn(point-level) → ffn → gather

        1. 实例维度 Self-Attention: 仅实例之间交互
        2. Scatter: 每条实例 query 复制 num_points 份, 叠加参考点位置编码
        3. Deformable Cross-Attention: 点粒度 query 采样 BEV 特征
        4. Gather: 聚合回实例 query
        """
        B, num_q, _ = query_bf.shape
        P = self.num_points

        # --- 1. 实例维度 Self-Attention ---
        identity = query_bf
        q = self.norm1(query_bf)
        q = self.self_attn(q, q, q, key_padding_mask=q_k_mask)[0]
        query_bf = identity + q  # (B, num_q, embed_dims)
        sgq_identity = query_bf  # scatter 前的实例 query, 供 residual='identity' 使用

        # --- 2. Scatter ---
        # (B, num_q, embed_dims) → (B, num_q, num_points, embed_dims)
        query_scattered = query_bf.unsqueeze(2).expand(-1, -1, P, -1)
        # 叠加参考点位置编码
        pos_embed = self.point_pos_embed(reference_points)  # (B, num_q, num_points, embed_dims)
        query_scattered = query_scattered + pos_embed

        # --- 3. Deformable Cross-Attention ---
        # 展平为点粒度 query: (B, num_q*num_points, embed_dims)
        query_cross = query_scattered.flatten(1, 2)
        # 参考点: (B, num_q, num_points, 2) → (B, num_q*num_points, 1, 2) → (B, num_q*num_points, num_points, 2)
        # 每个点query对应自己的参考点, 重复num_points次以满足CustomMSDeformableAttention的num_pts==num_points要求
        ref_points_cross = reference_points.reshape(B, num_q * P, 1, 2).expand(-1, -1, P, -1)  # (B, num_q*num_points, num_points, 2)

        identity = query_cross
        q = self.norm2(query_cross)
        q = self.cross_attn(
            q, key, value, identity=None,
            key_padding_mask=k_mask,
            reference_points=ref_points_cross,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query_cross = identity + q  # (B, num_q*num_points, embed_dims)

        # --- 4. FFN ---
        identity = query_cross
        q = self.norm3(query_cross)
        q = self.ffn(q)
        query_cross = identity + q  # (B, num_q*num_points, embed_dims)

        # --- 5. Gather 聚合 ---
        # (B, num_q*num_points, embed_dims) → (B, num_q, num_points, embed_dims) → (B, num_q, embed_dims)
        query_gathered = query_cross.view(B, num_q, P, -1)  # (B, num_q, num_points, embed_dims)
        query_gathered = self.gather(query_gathered, identity=sgq_identity)  # (B, num_q, embed_dims)

        return query_gathered

    def forward(self, query, key, value, reference_points=None,
                spatial_shapes=None, level_start_index=None,
                query_key_padding_mask=None, key_padding_mask=None):
        """
        入口: query=(num_q_total, B, embed_dims), key/value=(H*W, B, embed_dims)
        出口: output=(num_q_total, B, embed_dims)
        reference_points: (B, num_q_total, num_points, 2), 仅 SGQ 使用
        """
        bs = query.shape[1]
        query_bf = query.permute(1, 0, 2)  # (B, num_q_total, embed_dims)

        # 统一 key/value 到 (B, H*W, embed_dims) 格式
        key, value = self._convert_kv(key, value, bs)

        if self.use_sgq:
            out_bf = self._forward_sgq(
                query_bf, key, value, reference_points,
                spatial_shapes, level_start_index,
                query_key_padding_mask, key_padding_mask)
        else:
            out_bf = self._forward_original(
                query_bf, key, value, reference_points,
                spatial_shapes, level_start_index,
                query_key_padding_mask, key_padding_mask)

        # 转换回 (num_q_total, B, embed_dims) 返回
        return out_bf.permute(1, 0, 2)


class MapTransformerDecoder(nn.Module):
    """Decoder with reference point iteration (1 layer)."""

    def __init__(self, cfg):
        super().__init__()
        self.y_flip = getattr(cfg, 'y_flip', True)
        self.layers = nn.ModuleList([
            MapTransformerLayer(cfg) for _ in range(cfg.num_decoder_layers)
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
            # y-axis reversal: 仅当 BEV 特征 y=1 在顶部时反转 (原始 MapTR)
            tmp = reference_points.clone()
            tmp[..., 1:2] = (1.0 - reference_points[..., 1:2]) if self.y_flip else reference_points[..., 1:2]

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
            else:
                new_reference_points = reference_points

            intermediate.append(output.permute(1, 0, 2))
            intermediate_reference_points.append(new_reference_points)

        return intermediate, intermediate_reference_points
