import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_scale_deformable_attn import multi_scale_deformable_attn_pytorch


def _constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _xavier_init(module, distribution='uniform', bias=0):
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight)
    else:
        nn.init.xavier_normal_(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class MSDeformableAttention3D(nn.Module):
    """3D多尺度可变形注意力 (BEVFormer核心)"""

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=8,
                 im2col_step=64, dropout=0.1, batch_first=True):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims({embed_dims}) must be divisible by num_heads({num_heads})')
        dim_per_head = embed_dims // num_heads

        def _is_power_of_2(n):
            return (isinstance(n, int) and n > 0 and (n & (n - 1) == 0))
        if not _is_power_of_2(dim_per_head):
            warnings.warn("embed_dims should make dim_per_head a power of 2 for CUDA efficiency")

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.batch_first = batch_first

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        _constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        _constant_init(self.attention_weights, val=0., bias=0.)
        _xavier_init(self.value_proj, distribution='uniform', bias=0.)
        _xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self, query, value=None, identity=None, query_pos=None,
                key_padding_mask=None, reference_points=None,
                spatial_shapes=None, level_start_index=None,
                skip_residual=False):
        if value is None:
            value = query
        if identity is None and not skip_residual:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels,
                num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
        else:
            raise ValueError(f'Last dim of reference_points must be 2, got {reference_points.shape[-1]}')

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        output = self.dropout(output)
        if not skip_residual and identity is not None:
            output = output + identity
        return output


class SpatialCrossAttention(nn.Module):
    """跨空间交叉注意力: 每个相机只处理其可见的 BEV query (稀疏优化)"""

    def __init__(self, embed_dims=256, num_cams=6, dropout=0.1, batch_first=True,
                 deformable_attention=None, impl='vectorized', len_bucket=0,
                 cam_grouping=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first
        self.impl = impl
        self.len_bucket = int(len_bucket) if len_bucket else 0
        self.cam_grouping = bool(cam_grouping)

        if deformable_attention is None:
            self.deformable_attention = MSDeformableAttention3D(
                embed_dims=embed_dims, batch_first=batch_first)
        else:
            self.deformable_attention = deformable_attention
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weight()

    def init_weight(self):
        _xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self, query, key, value, reference_points_cam=None,
                bev_mask=None, spatial_shapes=None, level_start_index=None):
        if self.impl == 'legacy':
            return self._forward_legacy(query, key, value, reference_points_cam,
                                        bev_mask, spatial_shapes, level_start_index)
        if self.cam_grouping:
            return self._forward_vectorized(query, key, value, reference_points_cam,
                                            bev_mask, spatial_shapes, level_start_index)
        return self._forward_vectorized_padded(query, key, value, reference_points_cam,
                                               bev_mask, spatial_shapes, level_start_index)

    def _get_indexes(self, bev_mask):
        indexes = []
        for mask_per_img in bev_mask:
            indexes.append(mask_per_img[0].sum(-1).nonzero().squeeze(-1))
        return indexes

    def _forward_vectorized(self, query, key, value, reference_points_cam=None,
                            bev_mask=None, spatial_shapes=None, level_start_index=None):
        """向量化 + 按相机可见 query 数分组, 避免 padding 到 max_len 的浪费"""
        bs, num_query, _ = query.size()
        if reference_points_cam is None:
            raise ValueError('reference_points_cam must be provided')

        D = reference_points_cam.size(3)
        indexes = self._get_indexes(bev_mask)

        groups = {}
        for cam, idx in enumerate(indexes):
            n = int(idx.numel())
            if n == 0:
                continue
            groups.setdefault(n, []).append(cam)
        if not groups:
            return query

        slots = torch.zeros_like(query)
        num_cams, l, bs_k, embed_dims = key.shape

        for n, cams in groups.items():
            n_pad = n
            if self.len_bucket > 0:
                n_pad = ((n + self.len_bucket - 1) // self.len_bucket) * self.len_bucket
            g = len(cams)

            queries_group = query.new_zeros([bs, g, n_pad, self.embed_dims])
            ref_group = reference_points_cam.new_zeros([bs, g, n_pad, D, 2])
            for j, cam in enumerate(cams):
                idx = indexes[cam]
                queries_group[:, j, :n] = query[:, idx]
                ref_group[:, j, :n] = reference_points_cam[cam][:, idx]

            key_g = key[cams].permute(2, 0, 1, 3).reshape(bs * g, l, self.embed_dims)
            value_g = value[cams].permute(2, 0, 1, 3).reshape(bs * g, l, self.embed_dims)

            q_out = self.deformable_attention(
                query=queries_group.view(bs * g, n_pad, self.embed_dims),
                value=value_g,
                reference_points=ref_group.view(bs * g, n_pad, D, 2),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            ).view(bs, g, n_pad, self.embed_dims)

            for j, cam in enumerate(cams):
                idx = indexes[cam]
                slots[:, idx] += q_out[:, j, :n]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        return self.dropout(slots) + query

    def _forward_vectorized_padded(self, query, key, value, reference_points_cam=None,
                                   bev_mask=None, spatial_shapes=None, level_start_index=None):
        """向量化 rebatch/scatter: 所有相机 padding 到 max_len (旧实现, 保留用于 A/B)"""
        bs, num_query, _ = query.size()
        if reference_points_cam is None:
            raise ValueError('reference_points_cam must be provided')

        D = reference_points_cam.size(3)
        indexes = self._get_indexes(bev_mask)

        lens = []
        for idx in indexes:
            n = idx.numel()
            if self.len_bucket > 0:
                n = ((n + self.len_bucket - 1) // self.len_bucket) * self.len_bucket
            lens.append(n)
        max_len = max(lens) if lens else 0
        if max_len == 0:
            return query

        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])

        for cam, idx in enumerate(indexes):
            n = idx.numel()
            if n == 0:
                continue
            # index_select, 一次处理整个 batch
            queries_rebatch[:, cam, :n] = query[:, idx]
            reference_points_rebatch[:, cam, :n] = reference_points_cam[cam][:, idx]

        num_cams, l, bs_k, embed_dims = key.shape
        key_r = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value_r = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            value=value_r,
            reference_points=reference_points_rebatch.view(
                bs * self.num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        slots = torch.zeros_like(query)
        for cam, idx in enumerate(indexes):
            n = idx.numel()
            if n == 0:
                continue
            slots[:, idx] += queries[:, cam, :n]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        return self.dropout(slots) + query

    def _forward_legacy(self, query, key, value, reference_points_cam=None,
                        bev_mask=None, spatial_shapes=None, level_start_index=None):
        """
        query: (bs, num_query, embed_dims)  batch_first=True
        key, value: (num_cam, num_value, bs, embed_dims)
        reference_points_cam: (num_cam, bs, num_query, D, 2)
        bev_mask: (num_cam, bs, num_query, D)
        """
        bs, num_query, _ = query.size()

        if reference_points_cam is None:
            raise ValueError('reference_points_cam must be provided')

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes]) if indexes else 0

        slots = torch.zeros_like(query)

        if max_len == 0:
            return query

        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])

        for j in range(bs):
            for i in range(self.num_cams):
                idx = indexes[i]
                queries_rebatch[j, i, :len(idx)] = query[j, idx]
                reference_points_rebatch[j, i, :len(idx)] = reference_points_cam[i, j, idx]

        num_cams, l, bs_k, embed_dims = key.shape
        key_r = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value_r = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            value=value_r,
            reference_points=reference_points_rebatch.view(
                bs * self.num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        for j in range(bs):
            for i, idx in enumerate(indexes):
                slots[j, idx] += queries[j, i, :len(idx)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + query
