import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bevformer.multi_scale_deformable_attn import multi_scale_deformable_attn_pytorch


def constant_init(module, val, bias=0.):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, distribution='uniform', bias=0.):
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight)
    else:
        nn.init.xavier_normal_(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class CustomMSDeformableAttention(nn.Module):
    """多尺度可变形注意力 (用于 decoder cross-attention)"""

    def __init__(self, embed_dims=512, num_heads=8, num_levels=1,
                 num_points=16, im2col_step=64, dropout=0.1,
                 batch_first=False, use_sampling_offsets=True):
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
        self.use_sampling_offsets = use_sampling_offsets

        if use_sampling_offsets:
            self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        if self.use_sampling_offsets:
            constant_init(self.sampling_offsets, 0.)
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2. * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
                self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
            for i in range(self.num_points):
                grid_init[:, :, i, :] *= i + 1
            self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, key_padding_mask=None, reference_points=None,
                spatial_shapes=None, level_start_index=None):
        if value is None:
            value = query
        if identity is None:
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

        if self.use_sampling_offsets:
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        else:
            sampling_offsets = query.new_zeros(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        _, _, num_pts, _ = reference_points.shape
        reference_points = reference_points[:, :, None, None, :, :]
        sampling_locations = reference_points + \
            (sampling_offsets / offset_normalizer[None, None, None, :, None, :])
        assert list(sampling_locations.shape) == [bs, num_query, self.num_heads, self.num_levels, num_pts, 2]

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
