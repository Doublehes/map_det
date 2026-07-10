import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .spatial_cross_attention import SpatialCrossAttention, MSDeformableAttention3D


def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d',
                         bs=1, device='cuda', dtype=torch.float32):
    """生成 BEV 网格的归一化 [0,1] 参考点

    3D: (bs, num_points_in_pillar, H*W, 3) — 用于空间交叉注意力
    2D: (bs, H*W, 1, 2) — 用于自注意力
    """
    if dim == '3d':
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar,
                            dtype=dtype, device=device)
        zs = zs.view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        xs = xs.view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(H - 0.5, 0.5, H, dtype=dtype, device=device)
        ys = ys.view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d  # (bs, num_points_in_pillar, H*W, 3)
    elif dim == '2d':
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(H - 0.5, 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            indexing='ij')
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d  # (bs, H*W, 1, 2)
    else:
        raise ValueError(f'Unknown dim: {dim}')


def point_sampling(ref_3d, pc_range, intrinsics, extrinsics, img_h, img_w):
    """3D参考点 → 相机投影 → 归一化图像坐标 [0,1]

    Args:
        ref_3d: (B, D, num_query, 3) 归一化 [0,1] 3D点
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        intrinsics: (B, num_cams, 3, 3)
        extrinsics: (B, num_cams, 4, 4)
        img_h, img_w: 图像尺寸

    Returns:
        reference_points_cam: (num_cams, B, num_query, D, 2) 归一化 [0,1]
        bev_mask: (num_cams, B, num_query, D)
    """
    ref = ref_3d.clone()
    B, D, num_query = ref.shape[:3]
    num_cams = intrinsics.size(1)
    device = ref.device

    # 反归一化: [0,1] → 真实坐标
    ref[..., 0] = ref[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    ref[..., 1] = ref[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    ref[..., 2] = ref[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    # 齐次坐标: (B, D, num_query, 4)
    ref_h = torch.cat([ref, torch.ones_like(ref[..., :1])], -1)

    # 逐相机投影 (B,num_cams 较小, 循环性能可接受)
    ref_points_cam = torch.zeros(num_cams, B, num_query, D, 2, device=device, dtype=ref.dtype)
    bev_mask = torch.zeros(num_cams, B, num_query, D, device=device, dtype=torch.bool)

    eps = 1e-5
    for cam in range(num_cams):
        for b in range(B):
            E = extrinsics[b, cam]                     # (4, 4)
            pts_cam = (E @ ref_h[b].unsqueeze(-1)).squeeze(-1)  # (D, num_query, 4)
            z = pts_cam[..., 2:3]                      # (D, num_query, 1)
            valid = (z > eps).squeeze(-1)

            uv_norm = pts_cam[..., 0:2] / z.clamp(min=eps)  # (D, num_query, 2)
            K = intrinsics[b, cam]                     # (3, 3)
            uv_h = torch.cat([uv_norm, torch.ones_like(uv_norm[..., :1])], -1)  # (D, num_query, 3)
            uv_pix = (K @ uv_h.unsqueeze(-1)).squeeze(-1)  # (D, num_query, 3)

            u = uv_pix[..., 0] / img_w
            v = uv_pix[..., 1] / img_h
            valid = valid & (u > 0) & (u < 1) & (v > 0) & (v < 1)

            ref_points_cam[cam, b, :, :, 0] = u.T      # (num_query, D) from (D, num_query)
            ref_points_cam[cam, b, :, :, 1] = v.T
            bev_mask[cam, b] = valid.T

    return ref_points_cam, bev_mask


class BEVFormerLayer(nn.Module):
    """BEVFormer 单层编码器: deformable self-attn + cross-attn + FFN (pre-norm)"""

    def __init__(self, embed_dims, num_heads, num_levels, num_points, dropout, ffn_channels, num_cams):
        super().__init__()
        self.self_attn = MSDeformableAttention3D(
            embed_dims=embed_dims, num_heads=num_heads,
            num_levels=1, num_points=num_points,
            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)

        ms_deform_attn = MSDeformableAttention3D(
            embed_dims=embed_dims, num_heads=num_heads,
            num_levels=num_levels, num_points=num_points,
            dropout=dropout, batch_first=True)
        self.cross_attn = SpatialCrossAttention(
            embed_dims=embed_dims, num_cams=num_cams,
            dropout=dropout, batch_first=True,
            deformable_attention=ms_deform_attn)
        self.norm2 = nn.LayerNorm(embed_dims)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_channels, embed_dims),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, query, key, value, reference_points_cam=None,
                bev_mask=None, spatial_shapes=None, level_start_index=None,
                ref_2d=None, bev_pos=None, bev_h=None, bev_w=None):
        # Self-attention (deformable)
        shortcut = query
        q = self.norm1(query)
        if bev_pos is not None:
            q = q + bev_pos
        attn_shapes = torch.tensor([[bev_h, bev_w]], device=query.device, dtype=torch.long)
        attn_start = torch.tensor([0], device=query.device, dtype=torch.long)
        query = shortcut + self.self_attn(
            q, q, identity=None, skip_residual=True,
            reference_points=ref_2d,
            spatial_shapes=attn_shapes,
            level_start_index=attn_start,
        )

        # Cross-attention
        shortcut = query
        q = self.norm2(query)
        query = shortcut + self.cross_attn(
            q, key, value,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        # FFN
        shortcut = query
        q = self.norm3(query)
        query = shortcut + self.ffn(q)

        return query


class BEVFormerEncoder(nn.Module):
    """BEVFormer 编码器: N 层 BEVFormerLayer + 参考点生成 + 相机投影"""

    def __init__(self, cfg):
        super().__init__()
        self.pc_range = cfg.pc_range
        self.bev_h = cfg.bev_h
        self.bev_w = cfg.bev_w
        self.num_points_in_pillar = cfg.num_points_in_pillar
        self.num_cams = cfg.num_cams
        self.embed_dims = cfg.bev_embed_dims
        self.num_feat_levels = cfg.num_feat_levels
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w

        self.bev_embed = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.bev_pos = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feat_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))

        self.layers = nn.ModuleList([
            BEVFormerLayer(
                embed_dims=self.embed_dims,
                num_heads=cfg.num_heads,
                num_levels=self.num_feat_levels,
                num_points=cfg.num_sampling_points,
                dropout=cfg.dropout,
                ffn_channels=cfg.ffn_channels,
                num_cams=self.num_cams) for _ in range(cfg.num_layers)
        ])

        self.input_proj = nn.Sequential(
            nn.Conv2d(cfg.fpn_out_channels, self.embed_dims, kernel_size=1),
            nn.BatchNorm2d(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.debug_dir = None

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)
        nn.init.normal_(self.bev_pos.weight)

    def forward(self, img_feats, intrinsics, extrinsics, imgs=None):
        """
        img_feats: list of (B*num_cams, C, H_i, W_i)  — 多尺度FPN特征
        intrinsics: (B, num_cams, 3, 3)
        extrinsics: (B, num_cams, 4, 4)
        imgs: (B, num_cams, 3, H, W)  — 原始图像 (用于debug可视化)

        Returns:
            bev_feat: (B, C, bev_h, bev_w)
        """
        img_feats = [self.input_proj(f) for f in img_feats]
        
        B = img_feats[0].shape[0] // self.num_cams
        device = img_feats[0].device
        dtype = img_feats[0].dtype

        # 1. 特征准备: reshape + flatten + add embeddings
        feat_flatten = []
        spatial_shapes = []
        for level, feat in enumerate(img_feats):
            feat = feat.view(B, self.num_cams, -1, feat.shape[-2], feat.shape[-1])
            # feat: (B, num_cams, C, H, W)
            bs, ncam, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # feat: (num_cams, B, H*W, C)
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None, level:level+1, :].to(dtype)
            feat_flatten.append(feat)

        # 将所有 level concat: (num_cams, B, total_HW, C)
        feat_flatten = torch.cat(feat_flatten, dim=2)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)
        # (num_cams, total_HW, B, embed_dims)

        # 2. BEV queries
        bev_queries = self.bev_embed.weight.unsqueeze(1).repeat(1, B, 1).to(dtype)
        # (num_query, B, embed_dims)

        # 3. 生成3D参考点并投影到相机
        Z = self.pc_range[5] - self.pc_range[2]
        ref_3d = get_reference_points(
            self.bev_h, self.bev_w, Z, self.num_points_in_pillar,
            dim='3d', bs=B, device=device, dtype=dtype)
        # (B, D, num_query, 3)

        reference_points_cam, bev_mask = point_sampling(
            ref_3d, self.pc_range, intrinsics, extrinsics,
            self.img_h, self.img_w)
        # (num_cams, B, num_query, D, 2), (num_cams, B, num_query, D)

        if self.debug_dir is not None:
            self._debug_plot_ref_points(reference_points_cam, bev_mask, B, imgs=imgs)

        # 4. BEV 位置编码 + 2D 参考点
        bev_pos_embed = self.bev_pos.weight.unsqueeze(0).expand(B, -1, -1).to(dtype)
        ref_2d = get_reference_points(
            self.bev_h, self.bev_w, dim='2d', bs=B, device=device, dtype=dtype)

        # 5. 编码层
        bev_query = bev_queries.permute(1, 0, 2).contiguous()
        # (B, num_query, embed_dims)

        for layer in self.layers:
            bev_query = layer(
                bev_query, feat_flatten, feat_flatten,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                ref_2d=ref_2d,
                bev_pos=bev_pos_embed,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
            )

        # 6. reshape: (B, num_query, embed_dims) → (B, embed_dims, bev_h, bev_w)
        bev_feat = bev_query.permute(0, 2, 1).view(B, -1, self.bev_h, self.bev_w)

        return bev_feat

    def _debug_plot_ref_points(self, ref_pts_cam, bev_mask, B, imgs=None):
        """将每层BEV参考点投影画在原图上, 每层pillar单独一张图"""
        import cv2
        import numpy as np
        from pathlib import Path

        debug_dir = Path(self.debug_dir) / 'bev_proj'
        debug_dir.mkdir(parents=True, exist_ok=True)

        num_cams, _, num_q, D, _ = ref_pts_cam.shape
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255)]

        # 图像反归一化用
        mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)

        for d in range(D):
            for cam in range(num_cams):
                if imgs is not None:
                    img_t = imgs[0, cam].cpu().numpy().transpose(1, 2, 0)
                    canvas = np.ascontiguousarray(
                        np.clip(img_t + mean, 0, 255).astype(np.uint8))
                else:
                    canvas = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255

                mask = bev_mask[cam, 0, :, d]
                if mask.any():
                    uv = ref_pts_cam[cam, 0, mask, d]
                    px = (uv[:, 0] * (self.img_w - 1)).cpu().numpy().astype(int)
                    py = (uv[:, 1] * (self.img_h - 1)).cpu().numpy().astype(int)
                    for x, y in zip(px, py):
                        cv2.circle(canvas, (x, y), 1, colors[d % len(colors)], -1)

                out = debug_dir / f'pillar{d}_cam{cam}.png'
                cv2.imwrite(str(out), canvas)
                print(f'[debug] 保存BEV投影: {out}')
