import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVFormerEncoder(nn.Module):
    """将多相机图像特征转换为BEV特征图

    流程: BEV网格均匀采样3D点 → 投影到各相机 → 采样图像特征 → 跨相机融合 → 柱高度聚合
    输出: (B, 256, 40, 80) 的BEV特征图
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bev_h = cfg.bev_h          # 40
        self.bev_w = cfg.bev_w          # 80
        self.bev_embed_dims = cfg.bev_embed_dims  # 256
        self.num_points_in_pillar = cfg.num_points_in_pillar  # 4个高度层
        self.pc_range = cfg.pc_range    # [-10, -10, -3, 30, 10, 5]
        self.num_cams = cfg.num_cams    # 6

        # 可学习的BEV位置编码, 每个网格一个256维向量
        self.bev_embed = nn.Embedding(self.bev_h * self.bev_w, self.bev_embed_dims)

        # 3D位置编码器 (目前未使用, 预留)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.bev_embed_dims),
            nn.LayerNorm(self.bev_embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.bev_embed_dims, self.bev_embed_dims),
        )

        # 输出投影: 位置编码(256) + 图像特征(256) → 512 → 256
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.bev_embed_dims * 2, self.bev_embed_dims, kernel_size=1),
            nn.BatchNorm2d(self.bev_embed_dims),
            nn.ReLU(inplace=True),
        )

        self._init_bev_reference_points()

    def _init_bev_reference_points(self):
        """生成BEV网格的3D参考点 (40×80网格 × 4个高度 = 12800点)"""
        # X方向: pc_range[0](-10) ~ pc_range[3](30), 80个网格点中心
        xs = torch.linspace(
            self.pc_range[0], self.pc_range[3], self.bev_w + 1
        )[:-1] + (self.pc_range[3] - self.pc_range[0]) / (self.bev_w * 2)
        # Y方向: pc_range[1](-10) ~ pc_range[4](10), 40个网格点中心
        ys = torch.linspace(
            self.pc_range[1], self.pc_range[4], self.bev_h + 1
        )[:-1] + (self.pc_range[4] - self.pc_range[1]) / (self.bev_h * 2)
        # Z方向: pc_range[2](-3) ~ pc_range[5](5), 4个高度
        z_steps = torch.linspace(
            self.pc_range[2], self.pc_range[5], self.num_points_in_pillar
        )

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)
        grid_xy = grid_xy.unsqueeze(-2).expand(-1, -1, self.num_points_in_pillar, -1)
        z = z_steps.view(1, 1, -1, 1).expand_as(grid_xy[..., :1])
        ref_3d = torch.cat([grid_xy, z], dim=-1)  # (40, 80, 4, 3)
        self.register_buffer('ref_3d', ref_3d)

    def forward(self, img_feats, intrinsics, extrinsics):
        """
        img_feats: list of (B*6, 256, H_feat, W_feat)  多尺度特征图 (当前仅用[0])
        intrinsics: (B, 6, 3, 3)   相机内参
        extrinsics: (B, 6, 4, 4)   相机外参 [R|t], world→camera
        """
        B = img_feats[0].shape[0] // self.num_cams
        device = img_feats[0].device
        NC, C, feat_h, feat_w = img_feats[0].shape
        img_h, img_w = 176, 320

        # 1. 展平所有BEV参考点: (40,80,4,3) → (12800, 3)
        ref_3d = self.ref_3d.to(device)
        num_bev = self.bev_h * self.bev_w
        ref_3d_flat = ref_3d.reshape(1, num_bev * self.num_points_in_pillar, 3)

        # 2. 转齐次坐标: (x,y,z) → (x,y,z,1)
        num_pts = ref_3d_flat.shape[1]
        ones = torch.ones(1, num_pts, 1, device=device)
        ref_homo = torch.cat([ref_3d_flat, ones], dim=-1)
        ref_homo_5d = ref_homo.unsqueeze(1).unsqueeze(-1)

        # 3. 外参投影: world→camera, 得到每个点在每个相机坐标系下的 (x,y,z)
        points_cam_5d = extrinsics.unsqueeze(2) @ ref_homo_5d
        points_cam = points_cam_5d.squeeze(-1)
        x, y, z = points_cam[..., 0], points_cam[..., 1], points_cam[..., 2]

        # 4. 内参投影: 相机坐标 → 像素坐标 (u,v)
        K = intrinsics.unsqueeze(2)
        u = K[..., 0, 0] * x / (z + 1e-6) + K[..., 0, 2]
        v = K[..., 1, 1] * y / (z + 1e-6) + K[..., 1, 2]

        # 5. 有效掩码: 点在相机前方且投影在图像内
        valid_mask = (z > 1e-6)
        valid_mask = valid_mask & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        # dummy相机外参使z为负, valid_mask全False

        # 6. 归一化像素坐标到 [-1, 1], 用于grid_sample
        u_feat = u / max(feat_w, 1) * 2 - 1
        v_feat = v / max(feat_h, 1) * 2 - 1

        # 7. BEV位置编码: (3200, 256) → (B, 3200, 256)
        bev_queries = self.bev_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # 8. 特征采样: grid_sample在特征图上插值
        img_feat = img_feats[0]  # 当前仅用第1级特征图
        img_feat = img_feat.view(B * self.num_cams, -1, feat_h, feat_w)

        grid = torch.stack([u_feat, v_feat], dim=-1)
        grid_s = grid.reshape(B * self.num_cams, 1, num_bev * self.num_points_in_pillar, 2)
        sampled = F.grid_sample(
            img_feat, grid_s,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        # (B*6, 256, 1, 12800) → (B, 6, 12800, 256)
        sampled = sampled.view(B, self.num_cams, -1, num_bev * self.num_points_in_pillar)
        sampled = sampled.permute(0, 1, 3, 2)

        # 9. 跨相机融合: 仅平均有效相机 (dummy相机valid=0, 不贡献)
        m = valid_mask.float().unsqueeze(-1)
        sampled = sampled * m
        valid = m.sum(dim=1).clamp(min=1)
        sampled = sampled.sum(dim=1) / valid

        # 10. 柱高度聚合: 4个高度均匀平均
        hw = torch.ones(self.num_points_in_pillar, device=device) / self.num_points_in_pillar
        sampled = sampled.view(B, num_bev, self.num_points_in_pillar, -1)
        sampled = (sampled * hw.view(1, 1, -1, 1)).sum(dim=2)

        # 11. 输出: 位置编码 + 图像特征 → Conv1×1 → (B, 256, 40, 80)
        bev_feat = torch.cat([bev_queries, sampled], dim=-1)
        bev_feat = bev_feat.transpose(1, 2).view(B, -1, self.bev_h, self.bev_w)
        bev_feat = self.output_proj(bev_feat)

        return bev_feat


class GridMask(nn.Module):
    def __init__(self, use_grid_mask=True):
        super().__init__()
        self.use_grid_mask = use_grid_mask

    def forward(self, x):
        if not self.use_grid_mask or not self.training:
            return x
        B, C, H, W = x.shape
        h_mul = 8
        w_mul = 8
        mask_h = (H + h_mul - 1) // h_mul
        mask_w = (W + w_mul - 1) // w_mul
        grid = torch.rand(B, 1, mask_h, mask_w, device=x.device)
        grid = F.interpolate(grid, size=(H, W), mode='bilinear', align_corners=False)
        grid = (grid > 0.5).float()
        return x * grid
