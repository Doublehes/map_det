import torch
import torch.nn as nn
import torch.nn.functional as F

from .bevformer.encoder import BEVFormerEncoder


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
