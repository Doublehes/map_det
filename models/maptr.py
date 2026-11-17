import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .bev_encoder import BEVFormerEncoder, GridMask
from .head import MapTRHead, MapSegHead, BEVHeatMapHead


class BEVTransform:
    """BEV 特征变换: 翻转 + 旋转 + 平移, 统一接口

    归一化坐标 [-1,1] 到世界坐标:
        x_world = x_ndc * half_x + x_center
        y_world = -y_ndc * half_y + y_center

    theta 矩阵由三步推导: (x_ndc_out, y_ndc_out) → 世界 → 反向变换 → (x_ndc_in, y_ndc_in)

        x_in = c*x_out + (-s*yr)*y_out + ((c-1)*x_center + s*y_center) / half_x - dx/half_x
        y_in = (s*xr)*x_out + c*y_out + (s*x_center + (1-c)*y_center) / half_y + dy/half_y
    """

    def __init__(self, x_range=40.0, y_range=20.0, x_center=10.0, y_center=0.0):
        self.half_x = x_range / 2.0
        self.half_y = y_range / 2.0

        self.ratio_xy = x_range / y_range    # theta[1,0]: s * ratio_xy
        self.ratio_yx = y_range / x_range    # theta[0,1]: -s * ratio_yx

        self.x_center = x_center
        self.y_center = y_center

    @staticmethod
    def _as_list(val, B):
        """把 list/tuple/标量/tensor 统一成 Python 列表 (tensor 仅兜底, 会同步一次)"""
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val.detach().reshape(-1).tolist()
        if isinstance(val, (list, tuple)):
            return list(val)
        return [val] * B

    @staticmethod
    def _any_nonzero(vals) -> bool:
        return vals is not None and any(abs(float(v)) > 1e-6 for v in vals)

    def flip_bev(self, bev_feat: torch.Tensor, flip) -> torch.Tensor:
        B = bev_feat.shape[0]
        flags = self._as_list(flip, B)
        if flags is None or not any(bool(f) for f in flags):
            return bev_feat
        mask = torch.tensor([bool(f) for f in flags], device=bev_feat.device,
                            dtype=torch.bool).view(B, 1, 1, 1)
        return torch.where(mask, bev_feat.flip(dims=[-2]), bev_feat)

    def _affine_transform(self, bev_feat: torch.Tensor,
                          rot_angle=None, dx=None, dy=None) -> torch.Tensor:
        B = bev_feat.shape[0]
        device, dtype = bev_feat.device, bev_feat.dtype

        def _to_tensor(val):
            vals = self._as_list(val, B)
            if vals is None:
                return None
            t = torch.as_tensor(vals, device=device, dtype=dtype).reshape(-1)
            if t.numel() == 1 and B > 1:
                t = t.expand(B)
            return t

        rot = _to_tensor(rot_angle)
        if rot is not None:
            c = torch.cos(rot)
            s = torch.sin(rot)
        else:
            c = torch.ones(B, device=device, dtype=dtype)
            s = torch.zeros(B, device=device, dtype=dtype)

        dx_t = _to_tensor(dx)
        dy_t = _to_tensor(dy)
        tx = dx_t / self.half_x if dx_t is not None else 0.0
        ty = dy_t / self.half_y if dy_t is not None else 0.0

        theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = c
        theta[:, 0, 1] = -s * self.ratio_yx
        theta[:, 0, 2] = ((c - 1) * self.x_center + s * self.y_center) / self.half_x - tx
        theta[:, 1, 0] = s * self.ratio_xy
        theta[:, 1, 1] = c
        theta[:, 1, 2] = (s * self.x_center + (1 - c) * self.y_center) / self.half_y + ty

        grid = F.affine_grid(theta, bev_feat.shape, align_corners=False)
        return F.grid_sample(bev_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    def transform_bev(self, bev_feat: torch.Tensor, flip=None,
                      rot_angle=None, dx=None, dy=None) -> torch.Tensor:
        bev_feat = self.flip_bev(bev_feat, flip)
        B = bev_feat.shape[0]
        need_affine = (
            self._any_nonzero(self._as_list(rot_angle, B)) or
            self._any_nonzero(self._as_list(dx, B)) or
            self._any_nonzero(self._as_list(dy, B))
        )
        if need_affine:
            bev_feat = self._affine_transform(bev_feat, rot_angle, dx, dy)
        return bev_feat


class MapTR(nn.Module):
    """MapTR: 多相机 → BEV → Transformer解码 → 道路结构线检测"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = Backbone(cfg.img_backbone)
        self.grid_mask = GridMask(use_grid_mask=True)
        self.bev_encoder = BEVFormerEncoder(cfg.bev_encoder)

        self.head = MapTRHead(cfg.map_det_head) if cfg.map_det_head.get('enabled', True) else None
        self.seg_head = MapSegHead(cfg.map_seg_head) if cfg.map_seg_head.get('enabled', True) else None
        self.heatmap_head = BEVHeatMapHead(cfg.heatmap_head) if cfg.heatmap_head.get('enabled', True) else None

        pc = cfg.bev_encoder.pc_range
        self.bev_transform = BEVTransform(
            x_range=float(pc[3] - pc[0]),
            y_range=float(pc[4] - pc[1]),
            x_center=float((pc[3] + pc[0]) / 2),
            y_center=float((pc[4] + pc[1]) / 2),
        )

    def forward(self, imgs, intrinsics, extrinsics, batch=None,
                return_all_layers=False):
        batch_size, num_cams, C, H, W = imgs.shape

        imgs_flat = imgs.view(batch_size * num_cams, C, H, W)
        img_feats = self.backbone(imgs_flat)
        img_feats = [self.grid_mask(f) for f in img_feats]

        bev_feat = self.bev_encoder(img_feats, intrinsics, extrinsics, imgs=imgs)

        if batch is not None:
            bev_feat = self.bev_transform.transform_bev(
                bev_feat,
                flip=batch.get('flip'),
                rot_angle=batch.get('rot_angle'),
                dx=batch.get('dx'),
                dy=batch.get('dy'),
            )

        cls_scores, reg_preds = None, None
        if self.head is not None:
            cls_scores, reg_preds, cls_scores_aux, reg_preds_aux = self.head(bev_feat, return_all_layers=return_all_layers)
        if self.head is not None and cls_scores_aux is not None and any(c is not None for c in cls_scores_aux):
            self._aux_outputs = (cls_scores_aux, reg_preds_aux)
        else:
            self._aux_outputs = None

        seg_pred = self.seg_head(bev_feat) if self.seg_head else None
        heatmap_pred = self.heatmap_head(bev_feat) if self.heatmap_head else None

        return cls_scores, reg_preds, seg_pred, heatmap_pred, bev_feat

    def compute_loss(self, cls_scores, reg_preds, seg_preds, batch, heatmap_pred=None):
        loss_dict = {}
        if self.head is not None and cls_scores is not None:
            aux_outputs = getattr(self, '_aux_outputs', None)
            if aux_outputs is not None:
                cls_scores_aux, reg_preds_aux = aux_outputs
                loss_dict.update(self.head.loss(cls_scores, reg_preds, batch['vectors'],
                                                cls_scores_aux, reg_preds_aux))
            else:
                loss_dict.update(self.head.loss(cls_scores, reg_preds, batch['vectors']))
        if self.seg_head is not None and seg_preds is not None and batch.get('semantic_mask') is not None:
            loss_dict.update(self.seg_head.loss(seg_preds, batch['semantic_mask']))
        if self.heatmap_head is not None and heatmap_pred is not None and batch.get('soft_heatmap') is not None:
            loss_dict.update(self.heatmap_head.loss(heatmap_pred, batch['soft_heatmap']))
        return loss_dict
