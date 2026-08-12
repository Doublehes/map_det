import copy
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import LineString
from torch.utils.data import Dataset

from data.pipeline import vectorize_map


def rasterize_rgb(vectors, canvas_size, roi_size, thickness=2):
    """将向量化线绘制为 RGB 栅格图: 中心线(0)绿色, 边界线(1)红色, 黑底

    Args:
        vectors: {cls_id: (N, 1|2, num_points, 2)} 归一化[0,1]的向量线
        canvas_size: (H, W) 画布尺寸
        roi_size: (x_range, y_range) 米
        thickness: 线宽(px)
    Returns:
        (H, W, 3) float32 栅格图, 值域[0,1]
    """
    h, w = canvas_size
    img = np.zeros((h, w, 3), dtype=np.float32)
    colors = {0: (0.0, 1.0, 0.0), 1: (1.0, 0.0, 0.0)}   # BGR: 中心线绿, 边界线红

    for cls_id, lines in vectors.items():
        color = colors.get(cls_id)
        if color is None:
            continue
        for line in lines:
            pts_2d = line[0] if line.ndim == 3 else line
            denormalized = pts_2d * np.array([roi_size[0], roi_size[1]], dtype=np.float32)

            pts = []
            for p in denormalized:
                px = int(p[0] / roi_size[0] * w)
                py = int(p[1] / roi_size[1] * h)
                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], False, color, thickness=thickness)
    return img


def add_gaussian_noise(raster, sigma):
    """叠加高斯噪声: raster (H,W,3) float [0,1]"""
    noise = np.random.randn(*raster.shape).astype(np.float32) * sigma
    return raster + noise


class SlimDataset(Dataset):
    """精简数据集: 从pkl读取GT线, 栅格化为RGB图 (+高斯噪声), 供检测头解码"""

    def __init__(self, ann_file, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        with open(ann_file, 'rb') as f:
            self.samples = pickle.load(f)
        print(f'[SlimDataset] 加载 {len(self.samples)} 个样本')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # sample = self._augment_map_geom(sample)
        if self.is_train:
            sample = self._augment_map_geom(sample)
        vectors = self._load_map(sample)

        raster = rasterize_rgb(
            vectors,
            canvas_size=self.cfg.canvas_size,
            roi_size=self.cfg.roi_size,
            thickness=self.cfg.thickness,
        )

        noise_cfg = self.cfg.noise
        add_noise = noise_cfg.enabled and (
            (self.is_train and noise_cfg.train) or (not self.is_train and noise_cfg.eval))
        if add_noise:
            raster = add_gaussian_noise(raster, noise_cfg.sigma)

        return {
            'raster': torch.from_numpy(raster).permute(2, 0, 1).float(),  # (3, H, W)
            'vectors': vectors,
            'token': sample['token'],
        }

    def _augment_map_geom(self, sample):
        """对 3D 线做翻转/旋转/平移增强 (世界坐标, 训练时), 栅格与 GT 同源生成

        逻辑与主数据集 MapTRDataset.__getitem__ 一致; 返回增强后的 sample (可能为新副本)
        """
        need_copy = False

        if random.random() < getattr(self.cfg, 'bev_flip_prob', 0.0):
            sample = copy.deepcopy(sample)
            need_copy = True
            for cls_id in sample['map_geom']:
                for line in sample['map_geom'][cls_id]:
                    for pt in line:
                        pt[1] = -pt[1]

        max_angle = getattr(self.cfg, 'bev_rot_angle', 0.0)
        if max_angle > 0 and random.random() < 0.5:
            angle_deg = random.uniform(-max_angle, max_angle)
            rot_angle = np.radians(angle_deg)
            if not need_copy:
                sample = copy.deepcopy(sample)
                need_copy = True
            c, s = np.cos(rot_angle), np.sin(rot_angle)
            for cls_id in sample['map_geom']:
                for line in sample['map_geom'][cls_id]:
                    for pt in line:
                        x, y = pt[0], pt[1]
                        pt[0] = x * c - y * s
                        pt[1] = x * s + y * c

        scale = getattr(self.cfg, 'bev_scale', 0.0)
        if scale > 0 and random.random() < 0.5:
            s = random.uniform(1.0 - scale, 1.0 + scale)
            if not need_copy:
                sample = copy.deepcopy(sample)
                need_copy = True
            for cls_id in sample['map_geom']:
                for line in sample['map_geom'][cls_id]:
                    for pt in line:
                        pt[0] *= s
                        pt[1] *= s

        trans_x = getattr(self.cfg, 'bev_trans_x', 0.0)
        trans_y = getattr(self.cfg, 'bev_trans_y', 0.0)
        if (trans_x > 0 or trans_y > 0) and random.random() < 0.5:
            dx = random.uniform(-trans_x, trans_x) if trans_x > 0 else 0.0
            dy = random.uniform(-trans_y, trans_y) if trans_y > 0 else 0.0
            if not need_copy:
                sample = copy.deepcopy(sample)
                need_copy = True
            for cls_id in sample['map_geom']:
                for line in sample['map_geom'][cls_id]:
                    for pt in line:
                        pt[0] += dx
                        pt[1] += dy

        if need_copy:
            x_min, y_min, _, x_max, y_max, _ = self.cfg.pc_range
            for cls_id in list(sample['map_geom'].keys()):
                new_lines = []
                for line in sample['map_geom'][cls_id]:
                    filtered = [pt for pt in line if x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max]
                    if len(filtered) >= 2:
                        new_lines.append(filtered)
                sample['map_geom'][cls_id] = new_lines

        return sample

    def _load_map(self, sample):
        map_geoms = {i: [] for i in range(self.cfg.num_classes)}
        pc0, pc1 = self.cfg.pc_range[0], self.cfg.pc_range[1]
        for cls_id_s, lines in sample['map_geom'].items():
            cls_id = int(cls_id_s)
            if cls_id not in map_geoms:
                continue
            for line in lines:
                if len(line) >= 2:
                    pts = np.array(line, dtype=np.float32)
                    pts[:, 0] -= pc0
                    pts[:, 1] -= pc1
                    map_geoms[cls_id].append(LineString(pts[:, :2]))

        vectors = vectorize_map(
            map_geoms,
            roi_size=self.cfg.roi_size,
            num_points=self.cfg.num_points,
            normalize=True,
            permute={1},  # 仅boundary做方向增强; guide_line保留原始方向
        )
        return vectors


def collate_fn(batch):
    """合并 batch: raster stack, vectors 保持 list-of-dict"""
    rasters = torch.stack([b['raster'] for b in batch])   # (B, 3, H, W)

    vec_list = []
    for b in batch:
        cls_vecs = {int(cid): torch.from_numpy(arr) for cid, arr in b['vectors'].items()}
        vec_list.append(cls_vecs)

    return {
        'raster': rasters,
        'vectors': vec_list,
        'token': [b['token'] for b in batch],
    }
