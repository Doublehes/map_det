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

from data.pipeline import vectorize_map, rasterize_map


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

        sem_mask = self._load_semantic_mask(vectors)
        return {
            'raster': torch.from_numpy(raster).permute(2, 0, 1).float(),  # (3, H, W)
            'vectors': vectors,
            'semantic_mask': sem_mask,   # (num_classes, seg_h, seg_w)
            'token': sample['token'],
        }

    def _load_semantic_mask(self, vectors):
        """将向量线栅格化为 per-class 分割掩码 (不 flip, 与 slim 的 y 约定一致)"""
        sem_mask = rasterize_map(
            vectors,
            canvas_size=getattr(self.cfg, 'seg_canvas_size', self.cfg.canvas_size),
            roi_size=self.cfg.roi_size,
            thickness=2,
            num_classes=self.cfg.num_classes,
        )
        return torch.from_numpy(sem_mask).float()

    def _gen_synthetic_map(self):
        """生成合成边界线样本: 仅 cls1, 大曲率弯曲 + 90° L 型, 拒绝采样保证间距"""
        lo = getattr(self.cfg, 'syn_boundary_min', 3)
        hi = getattr(self.cfg, 'syn_boundary_max', 10)
        target = random.randint(lo, min(hi, 10))
        clearance = getattr(self.cfg, 'syn_clearance', 0.5)
        accepted = []
        for _ in range(target):
            for _ in range(50):   # 每条线最多试 50 次, 放不进则少一条 (条数自适应)
                cand = self._gen_candidate()
                geom = LineString([[p[0], p[1]] for p in cand])
                if all(geom.distance(a) > clearance for a in accepted):
                    accepted.append(geom)
                    break
        return {1: [[[float(x), float(y)] for x, y in g.coords] for g in accepted]}

    def _gen_candidate(self):
        """三分生成边界线: 陡峭直线 / L 型 / 大曲率弯线 (均 cls1)

        分布: 斜线 syn_slanted_prob, 其余按 syn_lshape_prob 分 L 型与弯线
        """
        slanted = getattr(self.cfg, 'syn_slanted_prob', 0.0)
        r = random.random()
        if r < slanted:
            return self._gen_slanted_boundary()
        lshape = getattr(self.cfg, 'syn_lshape_prob', 0.5)
        if r < slanted + (1 - slanted) * lshape:
            return self._gen_lshape_boundary()
        return self._gen_curved_boundary()

    def _gen_curved_boundary(self):
        """大曲率弯线: 沿近水平方向铺点, 垂直方向加 A·sin(2πt/wl+φ), 基线控制在 ROI 内"""
        x_min, y_min, _, x_max, y_max, _ = self.cfg.pc_range
        L = random.uniform(*getattr(self.cfg, 'syn_len_range', (5.0, 30.0)))
        A = random.uniform(1.0, getattr(self.cfg, 'syn_bend_amp', 2.5))
        wl = random.uniform(*getattr(self.cfg, 'syn_bend_wavelength', (2.0, 6.0)))
        phi = random.uniform(0, 2 * np.pi)
        theta = random.choice([0.0, np.pi]) + random.uniform(-np.pi / 6, np.pi / 6)

        hx = abs(np.cos(theta)) * L / 2
        hy = abs(np.sin(theta)) * L / 2 + A
        cx = random.uniform(x_min + hx, x_max - hx) if x_min + hx < x_max - hx else (x_min + x_max) / 2
        cy = random.uniform(y_min + hy, y_max - hy) if y_min + hy < y_max - hy else (y_min + y_max) / 2

        n = np.array([-np.sin(theta), np.cos(theta)])
        t = np.linspace(-L / 2, L / 2, 24)
        base = np.stack([cx + t * np.cos(theta), cy + t * np.sin(theta)], -1)
        disp = A * np.sin(2 * np.pi * t / wl + phi)
        pts = base + n[None, :] * disp[:, None]
        return [[float(x), float(y)] for x, y in pts]

    def _gen_lshape_boundary(self):
        """90° 垂直 L 型线: 沿 x 走 L1, 转 90° 沿 y 走 L2, 臂长钳制以适配 ROI"""
        x_min, y_min, _, x_max, y_max, _ = self.cfg.pc_range
        total = random.uniform(*getattr(self.cfg, 'syn_len_range', (5.0, 30.0)))
        L1 = total * random.uniform(0.3, 0.7)
        L2 = total - L1
        # 钳制臂长, 保证两臂在 ROI 内 (留 1m 边)
        L1 = min(L1, (x_max - x_min) / 2 - 2.0)
        L2 = min(L2, (y_max - y_min) / 2 - 2.0)
        sx = random.choice([1, -1])
        sy = random.choice([1, -1])
        cx = random.uniform(x_min + L1 + 1, x_max - L1 - 1)
        cy = random.uniform(y_min + L2 + 1, y_max - L2 - 1)

        pts = []
        for t in np.linspace(0, 1, 13)[:-1]:
            pts.append([cx - sx * L1 + sx * L1 * t, cy])
        for t in np.linspace(0, 1, 13):
            pts.append([cx, cy + sy * L2 * t])
        return pts

    def _gen_slanted_boundary(self):
        """陡峭直线边界: y = kx + b, 方向与 x 轴夹角 θ ∈ [60°,120°] (|k| ≥ tan60°≈1.73)"""
        x_min, y_min, _, x_max, y_max, _ = self.cfg.pc_range
        theta = np.radians(random.uniform(60.0, 120.0))
        L = random.uniform(*getattr(self.cfg, 'syn_len_range', (5.0, 30.0)))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # 按 ROI 可用空间钳制长度 (每侧留 1m 边)
        max_l = min(
            (x_max - x_min - 2.0) / (2 * abs(cos_t)) if abs(cos_t) > 1e-6 else 1e9,
            (y_max - y_min - 2.0) / (2 * abs(sin_t)),
        )
        L = min(L, max_l)
        x_half = (L / 2) * abs(cos_t)
        y_half = (L / 2) * abs(sin_t)
        # 中点放在可行区间内, 保证整段在 ROI 中
        mx = random.uniform(x_min + x_half, x_max - x_half)
        my = random.uniform(y_min + y_half, y_max - y_half)
        d = np.array([cos_t, sin_t])
        p0 = np.array([mx, my]) - (L / 2) * d
        p1 = np.array([mx, my]) + (L / 2) * d
        return [[float(p0[0] + (p1[0] - p0[0]) * t),
                 float(p0[1] + (p1[1] - p0[1]) * t)]
                for t in np.linspace(0, 1, 16)]

    def _gen_synthetic_center_map(self):
        """生成合成中心线样本: 仅 cls0, L 型且沿 x 增大方向, 拒绝采样保证间距"""
        lo = getattr(self.cfg, 'syn_center_min', 1)
        hi = getattr(self.cfg, 'syn_center_max', 3)
        target = random.randint(lo, hi)
        clearance = getattr(self.cfg, 'syn_clearance', 0.5)
        accepted = []
        for _ in range(target):
            for _ in range(50):
                cand = self._gen_lshape_center()
                geom = LineString([[p[0], p[1]] for p in cand])
                if all(geom.distance(a) > clearance for a in accepted):
                    accepted.append(geom)
                    break
        return {0: [[[float(x), float(y)] for x, y in g.coords] for g in accepted]}

    def _gen_lshape_center(self):
        """L 型中心线: 起点低 x → 沿 +X 到转角 → 转 ±90° 沿 Y; 点序 x 单调增 (方向沿 +X)"""
        x_min, y_min, _, x_max, y_max, _ = self.cfg.pc_range
        total = random.uniform(*getattr(self.cfg, 'syn_center_len_range', (10.0, 40.0)))
        L1 = total * random.uniform(0.3, 0.7)
        L2 = total - L1
        # 钳制臂长适配 ROI (留 1m 边)
        L1 = min(L1, (x_max - x_min) / 2 - 2.0)
        L2 = min(L2, (y_max - y_min) / 2 - 2.0)
        sy = random.choice([1, -1])
        # 起点 x0 低, 沿 +X 到转角 cx = x0 + L1; 起点 y 约束在 ±3m 内
        x0 = random.uniform(x_min + 1, x_max - L1 - 1)
        y0 = random.uniform(-3.0, 3.0)
        cx = x0 + L1
        cy = y0
        pts = []
        for t in np.linspace(0, 1, 13)[:-1]:
            pts.append([x0 + L1 * t, cy])
        for t in np.linspace(0, 1, 13):
            pts.append([cx, cy + sy * L2 * t])
        return pts

    def _augment_map_geom(self, sample):
        """对 3D 线做翻转/旋转/平移增强 (世界坐标, 训练时), 栅格与 GT 同源生成

        逻辑与主数据集 MapTRDataset.__getitem__ 一致; 返回增强后的 sample (可能为新副本)
        """
        need_copy = False

        # 合成数据: 每帧二选一 (中心线 / 边界线), 触发则清空原 map_geom
        sc = getattr(self.cfg, 'syn_center_prob', 0.0)
        sb = getattr(self.cfg, 'syn_boundary_prob', 0.0)
        r = random.random()
        if r < sc:
            sample = copy.deepcopy(sample)
            sample['map_geom'] = self._gen_synthetic_center_map()   # 只合成中心线
            need_copy = True
        elif r < sc + sb:
            sample = copy.deepcopy(sample)
            sample['map_geom'] = self._gen_synthetic_map()          # 只合成边界线
            need_copy = True

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

        bend = getattr(self.cfg, 'bev_bend_amp', 0.0)
        if bend and random.random() < getattr(self.cfg, 'bev_bend_prob', 0.5):
            if not need_copy:
                sample = copy.deepcopy(sample)
                need_copy = True
            # A: 支持标量(最大)或 tuple(min,max)
            if isinstance(bend, (list, tuple)):
                A = random.uniform(bend[0], bend[1])
            else:
                A = random.uniform(0.5, bend)
            wl = random.uniform(*getattr(self.cfg, 'bev_bend_wavelength', (4.0, 12.0)))
            phase = random.uniform(0, 2 * np.pi)
            # 全局一致弯曲: (x,y) → (x, y + A·sin(2πx/wl+φ)), 剪切映射不交叉/不自交
            for cls_id in sample['map_geom']:                 # 全部类
                for line in sample['map_geom'][cls_id]:
                    for pt in line:                           # 每个点同一变换
                        pt[1] = pt[1] + A * np.sin(2 * np.pi * pt[0] / wl + phase)

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

    sem_masks = torch.stack([b['semantic_mask'] for b in batch])   # (B, num_classes, seg_h, seg_w)

    return {
        'raster': rasters,
        'vectors': vec_list,
        'semantic_mask': sem_masks,
        'token': [b['token'] for b in batch],
    }
