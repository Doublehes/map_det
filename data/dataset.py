import pickle
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from shapely.geometry import LineString
from torch.utils.data import Dataset
import torch

MAPTR_DIR = str(Path(__file__).resolve().parent.parent / '../MapTracker')
if MAPTR_DIR not in sys.path:
    sys.path.insert(0, MAPTR_DIR)

from .pipeline import (
    normalize_image, resize_image, resize_intrinsics,
    vectorize_map, rasterize_map, compute_soft_heatmap
)

import random as _random


def _photo_metric_distortion(img, brightness_delta=32, contrast_range=(0.5, 1.5),
                              saturation_range=(0.5, 1.5), hue_delta=18):
    """像素级光度增强: brightness, contrast, saturation, hue (不影响 3D 信息)"""
    # 随机亮度
    if _random.randint(0, 1):
        delta = _random.uniform(-brightness_delta, brightness_delta)
        img += delta

    mode = _random.randint(0, 1)
    # mode=1 时先做对比度
    if mode == 1 and _random.randint(0, 1):
        alpha = _random.uniform(*contrast_range)
        img *= alpha

    # BGR → HSV
    img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    # 随机饱和度
    if _random.randint(0, 1):
        img_hsv[..., 1] *= _random.uniform(*saturation_range)

    # 随机色相
    if _random.randint(0, 1):
        img_hsv[..., 0] += _random.uniform(-hue_delta, hue_delta)
        img_hsv[..., 0][img_hsv[..., 0] > 360] -= 360
        img_hsv[..., 0][img_hsv[..., 0] < 0] += 360

    # HSV → BGR
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # mode=0 时后做对比度
    if mode == 0 and _random.randint(0, 1):
        alpha = _random.uniform(*contrast_range)
        img *= alpha

    return img


class MapTRDataset(Dataset):
    """MapTR数据集: 从pkl加载多视角图像 + 道路线标注"""

    def __init__(self, ann_file, data_root, cfg, is_train=True):
        self.cfg = cfg
        self.data_root = Path(data_root)
        self.is_train = is_train

        with open(ann_file, 'rb') as f:
            self.samples = pickle.load(f)

        print(f'[MapTRDataset] 加载 {len(self.samples)} 个样本')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回一个样本: 多相机图像 + 内参外参 + 向量化线 + 分割掩码(训练)"""
        sample = self.samples[idx]

        imgs, intrinsics, extrinsics = self._load_images(sample)
        map_vectors = self._load_map(sample)

        ret = {
            'imgs': imgs,                     # (6, 3, 176, 320)
            'intrinsics': intrinsics,          # (6, 3, 3)
            'extrinsics': extrinsics,          # (6, 4, 4)
            'vectors': map_vectors,            # dict {cls_id: (N, 2, 16, 2)}
            'token': sample['token'],
            'scene_name': sample['scene_name'],
            'sample_idx': sample['sample_idx'],
        }

        sem_mask = self._load_semantic_mask(sample)
        ret['semantic_mask'] = sem_mask    # (1, 80, 160)

        soft_heatmap = self._load_soft_heatmap(sample)
        ret['soft_heatmap'] = soft_heatmap  # (1, 80, 160)

        return ret

    def _load_images(self, sample):
        """
        加载多相机图像, 统一按相机名排序, 不足6个补dummy

        样本可能有4或6个相机, 统一输出到6个槽位
        dummy相机(零图像 + 远距离外参)在BEV编码时被valid_mask过滤掉
        """
        num_cams = self.cfg.num_cams
        available = {}
        for cam_name, cam_data in sample['cams'].items():
            if cam_data['img_fpath'] is not None:
                available[cam_name] = cam_data

        sorted_cam_names = sorted(available.keys())
        # sorted_cam_names = available.keys()

        imgs, intrinsics_list, extrinsics_list = [], [], []
        for cam_name in sorted_cam_names:
            cam_data = available[cam_name]
            img_path = self.data_root / cam_data['img_fpath']
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = img.astype(np.float32)
            orig_h, orig_w = img.shape[:2]

            # resize + photometric augmentation (train only) + normalize
            img = resize_image(img, (self.cfg.img_h, self.cfg.img_w))
            if self.is_train:
                img = _photo_metric_distortion(img)
            img = normalize_image(img, self.cfg.img_norm['mean'], self.cfg.img_norm['std'])
            img = torch.from_numpy(img).permute(2, 0, 1)

            K = np.array(cam_data['intrinsics'], dtype=np.float32).copy()
            K = resize_intrinsics(K, (orig_h, orig_w), (self.cfg.img_h, self.cfg.img_w))

            extr = np.array(cam_data['extrinsics'], dtype=np.float32).copy()

            imgs.append(img)
            intrinsics_list.append(K)
            extrinsics_list.append(extr)

        # 不足 num_cams 的补 dummy (零图像 + identity内参 + 远处外参)
        while len(imgs) < num_cams:
            dummy_img = torch.zeros((3, self.cfg.img_h, self.cfg.img_w), dtype=torch.float32)
            dummy_K = np.eye(3, dtype=np.float32)
            dummy_extr = np.array([
                [0, -1, 0, 1000000],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            imgs.append(dummy_img)
            intrinsics_list.append(dummy_K)
            extrinsics_list.append(dummy_extr)

        imgs = torch.stack(imgs[:num_cams], dim=0)
        intrinsics = np.stack(intrinsics_list[:num_cams], axis=0)
        extrinsics = np.stack(extrinsics_list[:num_cams], axis=0)
        return imgs, intrinsics, extrinsics

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
            roi_size=self.cfg.roi_size,    # (40, 20)
            num_points=self.cfg.num_points, # 16
            normalize=True,
            permute={1},  # 仅boundary做方向增强; guide_line保留原始方向
        )
        return vectors

    def _load_semantic_mask(self, sample):
        """将向量线栅格化为分割掩码 (辅助监督)"""
        vectors = self._load_map(sample)
        sem_mask = rasterize_map(
            vectors,
            canvas_size=self.cfg.canvas_size,  # (80, 160) = (H=行/Y, W=列/X)
            roi_size=self.cfg.roi_size,        # (40, 20)
            thickness=2,
            num_classes=self.cfg.num_classes,
        )
        return torch.from_numpy(sem_mask).float()

    def _load_soft_heatmap(self, sample):
        """计算BEV自适应高斯热力图: sigma=min(d_center+d_boundary, 5m)"""
        vectors = self._load_map(sample)
        heatmap = compute_soft_heatmap(
            vectors,
            canvas_size=self.cfg.canvas_size,
            roi_size=self.cfg.roi_size,
            max_sigma=getattr(self.cfg, 'heatmap_max_sigma', 5.0),
        )
        return torch.from_numpy(heatmap).float()


def collate_fn(batch):
    """将batch内多个样本合并为一个batch tensor

    图像/内参/外参: stack → (B, 6, ...)
    向量线: 保持dict格式, 每个样本一个 {cls_id: tensor(N, 2, 16, 2)}
    """
    imgs = torch.stack([b['imgs'] for b in batch], dim=0)
    intrinsics = torch.from_numpy(np.stack([b['intrinsics'] for b in batch], axis=0))
    extrinsics = torch.from_numpy(np.stack([b['extrinsics'] for b in batch], axis=0))

    vec_list = []
    for b in batch:
        v = b['vectors']
        cls_vecs = {}
        for cid, arr in v.items():
            cls_vecs[int(cid)] = torch.from_numpy(arr)
        vec_list.append(cls_vecs)

    ret = {
        'imgs': imgs,                    # (B, 6, 3, 176, 320)
        'intrinsics': intrinsics,        # (B, 6, 3, 3)
        'extrinsics': extrinsics,        # (B, 6, 4, 4)
        'vectors': vec_list,             # list of dict
        'token': [b['token'] for b in batch],
        'scene_name': [b['scene_name'] for b in batch],
    }

    if 'semantic_mask' in batch[0]:
        sem_masks = torch.stack([b['semantic_mask'] for b in batch], dim=0)
        ret['semantic_mask'] = sem_masks  # (B, 1, 80, 160)

    if 'soft_heatmap' in batch[0]:
        heatmaps = torch.stack([b['soft_heatmap'] for b in batch], dim=0)
        ret['soft_heatmap'] = heatmaps  # (B, 1, 80, 160)

    return ret
