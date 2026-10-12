"""model_slim2 数据集: 2D 分割图 → one-hot 输入 + 同源代价图/距离场 (自监督)

纯 2D 任务: 读取分割标签图 (不 resize, 不矫正朝向), 输入与监督信号共享同一图像坐标系。
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def mask_to_onehot(mask: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    """标签图 → (num_classes+1, H, W) one-hot: 前 C 通道类别, 最后通道 ignore"""
    h, w = mask.shape
    out = np.zeros((num_classes + 1, h, w), dtype=np.float32)
    for c in range(num_classes):
        out[c][mask == c] = 1.0
    out[num_classes][mask == ignore_index] = 1.0
    return out


def mask_to_cost(mask: np.ndarray, cost_values: dict, ignore_cost: float,
                 ignore_index: int, sigma: float) -> np.ndarray:
    """标签图 → 代价图 (H, W) float32 [0,1], sigma>0 高斯平滑"""
    lut = np.full(256, float(ignore_cost), dtype=np.float32)
    lut[ignore_index] = float(ignore_cost)
    for cls_id, cost in cost_values.items():
        lut[cls_id] = float(cost)
    cost = lut[mask]
    if sigma > 0:
        k = max(3, int(sigma * 4) | 1)
        cost = cv2.GaussianBlur(cost, (k, k), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REPLICATE)
    return cost.astype(np.float32)


def mask_to_dist_field(mask: np.ndarray, drivable_classes: tuple,
                       ignore_index: int, max_dist_px: float) -> np.ndarray:
    """可通行区距离场: 到非可通行边界的距离, 归一化 [0,1]; 非可通行区置 0"""
    drivable = np.zeros(mask.shape, dtype=np.uint8)
    for c in drivable_classes:
        drivable[mask == c] = 255
    drivable[mask == ignore_index] = 0
    dist_px = cv2.distanceTransform((drivable > 0).astype(np.uint8), cv2.DIST_L2, 5)
    out = np.clip(dist_px / max(max_dist_px, 1e-6), 0.0, 1.0).astype(np.float32)
    out[drivable == 0] = 0.0
    return out


def augment_mask(mask: np.ndarray, cfg) -> np.ndarray:
    """对标签图做几何增强: flip → rotate → bend (全部最近邻, 标签安全)"""
    h, w = mask.shape
    out = mask

    if random.random() < cfg.get('flip_prob', 0.0):
        out = np.fliplr(out)

    if random.random() < cfg.get('rot_prob', 0.0):
        max_deg = cfg.get('rot_max_deg', 15.0)
        angle = random.uniform(-max_deg, max_deg)
        mat = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, 1.0)
        out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    if random.random() < cfg.get('bend_prob', 0.0):
        amp = random.uniform(0.2, 1.0) * cfg.get('bend_amp', 0.04) * w
        wl_lo, wl_hi = cfg.get('bend_wavelength', (0.3, 0.8))
        wl = random.uniform(wl_lo, wl_hi) * w
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        shift = amp * np.sin(2.0 * np.pi * grid_x / wl)
        map_x = grid_x
        map_y = np.clip(grid_y + shift, 0, h - 1)
        out = cv2.remap(out, map_x, map_y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return np.ascontiguousarray(out)


class Slim2Dataset(Dataset):
    """分割标签图数据集: 输入 one-hot, 监督信号 (代价图/距离场) 同源自标签图"""

    def __init__(self, seg_dir, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.seg_dir = Path(seg_dir)

        files = [p for p in self.seg_dir.iterdir() if p.suffix.lower() == '.png']
        if not files:
            raise FileNotFoundError(f'目录中没有 PNG 文件: {self.seg_dir}')
        self.files = sorted(files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

        indices = list(range(len(self.files)))
        rng = random.Random(cfg.get('seed', 42))
        rng.shuffle(indices)
        num_val = max(1, int(len(indices) * cfg.get('val_ratio', 0.1)))
        val_set = set(indices[:num_val])
        self.split = [i for i in range(len(self.files)) if (i in val_set) == (not is_train)]
        print(f'[Slim2Dataset] {"train" if is_train else "val"}: {len(self.split)}/{len(self.files)} 样本')

    def __len__(self):
        return len(self.split)

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f'无法读取图像: {path}')
        if mask.ndim == 3:
            mask = mask[..., 0]
        return np.ascontiguousarray(mask)

    def __getitem__(self, idx):
        path = self.files[self.split[idx]]
        mask = self._load_mask(path)
        if self.is_train:
            mask = augment_mask(mask, self.cfg)

        inp = mask_to_onehot(mask, self.cfg.num_classes, self.cfg.ignore_index)
        cost = mask_to_cost(mask, self.cfg.cost_values, self.cfg.ignore_cost,
                            self.cfg.ignore_index, self.cfg.cost_sigma)
        dist = mask_to_dist_field(mask, tuple(self.cfg.drivable_classes),
                                  self.cfg.ignore_index, self.cfg.dist_max_px)

        return {
            'input': torch.from_numpy(inp),                      # (C+1, H, W)
            'cost': torch.from_numpy(cost).unsqueeze(0),         # (1, H, W)
            'dist_field': torch.from_numpy(dist).unsqueeze(0),   # (1, H, W)
            'mask': torch.from_numpy(mask.astype(np.int64)),     # (H, W)
            'token': path.stem,
        }


def collate_fn(batch):
    """堆叠张量字段, 保留 token 列表"""
    return {
        'input': torch.stack([b['input'] for b in batch], dim=0),
        'cost': torch.stack([b['cost'] for b in batch], dim=0),
        'dist_field': torch.stack([b['dist_field'] for b in batch], dim=0),
        'mask': torch.stack([b['mask'] for b in batch], dim=0),
        'token': [b['token'] for b in batch],
    }
