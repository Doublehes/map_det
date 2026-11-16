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

import model_slim2.field_pipeline as fp


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


def build_supervision(mask: np.ndarray, cfg):
    """标签图 -> (修正后距离场, 关键点列表), 与 vis_dist_field 共用 field_pipeline"""
    fg = fp.drivable_binary(mask, cfg.drivable_classes, cfg.ignore_index)
    pre = fp.preclean_mask(
        fg,
        mask=mask,
        low_cost_classes=cfg.low_cost_classes,
        ignore_index=cfg.ignore_index,
    )
    dist = cv2.distanceTransform(pre['mask'], cv2.DIST_L2, 5)
    base = fp.build_field(pre['mask'], dist=dist)
    skel = fp.build_skeleton(pre['mask'], with_stats=False)
    field, _ = fp.skeleton_guided_field(pre['mask'], dist, base, skel)
    kp_list = fp.extract_keypoints_from_skeleton(
        skel,
        shape=mask.shape,
        spacing=cfg.kp_spacing,
        v_front=cfg.kp_vfront,
        max_pts=int(cfg.kp_max),
    )
    return field, kp_list


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


def _resolve_dirs(data_dir):
    """把 str / list[str] 统一成 list[Path]"""
    if isinstance(data_dir, (list, tuple)):
        return [Path(d) for d in data_dir]
    return [Path(data_dir)]


def _list_pngs(dirs):
    """按目录顺序收集所有 PNG; 每个目录内按文件名(数字优先)排序"""
    files = []
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(f'目录不存在: {d}')
        ps = [p for p in d.iterdir() if p.suffix.lower() == '.png']
        if not ps:
            raise FileNotFoundError(f'目录中没有 PNG 文件: {d}')
        files.extend(sorted(ps, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem))
    return files


class Slim2Dataset(Dataset):
    """分割标签图数据集: 输入 one-hot, 监督信号 (代价图/距离场) 同源自标签图"""

    def __init__(self, data_dir, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.dirs = _resolve_dirs(data_dir)
        self.files = _list_pngs(self.dirs)
        self.split = list(range(len(self.files)))     # 训练/验证目录已分开, 不再切分
        print(f'[Slim2Dataset] {"train" if is_train else "val"}: '
              f'{len(self.files)} 样本, dirs={[str(d) for d in self.dirs]}')

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
        dist, kp_list = build_supervision(mask, self.cfg)

        kp_max = int(self.cfg.kp_max)
        keypoints = np.zeros((kp_max, 2), dtype=np.float32)      # (K, 2) 补 0
        keypoints_valid = np.zeros((kp_max,), dtype=np.float32)  # (K,) 真实=1
        for i, (u, v, _) in enumerate(kp_list[:kp_max]):
            keypoints[i] = (u, v)
            keypoints_valid[i] = 1.0

        return {
            'input': torch.from_numpy(inp),                      # (C+1, H, W)
            'cost': torch.from_numpy(cost).unsqueeze(0),         # (1, H, W)
            'dist_field': torch.from_numpy(dist).unsqueeze(0),   # (1, H, W)
            'mask': torch.from_numpy(mask.astype(np.int64)),     # (H, W)
            'keypoints': torch.from_numpy(keypoints),            # (K, 2)
            'keypoints_valid': torch.from_numpy(keypoints_valid),  # (K,)
            'token': path.stem,
        }


def collate_fn(batch):
    """堆叠张量字段, 保留 token 列表"""
    return {
        'input': torch.stack([b['input'] for b in batch], dim=0),
        'cost': torch.stack([b['cost'] for b in batch], dim=0),
        'dist_field': torch.stack([b['dist_field'] for b in batch], dim=0),
        'mask': torch.stack([b['mask'] for b in batch], dim=0),
        'keypoints': torch.stack([b['keypoints'] for b in batch], dim=0),
        'keypoints_valid': torch.stack([b['keypoints_valid'] for b in batch], dim=0),
        'token': [b['token'] for b in batch],
    }
