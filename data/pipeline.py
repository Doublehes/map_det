import numpy as np
import cv2
from shapely.geometry import LineString

from typing import List, Tuple, Dict, Set, Union


def normalize_image(img: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    img = img.astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (img - mean) / std


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def resize_intrinsics(K: np.ndarray, orig_size: Tuple[int, int], new_size: Tuple[int, int]) -> np.ndarray:
    h_scale = new_size[0] / orig_size[0]
    w_scale = new_size[1] / orig_size[1]
    K = K.copy()
    K[0, 0] *= w_scale
    K[0, 2] *= w_scale
    K[1, 1] *= h_scale
    K[1, 2] *= h_scale
    return K


def vectorize_map(
    map_geoms: Dict[int, List],
    roi_size: Tuple[float, float],
    num_points: int,
    normalize: bool = True,
    permute: Union[bool, Set[int]] = True,
    simplify_tol: float = 0.2,
) -> Dict:
    """将Shapely线 → 均匀采样的向量化点序列

    Args:
        permute: bool或set。True=全类permute, {1}=仅boundary permute
    返回:
        {cls_id: (num_lines, num_permute, num_points, 2)}  每点归一化到[0,1]
    """
    vectors = {}
    for cls_id, lines in map_geoms.items():
        cls_vectors = []
        for line in lines:
            # 1. Douglas-Peucker简化 (去除冗余点)
            if simplify_tol > 0 and line is not None:
                line = line.simplify(simplify_tol, preserve_topology=True)
            if line is None or line.is_empty:
                continue
            coords = np.array(line.coords, dtype=np.float32)

            # 2. 沿曲线均匀采样num_points个点
            if len(coords) >= 2:
                sampled = resample_line(coords, num_points)
            else:
                sampled = np.zeros((num_points, 2), dtype=np.float32)

            # 3. 归一化: _load_map已减去pc_range[0/1], 坐标在[0, roi], 直接除以roi → [0, 1]
            if normalize:
                sampled = sampled / np.array([roi_size[0], roi_size[1]], dtype=np.float32)

            # 4. 生成正向+反向两个排列 (开放线遍历方向歧义)
            should_permute = (cls_id in permute) if isinstance(permute, set) else permute
            if should_permute:
                line_permutes = [sampled, np.flip(sampled, axis=0)]
                cls_vectors.append(np.stack(line_permutes, axis=0))
            else:
                cls_vectors.append(sampled[np.newaxis, ...])

        if len(cls_vectors) > 0:
            vectors[cls_id] = np.stack(cls_vectors, axis=0)
    return vectors


def resample_line(coords: np.ndarray, num_points: int) -> np.ndarray:
    """沿折线路径均匀采样 num_points 个点

    计算累积距离 → 生成均匀间隔 → 线性插值
    返回: (num_points, 2)
    """
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    coord_dim = coords.shape[1]
    if coord_dim > 2:
        coords = coords[:, :2]

    dists = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    cum_dists = np.concatenate([np.zeros(1), np.cumsum(dists)])
    total_len = cum_dists[-1]

    if total_len < 1e-6:
        return np.tile(coords[0], (num_points, 1))

    uniform_dists = np.linspace(0, total_len, num_points)
    result = np.zeros((num_points, 2), dtype=np.float32)
    for i, d in enumerate(uniform_dists):
        idx = np.searchsorted(cum_dists, d) - 1
        idx = max(0, min(idx, len(coords) - 2))
        t = (d - cum_dists[idx]) / max(cum_dists[idx + 1] - cum_dists[idx], 1e-6)
        t = np.clip(t, 0, 1)
        result[i] = coords[idx, :2] * (1 - t) + coords[idx + 1, :2] * t
    return result


def rasterize_map(
    vectors: Dict[int, np.ndarray],
    canvas_size: Tuple[int, int],
    roi_size: Tuple[float, float],
    thickness: int = 2,
) -> np.ndarray:
    """将向量化线绘制为分割掩码 (辅助监督用)"""
    h, w = canvas_size
    num_classes = max(vectors.keys()) + 1 if vectors else 1
    sem_mask = np.zeros((num_classes, h, w), dtype=np.uint8)

    for cls_id, lines in vectors.items():
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
                cv2.polylines(sem_mask[cls_id], [pts], False, 1, thickness=thickness)
    return sem_mask


def compute_soft_heatmap(
    vectors: Dict[int, np.ndarray],
    canvas_size: Tuple[int, int],
    roi_size: Tuple[float, float],
    max_sigma: float = 5.0,
) -> np.ndarray:
    """计算BEV道路热力图, 边界处值≈0.1

    在中心线处估算道路半宽 hw, 全局常量 sigma = hw:
        hw = median(d_boundary at centerline pixels)
        heatmap = exp(-2.303 × (d_center / hw)²)    边界处≈0.1

    Args:
        vectors: {cls_id: (N, 2|1, num_points, 2)} 归一化[0,1]的向量线
        canvas_size: (H, W) 输出画布尺寸
        roi_size: (x_range, y_range) 米
        max_sigma: sigma上限(米), 默认5.0
    Returns:
        (1, H, W) float32 heatmap, 值域[0,1]
    """
    h, w = canvas_size
    pixels_per_meter = w / roi_size[0]

    center_canvas = np.zeros((h, w), dtype=np.uint8)
    boundary_canvas = np.zeros((h, w), dtype=np.uint8)

    for cls_id, lines in vectors.items():
        canvas = np.zeros((h, w), dtype=np.uint8)
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
                cv2.polylines(canvas, [pts], False, 255, thickness=1)

        if cls_id == 0:
            center_canvas = np.maximum(center_canvas, canvas)
        elif cls_id == 1:
            boundary_canvas = np.maximum(boundary_canvas, canvas)

    if center_canvas.max() == 0:
        return np.zeros((1, h, w), dtype=np.float32)

    d_center = cv2.distanceTransform(255 - center_canvas, cv2.DIST_L2, 5).astype(np.float32)
    d_c_m = d_center / pixels_per_meter

    if boundary_canvas.max() > 0:
        d_boundary = cv2.distanceTransform(255 - boundary_canvas, cv2.DIST_L2, 5).astype(np.float32)
        d_b_m = d_boundary / pixels_per_meter
        hw = np.median(d_b_m[center_canvas > 0])
        hw = np.clip(hw, 0.5, max_sigma)
    else:
        hw = max_sigma

    heatmap = np.exp(-2.303 * (d_c_m / hw) ** 2)

    return heatmap[np.newaxis, ...]


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, data: Dict):
        for t in self.transforms:
            data = t(data)
        return data
