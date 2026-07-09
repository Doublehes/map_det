import numpy as np
from scipy.spatial import distance as scipy_distance
from numpy.typing import NDArray


def chamfer_distance(line1: NDArray, line2: NDArray) -> float:
    dist_matrix = scipy_distance.cdist(line1, line2, 'euclidean')
    dist12 = dist_matrix.min(-1).sum() / len(line1)
    dist21 = dist_matrix.min(-2).sum() / len(line2)
    return float((dist12 + dist21) / 2)


def chamfer_distance_batch(pred_lines: NDArray, gt_lines: NDArray) -> np.ndarray:
    M, num_pts, coord_dims = pred_lines.shape
    N = gt_lines.shape[0]

    pred_flat = pred_lines.reshape(-1, coord_dims)
    gt_flat = gt_lines.reshape(-1, coord_dims)

    dist_mat = scipy_distance.cdist(pred_flat, gt_flat, 'euclidean')
    dist_mat = dist_mat.reshape(M, num_pts, N, num_pts).transpose(2, 0, 1, 3)

    dist1 = dist_mat.min(axis=-1).sum(axis=-1)
    dist2 = dist_mat.min(axis=-2).sum(axis=-1)
    dist_matrix = (dist1 + dist2).T / (2 * num_pts)

    return dist_matrix
