import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Union

from .distance import chamfer_distance, chamfer_distance_batch


def average_precision(recalls: np.ndarray, precisions: np.ndarray, mode: str = 'area') -> float:
    recalls = recalls[np.newaxis, :]
    precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = 0.

    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])

        ind = np.where(mrec[0, 1:] != mrec[0, :-1])[0]
        ap = np.sum(
            (mrec[0, ind + 1] - mrec[0, ind]) * mpre[0, ind + 1])

    elif mode == '11points':
        for thr in np.arange(0, 1 + 1e-3, 0.1):
            precs = precisions[recalls >= thr]
            prec = precs.max() if precs.size > 0 else 0
            ap += prec
        ap /= 11
    else:
        raise ValueError(f'Unrecognized mode "{mode}", only "area" and "11points" are supported')

    return ap


def instance_match(
    pred_lines: NDArray,
    scores: NDArray,
    gt_lines: NDArray,
    thresholds: Union[Tuple, List],
    metric: str = 'chamfer',
) -> List:
    if metric == 'chamfer':
        distance_fn = chamfer_distance
    elif metric == 'frechet':
        raise NotImplementedError('Frechet distance is not implemented')
    else:
        raise ValueError(f'unknown distance function {metric}')

    num_preds = pred_lines.shape[0]
    num_gts = gt_lines.shape[0]

    if num_gts == 0:
        tp_fp_list = []
        for thr in thresholds:
            tp = np.zeros(num_preds, dtype=np.float32)
            fp = np.ones(num_preds, dtype=np.float32)
            tp_fp_list.append((tp, fp))
        return tp_fp_list

    if num_preds == 0:
        tp_fp_list = []
        for thr in thresholds:
            tp = np.zeros(0, dtype=np.float32)
            fp = np.zeros(0, dtype=np.float32)
            tp_fp_list.append((tp, fp))
        return tp_fp_list

    assert pred_lines.shape[1] == gt_lines.shape[1], \
        "sample points num should be the same"

    matrix = chamfer_distance_batch(pred_lines, gt_lines)
    matrix_min = matrix.min(axis=1)
    matrix_argmin = matrix.argmin(axis=1)
    sort_inds = np.argsort(-scores)

    tp_fp_list = []
    for thr in thresholds:
        tp = np.zeros(num_preds, dtype=np.float32)
        fp = np.zeros(num_preds, dtype=np.float32)
        gt_covered = np.zeros(num_gts, dtype=bool)
        for i in sort_inds:
            if matrix_min[i] <= thr:
                matched_gt = matrix_argmin[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        tp_fp_list.append((tp, fp))

    return tp_fp_list
