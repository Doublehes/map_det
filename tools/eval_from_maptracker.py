"""使用 map_det 的评测代码评测 MapTracker 产出的检测结果"""
import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from configs.default import AttrDict
from metrics.vector_eval import VectorEvaluate


def load_maptracker_gt(pkl_path):
    """加载 MapTracker 的 GT pickle 文件

    MapTracker GT 格式: {token: {cls_id: [line_ndarray, ...]}}
    坐标: 世界坐标 (x: -10~30, y: -10~10)
    """
    print(f'加载 MapTracker GT: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        gts = pickle.load(f)
    print(f'  样本数: {len(gts)}')
    total_lines = sum(len(lines) for gt in gts.values() for lines in gt.values())
    print(f'  GT 线总数: {total_lines}')
    return gts


def load_maptracker_preds(json_path):
    """加载 MapTracker 的预测结果 JSON 文件

    MapTracker 预测格式: {'meta': ..., 'results': {token: {'vectors': [...], 'scores': [...], 'labels': [...]}}}
    vectors 是 list of list (每个点 [x, y]), 世界坐标
    """
    print(f'加载 MapTracker 预测: {json_path}')
    with open(json_path, 'r') as f:
        raw = json.load(f)

    results = raw.get('results', raw)
    print(f'  样本数: {len(results)}')

    preds = {}
    for token, pred in results.items():
        preds[token] = {
            'vectors': [np.array(v, dtype=np.float32) for v in pred.get('vectors', [])],
            'scores': pred.get('scores', []),
            'labels': pred.get('labels', []),
        }
    total = sum(len(p['scores']) for p in preds.values())
    print(f'  预测线总数: {total}')
    return preds


def build_minimal_cfg():
    """构造一个精简的 cfg 供 VectorEvaluate 使用"""
    cat2id = {'guide_line': 0, 'boundary': 1}
    data_cfg = AttrDict({
        'cat2id': cat2id,
        'num_classes': len(cat2id),
        'roi_size': (40, 20),
        'pc_range': [-10, -10, -3, 30, 10, 5],
        'val_ann_file': '',
    })
    cfg = AttrDict({
        'data': data_cfg,
        'eval_thresholds': [0.5, 1.0, 1.5],
        'score_thr': 0.0,
    })
    return cfg


def main():
    gt_path = '/home/flow/Code/MilitaryModel/maptracker_multiview/tmp_gts_nusc_40x20.pkl'
    pred_path = '/home/flow/Code/MilitaryModel/maptracker_multiview/work_dirs/config_vis/submission_vector.json'

    gts = load_maptracker_gt(gt_path)
    preds = load_maptracker_preds(pred_path)

    cfg = build_minimal_cfg()
    evaluator = VectorEvaluate(cfg, n_workers=8)

    evaluator.gts = gts
    evaluator._prepare_gts_done = True

    results = evaluator.evaluate(preds)
    evaluator.print_results(results)

    return results


if __name__ == '__main__':
    main()
