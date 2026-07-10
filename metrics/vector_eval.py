import os
import pickle
import numpy as np
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple
from time import time

from .ap import instance_match, average_precision

INTERP_NUM = 200
THRESHOLDS = [0.5, 1.0, 1.5]


class VectorEvaluate:
    def __init__(self, cfg, n_workers=0):
        self.cfg = cfg
        self.n_workers = n_workers
        self.cat2id = cfg.data.cat2id
        self.id2cat = {v: k for k, v in self.cat2id.items()}
        self.roi_size = cfg.data.roi_size
        self.pc_range = cfg.data.pc_range
        self.thresholds = getattr(cfg, 'eval_thresholds', THRESHOLDS)
        self.gts = {}
        self._prepare_gts_done = False

    def _denormalize(self, lines: np.ndarray) -> np.ndarray:
        lines = lines.copy()
        lines[..., 0] = lines[..., 0] * self.roi_size[0] + self.pc_range[0]
        lines[..., 1] = lines[..., 1] * self.roi_size[1] + self.pc_range[1]
        return lines

    def prepare_gts(self, dataset):
        ann_path = self.cfg.data.val_ann_file
        pkl_name = Path(ann_path).stem
        cache_dir = Path('./work_dirs')
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f'tmp_gts_{pkl_name}.pkl'

        if os.path.exists(cache_file):
            print(f'加载缓存 GT: {cache_file}')
            with open(cache_file, 'rb') as f:
                self.gts = pickle.load(f)
            self._prepare_gts_done = True
            num_lines = sum(len(v) for gt in self.gts.values() for v in gt.values())
            print(f'缓存加载完成, {len(self.gts)} 个样本, {num_lines} 条 GT 线')
            return

        print('收集 GT...')
        gts = {}
        for i in range(len(dataset)):
            sample = dataset[i]
            token = sample['token']
            vectors = sample['vectors']
            gt_by_cls = {}
            for cls_id, arr in vectors.items():
                lines_list = []
                for j in range(arr.shape[0]):
                    line = arr[j, 0].copy()
                    line = self._denormalize(line)
                    lines_list.append(line)
                gt_by_cls[cls_id] = lines_list
            gts[token] = gt_by_cls
            if (i + 1) % 100 == 0:
                print(f'  collected {i+1}/{len(dataset)} gts')

        print(f'保存 GT 缓存到: {cache_file}')
        with open(cache_file, 'wb') as f:
            pickle.dump(gts, f)

        self.gts = gts
        self._prepare_gts_done = True
        num_lines = sum(len(v) for gt in gts.values() for v in gt.values())
        print(f'done, {len(gts)} samples, {num_lines} GT lines total')

    def _interp_fixed_num(self, vector: np.ndarray, num_pts: int = INTERP_NUM) -> np.ndarray:
        from data.pipeline import resample_line
        return resample_line(vector, num_pts)

    def _evaluate_single(self, pred_vectors: List, scores: List,
                         groundtruth: List, thresholds: List,
                         metric: str = 'chamfer') -> Dict[float, np.ndarray]:
        pred_lines = []
        for vector in pred_vectors:
            vector = np.array(vector)
            vector_interp = self._interp_fixed_num(vector, INTERP_NUM)
            pred_lines.append(vector_interp)
        pred_lines = np.stack(pred_lines) if pred_lines else np.zeros((0, INTERP_NUM, 2))

        gt_lines = []
        for vector in groundtruth:
            vector_interp = self._interp_fixed_num(vector, INTERP_NUM)
            gt_lines.append(vector_interp)
        gt_lines = np.stack(gt_lines) if gt_lines else np.zeros((0, INTERP_NUM, 2))

        scores = np.array(scores)
        tp_fp_list = instance_match(pred_lines, scores, gt_lines, thresholds, metric)

        tp_fp_score_by_thr = {}
        for i, thr in enumerate(thresholds):
            tp, fp = tp_fp_list[i]
            tp_fp_score = np.hstack([tp[:, None], fp[:, None], scores[:, None]])
            tp_fp_score_by_thr[thr] = tp_fp_score
        return tp_fp_score_by_thr

    def evaluate(self, predictions: Dict, metric: str = 'chamfer') -> Dict[str, float]:
        assert self._prepare_gts_done, 'call prepare_gts() first'

        samples_by_cls = {label: [] for label in self.id2cat.keys()}
        num_gts = {label: 0 for label in self.id2cat.keys()}
        num_preds = {label: 0 for label in self.id2cat.keys()}

        for token, gt in self.gts.items():
            pred = predictions.get(token, {'vectors': [], 'scores': [], 'labels': []})

            vectors_by_cls = {label: [] for label in self.id2cat.keys()}
            scores_by_cls = {label: [] for label in self.id2cat.keys()}

            for i in range(len(pred['labels'])):
                label = pred['labels'][i]
                vector = pred['vectors'][i]
                score = pred['scores'][i]
                vectors_by_cls[label].append(vector)
                scores_by_cls[label].append(score)

            for label in self.id2cat.keys():
                samples_by_cls[label].append(
                    (vectors_by_cls[label], scores_by_cls[label], gt.get(label, [])))
                num_gts[label] += len(gt.get(label, []))
                num_preds[label] += len(vectors_by_cls[label])

        result_dict = {}
        print(f'\nevaluating {len(self.id2cat)} categories...')

        for label in self.id2cat.keys():
            samples = samples_by_cls[label]
            result_dict[self.id2cat[label]] = {
                'num_gts': num_gts[label],
                'num_preds': num_preds[label],
            }

            fn = partial(self._evaluate_single, thresholds=self.thresholds, metric=metric)
            if self.n_workers > 0:
                with Pool(self.n_workers) as pool:
                    tpfp_score_list = pool.starmap(fn, samples)
            else:
                tpfp_score_list = [fn(*sample) for sample in samples]

            sum_ap = 0.
            for thr in self.thresholds:
                tp_fp_score = np.vstack([s[thr] for s in tpfp_score_list])
                sort_inds = np.argsort(-tp_fp_score[:, -1])
                tp = tp_fp_score[sort_inds, 0]
                fp = tp_fp_score[sort_inds, 1]
                tp = np.cumsum(tp, axis=0)
                fp = np.cumsum(fp, axis=0)
                eps = np.finfo(np.float32).eps
                recalls = tp / np.maximum(num_gts[label], eps)
                precisions = tp / np.maximum(tp + fp, eps)
                ap = average_precision(recalls, precisions, 'area')
                sum_ap += ap
                result_dict[self.id2cat[label]][f'AP@{thr}'] = ap

            ap = sum_ap / len(self.thresholds)
            result_dict[self.id2cat[label]]['AP'] = ap

        mAP = sum(result_dict[cat]['AP'] for cat in self.id2cat.values()) / len(self.id2cat)
        result_dict['mAP'] = mAP
        return result_dict

    def print_results(self, result_dict: Dict[str, float]):
        try:
            import prettytable
            table = prettytable.PrettyTable(
                ['category', 'num_preds', 'num_gts'] +
                [f'AP@{thr}' for thr in self.thresholds] + ['AP'])
            for label in self.id2cat.values():
                r = result_dict[label]
                table.add_row([
                    label,
                    r['num_preds'],
                    r['num_gts'],
                    *[round(r[f'AP@{thr}'], 4) for thr in self.thresholds],
                    round(r['AP'], 4),
                ])
            print(table)
        except ImportError:
            self._print_results_fallback(result_dict)

        mAP = result_dict['mAP']
        mAP_normal = sum(
            result_dict[cat][f'AP@{thr}']
            for cat in self.id2cat.values()
            for thr in self.thresholds
        ) / (len(self.id2cat) * len(self.thresholds))

        print(f'mAP = {mAP:.4f}')
        print(f'mAP_normal = {mAP_normal:.4f}')

    def _print_results_fallback(self, result_dict: Dict):
        header = f"{'category':<12} {'preds':<8} {'gts':<8}"
        for thr in self.thresholds:
            header += f" {'AP@'+str(thr):<10}"
        header += f" {'AP':<8}"
        print(header)
        print('-' * len(header))
        for label in self.id2cat.values():
            r = result_dict[label]
            row = f"{label:<12} {r['num_preds']:<8} {r['num_gts']:<8}"
            for thr in self.thresholds:
                row += f" {round(r[f'AP@{thr}'], 4):<10}"
            row += f" {round(r['AP'], 4):<8}"
            print(row)
