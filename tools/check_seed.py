"""Verify that the seed controls model init and DataLoader randomness.

Usage:
    python tools/check_seed.py --config configs_4090/stage2_joint_freeze_ec2_e72_d3_sgq_identity_refine.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models.backbone as B

_ORIG_RESNET50 = B.resnet50
B.resnet50 = lambda weights=None, **kw: _ORIG_RESNET50(weights=None)

from configs.loader import load_config
from data.dataset import MapTRDataset, collate_fn
from models.maptr import MapTR
from train import set_seed, _worker_init_fn


def _assert_state_equal(a, b, label):
    if set(a) != set(b):
        raise AssertionError(f'{label}: state_dict keys differ')
    for k in a:
        if not torch.equal(a[k], b[k]):
            raise AssertionError(f'{label}: tensor differs at {k}')
    print(f'[PASS] {label}')


def _assert_state_diff(a, b, label):
    for k in a:
        if not torch.equal(a[k], b[k]):
            print(f'[PASS] {label} (first diff: {k})')
            return
    raise AssertionError(f'{label}: state_dict is identical')


def _tree_equal(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def _assert_batch_equal(a, b, label):
    for key in ('imgs', 'intrinsics', 'extrinsics', 'vectors'):
        if not _tree_equal(a[key], b[key]):
            raise AssertionError(f'{label}: batch differs at {key}')
    print(f'[PASS] {label}')


def _assert_batch_diff(a, b, label):
    for key in ('imgs', 'intrinsics', 'extrinsics', 'vectors'):
        if not _tree_equal(a[key], b[key]):
            print(f'[PASS] {label} (first diff: {key})')
            return
    raise AssertionError(f'{label}: batches are identical')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs_4090/stage2_joint_freeze_ec2_e72_d3_sgq_identity_refine.py')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.model.img_backbone.weights = ''
    cfg.data.compute_heatmap = False

    def build_model(seed):
        set_seed(seed)
        return MapTR(cfg.model).state_dict()

    sd0 = build_model(0)
    sd0b = build_model(0)
    sd1 = build_model(1)
    _assert_state_equal(sd0, sd0b, 'same seed -> same model init')
    _assert_state_diff(sd0, sd1, 'different seed -> different model init')

    ds = MapTRDataset(
        ann_file=cfg.data.train_ann_file,
        data_root=cfg.data.data_root,
        cfg=cfg.data,
        is_train=True,
    )

    def first_batch(seed, num_workers):
        set_seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            ds,
            batch_size=2,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            worker_init_fn=_worker_init_fn(1, seed),
            generator=generator,
        )
        return next(iter(loader))

    b0 = first_batch(0, 0)
    b0b = first_batch(0, 0)
    b0w = first_batch(0, 2)
    b0wb = first_batch(0, 2)
    b1 = first_batch(1, 0)
    _assert_batch_equal(b0, b0b, 'same seed, workers=0 -> same first batch')
    _assert_batch_equal(b0w, b0wb, 'same seed, workers=2 -> same first batch')
    _assert_batch_diff(b0, b1, 'different seed -> different first batch')

    print('SEED CHECK PASS')


if __name__ == '__main__':
    main()
