#!/bin/bash
set -e

STAGE1_DIR="./work_dirs/maptr_stage1"
STAGE2_DIR="./work_dirs/maptr_stage2"
PRETRAINED=""    # 用户自行填入 backbone 预训练权重路径, e.g. ./pretrained/resnet50.pth

# Stage 1: 分割预训练 (map_det_head.enabled=False, 仅训练 seg_head + bev_encoder, backbone 冻结)
echo "========== Stage 1: Segment-only pretraining =========="
python train.py \
    --work-dir "$STAGE1_DIR" \
    --pretrained "$PRETRAINED" \
    --freeze-backbone \
    --epochs 6 \
    configs/head_seg_pretrain.py

# Stage 2: 联合训练 (加载 stage1 权重, 全部参数可训练)
echo "========== Stage 2: Joint training (seg + cls + reg) =========="
python train.py \
    --work-dir "$STAGE2_DIR" \
    --pretrained "$STAGE1_DIR/latest.pth" \
    --epochs 36 \
    configs/head_seg.py
