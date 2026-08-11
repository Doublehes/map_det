#!/bin/bash
set -e

# 所有训练参数均在 config 中配置 (work_dir / pretrained / freeze_backbone / num_epochs 等)
# backbone 预训练权重路径在 configs/stage1_seg_pretrain.py 的 pretrained 字段填写

# Stage 1: 分割预训练 (关闭检测头, 冻结 backbone)
echo "========== Stage 1: Segment-only pretraining =========="
python train.py configs/stage1_seg_pretrain.py

# Stage 2: 联合训练 (加载 stage1 权重, 全部参数可训练)
echo "========== Stage 2: Joint training (seg + cls + reg) =========="
python train.py configs/stage2_joint.py
