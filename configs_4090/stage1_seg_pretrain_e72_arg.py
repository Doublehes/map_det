"""
Stage-1 分割预训练配置: 关闭检测头, 冻结 backbone, 仅训练 seg_head + bev_encoder.

使用:
    python train.py configs/stage1_seg_pretrain.py

backbone 预训练权重路径请在下方 pretrained 字段填写, 空串=不加载.
"""
import copy

from configs.default import config_default as base_cfg
from configs.loader import AttrDict, update_config

config_default = AttrDict(copy.deepcopy(base_cfg))

# ── 模型: 仅分割头, 关闭检测头和热力图头 ──
config_default.model.map_det_head.enabled = False
config_default.model.map_seg_head.enabled = True
config_default.model.heatmap_head.enabled = False

# ── 数据 ──
config_default.data.batch_size = 8
config_default.data.bev_flip_prob = 0.5
config_default.data.bev_rot_angle = 10.0
config_default.data.bev_trans_x = 0.0
config_default.data.bev_trans_y = 3.0

# ── 训练入口 ──
config_default.work_dir = ''
config_default.pretrained = 'work_dirs/stage1_seg_pretrain/latest.pth'    # ← 用户在此填入 backbone 预训练权重路径
config_default.freeze_backbone = False
config_default.num_epochs = 72
