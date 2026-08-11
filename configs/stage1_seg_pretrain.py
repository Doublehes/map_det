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

# ── 训练入口 ──
config_default.work_dir = './work_dirs/maptr_stage1'
config_default.pretrained = ''    # ← 用户在此填入 backbone 预训练权重路径
config_default.freeze_backbone = True
config_default.num_epochs = 6
