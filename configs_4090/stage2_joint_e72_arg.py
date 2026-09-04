"""
Stage-2 联合训练配置: det + seg (热力图头关闭), 加载 stage-1 权重, 全部参数可训练.

使用:
    python train.py configs/stage2_joint.py
"""
import copy

from configs.default import config_default as base_cfg
from configs.loader import AttrDict, update_config

config_default = AttrDict(copy.deepcopy(base_cfg))

# ── 模型: 检测头 + 分割头, 关闭热力图头 ──
config_default.model.map_det_head.enabled = True
config_default.model.map_seg_head.enabled = True
config_default.model.heatmap_head.enabled = False

# config_default.freeze_modules = ['backbone.', 'bev_encoder.', 'seg_head.', 'heatmap_head.']

# ── 数据 ──
config_default.data.batch_size = 8
config_default.data.bev_flip_prob = 0.5
config_default.data.bev_rot_angle = 10.0
config_default.data.bev_trans_x = 0.0
config_default.data.bev_trans_y = 3.0

# ── 训练入口 ──
config_default.work_dir = ''
config_default.pretrained = './work_dirs/stage1_seg_pretrain_e72_arg/latest.pth'
config_default.freeze_backbone = False
config_default.num_epochs = 72
