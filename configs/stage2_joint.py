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

# ── 训练入口 ──
config_default.work_dir = './work_dirs/maptr_stage2'
config_default.pretrained = './work_dirs/maptr_stage1/latest.pth'
config_default.freeze_backbone = False
config_default.num_epochs = 36
