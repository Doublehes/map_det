"""
Stage-2 联合训练配置 (d3): 直线 grid 初始化 reference points (可学习) + delta 细化.

与 stage2_joint_freeze_ec2_e72_d3.py 相比改:
    map_det_head.ref_init='line'
    map_det_head.predict_refine=True

使用:
    python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3_line_refine.py
"""
import copy

from configs.default import config_default as base_cfg
from configs.loader import AttrDict, update_config

config_default = AttrDict(copy.deepcopy(base_cfg))

# ── 模型: 检测头 + 分割头, 关闭热力图头 ──
config_default.model.map_det_head.enabled = True
config_default.model.map_det_head.aux_loss = True
config_default.model.map_det_head.num_decoder_layers = 3
config_default.model.map_det_head.ref_init = 'line'
config_default.model.map_det_head.predict_refine = True
config_default.model.map_seg_head.enabled = True
config_default.model.heatmap_head.enabled = False
config_default.model.map_det_head.bev_feat_net.enabled = True

config_default.freeze_modules = ['backbone.', 'bev_encoder.', 'seg_head.', 'heatmap_head.']

# ── 数据 ──
config_default.data.batch_size = 16

# ── 训练入口 ──
config_default.work_dir = ''
config_default.pretrained = './work_dirs/stage1_seg_pretrain_e72_arg/latest.pth'
config_default.freeze_backbone = False
config_default.num_epochs = 72
