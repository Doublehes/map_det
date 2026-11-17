"""
NPU Stage-2 联合训练: det + seg, 加载 Stage-1 权重, 全部参数可训练.

使用 (容器内, 工作目录 /workspace/code):
    python train.py configs_npu/stage2.py
"""
import copy

from configs.default import config_default as base_cfg
from configs.loader import AttrDict

config_default = AttrDict(copy.deepcopy(base_cfg))

# ── 数据 (容器内挂载路径) ──
config_default.data.data_root = '/workspace/data/trainlabel_line_data_multiview'
config_default.data.train_ann_file = (
    '/workspace/data/trainlabel_line_multiview/'
    'sample_own_map_infos_merge_extra_train_len27139.pkl')
config_default.data.val_ann_file = (
    '/workspace/data/trainlabel_line_multiview/'
    'sample_own_map_infos_merge_extra_val_len272.pkl')
config_default.data.batch_size = 8
config_default.data.num_workers = 8
config_default.data.bev_flip_prob = 0.5
config_default.data.bev_rot_angle = 10.0
config_default.data.bev_trans_x = 0.0
config_default.data.bev_trans_y = 3.0

# ── ImageNet 权重 (无外网, 用本地文件; 会被 stage1 权重覆盖) ──
config_default.model.img_backbone.weights = (
    '/workspace/weights/torch_home/hub/checkpoints/resnet50-0676ba61.pth')

# ── 模型: 检测头 + 分割头 ──
config_default.model.map_det_head.enabled = True
config_default.model.map_seg_head.enabled = True
config_default.model.heatmap_head.enabled = False

# ── 训练入口: 从 stage1 加载, 全量微调 ──
config_default.work_dir = '/workspace/work_dirs/stage2_joint'
config_default.pretrained = '/workspace/work_dirs/stage1_seg_pretrain/latest.pth'
config_default.freeze_backbone = False
config_default.freeze_modules = []
config_default.num_epochs = 36
