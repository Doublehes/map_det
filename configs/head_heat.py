"""
自定义配置示例

使用方法:
    python train.py --config configs/my_experiment.py
    python eval.py --config configs/my_experiment.py --checkpoint model.pth

说明:
    load_config() 执行此文件时，会注入 AttrDict 和 update_config 到命名空间，
    所以可以直接使用这两个名字。
"""
import copy

# ── step 1: 继承默认配置 ──
from configs.default import config_default as base_cfg
from configs.loader import AttrDict, update_config

config_default = AttrDict(copy.deepcopy(base_cfg))

# ── step 2: 用 update_config 批量覆盖（支持嵌套路径） ──
config_default.model.map_seg_head.enabled = False
config_default.model.heatmap_head.enabled = True
