"""model_slim2 配置: 2D 分割图 → 线检测 (自由折线, 自监督)

纯 2D 任务: 输入分割图, 输出每条线 16 个 (u,v) 归一化坐标点。
无 3D/BEV 参数 (pc_range/roi_size/canvas_size), 无朝向矫正。
"""
import torch
from copy import deepcopy

from configs.default import AttrDict


cfg_default = AttrDict({
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    'data': AttrDict({
        'seg_dir': '/home/double/Documents/wangjiang/code/map_det/data_seg/rt_bev_semantic',
        'num_classes': 4,
        'drivable_classes': (2,),      # 可通行类: 低速区 + 路面
        'ignore_index': 255,
        'cost_values': {0: 1.0, 1: 0.5, 2: 0.01, 3: 1.0},  # [0,1] 1.0=最高代价
        'ignore_cost': 1.0,
        'cost_sigma': 3.0,               # 代价图高斯平滑 sigma (像素)
        'dist_max_px': 10.0,             # 距离场归一化上限 (像素)
        'val_ratio': 0.9,
        'seed': 42,
        # 几何增强 (只作用于标签图, 信号同源派生)
        'flip_prob': 0.5,
        'rot_prob': 0.5,
        'rot_max_deg': 15.0,
        'bend_prob': 0.5,
        'bend_amp': 0.04,                # 归一化幅度 (相对图像宽)
        'bend_wavelength': (0.3, 0.8),   # 归一化波长 (相对图像宽)
        'batch_size': 16,
        'num_workers': 4,
    }),

    'backbone': AttrDict({
        'type': 'resnet',
        'depth': 18,
        'pretrained': False,
        'in_channels': 5,                # one-hot 4 + ignore 通道
    }),

    'neck': AttrDict({
        'type': 'fpn',
        'out_channels': 128,
        'use_level': 0,                  # P2 (stride4)
    }),

    'map_det_head': AttrDict({
        'type': 'maptr',
        'num_classes': 1,                # 只中心线一类
        'num_queries': 1,
        'num_points': 16,                # 每条线 16 个点
        'embed_dims': 128,
        'bev_embed_dims': 128,
        'num_heads': 4,
        'num_decoder_layers': 1,
        'dropout': 0.1,
        'ffn_channels': 256,
        'y_flip': False,                 # 特征图 row0=顶部, 与点 v=0 一致
        'bev_feat_net': AttrDict({'enabled': False}),
        # 自监督损失权重 (归一化坐标单位)
        'cost_weight': 1.0,
        'interior_weight': 0.5,
        'smooth_weight': 2.0,
        'repel_weight': 1.0,
        'repel_margin': 0.15,
        'length_weight': 1.0,
        'min_length': 0.8,
        'spread_weight': 0.0,
        'spread_thr': 0.9,
        'order_weight': 0.0,
        'grow_weight': 0.0,   # 增长激励权重 (0=关闭; 弧长度量易诱发锯齿, 调大需同步加大 smooth/order)
    }),

    'num_epochs': 100,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'grad_clip_max_norm': 35.0,
    'scheduler': 'cosine',
    'warmup_iters': 100,
    'work_dir': './work_dirs/slim2',
    'pretrained': '',
    'val_interval': 5,
    'cost_filter_thr': 0.3,
    'log_interval': 20,
    'checkpoint_interval': 10,
})

CONFIGS = {
    'default': cfg_default,
}

cfg = cfg_default
