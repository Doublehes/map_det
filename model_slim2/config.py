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
        'seg_dir': '/home/flow/Code/MilitaryModel/map_det/data_seg/rt_bev_semantic',
        'num_classes': 4,
        'drivable_classes': (2,),      # 可通行类: 低速区 + 路面
        'ignore_index': 255,
        'cost_values': {0: 1.0, 1: 0.5, 2: 0.01, 3: 1.0},  # [0,1] 1.0=最高代价
        'ignore_cost': 1.0,
        'cost_sigma': 3.0,               # 代价图高斯平滑 sigma (像素)
        'dist_max_px': 10.0,             # [旧场] 距离场归一化上限 (像素)
        'dist_field_mode': 'new',        # 距离场模式: 'new'=窗口归一化场(默认) | 'old'=原实现
        'dist_min_road_width': 5.0,      # [新场] 开运算剔除 < 该宽度的窄结构 (像素)
        'dist_ridge_window': 30.0,       # [新场] 局部最大(脊线高度)窗口半径 (像素)
        'val_ratio': 0.1,
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
        # 有效项: cost / interior / smooth / spacing / arc / span (均 >=0, 线内逐点/逐段用 Lp 聚合)
        # 关闭项: repel / spread (单线场景下为 no-op)
        'cost_weight': 0.0,
        'interior_weight': 1.0,
        'smooth_weight': 2.0,            # 形状正则: 转向(防打结) + 最小步长(防挤堆)
        'smooth_min_step_px': 2.0,       # 相邻段最小步长(像素), 段更短时惩罚
        'spacing_weight': 1.0,           # 等距正则: 段长相对偏差(尺度无关)
        
        'arc_weight': 1.0,            # 弧长下界 (真实长度), 逼线张开
        'min_arc': 0.4,
        'span_weight': 1.0,             # 端点距下界: 逼线张开成条(防紧凑团/环), 与弧长互补
        'min_span': 0.4,

        'repel_weight': 0.0,
        'repel_margin': 0.15,
        'spread_weight': 0.0,
        'spread_thr': 0.9,

        # 多线结构损失 (默认权重 0; 具体数值见 exp3)
        'start_gather_weight': 0.0,      # 低端聚合: 各线共享车后起点
        'diverge_weight': 0.0,           # 顶端发散: 各线车前端点互相拉开
        'diverge_margin': 0.2,           # 顶端最小间距 (归一化)

        'lp_p': 4.0,                     # 线内 Lp 聚合指数 (>1 强调最差点/段, 1=均值)
    }),

    'num_epochs': 100,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'grad_clip_max_norm': 35.0,
    'scheduler': 'cosine',
    'warmup_iters': 100,
    'work_dir': './work_dirs/default',
    'pretrained': '',
    'val_interval': 5,
    'cost_filter_thr': 0.99,
    'log_interval': 20,
    'checkpoint_interval': 50,
})

exp1 = deepcopy(cfg_default)
exp1['map_det_head']['interior_weight'] = 1.0
exp1['num_epochs'] = 200
exp1['map_det_head']['min_arc'] = 0.8
exp1['work_dir'] = './work_dirs/exp1'

exp2 = deepcopy(cfg_default)
exp2['map_det_head']['cost_weight'] = 1.0
exp2['map_det_head']['interior_weight'] = 0.0
exp2['map_det_head']['min_arc'] = 0.8
exp2['num_epochs'] = 200
exp2['work_dir'] = './work_dirs/exp2'

exp3 = deepcopy(cfg_default)
exp3['map_det_head']['num_queries'] = 3
exp3['map_det_head']['spread_weight'] = 0.0
exp3['map_det_head']['start_gather_weight'] = 1.0
exp3['map_det_head']['diverge_weight'] = 1.0
exp3['map_det_head']['diverge_margin'] = 0.2
exp3['work_dir'] = './work_dirs/exp3'

CONFIGS = {
    'default': cfg_default,
    'exp1': exp1,
    'exp2': exp2,
    'exp3': exp3,
}

cfg = cfg_default
