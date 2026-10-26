"""model_slim2 配置: 2D 分割图 → 线检测 (自由折线, 自监督)

纯 2D 任务: 输入分割图, 输出每条线 16 个 (u,v) 归一化坐标点。
无 3D/BEV 参数 (pc_range/roi_size/canvas_size), 无朝向矫正。
"""
import torch
from copy import deepcopy

from configs.default import AttrDict


# ── 可视化颜色 (BGR; infer.py / vis_dist_field.py 共用) ──
TARGET_CLASSES_4 = [
    {"id": 0, "name": "high_cost_obstacle", "rgb": [255, 41, 10]},
    {"id": 1, "name": "low_cost_obstacle", "rgb": [4, 250, 7]},
    {"id": 2, "name": "road", "rgb": [230, 230, 230]},
    {"id": 3, "name": "other", "rgb": [80, 50, 50]},
]
TARGET_CLASSES_6 = [
    {"id": 0, "name": "person", "rgb": [255, 0, 255]},
    {"id": 1, "name": "vehicle", "rgb": [255, 194, 7]},
    {"id": 2, "name": "high_cost_obstacle", "rgb": [255, 41, 10]},
    {"id": 3, "name": "low_cost_obstacle", "rgb": [4, 250, 7]},
    {"id": 4, "name": "road", "rgb": [230, 230, 230]},
    {"id": 5, "name": "other", "rgb": [80, 50, 50]},
]
use_6_classes = True
IGNORE_COLOR = (60, 60, 60)
CLASS_COLORS = {cls["id"]: tuple(cls["rgb"][::-1]) for cls in (TARGET_CLASSES_6 if use_6_classes else TARGET_CLASSES_4)}
NUM_CLASSES = len(CLASS_COLORS)
DRIVABLE_CLASSES = (4,) if use_6_classes else (2,)
IGNORE_INDEX = 255
COST_VALUES = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5, 4: 0.01, 5: 1.0} if use_6_classes else {0: 1.0, 1: 0.5, 2: 0.01, 3: 1.0}
IGNORE_COST = 1.0

cfg_default = AttrDict({
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    'data': AttrDict({
        # 训练/验证分割图目录: str 或 [str, ...] (list 时合并所有目录的 PNG)
        'train_dir': ['/home/flow/Code/bev_semantic/dazu_tms_1_cvt_label__norm__tms__scene1__bev_semantic',
                      '/home/flow/Code/bev_semantic/dazu_tms_2_cvt_label__norm__tms__scene2__bev_semantic'],
        'val_dir':   '/home/flow/Code/bev_semantic/dazu_tms_2_cvt_label__norm__tms__scene2__bev_semantic',  # 暂与 train 相同
        'num_classes': NUM_CLASSES,
        'drivable_classes': DRIVABLE_CLASSES,      # 可通行类: 低速区 + 路面
        'ignore_index': IGNORE_INDEX,
        'cost_values': COST_VALUES,  # [0,1] 1.0=最高代价
        'ignore_cost': IGNORE_COST,
        'cost_sigma': 3.0,               # 代价图高斯平滑 sigma (像素)
        'dist_max_px': 10.0,             # [旧场] 距离场归一化上限 (像素)
        'dist_field_mode': 'new',        # 距离场模式: 'new'=窗口归一化场(默认) | 'old'=原实现
        'dist_min_road_width': 5.0,      # [新场] 开运算剔除 < 该宽度的窄结构 (像素)
        'dist_ridge_window': 30.0,       # [新场] 局部最大(脊线高度)窗口半径 (像素)
        # 车前关键点提取 (从窗口场 dist_field 局部极大提取; 见 dataset.extract_keypoints_lm)
        'kp_win': 2.0,                   # 局部极大窗口半径 (px)
        'kp_eps': 0.02,                  # 平台容差 (字段单位 [0,1])
        'kp_field_floor': 0.5,           # 字段强度下限 [0,1]
        'kp_dmin': 0.05,                  # 关键点最小间隔 (归一化, NMS)
        'kp_max': 100,                   # 最多关键点个数 (同时也是 padding 长度)
        'kp_vfront': 0.7,                # 只看车前: 保留 v <= 该值 (v=0 顶部=车前)

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
        'in_channels': NUM_CLASSES + 1,                # one-hot 4 + ignore 通道
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

        # 关键点覆盖率 (默认权重 0; 见 exp4)
        'keypoint_coverage_weight': 0.0,  # 覆盖损失权重
        'keypoint_coverage_thr': 0.05,    # 覆盖阈值 (归一化): 关键点到最近预测点 <= thr 视为覆盖

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

# exp2: cost损失, 取消 interior_weight
exp2 = deepcopy(cfg_default)
exp2['map_det_head']['cost_weight'] = 1.0
exp2['map_det_head']['interior_weight'] = 0.0
exp2['map_det_head']['min_arc'] = 0.8
exp2['num_epochs'] = 200
exp2['work_dir'] = './work_dirs/exp2'

# exp3: 多线结构损失 (要求输出线共享车后起点)
exp3 = deepcopy(cfg_default)
exp3['map_det_head']['num_queries'] = 3
exp3['map_det_head']['spread_weight'] = 0.0
exp3['map_det_head']['start_gather_weight'] = 1.0
exp3['map_det_head']['diverge_weight'] = 1.0
exp3['map_det_head']['diverge_margin'] = 0.2
exp3['work_dir'] = './work_dirs/exp3'

# exp4: 关键点覆盖率损失 (要求输出线覆盖车前关键点)
exp4 = deepcopy(cfg_default)
exp4['map_det_head']['num_queries'] = 3
exp4['map_det_head']['keypoint_coverage_weight'] = 2.0
exp4['map_det_head']['keypoint_coverage_thr'] = 0.05
exp4['map_det_head']['start_gather_weight'] = 1.0
# exp4['map_det_head']['diverge_weight'] = 1.0
# exp4['map_det_head']['diverge_margin'] = 0.2
exp4['work_dir'] = './work_dirs/exp4'


exp5 = deepcopy(exp4)
# exp5['num_epochs'] = 200
exp5['map_det_head']['keypoint_coverage_weight'] = 5.0
exp5['work_dir'] = './work_dirs/exp5'


exp6 = deepcopy(exp5)
exp6['map_det_head']['cost_weight'] = 1.0
exp6['map_det_head']['interior_weight'] = 0.0
exp6['map_det_head']['min_arc'] = 0.2
exp6['map_det_head']['min_span'] = 0.2
exp6['work_dir'] = './work_dirs/exp6'


# exp7: 3 层 decoder, 每层都回传 loss (权重 0.5 / 0.5 / 1.0), 共享 reg head
exp7 = deepcopy(exp5)
exp7['map_det_head']['num_decoder_layers'] = 3
exp7['map_det_head']['decoder_layer_loss_weights'] = [0.5, 0.5, 1.0]
exp7['work_dir'] = './work_dirs/exp7'

exp8 = deepcopy(exp5)
exp8['map_det_head']['num_decoder_layers'] = 6
exp8['map_det_head']['decoder_layer_loss_weights'] = [0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
exp8['work_dir'] = './work_dirs/exp8'

exp9 = deepcopy(exp7)
exp9['backbone']['depth'] = 50
exp9['backbone']['pretrained'] = True
exp9['work_dir'] = './work_dirs/exp9'


CONFIGS = {
    'default': cfg_default,
    'exp1': exp1,
    'exp2': exp2,
    'exp3': exp3,
    'exp4': exp4,
    'exp5': exp5,
    'exp6': exp6,
    'exp7': exp7,
    'exp8': exp8,
    'exp9': exp9,
}

cfg = cfg_default
