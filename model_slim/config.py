import torch
from copy import deepcopy

from configs.default import AttrDict


cfg_default = AttrDict({
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    'data': AttrDict({
        'train_ann_file': '/home/double/Documents/wangjiang/line_data/original_label/label_I5_S0__dctj218_yubei__tms_dazu_20260416__len2471.pkl',
        'val_ann_file': '/home/double/Documents/wangjiang/line_data/original_label/label_I100_S5__dctj218_yubei__tms_dazu_20260416__x30_jialing_20260612__x30_jialing_road__len230.pkl',
        'pc_range': [-10, -10, -3, 30, 10, 5],
        'num_classes': 2,
        'num_points': 16,
        'cat2id': {'guide_line': 0, 'boundary': 1},
        'roi_size': (40, 20),
        'canvas_size': (160, 320),  # 栅格 (H, W), 8×8 px/m
        'seg_canvas_size': (80, 160),  # 分割 GT 分辨率 (低分辨率算 loss)
        'thickness': 4,
        'noise': AttrDict({
            'enabled': True,
            'sigma': 0.05,          # [0,1] 单位
            'train': True,          # 训练时加
            'eval': True,           # 推理/评测时加
        }),
        # 线增强 (训练时对 3D 线做变换, 栅格与 GT 同源生成, 天然一致)
        'bev_flip_prob': 0.0,   # 左右翻转概率 (y 反射), 0=关闭
        'bev_rot_angle': 0.0,  # 最大旋转角度(度), 0=关闭
        'bev_scale': 0.0,      # 尺度变换幅度: 系数 ∈ [1-0.15, 1+0.15], 0=关闭
        'bev_trans_x': 0.0,     # x 方向最大平移(米), 0=关闭
        'bev_trans_y': 0.0,     # y 方向最大平移(米), 0=关闭

        # 弯曲增强,sin wave. Y = Y0 + A * sin(2 * pi * x / wavelength)
        'bev_bend_amp': 2.0,    # 最大弯曲幅度(米), 实际随机 [0.5, 2.0], 0=关闭
        'bev_bend_prob': 0.0,   # 弯曲应用概率
        'bev_bend_wavelength': (4.0, 12.0),  # 正弦波长范围(米)

        # 合成边界线数据 (触发时清空原 map_geom, 只保留合成的 cls1 边界线)
        'syn_boundary_prob': 0.0,       # 合成数据概率, 0=关闭
        'syn_boundary_min': 3,          # 合成线总数下限
        'syn_boundary_max': 10,         # 合成线总数上限
        'syn_lshape_prob': 0.5,         # L 型(90°垂直)占比 (非斜线中)
        'syn_slanted_prob': 0.0,        # 陡峭直线边界占比 (θ∈[60°,120°]), 0=关闭
        'syn_len_range': (8.0, 30.0),   # 合成线总长范围(米)
        'syn_bend_amp': 4,            # 大曲率弯曲幅度(米)
        'syn_bend_wavelength': (15.0, 30.0),  # 弯曲波长范围(米)
        'syn_clearance': 1.5,           # 线间最小间距(米)

        # 合成中心线数据 (L 型, 沿 x 增大方向; 与边界线每帧互斥)
        'syn_center_prob': 0.0,         # 合成中心线概率(每帧), 0=关闭
        'syn_center_min': 1,            # 中心线条数下限
        'syn_center_max': 3,            # 中心线条数上限
        'syn_center_len_range': (10.0, 40.0),  # 中心线总长范围(米)

        'batch_size': 16,
        'num_workers': 4,
    }),

    'backbone': AttrDict({
        'type': 'resnet',
        'depth': 18,          # 18/34/50/101/152
        'pretrained': False,  # ImageNet 权重 (合成图任务建议 False)
    }),

    'neck': AttrDict({
        'type': 'fpn',
        'in_channels': [],    # 空=自动取 backbone.out_channels
        'out_channels': 256,  # = bev_embed_dims
        'use_level': 0,       # 0=P2 最细层 (stride4, 40×80)
    }),

    'map_det_head': AttrDict({
        'type': 'maptr',
        'num_classes': 2,
        'num_queries': 32,
        'num_points': 16,
        'embed_dims': 256,
        'bev_embed_dims': 256,
        'num_heads': 8,
        'num_decoder_layers': 1,
        'dropout': 0.1,
        'ffn_channels': 512,
        'y_flip': False,    # slim 栅格 y=0 在顶部, 无需解码器 y 反转 (原始 MapTR 为 True)
        'bev_feat_net': AttrDict({
            'enabled': False,
        }),
        'matcher_cls_weight': 5.0,
        'matcher_reg_weight': 50.0,
        'loss_cls_weight': 5.0,
        'loss_reg_weight': 50.0,
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'l1_beta': 0.01,
        'aux_loss': False,   # 多层解码时是否每层监督, 需显式开启
        'aux_weight': 1.0,   # 中间层损失权重, 最后一层恒为 1.0
    }),

    'map_seg_head': AttrDict({
        'enabled': False,
        'bev_embed_dims': 256,
        'num_classes': 2,
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'loss_seg_weight': 100.0,
        'loss_dice_weight': 1.0,
    }),

    'num_epochs': 48,
    'lr': 5e-4,
    'weight_decay': 1e-2,
    'grad_clip_max_norm': 35.0,
    'scheduler': 'cosine',
    'warmup_iters': 200,
    'work_dir': './work_dirs/default',
    'pretrained': '',          # 预训练权重路径, 空=不加载
    'eval_thresholds': [0.5, 1.0, 1.5],
    'val_interval': 2,         # 每 N 个 epoch 评测一次, 0=禁用
    'score_thr': 0.3,          # 评测置信度阈值
    'eval_workers': 4,         # 评测匹配多进程数
    'log_interval': 50,
    'checkpoint_interval': 6,
})

cfg_decode_layer3 = deepcopy(cfg_default)
cfg_decode_layer3['map_det_head']['num_decoder_layers'] = 3
cfg_decode_layer3['work_dir'] = './work_dirs/decode_layer3'

cfg_resnet34 = deepcopy(cfg_default)
cfg_resnet34['backbone']['depth'] = 34
cfg_resnet34['work_dir'] = './work_dirs/resnet34'

cfg_argument1 = deepcopy(cfg_default)
cfg_argument1['data']['bev_flip_prob'] = 0.5
cfg_argument1['data']['bev_rot_angle'] = 10.0
cfg_argument1['data']['bev_scale'] = 0.15
cfg_argument1['data']['bev_trans_x'] = 0.0
cfg_argument1['data']['bev_trans_y'] = 1.0
cfg_argument1['work_dir'] = './work_dirs/argument1'

cfg_argument2 = deepcopy(cfg_default)
cfg_argument2['data']['bev_flip_prob'] = 0.5
cfg_argument2['data']['bev_rot_angle'] = 30.0
cfg_argument2['data']['bev_scale'] = 0.5
cfg_argument2['data']['bev_trans_x'] = 5.0
cfg_argument2['data']['bev_trans_y'] = 3.0
cfg_argument2['work_dir'] = './work_dirs/argument2'

cfg_argu2_dlayer3 = deepcopy(cfg_argument2)
cfg_argu2_dlayer3['map_det_head']['num_decoder_layers'] = 3
cfg_argu2_dlayer3['work_dir'] = './work_dirs/argument2_dlayer3'

cfg_argu2_res34 = deepcopy(cfg_argument2)
cfg_argu2_res34['backbone']['depth'] = 34
cfg_argu2_res34['work_dir'] = './work_dirs/argument2_res34'


cfg_argu2_dlayer3_no_aux = deepcopy(cfg_argument2)
cfg_argu2_dlayer3_no_aux['map_det_head']['num_decoder_layers'] = 3
cfg_argu2_dlayer3_no_aux['work_dir'] = './work_dirs/argument2_dlayer3_no_aux'

cfg_argu2_dlayer3_aux = deepcopy(cfg_argument2)
cfg_argu2_dlayer3_aux['map_det_head']['num_decoder_layers'] = 3
cfg_argu2_dlayer3_aux['map_det_head']['aux_loss'] = True
cfg_argu2_dlayer3_aux['work_dir'] = './work_dirs/argument2_dlayer3_aux'

cfg_argu2_dlayer3_aux_w0_5 = deepcopy(cfg_argument2)
cfg_argu2_dlayer3_aux_w0_5['map_det_head']['num_decoder_layers'] = 3
cfg_argu2_dlayer3_aux_w0_5['map_det_head']['aux_loss'] = True
cfg_argu2_dlayer3_aux_w0_5['map_det_head']['aux_weight'] = 0.5
cfg_argu2_dlayer3_aux_w0_5['work_dir'] = './work_dirs/argument2_dlayer3_aux_w0.5'

cfg_argu2_dlayer6_aux = deepcopy(cfg_argument2)
cfg_argu2_dlayer6_aux['map_det_head']['num_decoder_layers'] = 6
cfg_argu2_dlayer6_aux['map_det_head']['aux_loss'] = True
cfg_argu2_dlayer6_aux['work_dir'] = './work_dirs/argument2_dlayer6_aux'

cfg_argu2_dlayer3_aux_e90 = deepcopy(cfg_argu2_dlayer3_aux)
cfg_argu2_dlayer3_aux_e90['num_epochs'] = 90
cfg_argu2_dlayer3_aux_e90['work_dir'] = './work_dirs/argument2_dlayer3_aux_e90'

cfg_argu2_dlayer3_aux_q64 = deepcopy(cfg_argu2_dlayer3_aux)
cfg_argu2_dlayer3_aux_q64['map_det_head']['num_queries'] = 64
cfg_argu2_dlayer3_aux_q64['work_dir'] = './work_dirs/argument2_dlayer3_aux_q64'

cfg_argu2_dlayer3_aux_e90_q64 = deepcopy(cfg_argu2_dlayer3_aux)
cfg_argu2_dlayer3_aux_e90_q64['num_epochs'] = 90
cfg_argu2_dlayer3_aux_e90_q64['map_det_head']['num_queries'] = 64
cfg_argu2_dlayer3_aux_e90_q64['work_dir'] = './work_dirs/argument2_dlayer3_aux_e90_q64'

cfg_argu2_e90 = deepcopy(cfg_argument2)
cfg_argu2_e90['num_epochs'] = 90
cfg_argu2_e90['work_dir'] = './work_dirs/argument2_e90'

cfg_argu2_e90_q64 = deepcopy(cfg_argument2)
cfg_argu2_e90_q64['num_epochs'] = 90
cfg_argu2_e90_q64['map_det_head']['num_queries'] = 64
cfg_argu2_e90_q64['work_dir'] = './work_dirs/argument2_e90_q64'

cfg_argu2_q64 = deepcopy(cfg_argument2)
cfg_argu2_q64['map_det_head']['num_queries'] = 64
cfg_argu2_q64['work_dir'] = './work_dirs/argument2_q64'


cfg_argument3 = deepcopy(cfg_argument2)
cfg_argument3['data']['bev_bend_amp'] = 4.0                  # 最大弯曲幅度(米), 0=关闭
cfg_argument3['data']['bev_bend_prob'] = 0.5                 # 弯曲应用概率
cfg_argument3['data']['bev_bend_wavelength'] = (15.0, 30.0)   # 正弦波长范围(米)
cfg_argument3['work_dir'] = './work_dirs/argument3'

cfg_argu3_dlayer3_aux = deepcopy(cfg_argument3)
cfg_argu3_dlayer3_aux['map_det_head']['num_decoder_layers'] = 3
cfg_argu3_dlayer3_aux['map_det_head']['aux_loss'] = True
cfg_argu3_dlayer3_aux['work_dir'] = './work_dirs/argument3_dlayer3_aux'


cfg_argument4 = deepcopy(cfg_argument2)
cfg_argument4['data']['bev_rot_angle'] = 70.0
cfg_argument4['work_dir'] = './work_dirs/argument4'

cfg_argu4_dlayer3_aux = deepcopy(cfg_argument4)
cfg_argu4_dlayer3_aux['map_det_head']['num_decoder_layers'] = 3
cfg_argu4_dlayer3_aux['map_det_head']['aux_loss'] = True
cfg_argu4_dlayer3_aux['work_dir'] = './work_dirs/argument4_dlayer3_aux'


cfg_argument5 = deepcopy(cfg_argument3)
cfg_argument5['data']['bev_bend_prob'] = 0.7
cfg_argument5['data']['syn_boundary_prob'] = 0.3   # 开启合成边界线数据
cfg_argument5['data']['syn_center_prob'] = 0.3     # 开启合成中心线数据 (与边界线分段互斥)
cfg_argument5['work_dir'] = './work_dirs/argument5'

cfg_argu5_dlayer3_aux = deepcopy(cfg_argument5)
cfg_argu5_dlayer3_aux['map_det_head']['num_decoder_layers'] = 3
cfg_argu5_dlayer3_aux['map_det_head']['aux_loss'] = True
cfg_argu5_dlayer3_aux['work_dir'] = './work_dirs/argument5_dlayer3_aux2'

cfg_argu5_dlayer3_aux_e90 = deepcopy(cfg_argu5_dlayer3_aux)
cfg_argu5_dlayer3_aux_e90['num_epochs'] = 90
cfg_argu5_dlayer3_aux_e90['work_dir'] = './work_dirs/argument5_dlayer3_aux_e90'

cfg_argu5_dlayer3_aux_q64 = deepcopy(cfg_argu5_dlayer3_aux)
cfg_argu5_dlayer3_aux_q64['map_det_head']['num_queries'] = 64
cfg_argu5_dlayer3_aux_q64['work_dir'] = './work_dirs/argument5_dlayer3_aux_q64'

cfg_argu5_dlayer3_aux_q64_e90 = deepcopy(cfg_argu5_dlayer3_aux_q64)
cfg_argu5_dlayer3_aux_q64_e90['num_epochs'] = 90
cfg_argu5_dlayer3_aux_q64_e90['work_dir'] = './work_dirs/argument5_dlayer3_aux_q64_e90_2'

cfg_argu5_dlayer3_aux_q64_e90_slanted = deepcopy(cfg_argu5_dlayer3_aux_q64_e90)
cfg_argu5_dlayer3_aux_q64_e90_slanted['data']['syn_slanted_prob'] = 0.2
cfg_argu5_dlayer3_aux_q64_e90_slanted['work_dir'] = './work_dirs/argument5_dlayer3_aux_q64_e90_slanted'

cfg_argu5_dlayer6_aux_q64_e90 = deepcopy(cfg_argu5_dlayer3_aux_q64_e90)
cfg_argu5_dlayer6_aux_q64_e90['map_det_head']['num_decoder_layers'] = 6
cfg_argu5_dlayer6_aux_q64_e90['work_dir'] = './work_dirs/argument5_dlayer6_aux_q64_e90'

cfg_argu5_dlayer3_aux_seg = deepcopy(cfg_argu5_dlayer3_aux)
cfg_argu5_dlayer3_aux_seg['map_seg_head']['enabled'] = True
cfg_argu5_dlayer3_aux_seg['work_dir'] = './work_dirs/argument5_dlayer3_aux_seg'

cfg_argu5_dlayer3_aux_dim512 = deepcopy(cfg_argu5_dlayer3_aux)
cfg_argu5_dlayer3_aux_dim512['map_det_head']['embed_dims'] = 512
cfg_argu5_dlayer3_aux_dim512['work_dir'] = './work_dirs/argument5_dlayer3_aux_dim512'



# 配置变体注册表 (train.py / infer.py 通过 --config 选择)
CONFIGS = {
    'default': cfg_default,
    'resnet34': cfg_resnet34,
    'decode_layer3': cfg_decode_layer3,
    'argument1': cfg_argument1,
    'argument2': cfg_argument2,
    'argument2_dlayer3': cfg_argu2_dlayer3,
    'argument2_res34': cfg_argu2_res34,
    'argument2_dlayer3_no_aux': cfg_argu2_dlayer3_no_aux,
    'argument2_dlayer3_aux': cfg_argu2_dlayer3_aux,
    'argument2_dlayer3_aux_w0_5': cfg_argu2_dlayer3_aux_w0_5,
    'argument2_dlayer6_aux': cfg_argu2_dlayer6_aux,
    'argument2_dlayer3_aux_e90': cfg_argu2_dlayer3_aux_e90,
    'argument2_e90': cfg_argu2_e90,
    'argument2_e90_q64': cfg_argu2_e90_q64,
    'argument2_dlayer3_aux_e90_q64': cfg_argu2_dlayer3_aux_e90_q64,
    'argument2_dlayer3_aux_q64': cfg_argu2_dlayer3_aux_q64,
    'argument3': cfg_argument3,
    'argument3_dlayer3_aux': cfg_argu3_dlayer3_aux,
    'argument4': cfg_argument4,
    'argument4_dlayer3_aux': cfg_argu4_dlayer3_aux,
    'argument5': cfg_argument5,
    'argument5_dlayer3_aux': cfg_argu5_dlayer3_aux,
    'argument5_dlayer3_aux_e90': cfg_argu5_dlayer3_aux_e90,
    'argument5_dlayer3_aux_q64': cfg_argu5_dlayer3_aux_q64,
    'argument5_dlayer3_aux_q64_e90': cfg_argu5_dlayer3_aux_q64_e90,
    'argument5_dlayer3_aux_q64_e90_slanted': cfg_argu5_dlayer3_aux_q64_e90_slanted,
    'argument5_dlayer6_aux_q64_e90': cfg_argu5_dlayer6_aux_q64_e90,
    'argument5_dlayer3_aux_seg': cfg_argu5_dlayer3_aux_seg,
    'argument5_dlayer3_aux_dim512': cfg_argu5_dlayer3_aux_dim512,
}

cfg = deepcopy(cfg_default)