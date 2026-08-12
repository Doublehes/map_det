import torch
from copy import deepcopy

from configs.default import AttrDict


cfg_default = AttrDict({
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    'data': AttrDict({
        'train_ann_file': '/home/double/Documents/wangjiang/line_data/dctj218_yubei.pkl',
        'val_ann_file': '/home/double/Documents/wangjiang/line_data/trainlabel_sampled_209.pkl',
        'pc_range': [-10, -10, -3, 30, 10, 5],
        'num_classes': 2,
        'num_points': 16,
        'cat2id': {'guide_line': 0, 'boundary': 1},
        'roi_size': (40, 20),
        'canvas_size': (160, 320),  # 栅格 (H, W), 8×8 px/m
        'thickness': 4,
        'noise': AttrDict({
            'enabled': True,
            'sigma': 0.05,          # [0,1] 单位
            'train': True,          # 训练时加
            'eval': True,           # 推理/评测时加
        }),
        # 线增强 (训练时对 3D 线做变换, 栅格与 GT 同源生成, 天然一致)
        'bev_flip_prob': 0.5,   # 左右翻转概率 (y 反射), 0=关闭
        'bev_rot_angle': 10.0,  # 最大旋转角度(度), 0=关闭
        'bev_trans_x': 0.0,     # x 方向最大平移(米), 0=关闭
        'bev_trans_y': 3.0,     # y 方向最大平移(米), 0=关闭
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

# 配置变体注册表 (train.py / infer.py 通过 --config 选择)
CONFIGS = {
    'default': cfg_default,
    'resnet34': cfg_resnet34,
    'decode_layer3': cfg_decode_layer3,
}

cfg = deepcopy(cfg_default)