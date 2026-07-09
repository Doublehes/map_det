import torch
import copy


class AttrDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(key)

    def __setattr__(self, key, val):
        self[key] = val


def update_config(config, update_items):
    for k, v in update_items.items():
        if isinstance(v, dict):
            update_config(config[k], update_items[k])
        else:
            config[k] = v


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════
#  默认配置（扁平结构，按功能组排列）
# ══════════════════════════════════════════════════
config_default = AttrDict({
    'device': device,

    # ── 数据 ──
    'data_root': "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_data_multiview",
    'train_ann_file': "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_multiview/origin_label/owndata/dctj218_yubei_sampled_330.pkl",
    'val_ann_file': "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_multiview/origin_label/owndata/dctj218_yubei_sampled_330.pkl",
    'cat2id': {'guide_line': 0, 'boundary': 1},
    'num_classes': 2,
    'img_h': 176,
    'img_w': 320,
    'img_org_h': 352,
    'img_org_w': 640,
    'num_cams': 6,
    'img_norm': dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0]),
    'pc_range': [-10, -10, -3, 30, 10, 5],
    'roi_size': (40, 20),
    'bev_h': 40,
    'bev_w': 80,
    'canvas_size': (80, 160),
    'history_steps': 0,

    # ── 模型架构 ──
    'bev_embed_dims': 256,
    'embed_dims': 256,
    'num_feat_levels': 2,
    'num_points': 16,
    'num_queries': 32,
    'backbone_depth': 50,
    'fpn_in_channels': [512, 1024, 2048],
    'fpn_out_channels': 256,
    'bevformer_num_layers': 1,
    'num_points_in_pillar': 4,
    'num_sampling_points': 8,
    'decoder_num_layers': 1,
    'num_heads': 8,
    'ffn_channels': 512,
    'dropout': 0.1,

    # ── 训练超参数 ──
    'batch_size': 2,
    'num_epochs': 3,
    'num_workers': 4,
    'lr': 5e-4,
    'backbone_lr_mult': 0.1,
    'weight_decay': 1e-2,
    'grad_clip_max_norm': 35.0,
    'scheduler': 'step',
    'lr_milestones': [1, 2, 3],
    'lr_gamma': 0.1,
    'warmup_iters': 50,
    'warmup_ratio': 1.0 / 3,
    'min_lr_ratio': 1e-2,
    'log_interval': 50,

    # ── 损失权重 ──
    'loss_cls_weight': 5.0,
    'loss_reg_weight': 50.0,
    'loss_seg_weight': 10.0,
    'loss_dice_weight': 1.0,
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,
    'l1_beta': 0.01,

    # ── 评测 ──
    'score_thr': 0.3,
    'eval_thresholds': [0.5, 1.0, 1.5],
})

# 派生字段
config_default['num_iters_per_epoch'] = 135693 // config_default['batch_size']
config_default['total_iters'] = config_default['num_epochs'] * config_default['num_iters_per_epoch']


# ══════════════════════════════════════════════════
#  配置变体
# ══════════════════════════════════════════════════
config_tiny = AttrDict(copy.deepcopy(config_default))
config_tiny.num_queries = 16
config_tiny.num_points = 8
config_tiny.num_epochs = 5
