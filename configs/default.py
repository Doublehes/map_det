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


# ══════════════════════════════════════════════════
#  默认配置
# ══════════════════════════════════════════════════

# 基础配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_cams = 6
bev_h = 40
bev_w = 80
pc_range = [-10, -10, -3, 30, 10, 5]
img_h = 176
img_w = 320
batch_size = 2
num_workers = 4
cat2id = {'guide_line': 0, 'boundary': 1}
num_classes = len(cat2id)
canvas_size = (80, 160)
num_points = 16

data = AttrDict({
    # ── 数据 ──
    'data_root': "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_data_multiview",
    'train_ann_file': "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_multiview/origin_label/owndata/dctj218_yubei_sampled_330.pkl",
    'val_ann_file': "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_multiview/origin_label/owndata/dctj218_yubei_sampled_330.pkl",
    'cat2id': cat2id,
    'num_classes': num_classes,
    'num_points': num_points,
    'img_h': img_h,
    'img_w': img_w,
    'img_org_h': 352,
    'img_org_w': 640,
    'num_cams': num_cams,
    'img_norm': dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0]),
    'pc_range': pc_range,
    'roi_size': (40, 20),
    'bev_h': bev_h,
    'bev_w': bev_w,
    'canvas_size': canvas_size,
    'batch_size': batch_size,
    'num_workers': num_workers,
})

num_feat_levels = 2
fpn_out_channels = 256
bev_embed_dims = 256
model = AttrDict({
    'img_backbone': AttrDict({
        'type': 'resnet',
        'depth': 50,
        'fpn': AttrDict({
            'in_channels': [512, 1024, 2048],
            'out_channels': fpn_out_channels,
        }),
        'num_feat_levels': num_feat_levels,
    }),

    'bev_encoder': AttrDict({
        'type': 'bevformer',
        'num_cams': num_cams,
        'pc_range': pc_range,
        'bev_h': bev_h,
        'bev_w': bev_w,
        'img_h': img_h,
        'img_w': img_w,

        'fpn_out_channels': fpn_out_channels,  # input projection
        'num_points_in_pillar': 4,
        'bev_embed_dims': bev_embed_dims,
        'num_feat_levels': num_feat_levels,
        'num_layers': 1,
        'num_heads': 8,
        'num_sampling_points': 8,
        'dropout': 0.1,
        'ffn_channels': 512,
    }),

    'map_det_head': AttrDict({
        'type': 'maptr',
        'num_classes': num_classes,
        'num_queries': 32,
        'num_points': num_points,
        'embed_dims': 256,
        'bev_embed_dims': bev_embed_dims,
        'num_heads': 8,
        'num_decoder_layers': 1,
        'dropout': 0.1,
        'ffn_channels': 512,
    }),
    'map_seg_head': AttrDict({
        'enabled': True,
        'type': 'mapseg',
        'bev_embed_dims': bev_embed_dims,
        'num_classes': num_classes,
    }),
    'heatmap_head': AttrDict({
        'enabled': True,
        'bev_embed_dims': bev_embed_dims,
    }),
    'loss': AttrDict({
        'loss_cls_weight': 5.0,
        'loss_reg_weight': 50.0,
        'loss_seg_weight': 100.0,
        'loss_dice_weight': 1.0,
        'loss_heatmap_weight': 10.0,
        'heatmap_loss_threshold': 0.05,
        'heatmap_loss_beta': 0.01,
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'l1_beta': 0.01,
        'num_points': num_points,
    }),
})



config_default = AttrDict({
    'data': data,
    'model': model,

    # ── 训练超参数 ──
    'num_epochs': 3,
    'lr': 5e-4,
    'backbone_lr_mult': 0.1,
    'weight_decay': 1e-2,
    'grad_clip_max_norm': 35.0,
    'scheduler': 'cosine',
    'lr_milestones': [1, 2, 3],
    'lr_gamma': 0.1,
    'warmup_iters': 500,
    'warmup_ratio': 1.0 / 3,
    'min_lr_ratio': 1e-2,
    'log_interval': 50,

    # ── 评测 ──
    'score_thr': 0.3,
    'eval_thresholds': [0.5, 1.0, 1.5],
})

# 派生字段
config_default['device'] = device


# ══════════════════════════════════════════════════
#  配置变体
# ══════════════════════════════════════════════════
config_tiny = AttrDict(copy.deepcopy(config_default))
config_tiny.num_queries = 16
config_tiny.num_points = 8
config_tiny.num_epochs = 5
