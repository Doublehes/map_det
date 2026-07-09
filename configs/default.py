import torch


class Config:
    pass


def get_default_cfg():
    cfg = Config()

    cfg.data_root = "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_data_multiview"
    cfg.train_ann_file = "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_multiview/origin_label/owndata/dctj218_yubei_sampled_330.pkl"
    # cfg.train_ann_file = "/media/double/SAMSUNG/datasets/trainlabel_line_multiview/trainlabel_sampled.pkl"
    # cfg.val_ann_file = "/media/double/SAMSUNG/datasets/trainlabel_line_multiview/dctj218_yubei.pkl"
    # cfg.val_ann_file = "/media/flow/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_multiview/train_line_only/own_map_infos_merge_extra_val.pkl"
    cfg.val_ann_file = "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_multiview/origin_label/owndata/dctj218_yubei_sampled_330.pkl"

    cfg.cat2id = {'guide_line': 0, 'boundary': 1}
    cfg.num_classes = len(cfg.cat2id)

    cfg.img_h = 176
    cfg.img_w = 320
    cfg.img_org_h = 352
    cfg.img_org_w = 640
    cfg.num_cams = 6

    cfg.img_norm = dict(
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
    )

    cfg.pc_range = [-10, -10, -3, 30, 10, 5]
    cfg.roi_size = (40, 20)
    cfg.bev_h = 40
    cfg.bev_w = 80
    cfg.canvas_size = (80, 160)

    cfg.bev_embed_dims = 256
    cfg.embed_dims = 256
    cfg.num_feat_levels = 2
    cfg.num_points = 16
    cfg.num_queries = 32

    cfg.backbone_depth = 50
    cfg.fpn_in_channels = [512, 1024, 2048]
    cfg.fpn_out_channels = 256

    cfg.bevformer_num_layers = 1
    cfg.num_points_in_pillar = 4
    cfg.num_sampling_points = 8   # BEVFormer encoder 中的 deformable attention 采样点数

    cfg.decoder_num_layers = 1
    cfg.num_heads = 8
    cfg.ffn_channels = cfg.embed_dims * 2
    cfg.dropout = 0.1

    cfg.score_thr = 0.3
    cfg.eval_thresholds = [0.5, 1.0, 1.5]

    cfg.loss_cls_weight = 5.0
    cfg.loss_reg_weight = 50.0
    cfg.loss_seg_weight = 10.0
    cfg.loss_dice_weight = 1.0
    cfg.focal_gamma = 2.0
    cfg.focal_alpha = 0.25
    cfg.l1_beta = 0.01

    cfg.batch_size = 2
    cfg.num_epochs = 3
    cfg.num_workers = 4
    cfg.lr = 5e-4
    cfg.backbone_lr_mult = 0.1
    cfg.weight_decay = 1e-2
    cfg.grad_clip_max_norm = 35.0
    cfg.scheduler = 'step'       # 调度器: 'cosine', 'step', 'none'
    cfg.lr_milestones = [1, 2, 3]  # MultiStepLR 衰减节点 (epoch)
    cfg.lr_gamma = 0.1             # StepLR/Cosine 衰减因子
    cfg.warmup_iters = 50
    cfg.warmup_ratio = 1.0 / 3
    cfg.min_lr_ratio = 1e-2

    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.history_steps = 0
    cfg.num_iters_per_epoch = 135693 // cfg.batch_size
    cfg.total_iters = cfg.num_epochs * cfg.num_iters_per_epoch

    cfg.log_interval = 50    # iter interval

    return cfg


cfg = get_default_cfg()
