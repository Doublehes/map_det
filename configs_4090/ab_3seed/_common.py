import copy

from configs.default import config_default as base_cfg
from configs.loader import AttrDict


def make_cfg(use_sgq, predict_refine, seed, work_dir):
    cfg = AttrDict(copy.deepcopy(base_cfg))
    head = cfg.model.map_det_head

    head.enabled = True
    head.aux_loss = True
    head.num_decoder_layers = 3
    head.use_sgq = use_sgq
    head.gather_residual = 'identity'
    head.point_pos_encoding = 'legacy'
    head.predict_refine = predict_refine
    head.different_heads = False
    head.ref_init = 'mlp'

    cfg.model.map_seg_head.enabled = True
    cfg.model.heatmap_head.enabled = False
    head.bev_feat_net.enabled = True

    cfg.freeze_modules = ['backbone.', 'bev_encoder.', 'seg_head.', 'heatmap_head.']
    cfg.data.batch_size = 16

    cfg.seed = int(seed)
    cfg.work_dir = work_dir
    cfg.pretrained = './work_dirs/stage1_seg_pretrain_e72_arg/latest.pth'
    cfg.resume = ''
    cfg.freeze_backbone = False
    cfg.num_epochs = 72
    cfg.amp = False
    cfg.eval_interval = 1
    cfg.checkpoint_interval = 6
    return cfg
