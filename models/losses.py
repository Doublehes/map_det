import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


def focal_loss(pred, target, gamma=2.0, alpha=0.25, reduction='mean'):
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = pred.sigmoid()
    pt = p_t * target + (1 - p_t) * (1 - target)
    focal_weight = (1 - pt) ** gamma
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        focal_weight = focal_weight * alpha_t
    loss = focal_weight * ce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def l1_loss(pred, target, beta=0.01):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    # 先对每个实例求和 (num_points*2 个坐标), 再对实例取平均, 最后除以 num_points
    per_instance = loss.view(loss.shape[0], -1).sum(dim=1)
    return per_instance.mean() / loss.shape[1]


class MaskFocalLoss(nn.Module):
    """Per-class sigmoid focal loss for segmentation mask"""

    def __init__(self, loss_weight=10.0, gamma=2.0, alpha=0.25):
        super().__init__()
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target, **kwargs):
        num_classes = pred.size(1)
        loss = 0.
        for c in range(num_classes):
            pred_sigmoid = pred[:, c].sigmoid()
            target_c = target[:, c].type_as(pred)
            pt = (1 - pred_sigmoid) * target_c + pred_sigmoid * (1 - target_c)
            focal_weight = (self.alpha * target_c + (1 - self.alpha) * (1 - target_c)) * pt.pow(self.gamma)
            ce = F.binary_cross_entropy_with_logits(pred[:, c], target_c, reduction='none')
            loss = loss + (focal_weight * ce).mean()
        return self.loss_weight * loss / num_classes


class MaskDiceLoss(nn.Module):
    """Dice loss for segmentation mask"""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.smooth = 1e-5

    def forward(self, pred, target):
        bs, num_classes = pred.shape[:2]
        pred_flat = pred.sigmoid().flatten(2)   # (B, C, H*W)
        target_flat = target.flatten(2)          # (B, C, H*W)
        intersection = (pred_flat * target_flat).sum(2)
        union = pred_flat.pow(2).sum(2) + target_flat.sum(2)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return self.loss_weight * (1 - dice.mean())


class HungarianMatcher(nn.Module):
    """匈牙利匹配: 在预测query和GT线之间做二分图匹配

    回归成本取 forward/backward 两个方向的最小值 (开放线的方向歧义)
    返回: (匹配到的pred索引, 匹配到的GT索引, 每个pair的最优方向)
    """
    def __init__(self, cls_weight=5.0, reg_weight=50.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    @torch.no_grad()
    def forward(self, cls_scores, reg_preds, gt_cls, gt_lines):
        bs, num_queries = cls_scores.shape[:2]

        indices = []
        for i in range(bs):
            pred_cls = cls_scores[i].sigmoid()    # (num_q, num_classes)
            pred_lines = reg_preds[i]              # (num_q, num_points, 2)

            tgt_cls = gt_cls[i]                    # (num_gt,)
            tgt_lines = gt_lines[i]                # (num_gt, [num_permute,] num_points, 2)

            if len(tgt_cls) == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=cls_scores.device),
                                torch.tensor([], dtype=torch.long, device=cls_scores.device)))
                continue

            # 分类成本: pred对每个GT类别的负对数置信度
            cls_cost = -torch.log(pred_cls[:, tgt_cls] + 1e-8)  # (num_q, num_gt)

            num_pts = pred_lines.shape[1]
            pred_flat = pred_lines.flatten(1)  # (num_q, num_points*2)

            if tgt_lines.dim() == 4:
                # 计算forward+backward两个方向的回归成本, 取最小值
                num_gt, num_permute, num_points, _ = tgt_lines.shape
                tgt_all = tgt_lines.flatten(2).reshape(num_gt * num_permute, -1)
                reg_full = (pred_flat.unsqueeze(1) - tgt_all.unsqueeze(0)).abs().sum(dim=-1)
                reg_full = reg_full / num_pts  # 除 num_points, 得到平均每点距离
                reg_full = reg_full.view(num_queries, num_gt, num_permute)
                reg_cost, best_perm = reg_full.min(dim=-1)
            else:
                tgt_flat = tgt_lines.flatten(1)
                reg_cost = (pred_flat.unsqueeze(1) - tgt_flat.unsqueeze(0)).abs().sum(dim=-1)
                reg_cost = reg_cost / num_pts  # 除 num_points
                best_perm = None

            # 总成本 = w_cls * cls_cost + w_reg * reg_cost
            cost = self.cls_weight * cls_cost + self.reg_weight * reg_cost

            # scipy匈牙利算法求解二分图最优匹配
            cost = cost.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            if best_perm is not None:
                perm_for_match = best_perm[row_ind, col_ind]
            else:
                perm_for_match = None
            indices.append((
                torch.from_numpy(row_ind).to(cls_scores.device),
                torch.from_numpy(col_ind).to(cls_scores.device),
                perm_for_match,
            ))

        return indices


class MapTRCriterion(nn.Module):
    """损失计算: Hungarian匹配后计算分类FocalLoss + 回归SmoothL1Loss + 分割损失"""
    def __init__(self, cfg):
        super().__init__()
        self.matcher = HungarianMatcher(
            cls_weight=cfg.loss_cls_weight,
            reg_weight=cfg.loss_reg_weight,
        )
        self.loss_cls_weight = cfg.loss_cls_weight
        self.loss_reg_weight = cfg.loss_reg_weight
        self.loss_seg_weight = cfg.loss_seg_weight
        self.loss_dice_weight = cfg.loss_dice_weight
        self.focal_gamma = cfg.focal_gamma
        self.focal_alpha = cfg.focal_alpha
        self.l1_beta = cfg.l1_beta
        self.num_points = cfg.num_points
        self.mask_focal_loss = MaskFocalLoss(loss_weight=1.0, gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)
        self.mask_dice_loss = MaskDiceLoss(loss_weight=1.0)

    def forward(self, cls_scores, reg_preds, gt_vectors, gt_semantic_mask=None, seg_preds=None, seg_only=False):
        loss_dict = {}

        if not seg_only:
            bs = len(gt_vectors)
            device = cls_scores.device

            # 1. 展平GT: 将各个类别的线合并为统一的列表
            gt_labels_list, gt_lines_list = [], []
            for i in range(bs):
                vec = gt_vectors[i]
                labels, lines = [], []
                for cls_id in sorted(vec.keys()):
                    cls_lines = vec[cls_id]
                    for j in range(cls_lines.shape[0]):
                        labels.append(cls_id)
                        lines.append(cls_lines[j].to(device).float())
                if len(labels) == 0:
                    labels.append(0)
                    lines.append(torch.zeros((2, self.num_points, 2), device=device, dtype=torch.float))
                gt_labels_list.append(torch.tensor(labels, device=device, dtype=torch.long))
                gt_lines_list.append(torch.stack(lines, dim=0))

            # 2. Hungarian匹配: 每个query匹配到最优GT线
            indices = self.matcher(cls_scores, reg_preds, gt_labels_list, gt_lines_list)

            # 3. 计算匹配后的损失
            total_cls_loss = 0.0
            total_reg_loss = 0.0
            num_matched = 0

            for i in range(bs):
                pred_idx, tgt_idx = indices[i][:2]
                best_perm = indices[i][2] if len(indices[i]) > 2 else None
                num_q, num_cls = cls_scores.shape[1], cls_scores.shape[2]

                # 所有 query 的 cls target: 负样本全0, 正样本 one-hot
                cls_target = cls_scores[i].new_zeros(num_q, num_cls)

                if len(pred_idx) > 0:
                    tgt_cls = gt_labels_list[i][tgt_idx]
                    cls_target[pred_idx, tgt_cls] = 1.0

                    matched_reg = reg_preds[i, pred_idx]
                    tgt_lines = gt_lines_list[i][tgt_idx]
                    if best_perm is not None and tgt_lines.dim() == 4:
                        M = len(pred_idx)
                        tgt_lines = tgt_lines[torch.arange(M, device=tgt_lines.device), best_perm]
                    total_reg_loss += l1_loss(matched_reg, tgt_lines, self.l1_beta)
                    num_matched += len(pred_idx)

                # cls loss: 所有 query 都计算, 负样本被推向全0
                total_cls_loss += focal_loss(
                    cls_scores[i], cls_target, self.focal_gamma, self.focal_alpha, reduction='sum')

            if num_matched > 0:
                total_cls_loss = total_cls_loss / num_matched
                total_reg_loss = total_reg_loss / num_matched
            else:
                total_cls_loss = cls_scores.mean() * 0.0
                total_reg_loss = reg_preds.mean() * 0.0

            loss_dict['cls_loss'] = self.loss_cls_weight * total_cls_loss
            loss_dict['reg_loss'] = self.loss_reg_weight * total_reg_loss
        else:
            device = seg_preds.device if seg_preds is not None else cls_scores.device
            loss_dict['cls_loss'] = torch.tensor(0.0, device=device)
            loss_dict['reg_loss'] = torch.tensor(0.0, device=device)

        # 4. 分割损失
        if seg_preds is not None and gt_semantic_mask is not None:
            loss_dict['seg_loss'] = self.loss_seg_weight * self.mask_focal_loss(seg_preds, gt_semantic_mask)
            loss_dict['dice_loss'] = self.loss_dice_weight * self.mask_dice_loss(seg_preds, gt_semantic_mask)

        return loss_dict
