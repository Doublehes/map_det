"""推理可视化: 逐张展示 val 样本的 输入(+噪声) / GT向量 / 预测线, 不保存结果

运行: python model_slim/infer.py [--config resnet34|decode_layer3|default]
checkpoint 路径从配置推导: cfg.work_dir/latest.pth
"""
import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_slim.config as cfg_module
from model_slim.dataset import SlimDataset
from model_slim.model import SlimModel

SCORE_THRESH = 0.3


def decode_predictions(cls_scores, reg_preds, score_thresh):
    """解码预测线: 每 query 取最大置信度类别, 超过阈值则保留

    cls_scores: (num_queries, num_classes) logits
    reg_preds:  (num_queries, num_points, 2) 归一化[0,1]
    Returns: {cls_id: [line, ...]} (与 GT dict 同构)
    """
    probs = torch.sigmoid(cls_scores).cpu().numpy()
    reg_preds = reg_preds.cpu().numpy()

    out = {0: [], 1: []}
    for qi in range(len(probs)):
        max_score, cls_id = probs[qi].max(), probs[qi].argmax()
        if max_score < score_thresh:
            continue
        out[int(cls_id)].append(reg_preds[qi])
    return out


def draw_lines(vectors, canvas_size, roi_size, thickness=1):
    """将归一化向量线绘制为 RGB 画布 (与 rasterize_rgb 同像素映射)

    cls0 中心线: 绿色箭头线; cls1 边界线: 红色实线
    """
    h, w = canvas_size
    img = np.zeros((h, w, 3), dtype=np.float32)
    colors = {0: (0.0, 1.0, 0.0), 1: (1.0, 0.0, 0.0)}   # BGR: 绿/红

    for cls_id, lines in vectors.items():
        color = colors.get(cls_id)
        if color is None:
            continue
        for line in lines:
            pts_2d = line[0] if line.ndim == 3 else line
            denormalized = pts_2d * np.array([roi_size[0], roi_size[1]], dtype=np.float32)

            pts = []
            for p in denormalized:
                px = int(p[0] / roi_size[0] * w)
                py = int(p[1] / roi_size[1] * h)
                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            if len(pts) < 2:
                continue
            if cls_id == 0:
                for i in range(len(pts) - 1):
                    cv2.arrowedLine(img, tuple(pts[i]), tuple(pts[i + 1]),
                                    color, thickness, tipLength=0.3)
            else:
                cv2.polylines(img, [pts], False, color, thickness, lineType=cv2.LINE_AA)
            cv2.circle(img, tuple(pts[0]), thickness + 1, color, -1)
            cv2.circle(img, tuple(pts[-1]), thickness + 1, color, -1)
    return img


def main():
    parser = argparse.ArgumentParser(description='Slim 模型推理可视化')
    parser.add_argument('--config', default='default', choices=list(cfg_module.CONFIGS.keys()),
                        help='配置变体: default / resnet34 / decode_layer3')
    args = parser.parse_args()
    cfg = cfg_module.CONFIGS[args.config]

    ckpt_path = os.path.join(cfg.work_dir, 'latest.pth')
    print(f'[加载] {ckpt_path}')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'checkpoint 不存在: {ckpt_path}')

    model = SlimModel(cfg).to(cfg.device)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f'[加载] epoch={ckpt.get("epoch", "?")}')

    ds = SlimDataset(cfg.data.val_ann_file, cfg.data, is_train=False)
    print(f'[推理] 共 {len(ds)} 个样本, score_thresh={SCORE_THRESH}')

    with torch.no_grad():
        for i, sample in enumerate(ds):
            raster = sample['raster'].unsqueeze(0).to(cfg.device)
            cls_scores_all, reg_preds_all, _ = model(raster, return_all_layers=True)
            num_layers = len(cls_scores_all)

            inp = sample['raster'].permute(1, 2, 0).numpy()                    # ① 输入(+噪声), 只算一次
            gt = draw_lines(sample['vectors'], cfg.data.canvas_size, cfg.data.roi_size)   # ② GT, 只算一次
            gt_v = sample['vectors']

            for l in range(num_layers):
                preds = decode_predictions(cls_scores_all[l][0], reg_preds_all[l][0], SCORE_THRESH)
                pred = draw_lines(preds, cfg.data.canvas_size, cfg.data.roi_size)            # ③ 该层预测

                panel = np.concatenate([inp, gt, pred], axis=1) * 255
                panel = np.clip(panel, 0, 255).astype(np.uint8)
                h_panel, w_panel = panel.shape[:2]
                panel = cv2.resize(panel, (w_panel * 2, h_panel * 2),
                                   interpolation=cv2.INTER_NEAREST)
                col_w = panel.shape[1] // 3
                cv2.putText(panel, f'Input  (sample {i})', (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(panel, f'GT  (sample {i})', (col_w + 20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(panel, f'Layer {l} Pred  thr={SCORE_THRESH}  (sample {i})',
                            (2 * col_w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow(f'layer L{l}', panel)   # 每层独立窗口
                print(f'[样本 {i}/{len(ds)}] L{l}: Pred {len(preds[0])}中心/{len(preds[1])}边界')
            print(f'  GT: {len(gt_v.get(0, []))}中心/{len(gt_v.get(1, []))}边界')
            cv2.waitKey(0)   # 全部层窗口显示后, 按任意键看下一张

    print('[推理完成]')


if __name__ == '__main__':
    main()
