"""模型分析工具：可视化 query 嵌入、注意力图、参考点等"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.maptr import MapTR


def compute_similarity_matrix(q_weight: torch.Tensor) -> np.ndarray:
    q = F.normalize(q_weight, dim=-1)
    sim = torch.mm(q, q.T)
    return sim.detach().cpu().numpy()


def render_query_similarity(sim_mat: np.ndarray, save_path: str):
    n = sim_mat.shape[0]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    for i in range(n):
        for j in range(n):
            val = sim_mat[i, j]
            color = 'white' if abs(val) < 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=5.5, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=6)
    ax.set_yticklabels([str(i) for i in range(n)], fontsize=6)
    ax.set_xlabel('Query Index', fontsize=10)
    ax.set_ylabel('Query Index', fontsize=10)
    ax.set_title('Query Embedding Cosine Similarity', fontsize=12, fontweight='bold')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='Cosine Similarity')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[保存] {save_path}')


def compute_initial_ref_points(model) -> np.ndarray:
    """计算所有 query 的初始参考点 (num_queries, num_points, 2)"""
    q_weight = model.head.query_embedding.weight
    bs = 1
    query = q_weight.unsqueeze(1).repeat(1, bs, 1)
    ref_flat = model.head.reference_points_embed(query.permute(1, 0, 2)).sigmoid()
    ref = ref_flat.view(bs, model.head.num_queries, model.head.num_points, 2)
    return ref[0].detach().cpu().numpy()


def compute_reg_branch_direct(model) -> tuple:
    """reg_branch 直接输出 (skip decoder), 以及 cls_scores
    Returns: (num_queries, num_points, 2), (num_queries, num_classes)
    """
    q_weight = model.head.query_embedding.weight  # (32, 256)
    q = q_weight.unsqueeze(0)  # (1, 32, 256)
    raw_reg = model.head.reg_branches[-1](q)  # (1, 32, 32)
    reg = raw_reg.view(1, model.head.num_queries, model.head.num_points, 2).sigmoid()

    cls = model.head.cls_branches[-1](q).sigmoid()
    return reg[0].detach().cpu().numpy(), cls[0].detach().cpu().numpy()


def world_to_panel(pts, pc_min_x, pc_min_y, pc_max_x, pc_max_y, pw, ph):
    """车辆坐标 → 面板像素(col,row)"""
    col = ((pts[:, 0] - pc_min_x) / (pc_max_x - pc_min_x) * pw)
    row = ((pc_max_y - pts[:, 1]) / (pc_max_y - pc_min_y) * ph)
    return np.stack([col, row], axis=1)


def _draw_bev_grid(ax, pw, ph, pc_min_x, pc_min_y, pc_max_x, pc_max_y):
    """在 ax 上绘制 BEV 网格和 ego"""
    for x_m in range(-10, 31, 5):
        px = (x_m - pc_min_x) / (pc_max_x - pc_min_x) * pw
        ax.axvline(px, color='#e0e0e0', linewidth=0.6)
    for y_m in range(-10, 11, 5):
        py = (pc_max_y - y_m) / (pc_max_y - pc_min_y) * ph
        ax.axhline(py, color='#e0e0e0', linewidth=0.6)
    for x_m in range(0, 31, 10):
        px = (x_m - pc_min_x) / (pc_max_x - pc_min_x) * pw
        ax.text(px, ph - 10, f'{x_m}m', color='#666', fontsize=8, ha='center')
    for y_m in range(-10, 11, 10):
        py = (pc_max_y - y_m) / (pc_max_y - pc_min_y) * ph
        ax.text(8, py, f'{y_m}m', color='#666', fontsize=8, va='center')

    ex = (0 - pc_min_x) / (pc_max_x - pc_min_x) * pw
    ey = (pc_max_y - 0) / (pc_max_y - pc_min_y) * ph
    ax.plot(ex, ey, marker='^', markersize=14, color='#e74c3c',
            markeredgecolor='#c0392b', markeredgewidth=1, zorder=10)
    ax.text(ex + 12, ey + 12, 'Ego', color='#c0392b', fontsize=9, fontweight='bold')
    ax.text(pw // 2, ph - 28, 'Forward X', color='#666', fontsize=9, ha='center')
    ax.text(18, ph // 2, 'Left Y', color='#666', fontsize=9, va='center', rotation=90)


def _draw_query_lines(ax, preds, cmap, roi_w, roi_h,
                      pc_min_x, pc_min_y, pc_max_x, pc_max_y, pw, ph):
    """在 ax 上画出每个 query 的预测线"""
    num_q, num_pts, _ = preds.shape
    for qi in range(num_q):
        color = cmap(qi)
        pts = preds[qi]
        world_x = pts[:, 0] * roi_w + pc_min_x
        world_y = pts[:, 1] * roi_h + pc_min_y
        pix = world_to_panel(np.stack([world_x, world_y], axis=1),
                             pc_min_x, pc_min_y, pc_max_x, pc_max_y, pw, ph)
        ax.plot(pix[:, 0], pix[:, 1], '-', color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(pix[0, 0], pix[0, 1], color=color, s=30,
                   edgecolors='white', linewidth=0.6, zorder=5)
        ax.scatter(pix[1:, 0], pix[1:, 1], color=color, s=12, alpha=0.4)


def _class_color_map(best_cls: np.ndarray, num_q: int):
    """根据预测类别返回颜色列表: 类别0→蓝色系, 类别1→红色系"""
    colors = []
    cnt = [0, 0]
    total = [(best_cls == 0).sum(), (best_cls == 1).sum()]
    for qi in range(num_q):
        c = best_cls[qi]
        t = cnt[c] / max(1, total[c] - 1)
        if c == 0:
            colors.append(plt.cm.Blues(0.3 + 0.7 * t))
        else:
            colors.append(plt.cm.Reds(0.3 + 0.7 * t))
        cnt[c] += 1
    return lambda qi: colors[qi]


def render_points_comparison(ref_points: np.ndarray, reg_points: np.ndarray,
                              cls_scores: np.ndarray, cfg, save_path: str):
    """对比图: 左=初始参考点, 右=reg_branch直接输出, 颜色按预测类别区分"""
    pc_min_x, pc_min_y, _, pc_max_x, pc_max_y, _ = cfg.data.pc_range
    roi_w = cfg.data.roi_size[0]
    roi_h = cfg.data.roi_size[1]
    pw, ph = 800, 400
    num_q, num_pts, _ = ref_points.shape
    best_cls = cls_scores.argmax(axis=1)
    cmap = _class_color_map(best_cls, num_q)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    for ax in (ax1, ax2):
        ax.set_xlim(0, pw)
        ax.set_ylim(ph, 0)
        _draw_bev_grid(ax, pw, ph, pc_min_x, pc_min_y, pc_max_x, pc_max_y)

    # 左: 初始参考点
    _draw_query_lines(ax1, ref_points, cmap, roi_w, roi_h,
                      pc_min_x, pc_min_y, pc_max_x, pc_max_y, pw, ph)
    ax1.set_title('(a) Initial Reference Points\n(from reference_points_embed)',
                  fontsize=12, fontweight='bold', pad=10)
    ax1.text(pw - 10, 14, f'{num_q} queries, {num_pts} pts/query',
             color='#999', fontsize=8, ha='right')

    # 右: reg_branch 直接输出 (skip decoder)
    _draw_query_lines(ax2, reg_points, cmap, roi_w, roi_h,
                      pc_min_x, pc_min_y, pc_max_x, pc_max_y, pw, ph)
    ax2.set_title('(b) Reg Branch Direct Output\n(skip decoder, Blue=Lane  Red=Boundary)',
                  fontsize=12, fontweight='bold', pad=10)

    # 图例: 蓝色/红色 patch
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.cm.Blues(0.6), label=f'Lane ({int((best_cls==0).sum())} queries)'),
        Patch(facecolor=plt.cm.Reds(0.6), label=f'Boundary ({int((best_cls==1).sum())} queries)'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9,
               framealpha=0.9, edgecolor='#ccc')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[保存] {save_path}')


def render_initial_ref_points(ref_points: np.ndarray, cfg, save_path: str):
    """在 BEV 图上画出每个 query 的初始参考点"""
    pc_min_x, pc_min_y, _, pc_max_x, pc_max_y, _ = cfg.data.pc_range
    roi_w = cfg.data.roi_size[0]
    roi_h = cfg.data.roi_size[1]
    pw, ph = 800, 400

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, pw)
    ax.set_ylim(ph, 0)
    _draw_bev_grid(ax, pw, ph, pc_min_x, pc_min_y, pc_max_x, pc_max_y)

    num_q, num_pts, _ = ref_points.shape
    cmap = get_cmap('tab20', num_q)
    _draw_query_lines(ax, ref_points, cmap, roi_w, roi_h,
                      pc_min_x, pc_min_y, pc_max_x, pc_max_y, pw, ph)

    ax.set_title('Initial Reference Points (before decoder refine)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.text(pw - 10, 14, f'{num_q} queries, {num_pts} pts/query',
            color='#999', fontsize=9, ha='right')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[保存] {save_path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MapTR 模型分析工具')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--save-dir', type=str, default='.', help='输出保存目录')
    args = parser.parse_args()

    from configs.loader import load_config
    cfg = load_config(args.config)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f'[加载] config: {args.config}')
    print(f'[加载] checkpoint: {args.checkpoint}')

    model = MapTR(cfg.model).to(cfg.device)
    ckpt = torch.load(args.checkpoint, map_location=cfg.device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f'[加载完成] epoch={ckpt.get("epoch", "?")}')

    # 提取 query embedding 权重
    q_weight = model.head.query_embedding.weight  # (num_queries, embed_dims)
    num_q, dim = q_weight.shape
    print(f'[Query] num_queries={num_q}, embed_dims={dim}')

    # 相似度热力图
    sim_mat = compute_similarity_matrix(q_weight)
    save_path = save_dir / 'query_similarity.png'
    render_query_similarity(sim_mat, save_path)

    # 初始参考点可视化
    ref_points = compute_initial_ref_points(model)
    save_path = save_dir / 'initial_ref_points.png'
    render_initial_ref_points(ref_points, cfg, save_path)

    # reg_branch 直接输出 (skip decoder) + cls_scores
    reg_points, cls_scores = compute_reg_branch_direct(model)
    save_path = save_dir / 'query_predictions_comparison.png'
    render_points_comparison(ref_points, reg_points, cls_scores, cfg, save_path)

    # 打印 cls 统计
    best_cls = cls_scores.argmax(axis=1)
    best_conf = cls_scores.max(axis=1)
    print(f'[Cls] 预测分布:')
    for i in range(model.head.num_classes):
        cnt = (best_cls == i).sum()
        print(f'      类别{i}: {cnt} queries')
    print(f'[Cls] 平均置信度: {best_conf.mean():.3f}, '
          f'最高: {best_conf.max():.3f}, 最低: {best_conf.min():.3f}')

    print(f'[完成] 输出目录: {save_dir}')


if __name__ == '__main__':
    main()
