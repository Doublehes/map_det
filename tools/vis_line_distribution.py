import os
import sys
import argparse
import pickle

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.vis_pkl_multi import Visualizer

LINE_ID_CURB = 1
LINE_ID_CENTER = 0


def collect_lines(train_infos: list, cls_id: int, scope: list) -> list:
    """收集某一类线的所有线段（ego 坐标系，过滤出 scope 范围），返回 [(N,2), ...]"""
    x_max, y_max, x_min, y_min = scope
    lines = []
    for info in tqdm(train_infos, desc=f"collect cls_id={cls_id}"):
        if "map_geom" not in info or cls_id not in info["map_geom"]:
            continue
        for line in info["map_geom"][cls_id]:
            pts = np.array(line, dtype=np.float32)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            pts = pts[:, :2]
            mask = ((pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
                    (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max))
            pts = pts[mask]
            if len(pts) > 1:
                lines.append(pts)
    return lines


def render_line_overlay(lines: list, scope: list, img_size: tuple,
                        title: str, color: tuple, out_path: str):
    """所有线画在单张 BEV 图上（细线，叠加密度自然显色）"""
    vis = Visualizer(scope, img_size)

    for line in lines:
        # import pudb; pudb.set_trace()
        px = vis.world_to_pixel(line).reshape(-1, 1, 2)
        cv2.polylines(vis.img, [px], isClosed=False, color=color, thickness=1)

    vis.draw_grid(x_step=10, y_step=10)
    vis.draw_origin()
    vis.draw_text(title, (5, 20))
    
    cv2.imwrite(out_path, vis.img)
    print(f"saved: {out_path}")


def build_density_grid(lines: list, scope: list, img_size: tuple) -> np.ndarray:
    """像素级点计数网格 (h, w)，供热力图与剖面复用"""
    w, h = img_size
    vis = Visualizer(scope, img_size)
    grid = np.zeros((h, w), dtype=np.int32)
    for line in lines:
        px = vis.world_to_pixel(line)
        px = px[(px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)]
        np.add.at(grid, (px[:, 1], px[:, 0]), 1)
    return grid


def _slice_colors(n: int) -> list:
    """为剖面/切片标注循环分配 BGR 颜色（沿用 infer.py 的青/黄/品红）"""
    base = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
    return [base[i % len(base)] for i in range(n)]


def render_density_heatmap(grid: np.ndarray, scope: list, img_size: tuple,
                           title: str, out_path: str,
                           slice_x: list, slice_colors: list):
    """点密度热力图：像素级计数 → colormap 着色，叠加切片标注线"""
    w, h = img_size
    x_max, y_max, x_min, y_min = scope

    if grid.max() > 0:
        log_grid = np.log1p(grid)
        norm = log_grid / log_grid.max()
    else:
        norm = grid.astype(np.float32)

    cmap = plt.get_cmap("hot")
    colored = cmap(norm)[:, :, :3]
    img = (colored * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for d, sc in zip(slice_x, slice_colors):
        row = int(round((x_max - d) / (x_max - x_min) * (h - 1)))
        for u in range(0, w, 8):
            cv2.line(img, (u, row), (min(u + 4, w - 1), row), sc, 2)
        cv2.putText(img, f"x={d:.0f}m", (w - 70, row - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, sc, 1)

    cv2.imwrite(out_path, img)
    print(f"saved: {out_path}")


def render_profiles(grid: np.ndarray, scope: list, img_size: tuple,
                    slice_x: list, slice_colors: list, title: str, out_path: str):
    """剖面曲线: 指定 x 位置的密度沿横向 y 的分布"""
    x_max, y_max, x_min, y_min = scope
    w, h = img_size

    fig, ax = plt.subplots(figsize=(10, 5))
    y_axis = y_max - np.arange(w) / (w - 1) * (y_max - y_min)
    for d, sc in zip(slice_x, slice_colors):
        row = int(round((x_max - d) / (x_max - x_min) * (h - 1)))
        row = np.clip(row, 0, h - 1)
        ax.plot(y_axis, grid[row, :], label=f"x={d:.0f}m",
                color=tuple(c / 255 for c in reversed(sc)))
    ax.set_xlabel("y lateral (m)")
    ax.set_ylabel("point count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化 pkl 中边界线/中心线的分布")
    parser.add_argument("--pkl", required=True, help="pkl 文件路径")
    parser.add_argument("--output", default=".", help="输出目录")
    parser.add_argument("--scope", nargs=4, type=float, default=[30, 10, -10, -10],
                        help="BEV 范围 [x_max, y_max, x_min, y_min]")
    parser.add_argument("--img-size", nargs=2, type=int, default=[200, 600],
                        help="输出图尺寸 [w, h]")
    parser.add_argument("--slice-x", nargs="+", type=float, default=[0, 10, 20],
                        help="剖面位置 x (m)，如 --slice-x 0 10 20")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    scope = list(args.scope)
    img_size = tuple(args.img_size)
    slice_colors = _slice_colors(len(args.slice_x))

    with open(args.pkl, "rb") as f:
        train_infos = pickle.load(f)
    print(f"total frames: {len(train_infos)}")

    for cls_id, name, color in [(LINE_ID_CURB, "boundary", (0, 0, 255)),
                                (LINE_ID_CENTER, "center", (255, 0, 0))]:
        lines = collect_lines(train_infos, cls_id, scope)
        n_pts = sum(len(l) for l in lines)
        print(f"cls_id={cls_id} ({name}): {len(lines)} lines, {n_pts} points in scope")

        grid = build_density_grid(lines, scope, img_size)
        render_line_overlay(lines, scope, img_size, f"BEV {name} lines", color,
                            os.path.join(args.output, f"{name}_lines.png"))
        render_density_heatmap(grid, scope, img_size, f"BEV {name} heatmap",
                               os.path.join(args.output, f"{name}_heatmap.png"),
                               args.slice_x, slice_colors)
        render_profiles(grid, scope, img_size, args.slice_x, slice_colors,
                        f"{name} density profile",
                        os.path.join(args.output, f"{name}_profile.png"))


if __name__ == "__main__":
    main()
