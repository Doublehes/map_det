import os
import sys
import pickle
import cv2
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


linetype_id = dict(line_curb=0, line_center=1)
linetype_color = dict(line_curb=(0, 0, 255),     # red
                      line_center=(255, 0, 0))   # blue


class Visualizer:
    """BEV 可视化工具：统一坐标转换与绘图接口"""

    def __init__(self, scope, img_size, bg_color=(0, 0, 0)):
        self.x_max, self.y_max, self.x_min, self.y_min = scope
        self.w, self.h = img_size
        self.img = np.full((self.h, self.w, 3), bg_color, np.uint8)

    def world_to_pixel(self, points):
        pts = np.array(points, dtype=np.float32)
        u = (self.y_max - pts[..., 1]) / (self.y_max - self.y_min) * self.w
        v = (self.x_max - pts[..., 0]) / (self.x_max - self.x_min) * self.h
        return np.stack([u, v], axis=-1).astype(np.int32)

    def draw_grid(self, x_step=20, y_step=10, color=(64, 64, 64)):
        for x in range(int(self.x_min), int(self.x_max) + 1, x_step):
            v = int((self.x_max - x) / (self.x_max - self.x_min) * self.h)
            cv2.line(self.img, (0, v), (self.w, v), color, 1)
            cv2.putText(self.img, f"{x}m", (2, v - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        for y in range(int(self.y_min), int(self.y_max) + 1, y_step):
            u = int((self.y_max - y) / (self.y_max - self.y_min) * self.w)
            cv2.line(self.img, (u, 0), (u, self.h), color, 1)
            if y == 0:
                cv2.putText(self.img, "y=0", (u + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def draw_origin(self, label="ego", color=(255, 255, 255)):
        px = self.world_to_pixel([(0, 0)])[0]
        cv2.circle(self.img, tuple(px), 5, color, 2)
        cv2.putText(self.img, label, (px[0] + 4, px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def draw_box_bev(self, box_7d, color, label=""):
        x, y, z, l, w, h, yaw = box_7d
        corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = corners @ rot.T + np.array([x, y])
        pts = self.world_to_pixel(corners).reshape(-1, 1, 2)
        cv2.polylines(self.img, [pts], isClosed=True, color=color, thickness=2)

        front = np.array([x + l/2 * np.cos(yaw), y + l/2 * np.sin(yaw)])
        center_px = self.world_to_pixel([(x, y)])[0]
        front_px = self.world_to_pixel([front])[0]
        cv2.arrowedLine(self.img, tuple(center_px), tuple(front_px), color, 2, tipLength=0.3)

        if label:
            cv2.putText(self.img, label, (pts[0, 0, 0], pts[0, 0, 1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def draw_polyline(self, points_3d, color, style='line', thickness=2):
        pts = np.array(points_3d, dtype=np.float32)
        if pts.shape[1] == 3:
            pts = pts[:, :2]
        px = self.world_to_pixel(pts).reshape(-1, 1, 2)

        if style == 'line':
            cv2.polylines(self.img, [px], isClosed=False, color=color, thickness=thickness)
            for p in px:
                cv2.circle(self.img, tuple(p[0]), 2, color, -1)
        elif style == 'arrow':
            for i in range(len(px) - 1):
                cv2.arrowedLine(self.img, tuple(px[i, 0]), tuple(px[i+1, 0]),
                                color, thickness, tipLength=0.3)

    def draw_text(self, text, pos, color=(255, 255, 255)):
        cv2.putText(self.img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def show(self, window_name="bev"):
        cv2.imshow(window_name, self.img)


def vis_pkl(pkl_path, data_root):
    with open(pkl_path, "rb") as f:
        train_infos = pickle.load(f)
    print(f"len(train_infos): {len(train_infos)}")

    bev_scope = [ 30,10,-10, -10] # x_max, y_max, x_min, y_min
    bev_obj_scope = [ 80,20,-40, -20] # x_max, y_max, x_min, y_min
    
    for info in tqdm(train_infos[::5], desc=os.path.basename(pkl_path)):
        show_imgs = []
        for cam_name in info["cams"].keys():

            img_path = info["cams"][cam_name]["img_fpath"]
            tqdm.write(f"img_path: {img_path}")
            img_path = os.path.join(data_root, img_path)

            if not os.path.exists(img_path):
                print(f"img: {img_path} not exist")
                continue

            extrinsics = info["cams"][cam_name]["extrinsics"]
            # extrinsics = np.eye(4)
            intrinsics = info["cams"][cam_name]["intrinsics"]
            # print(f"calib cam_name: {cam_name}")
            # print(intrinsics)
            # print(np.round(extrinsics, 3))

            # img
            img = cv2.imread(img_path)
            # cv2.imshow(f"img_raw_{cam_name}", img)

            # line
            if "map_geom" in info:
                line = info["map_geom"]
                get_type_by_id = lambda id: "line_curb" if id == 0 else "line_center"
                line_count = 0
                for line_id, line_list in line.items():
                    for line_arr in line_list:
                        line_arr = np.array(line_arr)
                        # img draw
                        points_cam = (np.matmul(extrinsics[:3, :3], line_arr.T) + extrinsics[:3, 3].reshape(3, 1)).T
                        points_cam = points_cam[points_cam[:, 2] > 0]
                        points_uv = np.matmul(intrinsics, points_cam.T).T
                        points_uv = (points_uv / points_uv[:, 2:3]).astype(np.int32)
                        
                        points_uv = points_uv[:, :2]
                        # filter out points outside the image
                        points_uv = points_uv[(points_uv[:, 0] >= 0) & (points_uv[:, 0] < img.shape[1]) & 
                                            (points_uv[:, 1] >= 0) & (points_uv[:, 1] < img.shape[0])]
                        points_uv = points_uv.reshape(-1, 1, 2)
                        # cv2.polylines(img, [points_uv], isClosed=False, color=linetype_color[get_type_by_id(line_id)], thickness=2)
                        # for point in points_uv:
                        #     cv2.circle(img, tuple(point[0]), radius=3, color=linetype_color[get_type_by_id(line_id)], thickness=3)
                        points_uv = points_uv.reshape(-1, 2)
                        # 绘制箭头：从第一个点到第二个点，第二个点到第三个点...
                        for i in range(len(points_uv) - 1):
                            cv2.arrowedLine(img, tuple(points_uv[i]), tuple(points_uv[i+1]), 
                                            color=linetype_color[get_type_by_id(line_id)], thickness=2, tipLength=0.3)
                        line_count += 1
                cv2.putText(img, f"{cam_name}, line: {line_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            show_imgs.append(img)
        if len(show_imgs) == 6:
            show1 = np.concatenate(show_imgs[:3], axis=1)
            show2 = np.concatenate(show_imgs[3:], axis=1)
        elif len(show_imgs) == 4:
            show1 = np.concatenate(show_imgs[:2], axis=1)
            show2 = np.concatenate(show_imgs[2:], axis=1)
        elif len(show_imgs) == 5:
            h, w = show_imgs[0].shape[:2]
            show_imgs.append(np.zeros((h, w, 3), np.uint8))
            show1 = np.concatenate(show_imgs[:3], axis=1)
            show2 = np.concatenate(show_imgs[3:], axis=1)
        else:
            raise ValueError(f"show_imgs len: {len(show_imgs)}")
        show_concat = np.concatenate([show1, show2], axis=0)
        h, w = show_concat.shape[:2]
        cv2.imshow("show_imgs", cv2.resize(show_concat, (w // 2, h // 2)))


        if "map_geom" in info:
            # bev line xy
            vis_xy = Visualizer([30, 10, -10, -10], (150, 300))
            vis_xy.draw_origin()
            vis_xy.draw_grid(x_step=10, y_step=10)
            for line_id, line_list in info["map_geom"].items():
                for line_arr in line_list:
                    color = linetype_color["line_curb" if line_id == 0 else "line_center"]
                    vis_xy.draw_polyline(line_arr, color,
                                        style='line' if line_id == 0 else 'arrow')
            vis_xy.draw_text("BEV XY", (5, 20))

            # bev line xz
            vis_xz = Visualizer([30, 2, -10, -2], (150, 300))
            vis_xz.draw_grid(x_step=10, y_step=1)
            for line_id, line_list in info["map_geom"].items():
                for line_arr in line_list:
                    color = linetype_color["line_curb" if line_id == 0 else "line_center"]
                    pts = np.array(line_arr)[:, [0, 2]]
                    vis_xz.draw_polyline(pts, color,
                                        style='line' if line_id == 0 else 'arrow')
            vis_xz.draw_text("BEV XZ", (5, 20))

            separator = np.full((300, 5, 3), 255, np.uint8)
            bev_concat = np.concatenate([vis_xy.img, separator, vis_xz.img], axis=1)
            cv2.imshow("bev_show", bev_concat)

        # bev object
        if "gt_boxes" in info:
            vis = Visualizer(bev_obj_scope, (200, 600))
            vis.draw_grid()
            vis.draw_origin()
            vis.draw_text("BEV OBJ", (5, 20))

            cls_color = {"car": (0, 255, 0), "truck": (0, 200, 255), "bus": (0, 100, 255),
                         "pedestrian": (255, 0, 255), "cyclist": (255, 255, 0)}

            line_color_map = {0: (0, 0, 255), 1: (255, 0, 0)}
            line_style_map = {0: 'line', 1: 'arrow'}
            for line_id, line_list in info["map_geom"].items():
                for line_arr in line_list:
                    vis.draw_polyline(line_arr, line_color_map.get(line_id, (255, 255, 255)),
                                      style=line_style_map.get(line_id, 'line'))

            gt_boxes = info["gt_boxes"]
            gt_names = info["gt_names"]
            valid_flag = info.get("valid_flag", [True] * len(gt_boxes))

            for i in range(len(gt_boxes)):
                if not valid_flag[i]:
                    continue
                name = gt_names[i] if i < len(gt_names) else "obj"
                vis.draw_box_bev(gt_boxes[i], cls_color.get(name, (0, 255, 0)), name)

            vis.show("bev_obj")

        key = cv2.waitKey(100)
        if key == 32: # if space bar is pressed, pause the program
            key = cv2.waitKey(0)
        if key == 27 or key == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    data_root = "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_data_multiview"
    # pkl_path = "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_multiview/origin_label_for_evaluation/x30_jialing_road.pkl"
    pkl_path = "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_multiview/origin_label/public_argoverse2/argoverse2.pkl"

    # data_root = "/media/double/SAMSUNG/datasets"
    # pkl_path = "/media/double/SAMSUNG/datasets/trainlabel_line_multiview/old88_nanchuan.pkl"

    vis_pkl(pkl_path, data_root)
