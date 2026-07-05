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


def getPara_cam(filename):
    if not os.path.exists(filename):
        print("{} not exists!".format(filename))
        exit(-1)
    infile = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    para = {}
    para["K"] = infile.getNode("K").mat()
    para["R"] = infile.getNode("R").mat()
    para["D"] = infile.getNode("D").mat()
    para["T"] = (infile.getNode("T").mat()).reshape(-1)
    para["imgh"] = infile.getNode("imgh").real()
    para["imgw"] = infile.getNode("imgw").real()
    para["camera_type"] = infile.getNode("camera_type").string()
    if 'mid' in filename:
        para["camera_type"] = 'mid'
    elif 'fish' in filename:
        para["camera_type"] = 'fish'
    infile.release()
    if para["T"][2] > 10:
        para["T"][0] = para["T"][0] * 0.01
        para["T"][1] = para["T"][1] * 0.01
        para["T"][2] = para["T"][2] * 0.01
    return para


def vis_pkl(pkl_path, data_root):
    with open(pkl_path, "rb") as f:
        train_infos = pickle.load(f)
    print(f"len(train_infos): {len(train_infos)}")

    bev_scope = [ 30,10,-10, -10] # x_max, y_max, x_min, y_min
    
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
            # cv2.imshow(f"img_{cam_name}", img)
            show_imgs.append(img)
        if len(show_imgs) == 6:
            show1 = np.concatenate(show_imgs[:3], axis=1)
            show2 = np.concatenate(show_imgs[3:], axis=1)
        elif len(show_imgs) == 4:
            show1 = np.concatenate(show_imgs[:2], axis=1)
            show2 = np.concatenate(show_imgs[2:], axis=1)
        else:
            raise ValueError(f"show_imgs len: {len(show_imgs)}")
        cv2.imshow(f"show_imgs", np.concatenate([show1, show2], axis=0))


        # bev img
        bev_width, bev_height = 180, 300
        bev_img = np.zeros((bev_height, bev_width, 3), np.uint8)
        bev_z_img = np.zeros((bev_height, bev_width, 3), np.uint8)
        origin_v = (bev_scope[0] - 0) / (bev_scope[0] - bev_scope[2]) * bev_height
        origin_u = (bev_scope[1] - 0) / (bev_scope[1] - bev_scope[3]) * bev_width
        cv2.circle(bev_img, (int(origin_u), int(origin_v)), radius=4, color=(255, 255, 255), thickness=2)

        # line_bev
        line = info["map_geom"]
        get_type_by_id = lambda id: "line_curb" if id == 0 else "line_center"
        for line_id, line_list in line.items():
            for line_arr in line_list:
                line_arr = np.array(line_arr)
                # print(f"line_arr: {line_arr.shape}", line_arr.shape)
                # bev draw
                drawx = (bev_scope[0] - line_arr[:, 0]) / (bev_scope[0] - bev_scope[2]) * bev_height
                drawy = (bev_scope[1] - line_arr[:, 1]) / (bev_scope[1] - bev_scope[3]) * bev_width
                drawz = (line_arr[:, 2] + 4) / (4 + 4) * bev_width
                xy_draw_uv = np.concatenate([drawy.astype(np.int32).reshape(-1, 1), 
                                            drawx.astype(np.int32).reshape(-1, 1)], axis=1)
                xz_draw_uv = np.concatenate([drawz.astype(np.int32).reshape(-1, 1),
                                            drawx.astype(np.int32).reshape(-1, 1)], axis=1)
                # cv2.polylines(bev_img, [xy_draw_uv.reshape(-1, 1, 2)], isClosed=False, 
                #             color=linetype_color[get_type_by_id(line_id)], thickness=3)
                # cv2.polylines(bev_z_img, [xz_draw_uv.reshape(-1, 1, 2)], isClosed=False,
                #             color=linetype_color[get_type_by_id(line_id)], thickness=3)
                for point in xy_draw_uv:
                    cv2.circle(bev_img, tuple(point), radius=2, color=linetype_color[get_type_by_id(line_id)], thickness=-1)
                for point in xz_draw_uv:
                    cv2.circle(bev_z_img, tuple(point), radius=2, color=linetype_color[get_type_by_id(line_id)], thickness=-1)

        cv2.imshow(f"bev_img", bev_img)
        cv2.imshow(f"bev_z_img", bev_z_img)

        key = cv2.waitKey(100)
        if key == 32: # if space bar is pressed, pause the program
            key = cv2.waitKey(0)
        if key == 27 or key == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    # data_root = "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_data_multiview"
    # pkl_path = "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_multiview/origin_label_for_evaluation/evaluation.pkl"

    data_root = "/media/double/SAMSUNG/datasets"
    pkl_path = "/media/double/SAMSUNG/datasets/trainlabel_line_multiview/old88_nanchuan.pkl"

    vis_pkl(pkl_path, data_root)
