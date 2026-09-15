import os
import sys
import pickle
import cv2
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


pkl_dir = "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_multiview/train_line_only_260618_jialing"
# pkl_names = [name for name in os.listdir(pkl_dir) if name.endswith(".pkl") and "gt_tracks" not in name]
pkl_names = [
    # "dctj218_yubei.pkl",
    # "tms_dazu_20260416.pkl",
    # "x30_jialing_20260612.pkl",
    # "x30_jialing_road.pkl",
    "own_map_infos_merge_extra_train.pkl"
]
new_infos = []
sample_interval = 5
sample_start = 0
save_pkl_name = f"label_I{sample_interval}_S{sample_start}_"
for pkl_name in tqdm(pkl_names, desc="select subset"):
    pkl_path = os.path.join(pkl_dir, pkl_name)
    with open(pkl_path, "rb") as f:
        train_infos = pickle.load(f)
    print(f"{pkl_name} len: {len(train_infos)}")
    new_infos.extend(train_infos[sample_start::sample_interval])

    save_pkl_name = f"{save_pkl_name}_{pkl_name.split('.')[0]}_"

print(f"new len: {len(new_infos)}")
with open(os.path.join(pkl_dir, f"{save_pkl_name}_len{len(new_infos)}.pkl"), "wb") as f:
    pickle.dump(new_infos, f)


# copy data to new_data_dir
import shutil
src_data_dir = "/media/double/T7 Shield/LINE_OBJECT_DATA/trainlabel_line_data_multiview"
tgt_data_dir = "/media/double/T7 Shield/LINE_DATA_SAMPLE"
os.makedirs(tgt_data_dir, exist_ok=True)
for info in tqdm(new_infos, desc="copy data"):
    for cam_name, cam_data in info['cams'].items():
        if cam_data['img_fpath'] is None:
            print(f"cam {cam_name} img_fpath is None")
            continue
        
        img_path = os.path.join(src_data_dir, cam_data['img_fpath'])
        tgt_img_path = os.path.join(tgt_data_dir, cam_data['img_fpath'])
        os.makedirs(os.path.dirname(tgt_img_path), exist_ok=True)
        
        os.makedirs(os.path.dirname(tgt_img_path), exist_ok=True)
        shutil.copy(img_path, tgt_img_path)



