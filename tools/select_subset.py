import os
import sys
import pickle
import cv2
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


pkl_dir = "/media/double/SAMSUNG/datasets/trainlabel_line_multiview"
# pkl_names = [name for name in os.listdir(pkl_dir) if name.endswith(".pkl") and "gt_tracks" not in name]
pkl_names = [
    "dctj218_yubei.pkl",
    "old88_nanchuan.pkl",
    "old88_huailai_1.pkl",
    "tms_als_object.pkl",
    "tms_als_tracking.pkl",
    "tms_als_yueye.pkl"
]
new_infos = []
for pkl_name in tqdm(pkl_names, desc="select subset"):
    pkl_path = os.path.join(pkl_dir, pkl_name)
    with open(pkl_path, "rb") as f:
        train_infos = pickle.load(f)
    print(f"{pkl_name} len: {len(train_infos)}")
    new_infos.extend(train_infos[::10])

print(f"new len: {len(new_infos)}")
with open(os.path.join(pkl_dir, "trainlabel_sampled.pkl"), "wb") as f:
    pickle.dump(new_infos, f)
