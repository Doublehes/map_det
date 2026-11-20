import cv2
import os
import argparse
from tqdm import tqdm
import imageio

class VideoWriter:
    def __init__(self, output_video_path):
        self.video = None
        self.save_path = output_video_path

    def init_video_writer(self, img_w, img_h, fps=10):
        # 创建 VideoWriter 对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
        self.video = cv2.VideoWriter(self.save_path, fourcc, fps, (img_w, img_h))

    def write(self, img):
        if self.video is None:
            h, w, _ = img.shape
            self.init_video_writer(w, h)
        self.video.write(img)

    def release(self):
        print(f"Save video : {self.save_path}")
        self.video.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", help='img dir')
    parser.add_argument("--save_dir", help="video save path", default=".")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    img_dir = args.img_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    img_names = os.listdir(img_dir)
    img_names.sort(key=lambda s: int(s.split(".")[0].split("_")[-1]))
    
    v_writer = VideoWriter(os.path.join(save_dir, f"video.mp4"))
    imgs = []
    count = 200
    for i, img_name in enumerate(img_names):
        if i > count:
            continue
        img_path = img_dir + f"/{img_name}"
        assert os.path.exists(img_path), f"img {img_path} not exists"
        img = cv2.imread(img_path)
        v_writer.write(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    
    imageio.mimsave(os.path.join(save_dir, f"gif.gif"), imgs, "gif", duration=1, loop=0)
    v_writer.release()