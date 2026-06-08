import os
import sys
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_DIR = '/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1'

for subj in sorted(os.listdir(DATA_DIR)):
    subj_path = os.path.join(DATA_DIR, subj)
    vid_path_avi = os.path.join(subj_path, 'vid.avi')
    frames_path = os.path.join(subj_path, 'frames.npy')
    mask_path = os.path.join(subj_path, 'mask.npy')
    if not os.path.exists(mask_path):
        continue  # chỉ xử lý subject đã có mask
    if os.path.exists(frames_path):
        print(f'{subj} frames already exist, skipping.')
        continue

    # Tạo thư mục tạm cho ảnh
    tmp_dir = os.path.join(subj_path, 'tmp_frames')
    os.makedirs(tmp_dir, exist_ok=True)

    # Dùng ffmpeg để giải nén tất cả frame ra ảnh PNG
    cmd = [
        'ffmpeg', '-i', vid_path_avi,
        '-vsync', '0',
        os.path.join(tmp_dir, 'frame_%06d.png'),
        '-y', '-loglevel', 'error'
    ]
    print(f'Extracting frames for {subj}...')
    subprocess.run(cmd, check=True)

    # Đọc ảnh và tạo numpy array
    png_files = sorted([f for f in os.listdir(tmp_dir) if f.endswith('.png')])
    frames = []
    for fname in tqdm(png_files, desc=f'Loading {subj}'):
        img = Image.open(os.path.join(tmp_dir, fname))
        arr = np.array(img, dtype=np.float32) / 255.0  # RGB, [0,1]
        frames.append(arr)

    frames = np.array(frames)
    np.save(frames_path, frames)
    print(f'Saved {frames_path} with shape {frames.shape}')

    # Xóa thư mục tạm để tiết kiệm dung lượng
    import shutil
    shutil.rmtree(tmp_dir)
