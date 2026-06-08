import os, subprocess, numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

subj_path = '/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1/subject39'
vid_path = os.path.join(subj_path, 'vid.avi')

tmp_dir = os.path.join(subj_path, 'tmp_frames')
os.makedirs(tmp_dir, exist_ok=True)

# Giải nén frame từ AVI bằng ffmpeg
cmd = ['ffmpeg', '-i', vid_path, '-vsync', '0', os.path.join(tmp_dir, 'frame_%06d.png'), '-y', '-loglevel', 'error']
subprocess.run(cmd, check=True)

# Đọc ảnh và lưu frames.npy
png_files = sorted([f for f in os.listdir(tmp_dir) if f.endswith('.png')])
frames = []
for fname in tqdm(png_files, desc='Loading subject39'):
    img = Image.open(os.path.join(tmp_dir, fname))
    frames.append(np.array(img, dtype=np.float32) / 255.0)

frames = np.array(frames)
np.save(os.path.join(subj_path, 'frames.npy'), frames)
print(f'Đã lưu frames.npy với shape {frames.shape}')

shutil.rmtree(tmp_dir)
