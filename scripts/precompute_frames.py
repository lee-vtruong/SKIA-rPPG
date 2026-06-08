import os
import sys
import numpy as np
import cv2
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
    cap = cv2.VideoCapture(vid_path_avi)
    if not cap.isOpened():
        print(f'{subj} cannot open video, skipping.')
        continue
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in tqdm(range(total_frames), desc=f'Reading {subj}'):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    if len(frames) != total_frames:
        print(f'Warning: {subj} expected {total_frames} but got {len(frames)}')
    frames = np.array(frames, dtype=np.float32) / 255.0  # chuẩn hóa [0,1]
    np.save(frames_path, frames)
    print(f'Saved {frames_path} with shape {frames.shape}')
