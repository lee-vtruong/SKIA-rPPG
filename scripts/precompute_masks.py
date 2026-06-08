import os
import sys
sys.path.append('/raid/hvtham/whale/SIKA')
from utils.kinematic_mask_simple import get_mouth_mask
import cv2
import numpy as np
from tqdm import tqdm

DATA_DIR = '/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1'

for subj in sorted(os.listdir(DATA_DIR)):
    subj_path = os.path.join(DATA_DIR, subj)
    if not os.path.isdir(subj_path):
        continue

    mask_path = os.path.join(subj_path, 'mask.npy')
    if os.path.exists(mask_path):
        print(f'{subj} already has mask, skipping.')
        continue

    # Ưu tiên vid.mp4, nếu không có thì dùng vid.avi
    vid_path_mp4 = os.path.join(subj_path, 'vid.mp4')
    vid_path_avi = os.path.join(subj_path, 'vid.avi')
    if os.path.exists(vid_path_mp4):
        vid_path = vid_path_mp4
    else:
        vid_path = vid_path_avi

    if not os.path.exists(vid_path):
        print(f'{subj} no video found, skipping.')
        continue

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f'{subj} cannot open video, skipping.')
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    masks = []
    ret, prev = cap.read()
    if not ret:
        cap.release()
        continue
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)

    for _ in tqdm(range(total_frames - 1), desc=f'Processing {subj}'):
        ret, curr = cap.read()
        if not ret:
            break
        curr_rgb = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)
        mask = get_mouth_mask(curr_rgb, prev)
        masks.append(mask)
        prev = curr_rgb

    cap.release()

    if len(masks) > 0:
        masks = np.array(masks, dtype=np.float32)
        np.save(mask_path, masks)
        print(f'Saved {mask_path} with shape {masks.shape}')
    else:
        print(f'{subj} no frames processed.')
