import os, sys, numpy as np
sys.path.append('/raid/hvtham/whale/SIKA')
from utils.kinematic_mask_simple import get_mouth_mask
from tqdm import tqdm

DATA_DIR = '/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1'

for subj in sorted(os.listdir(DATA_DIR)):
    subj_path = os.path.join(DATA_DIR, subj)
    if not os.path.isdir(subj_path):
        continue
    frames_path = os.path.join(subj_path, 'frames.npy')
    mask_path = os.path.join(subj_path, 'mask.npy')
    if not os.path.exists(frames_path):
        continue

    recalc = False
    if os.path.exists(mask_path):
        try:
            mask_existing = np.load(mask_path, mmap_mode='r')
            frames_existing = np.load(frames_path, mmap_mode='r')
            if len(mask_existing) == len(frames_existing):
                print(f'{subj} mask already correct, skipping.')
                continue
            else:
                print(f'{subj} mask length mismatch, recalculating.')
                recalc = True
        except Exception as e:
            print(f'{subj} mask file corrupted ({e}), recalculating.')
            os.remove(mask_path)   # xóa file hỏng
            recalc = True
    else:
        recalc = True

    if recalc:
        # Đọc frames bằng memory-map
        frames_mmap = np.load(frames_path, mmap_mode='r')
        T = len(frames_mmap)
        masks = []
        for i in tqdm(range(T), desc=f'Mask {subj}'):
            if i == 0:
                mask = np.ones((frames_mmap.shape[1], frames_mmap.shape[2]), dtype=np.float32)
            else:
                prev_frame = (frames_mmap[i-1] * 255).astype(np.uint8)
                curr_frame = (frames_mmap[i]   * 255).astype(np.uint8)
                mask = get_mouth_mask(curr_frame, prev_frame)
            masks.append(mask)
        masks = np.array(masks, dtype=np.float32)
        np.save(mask_path, masks)
        print(f'Saved {mask_path} with shape {masks.shape}')
