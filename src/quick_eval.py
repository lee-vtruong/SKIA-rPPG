import numpy as np
from scipy.signal import butter, filtfilt
import os, sys

def chrom_ppg_from_frames(frames, fs=30, mask=None):
    """ frames: (T, H, W, C) float [0,1] hoặc mmap, mask: (T, H, W) hoặc None """
    if mask is not None:
        frames = frames * mask[..., None]
    r = np.mean(frames[:, :, :, 0], axis=(1,2))
    g = np.mean(frames[:, :, :, 1], axis=(1,2))
    b = np.mean(frames[:, :, :, 2], axis=(1,2))
    r = (r - np.mean(r)) / (np.std(r) + 1e-8)
    g = (g - np.mean(g)) / (np.std(g) + 1e-8)
    b = (b - np.mean(b)) / (np.std(b) + 1e-8)
    X = 0.77 * r - 0.51 * g
    Y = 0.77 * r + 0.51 * g - 0.77 * b
    nyq = 0.5 * fs
    b_b, a_b = butter(4, [0.7/nyq, 2.5/nyq], btype='band')
    X_f = filtfilt(b_b, a_b, X)
    Y_f = filtfilt(b_b, a_b, Y)
    return X_f - Y_f

# Đường dẫn
data_root = '/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1'
subjects = sorted(os.listdir(data_root))[:3]   # chỉ làm 3 subject để test nhanh
num_frames = 500                                # mỗi subject lấy 500 frame đầu

for subj in subjects:
    subj_dir = os.path.join(data_root, subj)
    frames_path = os.path.join(subj_dir, 'frames.npy')
    mask_path = os.path.join(subj_dir, 'mask.npy')
    gt_path = os.path.join(subj_dir, 'ground_truth.txt')
    if not all(os.path.exists(p) for p in [frames_path, mask_path, gt_path]):
        continue

    # Dùng memory‑map, chỉ đọc 500 frame đầu
    frames_mmap = np.load(frames_path, mmap_mode='r')
    masks_mmap  = np.load(mask_path, mmap_mode='r')
    gt = np.loadtxt(gt_path)

    T = min(len(frames_mmap), len(masks_mmap), gt.shape[1] if gt.ndim > 1 else len(gt))
    n = min(num_frames, T)
    frames = np.array(frames_mmap[:n])
    masks  = np.array(masks_mmap[:n])
    bvp_gt = gt[0, :n] if gt.ndim > 1 else gt[:n]

    # CHROM không mask
    ppg_no = chrom_ppg_from_frames(frames, mask=None)
    # CHROM với mask (dùng mask từ file, đã được làm mịn)
    ppg_masked = chrom_ppg_from_frames(frames, mask=masks)

    # Lọc ground truth
    nyq = 15
    b_b, a_b = butter(4, [0.7/nyq, 2.5/nyq], btype='band')
    bvp_gt_f = filtfilt(b_b, a_b, bvp_gt)

    p_no = np.corrcoef(ppg_no, bvp_gt_f)[0,1]
    p_masked = np.corrcoef(ppg_masked, bvp_gt_f)[0,1]
    print(f'{subj} (first {n} frames): Pearson no_mask={p_no:.3f}, masked={p_masked:.3f}')