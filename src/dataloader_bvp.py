import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=0.7, highcut=2.5, fs=30, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

class UBFCrPPG_BVP(Dataset):
    def __init__(self, root_dir, clip_length=64, bvp_row=0, fs=30):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.bvp_row = bvp_row          # dòng chứa BVP (0)
        self.fs = fs
        self.subjects = sorted([
            s for s in os.listdir(root_dir)
            if os.path.exists(os.path.join(root_dir, s, 'frames.npy'))
            and os.path.exists(os.path.join(root_dir, s, 'mask.npy'))
        ])
        print(f"Tìm thấy {len(self.subjects)} subjects.")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj_dir = os.path.join(self.root_dir, self.subjects[idx])
        frames_path = os.path.join(subj_dir, 'frames.npy')
        mask_path = os.path.join(subj_dir, 'mask.npy')
        gt_path = os.path.join(subj_dir, 'ground_truth.txt')

        all_frames = np.load(frames_path, mmap_mode='r')
        all_masks  = np.load(mask_path, mmap_mode='r')
        T = min(len(all_frames), len(all_masks))

        gt_data = np.loadtxt(gt_path)
        if gt_data.ndim == 1:
            bvp = gt_data
        else:
            bvp = gt_data[self.bvp_row, :]

        if len(bvp) != T:
            from scipy.signal import resample
            bvp = resample(bvp, T)

        # Lọc bandpass nhẹ nhàng
        bvp = bandpass_filter(bvp, 0.7, 2.5, self.fs)

        if T < self.clip_length:
            raise ValueError(f"{self.subjects[idx]} quá ngắn ({T} < {self.clip_length})")

        start = np.random.randint(0, T - self.clip_length + 1)
        frames_clip = np.array(all_frames[start:start+self.clip_length, ::2, ::2, :]).copy()
        masks_clip  = np.array(all_masks[start:start+self.clip_length, ::2, ::2]).copy()
        bvp_clip    = bvp[start:start+self.clip_length].copy()

        frames_tensor = torch.from_numpy(frames_clip).permute(3, 0, 1, 2).float()  # (C, T, H, W)
        masks_tensor  = torch.from_numpy(masks_clip).unsqueeze(0).float()         # (1, T, H, W)
        bvp_tensor    = torch.from_numpy(bvp_clip).float()                       # (T,)

        return frames_tensor, masks_tensor, bvp_tensor