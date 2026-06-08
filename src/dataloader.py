import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from scipy.signal import butter, filtfilt


def bandpass_filter(data, lowcut=0.7, highcut=2.5, fs=30, order=4):
    """Lọc bandpass - chỉ áp dụng cho tín hiệu BVP dạng sóng, không cần cho HR."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


class UBFCrPPG_HR(Dataset):
    """Dataset cho bài toán Heart Rate Estimation."""
    def __init__(self, root_dir, clip_length=64, hr_row=1, fs=30):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.hr_row = hr_row      # Dòng chứa HR (mặc định 1)
        self.fs = fs
        self.subjects = sorted([
            s for s in os.listdir(root_dir)
            if os.path.exists(os.path.join(root_dir, s, 'frames.npy'))
            and os.path.exists(os.path.join(root_dir, s, 'mask.npy'))
        ])
        print(f"Tìm thấy {len(self.subjects)} subject có đủ frames và mask.")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj_dir = os.path.join(self.root_dir, self.subjects[idx])
        frames_path = os.path.join(subj_dir, 'frames.npy')
        mask_path = os.path.join(subj_dir, 'mask.npy')
        gt_path = os.path.join(subj_dir, 'ground_truth.txt')

        # Đọc frames và masks bằng memory-map
        all_frames = np.load(frames_path, mmap_mode='r')   # (T, H, W, C)
        all_masks  = np.load(mask_path, mmap_mode='r')     # (T, H, W)
        T = min(len(all_frames), len(all_masks))

        # Đọc ground truth HR (dòng 1)
        gt_data = np.loadtxt(gt_path)          # shape (3, N) hoặc (N,)
        if gt_data.ndim == 1:
            hr_signal = gt_data
        else:
            hr_signal = gt_data[self.hr_row, :]    # dòng HR

        # Resample HR về đúng số frame nếu cần
        if len(hr_signal) != T:
            hr_signal = signal.resample(hr_signal, T)

        # Cắt clip ngẫu nhiên
        if T < self.clip_length:
            raise ValueError(f"{self.subjects[idx]} quá ngắn ({T} < {self.clip_length})")

        start = np.random.randint(0, T - self.clip_length + 1)

        # Downsample 2x (từ 480×640 → 240×320)
        frames_clip = np.array(
            all_frames[start:start + self.clip_length, ::2, ::2, :]
        ).copy()
        masks_clip = np.array(
            all_masks[start:start + self.clip_length, ::2, ::2]
        ).copy()

        # HR trung bình của clip (làm tròn 1 chữ số thập phân)
        hr_clip = hr_signal[start:start + self.clip_length]
        hr_mean = np.mean(hr_clip)

        # Chuyển sang tensor
        frames_tensor = torch.from_numpy(frames_clip).permute(3, 0, 1, 2).float()  # (C,T,H,W)
        masks_tensor  = torch.from_numpy(masks_clip).unsqueeze(0).float()          # (1,T,H,W)
        hr_tensor     = torch.tensor([hr_mean], dtype=torch.float32)               # (1,)

        return frames_tensor, masks_tensor, hr_tensor


# ─── Test nhanh ───
if __name__ == '__main__':
    ds = UBFCrPPG_HR(
        root_dir='/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1',
        clip_length=64,
        hr_row=1,
        fs=30
    )
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    frames, masks, hr = next(iter(loader))
    print(f"Frames: {frames.shape}")    # (B, 3, 64, 240, 320)
    print(f"Masks:  {masks.shape}")     # (B, 1, 64, 240, 320)
    print(f"HR:     {hr.numpy().flatten()}")   # [97.0, 102.0]