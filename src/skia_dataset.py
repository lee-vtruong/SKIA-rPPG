import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample


class SKIADataset(Dataset):
    def __init__(self, data_root, subjects, clip_len=128, stride=64, size=128, mode="raw"):
        self.data_root = data_root
        self.subjects = subjects
        self.clip_len = clip_len
        self.stride = stride
        self.size = size
        self.mode = mode
        self.samples = []

        for subj in subjects:
            subj_dir = os.path.join(data_root, subj)
            frames_path = os.path.join(subj_dir, "frames.npy")
            gt_path = os.path.join(subj_dir, "ground_truth.txt")

            if not os.path.exists(frames_path) or not os.path.exists(gt_path):
                continue

            frames = np.load(frames_path, mmap_mode="r")
            gt = np.loadtxt(gt_path)
            T = min(len(frames), gt.shape[1])

            for start in range(0, T - clip_len, stride):
                self.samples.append((subj, start))

    def __len__(self):
        return len(self.samples)

    def _resize_center_crop(self, x):
        # x: T,H,W,C
        T, H, W, C = x.shape
        top = int(H * 0.1)
        bottom = int(H * 0.9)
        left = int(W * 0.2)
        right = int(W * 0.8)

        x = x[:, top:bottom, left:right, :]

        # simple resize bằng torch
        x = torch.from_numpy(x.copy()).permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(
            x,
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False,
        )
        return x  # T,C,H,W

    def _static_mask(self, T, H, W):
        mask = torch.ones((T, 1, H, W), dtype=torch.float32)
        mask[:, :, int(H * 0.7):, :] = 0.0
        return mask

    def __getitem__(self, idx):
        subj, start = self.samples[idx]
        subj_dir = os.path.join(self.data_root, subj)

        frames = np.load(os.path.join(subj_dir, "frames.npy"), mmap_mode="r")
        gt = np.loadtxt(os.path.join(subj_dir, "ground_truth.txt"))

        clip = np.asarray(frames[start:start + self.clip_len], dtype=np.float32)
        bvp = gt[0, start:start + self.clip_len].astype(np.float32)

        x = self._resize_center_crop(clip)  # T,C,H,W

        # normalize video
        x = (x - x.mean()) / (x.std() + 1e-6)

        T, C, H, W = x.shape
        static_mask = self._static_mask(T, H, W)

        raw = x
        masked = x * static_mask
        diff = torch.zeros_like(x)
        diff[1:] = x[1:] - x[:-1]

        if self.mode == "raw":
            inp = raw
        elif self.mode == "masked":
            inp = masked
        elif self.mode == "raw_masked":
            inp = torch.cat([raw, masked], dim=1)
        elif self.mode == "full":
            inp = torch.cat([raw, masked, diff], dim=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        bvp = torch.from_numpy(bvp)
        bvp = (bvp - bvp.mean()) / (bvp.std() + 1e-6)

        # model expects B,C,T,H,W
        inp = inp.permute(1, 0, 2, 3)

        return inp, bvp
