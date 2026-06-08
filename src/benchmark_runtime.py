import os
import time
import numpy as np
import torch
from scipy.signal import butter, filtfilt

from skia_dataset import SKIADataset
from skia_model import SKIANetLite


DATA_ROOT = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"
FS = 30


def bandpass(x):
    b, a = butter(4, [0.7/(FS/2), 2.5/(FS/2)], btype="band")
    return filtfilt(b, a, x)


def chrom_from_rgb(r, g, b):
    r = (r - r.mean()) / (r.std() + 1e-8)
    g = (g - g.mean()) / (g.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)

    X = 3*r - 2*g
    Y = 1.5*r + g - 1.5*b

    Xf = bandpass(X)
    Yf = bandpass(Y)

    alpha = np.std(Xf) / (np.std(Yf) + 1e-8)

    return Xf - alpha * Yf


def benchmark_chrom(num_subjects=10, max_frames=1500, use_mask=False):
    subjects = sorted(os.listdir(DATA_ROOT))[-num_subjects:]

    total_frames = 0
    start_time = time.time()

    for subj in subjects:
        subj_dir = os.path.join(DATA_ROOT, subj)
        frames_path = os.path.join(subj_dir, "frames.npy")

        if not os.path.exists(frames_path):
            continue

        frames = np.load(frames_path, mmap_mode="r")
        T = min(len(frames), max_frames)

        x = np.asarray(frames[:T], dtype=np.float32)

        if use_mask:
            H, W = x.shape[1], x.shape[2]
            mask = np.ones((H, W), dtype=np.float32)
            mask[int(H * 0.7):, :] = 0.0
            denom = mask.sum() + 1e-8

            r = (x[..., 0] * mask).sum(axis=(1, 2)) / denom
            g = (x[..., 1] * mask).sum(axis=(1, 2)) / denom
            b = (x[..., 2] * mask).sum(axis=(1, 2)) / denom
        else:
            r = x[..., 0].mean(axis=(1, 2))
            g = x[..., 1].mean(axis=(1, 2))
            b = x[..., 2].mean(axis=(1, 2))

        _ = chrom_from_rgb(r, g, b)

        total_frames += T

    elapsed = time.time() - start_time
    fps = total_frames / elapsed

    return elapsed, fps


def get_in_channels(mode):
    if mode in ["raw", "masked"]:
        return 3
    if mode == "raw_masked":
        return 6
    if mode == "full":
        return 9
    raise ValueError(mode)


def benchmark_skia(mode, ckpt, max_batches=30):
    subjects = sorted(os.listdir(DATA_ROOT))
    test_subjects = subjects[32:]

    dataset = SKIADataset(
        DATA_ROOT,
        test_subjects,
        clip_len=64,
        stride=128,
        size=96,
        mode=mode,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SKIANetLite(
        in_channels=get_in_channels(mode)
    ).to(device)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    total_frames = 0
    total_time = 0.0

    with torch.no_grad():
        for i in range(min(len(dataset), max_batches)):
            x, _ = dataset[i]
            x = x.unsqueeze(0).to(device)

            if device == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()

            _ = model(x)

            if device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - start_time

            total_time += elapsed
            total_frames += x.shape[2]

    fps = total_frames / total_time

    return total_time, fps


def main():
    print("===== Runtime Benchmark =====")

    t, fps = benchmark_chrom(use_mask=False)
    print(f"CHROM              | time={t:.2f}s | FPS={fps:.2f}")

    t, fps = benchmark_chrom(use_mask=True)
    print(f"CHROM + StaticMask | time={t:.2f}s | FPS={fps:.2f}")

    t, fps = benchmark_skia(
        "raw",
        "../checkpoints/skia_raw_best.pt",
    )
    print(f"SKIA Raw           | time={t:.2f}s | FPS={fps:.2f}")

    t, fps = benchmark_skia(
        "raw_masked",
        "../checkpoints/skia_raw_masked_best.pt",
    )
    print(f"SKIA Raw+Masked    | time={t:.2f}s | FPS={fps:.2f}")

    t, fps = benchmark_skia(
        "full",
        "../checkpoints/skia_full_best.pt",
    )
    print(f"SKIA Full          | time={t:.2f}s | FPS={fps:.2f}")


if __name__ == "__main__":
    main()
