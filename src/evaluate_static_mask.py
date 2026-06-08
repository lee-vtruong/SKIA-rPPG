import numpy as np
from scipy.signal import butter, filtfilt
import os

def extract_rgb_means(frames, static_mask=None, max_frames=None, batch_size=256):
    T = len(frames) if max_frames is None else min(len(frames), max_frames)

    r_list, g_list, b_list = [], [], []

    if static_mask is not None:
        denom = np.sum(static_mask) + 1e-8

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        x = np.asarray(frames[start:end], dtype=np.float32)

        if static_mask is None:
            r = x[..., 0].mean(axis=(1, 2))
            g = x[..., 1].mean(axis=(1, 2))
            b = x[..., 2].mean(axis=(1, 2))
        else:
            r = (x[..., 0] * static_mask).sum(axis=(1, 2)) / denom
            g = (x[..., 1] * static_mask).sum(axis=(1, 2)) / denom
            b = (x[..., 2] * static_mask).sum(axis=(1, 2)) / denom

        r_list.append(r)
        g_list.append(g)
        b_list.append(b)

    return np.concatenate(r_list), np.concatenate(g_list), np.concatenate(b_list)


def chrom_from_rgb(r, g, b, fs=30):
    r = (r - np.mean(r)) / (np.std(r) + 1e-8)
    g = (g - np.mean(g)) / (np.std(g) + 1e-8)
    b = (b - np.mean(b)) / (np.std(b) + 1e-8)

    X = 3 * r - 2 * g
    Y = 1.5 * r + g - 1.5 * b

    nyq = 0.5 * fs
    bf, af = butter(4, [0.7 / nyq, 2.5 / nyq], btype="band")

    X_f = filtfilt(bf, af, X)
    Y_f = filtfilt(bf, af, Y)

    alpha = np.std(X_f) / (np.std(Y_f) + 1e-8)
    return X_f - alpha * Y_f


def bandpass(x, fs=30):
    nyq = 0.5 * fs
    bf, af = butter(4, [0.7 / nyq, 2.5 / nyq], btype="band")
    return filtfilt(bf, af, x)


data_root = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"
subjects = sorted(os.listdir(data_root))[:10]
results = {}

max_frames = 1500   # chạy nhanh trước, sau ổn thì đổi None
batch_size = 256
fs = 30

for subj in subjects:
    subj_dir = os.path.join(data_root, subj)
    frames_path = os.path.join(subj_dir, "frames.npy")
    gt_path = os.path.join(subj_dir, "ground_truth.txt")

    if not os.path.exists(frames_path) or not os.path.exists(gt_path):
        continue

    frames = np.load(frames_path, mmap_mode="r")
    gt = np.loadtxt(gt_path)

    T = min(len(frames), gt.shape[1], max_frames)

    H, W = frames.shape[1], frames.shape[2]
    static_mask = np.ones((H, W), dtype=np.float32)
    static_mask[int(H * 0.6):, :] = 0.0

    r0, g0, b0 = extract_rgb_means(frames, None, max_frames=T, batch_size=batch_size)
    r1, g1, b1 = extract_rgb_means(frames, static_mask, max_frames=T, batch_size=batch_size)

    ppg_no = chrom_from_rgb(r0, g0, b0, fs=fs)
    ppg_static = chrom_from_rgb(r1, g1, b1, fs=fs)

    bvp_gt = gt[0, :T]
    bvp_gt_f = bandpass(bvp_gt, fs=fs)

    p_no = np.corrcoef(ppg_no, bvp_gt_f)[0, 1]
    p_static = np.corrcoef(ppg_static, bvp_gt_f)[0, 1]

    results[subj] = {"no_mask": p_no, "static_mask": p_static}

    print(f"{subj}: no_mask={p_no:.3f}, static_mask={p_static:.3f}, delta={p_static - p_no:+.3f}")

avg_no = np.mean([v["no_mask"] for v in results.values()])
avg_static = np.mean([v["static_mask"] for v in results.values()])
wins = sum(v["static_mask"] > v["no_mask"] for v in results.values())

print(f"\nTrung bình: No Mask={avg_no:.4f}, Static Mask={avg_static:.4f}")
print(f"Static thắng: {wins}/{len(results)} subjects")