import os
import csv
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import wilcoxon


DATA_ROOT = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"
OUT_CSV = "static_final_results.csv"

FS = 30
MAX_FRAMES = 1500
BATCH_SIZE = 256
STATIC_RATIO = 0.7


def bandpass(x, fs=30):
    bf, af = butter(4, [0.7 / (fs / 2), 2.5 / (fs / 2)], btype="band")
    return filtfilt(bf, af, x)


def extract_rgb(frames, mask=None, max_frames=None, batch_size=256):
    T = len(frames) if max_frames is None else min(len(frames), max_frames)

    r_list, g_list, b_list = [], [], []

    if mask is not None:
        denom = mask.sum() + 1e-8

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        x = np.asarray(frames[start:end], dtype=np.float32)

        if mask is None:
            r = x[..., 0].mean(axis=(1, 2))
            g = x[..., 1].mean(axis=(1, 2))
            b = x[..., 2].mean(axis=(1, 2))
        else:
            r = (x[..., 0] * mask).sum(axis=(1, 2)) / denom
            g = (x[..., 1] * mask).sum(axis=(1, 2)) / denom
            b = (x[..., 2] * mask).sum(axis=(1, 2)) / denom

        r_list.append(r)
        g_list.append(g)
        b_list.append(b)

    return np.concatenate(r_list), np.concatenate(g_list), np.concatenate(b_list)


def chrom(r, g, b, fs=30):
    r = (r - r.mean()) / (r.std() + 1e-8)
    g = (g - g.mean()) / (g.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)

    X = 3.0 * r - 2.0 * g
    Y = 1.5 * r + g - 1.5 * b

    Xf = bandpass(X, fs)
    Yf = bandpass(Y, fs)

    alpha = np.std(Xf) / (np.std(Yf) + 1e-8)

    return Xf - alpha * Yf


def pearson(a, b):
    return np.corrcoef(a, b)[0, 1]


def make_static_mask(H, W, ratio=0.7):
    mask = np.ones((H, W), dtype=np.float32)
    mask[int(H * ratio):, :] = 0.0
    return mask


def main():
    subjects = sorted(os.listdir(DATA_ROOT))
    rows = []

    for subj in subjects:
        subj_dir = os.path.join(DATA_ROOT, subj)

        frames_path = os.path.join(subj_dir, "frames.npy")
        gt_path = os.path.join(subj_dir, "ground_truth.txt")

        if not os.path.exists(frames_path) or not os.path.exists(gt_path):
            continue

        frames = np.load(frames_path, mmap_mode="r")
        gt = np.loadtxt(gt_path)

        T = min(len(frames), gt.shape[1], MAX_FRAMES)

        H, W = frames.shape[1], frames.shape[2]
        static_mask = make_static_mask(H, W, STATIC_RATIO)

        r0, g0, b0 = extract_rgb(
            frames,
            mask=None,
            max_frames=T,
            batch_size=BATCH_SIZE,
        )

        rs, gs, bs = extract_rgb(
            frames,
            mask=static_mask,
            max_frames=T,
            batch_size=BATCH_SIZE,
        )

        ppg_no = chrom(r0, g0, b0, FS)
        ppg_static = chrom(rs, gs, bs, FS)

        gt_bvp = bandpass(gt[0, :T], FS)

        corr_no = pearson(ppg_no, gt_bvp)
        corr_static = pearson(ppg_static, gt_bvp)
        delta = corr_static - corr_no

        rows.append([subj, corr_no, corr_static, delta])

        print(
            f"{subj}: "
            f"no={corr_no:.3f}, "
            f"static={corr_static:.3f}, "
            f"delta={delta:+.3f}"
        )

    no_arr = np.array([r[1] for r in rows])
    static_arr = np.array([r[2] for r in rows])
    delta_arr = np.array([r[3] for r in rows])

    stat, p_value = wilcoxon(static_arr, no_arr)

    print("\n===== FINAL SUMMARY =====")
    print(f"Subjects: {len(rows)}")
    print(f"No Mask mean      = {no_arr.mean():.4f}")
    print(f"Static mean       = {static_arr.mean():.4f}")
    print(f"Mean delta        = {delta_arr.mean():+.4f}")
    print(f"Median delta      = {np.median(delta_arr):+.4f}")
    print(f"Winrate           = {(delta_arr > 0).mean() * 100:.1f}%")
    print(f"Wilcoxon statistic= {stat:.4f}")
    print(f"Wilcoxon p-value  = {p_value:.6f}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "no_mask", "static_mask", "delta"])
        writer.writerows(rows)

    print(f"\nSaved CSV to: {OUT_CSV}")


if __name__ == "__main__":
    main()
