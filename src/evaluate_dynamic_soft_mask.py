import os
import numpy as np
from scipy.signal import butter, filtfilt


def extract_rgb_means(frames, masks=None, max_frames=None, batch_size=256):
    T = len(frames) if max_frames is None else min(len(frames), max_frames)

    r_list, g_list, b_list = [], [], []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)

        x = np.asarray(frames[start:end], dtype=np.float32)

        if masks is None:
            r = x[..., 0].mean(axis=(1, 2))
            g = x[..., 1].mean(axis=(1, 2))
            b = x[..., 2].mean(axis=(1, 2))
        else:
            m = np.asarray(masks[start:end], dtype=np.float32)
            denom = m.sum(axis=(1, 2)) + 1e-8

            r = (x[..., 0] * m).sum(axis=(1, 2)) / denom
            g = (x[..., 1] * m).sum(axis=(1, 2)) / denom
            b = (x[..., 2] * m).sum(axis=(1, 2)) / denom

        r_list.append(r)
        g_list.append(g)
        b_list.append(b)

    return np.concatenate(r_list), np.concatenate(g_list), np.concatenate(b_list)


def chrom_from_rgb(r, g, b, fs=30):
    r = (r - r.mean()) / (r.std() + 1e-8)
    g = (g - g.mean()) / (g.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)

    X = 3.0 * r - 2.0 * g
    Y = 1.5 * r + g - 1.5 * b

    bf, af = butter(4, [0.7 / (fs / 2), 2.5 / (fs / 2)], btype="band")

    Xf = filtfilt(bf, af, X)
    Yf = filtfilt(bf, af, Y)

    alpha = np.std(Xf) / (np.std(Yf) + 1e-8)

    return Xf - alpha * Yf


def bandpass(x, fs=30):
    bf, af = butter(4, [0.7 / (fs / 2), 2.5 / (fs / 2)], btype="band")
    return filtfilt(bf, af, x)


def make_static_mask(H, W, ratio=0.7):
    mask = np.ones((H, W), dtype=np.float32)
    mask[int(H * ratio):, :] = 0.0
    return mask


def evaluate():
    data_root = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"

    subjects = sorted(os.listdir(data_root))

    fs = 30
    max_frames = 1500
    batch_size = 256

    results = {}

    for subj in subjects:
        subj_dir = os.path.join(data_root, subj)

        frames_path = os.path.join(subj_dir, "frames.npy")
        dynamic_mask_path = os.path.join(subj_dir, "mask.npy")
        gt_path = os.path.join(subj_dir, "ground_truth.txt")

        if not (
            os.path.exists(frames_path)
            and os.path.exists(dynamic_mask_path)
            and os.path.exists(gt_path)
        ):
            continue

        frames = np.load(frames_path, mmap_mode="r")
        dynamic_masks = np.load(dynamic_mask_path, mmap_mode="r")
        gt = np.loadtxt(gt_path)

        T = min(len(frames), len(dynamic_masks), gt.shape[1], max_frames)

        H, W = frames.shape[1], frames.shape[2]

        static_mask = make_static_mask(H, W, ratio=0.7)
        static_masks = np.broadcast_to(static_mask, (T, H, W))

        r0, g0, b0 = extract_rgb_means(
            frames,
            masks=None,
            max_frames=T,
            batch_size=batch_size,
        )

        rs, gs, bs = extract_rgb_means(
            frames,
            masks=static_masks,
            max_frames=T,
            batch_size=batch_size,
        )

        rd, gd, bd = extract_rgb_means(
            frames,
            masks=dynamic_masks,
            max_frames=T,
            batch_size=batch_size,
        )

        ppg_no = chrom_from_rgb(r0, g0, b0, fs=fs)
        ppg_static = chrom_from_rgb(rs, gs, bs, fs=fs)
        ppg_dynamic = chrom_from_rgb(rd, gd, bd, fs=fs)

        bvp_gt = gt[0, :T]
        bvp_gt_f = bandpass(bvp_gt, fs=fs)

        corr_no = np.corrcoef(ppg_no, bvp_gt_f)[0, 1]
        corr_static = np.corrcoef(ppg_static, bvp_gt_f)[0, 1]
        corr_dynamic = np.corrcoef(ppg_dynamic, bvp_gt_f)[0, 1]

        results[subj] = {
            "no_mask": corr_no,
            "static": corr_static,
            "dynamic": corr_dynamic,
        }

        print(
            f"{subj}: "
            f"no={corr_no:.3f}, "
            f"static={corr_static:.3f}, "
            f"dynamic={corr_dynamic:.3f}, "
            f"dyn-static={corr_dynamic - corr_static:+.3f}"
        )

    no_arr = np.array([v["no_mask"] for v in results.values()])
    static_arr = np.array([v["static"] for v in results.values()])
    dynamic_arr = np.array([v["dynamic"] for v in results.values()])

    print("\n===== SUMMARY =====")
    print(f"No Mask mean   = {no_arr.mean():.4f}")
    print(f"Static mean    = {static_arr.mean():.4f}")
    print(f"Dynamic mean   = {dynamic_arr.mean():.4f}")
    print(f"Static delta   = {(static_arr - no_arr).mean():+.4f}")
    print(f"Dynamic delta  = {(dynamic_arr - no_arr).mean():+.4f}")
    print(f"Dynamic-static = {(dynamic_arr - static_arr).mean():+.4f}")
    print(f"Static winrate over no   = {(static_arr > no_arr).mean() * 100:.1f}%")
    print(f"Dynamic winrate over no  = {(dynamic_arr > no_arr).mean() * 100:.1f}%")
    print(f"Dynamic winrate over static = {(dynamic_arr > static_arr).mean() * 100:.1f}%")


if __name__ == "__main__":
    evaluate()