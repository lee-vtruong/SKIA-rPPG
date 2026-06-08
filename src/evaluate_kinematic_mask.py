import numpy as np
from scipy.signal import butter, filtfilt, resample
import os
import matplotlib.pyplot as plt

def chrom_ppg_from_frames(frames, fs=30, lowcut=0.7, highcut=2.5, mask=None):
    """
    Trích xuất BVP bằng CHROM.
    frames: (T, H, W, C) numpy array float [0,1]
    mask: (T, H, W) hoặc None. Nếu có, nhân mask vào frames trước khi tính.
    """
    if mask is not None:
        frames = frames * mask[..., None]  # broadcast
    r = np.mean(frames[:, :, :, 0], axis=(1,2))
    g = np.mean(frames[:, :, :, 1], axis=(1,2))
    b = np.mean(frames[:, :, :, 2], axis=(1,2))
    # Chuẩn hóa
    r = (r - np.mean(r)) / (np.std(r) + 1e-8)
    g = (g - np.mean(g)) / (np.std(g) + 1e-8)
    b = (b - np.mean(b)) / (np.std(b) + 1e-8)
    X = 0.77 * r - 0.51 * g
    Y = 0.77 * r + 0.51 * g - 0.77 * b
    nyq = 0.5 * fs
    b_b, a_b = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
    X_f = filtfilt(b_b, a_b, X)
    Y_f = filtfilt(b_b, a_b, Y)
    ppg = X_f - Y_f
    return ppg

def evaluate_masks(data_root, subjects_list, output_dir='evaluation_results'):
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for subj in subjects_list:
        subj_dir = os.path.join(data_root, subj)
        frames_path = os.path.join(subj_dir, 'frames.npy')
        mask_path = os.path.join(subj_dir, 'mask.npy')
        gt_path = os.path.join(subj_dir, 'ground_truth.txt')
        if not all(os.path.exists(p) for p in [frames_path, mask_path, gt_path]):
            continue
        frames = np.load(frames_path, mmap_mode='r')
        masks = np.load(mask_path, mmap_mode='r')
        gt = np.loadtxt(gt_path)
        bvp_gt = gt[0, :]  # dòng 0 PPG
        T = min(len(frames), len(masks), len(bvp_gt))
        frames = np.array(frames[:T])
        masks = np.array(masks[:T])
        bvp_gt = bvp_gt[:T]
        if len(bvp_gt) != T:
            bvp_gt = resample(bvp_gt, T)

        # CHROM không mask
        ppg_no_mask = chrom_ppg_from_frames(frames, mask=None)
        # CHROM có mask
        ppg_masked = chrom_ppg_from_frames(frames, mask=masks)

        # Lọc ground truth
        nyq = 15
        b_b, a_b = butter(4, [0.7/nyq, 2.5/nyq], btype='band')
        bvp_gt_f = filtfilt(b_b, a_b, bvp_gt)

        pearson_no = np.corrcoef(ppg_no_mask, bvp_gt_f)[0,1]
        pearson_masked = np.corrcoef(ppg_masked, bvp_gt_f)[0,1]

        print(f'{subj}: Pearson no_mask={pearson_no:.3f}, masked={pearson_masked:.3f}')

        results[subj] = {'no_mask': pearson_no, 'masked': pearson_masked}

        # Vẽ minh họa một vài subject
        if len(results) <= 5:
            plt.figure(figsize=(12,4))
            plt.plot(bvp_gt_f[:500], label='GT')
            plt.plot(ppg_no_mask[:500], label='No Mask')
            plt.plot(ppg_masked[:500], label='Masked')
            plt.legend()
            plt.title(f'{subj} - Pearson: {pearson_no:.2f} (no) vs {pearson_masked:.2f} (masked)')
            plt.savefig(f'{output_dir}/{subj}_comparison.png')
            plt.close()

    # Tổng hợp
    avg_no = np.mean([v['no_mask'] for v in results.values()])
    avg_masked = np.mean([v['masked'] for v in results.values()])
    print(f'\nTrung bình Pearson: No Mask={avg_no:.4f}, Masked={avg_masked:.4f}')
    return results

if __name__ == '__main__':
    data_root = '/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1'
    subjects = sorted(os.listdir(data_root))[:10]  # Làm 10 subject trước để test nhanh
    evaluate_masks(data_root, subjects, output_dir='evaluation_results')