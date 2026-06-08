import numpy as np
from scipy.signal import butter, filtfilt
import os

def extract_rgb(frames, mask=None):
    if mask is None:
        r = frames[...,0].mean(axis=(1,2))
        g = frames[...,1].mean(axis=(1,2))
        b = frames[...,2].mean(axis=(1,2))
    else:
        denom = np.sum(mask) + 1e-8
        r = (frames[...,0] * mask).sum(axis=(1,2)) / denom
        g = (frames[...,1] * mask).sum(axis=(1,2)) / denom
        b = (frames[...,2] * mask).sum(axis=(1,2)) / denom
    return r, g, b

def chrom(r,g,b,fs=30):
    r = (r - r.mean()) / (r.std()+1e-8)
    g = (g - g.mean()) / (g.std()+1e-8)
    b = (b - b.mean()) / (b.std()+1e-8)

    X = 3*r - 2*g
    Y = 1.5*r + g - 1.5*b

    bf, af = butter(4, [0.7/(fs/2), 2.5/(fs/2)], btype='band')

    Xf = filtfilt(bf, af, X)
    Yf = filtfilt(bf, af, Y)

    alpha = np.std(Xf)/(np.std(Yf)+1e-8)

    return Xf - alpha*Yf

def bandpass(x, fs=30):
    bf, af = butter(4, [0.7/(fs/2), 2.5/(fs/2)], btype='band')
    return filtfilt(bf, af, x)

data_root = '/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1'

subjects = sorted(os.listdir(data_root))

ratios = [0.5, 0.6, 0.7, 0.8]

results = {r: [] for r in ratios}

max_frames = 1500

for subj in subjects:

    subj_dir = os.path.join(data_root, subj)

    frames_path = os.path.join(subj_dir, 'frames.npy')
    gt_path = os.path.join(subj_dir, 'ground_truth.txt')

    if not os.path.exists(frames_path):
        continue

    frames = np.load(frames_path, mmap_mode='r')

    gt = np.loadtxt(gt_path)

    T = min(len(frames), gt.shape[1], max_frames)

    frames = np.asarray(frames[:T], dtype=np.float32)

    gt_bvp = bandpass(gt[0,:T])

    H,W = frames.shape[1:3]

    r0,g0,b0 = extract_rgb(frames)
    base_ppg = chrom(r0,g0,b0)

    base_corr = np.corrcoef(base_ppg, gt_bvp)[0,1]

    print(f'\n{subj} baseline={base_corr:.3f}')

    for ratio in ratios:

        mask = np.ones((H,W), dtype=np.float32)
        mask[int(H*ratio):,:] = 0

        r,g,b = extract_rgb(frames, mask)

        ppg = chrom(r,g,b)

        corr = np.corrcoef(ppg, gt_bvp)[0,1]

        delta = corr - base_corr

        results[ratio].append(delta)

        print(f'  ratio={ratio}: {corr:.3f} ({delta:+.3f})')

print('\n===== SUMMARY =====')

for ratio in ratios:
    arr = np.array(results[ratio])

    print(
        f'ratio={ratio} | '
        f'mean_delta={arr.mean():+.4f} | '
        f'winrate={(arr>0).mean()*100:.1f}%'
    )