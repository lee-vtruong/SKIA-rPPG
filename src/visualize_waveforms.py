import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


DATA_ROOT = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"

FS = 30
MAX_FRAMES = 1500


def bandpass(x, fs=30):
    bf, af = butter(4, [0.7/(fs/2), 2.5/(fs/2)], btype='band')
    return filtfilt(bf, af, x)


def chrom(r, g, b):
    r = (r - r.mean()) / (r.std() + 1e-8)
    g = (g - g.mean()) / (g.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)

    X = 3*r - 2*g
    Y = 1.5*r + g - 1.5*b

    Xf = bandpass(X)
    Yf = bandpass(Y)

    alpha = np.std(Xf)/(np.std(Yf)+1e-8)

    return Xf - alpha*Yf


def extract_rgb(frames, mask=None):
    if mask is None:
        r = frames[...,0].mean(axis=(1,2))
        g = frames[...,1].mean(axis=(1,2))
        b = frames[...,2].mean(axis=(1,2))
    else:
        denom = mask.sum() + 1e-8

        r = (frames[...,0] * mask).sum(axis=(1,2)) / denom
        g = (frames[...,1] * mask).sum(axis=(1,2)) / denom
        b = (frames[...,2] * mask).sum(axis=(1,2)) / denom

    return r,g,b


def make_static_mask(H,W,ratio=0.7):
    mask = np.ones((H,W), dtype=np.float32)
    mask[int(H*ratio):,:] = 0
    return mask


subject = "subject42"

subj_dir = os.path.join(DATA_ROOT, subject)

frames = np.load(
    os.path.join(subj_dir, "frames.npy"),
    mmap_mode="r"
)

gt = np.loadtxt(
    os.path.join(subj_dir, "ground_truth.txt")
)

T = min(len(frames), gt.shape[1], MAX_FRAMES)

frames = np.asarray(frames[:T], dtype=np.float32)

H,W = frames.shape[1:3]

mask = make_static_mask(H,W,0.7)

r0,g0,b0 = extract_rgb(frames, None)
rs,gs,bs = extract_rgb(frames, mask)

ppg_no = chrom(r0,g0,b0)
ppg_static = chrom(rs,gs,bs)

gt_bvp = bandpass(gt[0,:T])

# normalize for visualization
def norm(x):
    return (x - x.mean()) / (x.std() + 1e-8)

gt_bvp = norm(gt_bvp)
ppg_no = norm(ppg_no)
ppg_static = norm(ppg_static)

plt.figure(figsize=(14,5))

plt.plot(gt_bvp[:500], label='GT BVP')
plt.plot(ppg_no[:500], label='CHROM Baseline')
plt.plot(ppg_static[:500], label='CHROM + Static Mask')

plt.legend()
plt.title(subject)

plt.tight_layout()

plt.savefig(f"{subject}_waveform.png", dpi=300)

print("saved")
