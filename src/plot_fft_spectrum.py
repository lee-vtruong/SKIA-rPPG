import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


DATA_ROOT = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"
FS = 30
SUBJECT = "subject42"
MAX_FRAMES = 1500


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


def normalize_spectrum(x):
    x = x - x.mean()
    freqs = np.fft.rfftfreq(len(x), d=1/FS)
    mag = np.abs(np.fft.rfft(x))
    mag = mag / (mag.max() + 1e-8)
    return freqs, mag


subj_dir = os.path.join(DATA_ROOT, SUBJECT)

frames = np.load(os.path.join(subj_dir, "frames.npy"), mmap_mode="r")
gt = np.loadtxt(os.path.join(subj_dir, "ground_truth.txt"))

T = min(len(frames), gt.shape[1], MAX_FRAMES)
x = np.asarray(frames[:T], dtype=np.float32)

r0 = x[..., 0].mean(axis=(1, 2))
g0 = x[..., 1].mean(axis=(1, 2))
b0 = x[..., 2].mean(axis=(1, 2))

H, W = x.shape[1], x.shape[2]
mask = np.ones((H, W), dtype=np.float32)
mask[int(H * 0.7):, :] = 0.0
denom = mask.sum() + 1e-8

rs = (x[..., 0] * mask).sum(axis=(1, 2)) / denom
gs = (x[..., 1] * mask).sum(axis=(1, 2)) / denom
bs = (x[..., 2] * mask).sum(axis=(1, 2)) / denom

ppg_no = chrom_from_rgb(r0, g0, b0)
ppg_static = chrom_from_rgb(rs, gs, bs)
gt_bvp = bandpass(gt[0, :T])

f_gt, m_gt = normalize_spectrum(gt_bvp)
f_no, m_no = normalize_spectrum(ppg_no)
f_static, m_static = normalize_spectrum(ppg_static)

valid = (f_gt >= 0.7) & (f_gt <= 2.5)

plt.figure(figsize=(8, 5))

plt.plot(f_gt[valid] * 60, m_gt[valid], label="GT BVP", linewidth=2)
plt.plot(f_no[valid] * 60, m_no[valid], label="CHROM", linewidth=2)
plt.plot(f_static[valid] * 60, m_static[valid], label="CHROM + Static Mask", linewidth=2)

plt.xlabel("Heart Rate Frequency (bpm)")
plt.ylabel("Normalized Magnitude")
plt.title(f"Frequency-domain Comparison ({SUBJECT})")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("fft_spectrum_comparison.png", dpi=300)

print("saved fft_spectrum_comparison.png")
