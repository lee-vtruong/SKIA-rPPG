import os
import numpy as np
from scipy.signal import butter, filtfilt


DATA_ROOT = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"

FS = 30
MAX_FRAMES = 1500
STATIC_RATIO = 0.7


def bandpass(x, fs=30):
    bf, af = butter(4, [0.7/(fs/2), 2.5/(fs/2)], btype='band')
    return filtfilt(bf, af, x)


def chrom(r,g,b):
    r = (r-r.mean())/(r.std()+1e-8)
    g = (g-g.mean())/(g.std()+1e-8)
    b = (b-b.mean())/(b.std()+1e-8)

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

        r = (frames[...,0]*mask).sum(axis=(1,2))/denom
        g = (frames[...,1]*mask).sum(axis=(1,2))/denom
        b = (frames[...,2]*mask).sum(axis=(1,2))/denom

    return r,g,b


def make_static_mask(H,W,ratio=0.7):
    mask = np.ones((H,W), dtype=np.float32)
    mask[int(H*ratio):,:] = 0
    return mask


def estimate_bpm(signal, fs=30):

    x = signal - signal.mean()

    freqs = np.fft.rfftfreq(len(x), d=1/fs)
    fft_mag = np.abs(np.fft.rfft(x))

    valid = (freqs >= 0.7) & (freqs <= 2.5)

    peak_freq = freqs[valid][np.argmax(fft_mag[valid])]

    bpm = peak_freq * 60.0

    return bpm


subjects = sorted(os.listdir(DATA_ROOT))

mae_no = []
mae_static = []

for subj in subjects:

    subj_dir = os.path.join(DATA_ROOT, subj)

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

    mask = make_static_mask(H,W,STATIC_RATIO)

    r0,g0,b0 = extract_rgb(frames, None)
    rs,gs,bs = extract_rgb(frames, mask)

    ppg_no = chrom(r0,g0,b0)
    ppg_static = chrom(rs,gs,bs)

    gt_bvp = bandpass(gt[0,:T])

    bpm_gt = estimate_bpm(gt_bvp)

    bpm_no = estimate_bpm(ppg_no)
    bpm_static = estimate_bpm(ppg_static)

    err_no = abs(bpm_no - bpm_gt)
    err_static = abs(bpm_static - bpm_gt)

    mae_no.append(err_no)
    mae_static.append(err_static)

    print(
        f"{subj}: "
        f"GT={bpm_gt:.1f}, "
        f"baseline={bpm_no:.1f}, "
        f"static={bpm_static:.1f}"
    )

print("\n===== HR SUMMARY =====")

print(f"Baseline MAE = {np.mean(mae_no):.3f} bpm")
print(f"Static MAE   = {np.mean(mae_static):.3f} bpm")
print(f"Delta        = {np.mean(mae_no)-np.mean(mae_static):+.3f} bpm")
