import os
import argparse
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.signal import butter, filtfilt

from skia_dataset import SKIADataset
from skia_model import SKIANetLite


FS = 30


def bandpass(x, fs=30):
    bf, af = butter(4, [0.7/(fs/2), 2.5/(fs/2)], btype='band')
    return filtfilt(bf, af, x)


def estimate_bpm(signal, fs=30):

    signal = signal - signal.mean()

    freqs = np.fft.rfftfreq(len(signal), d=1/fs)
    mag = np.abs(np.fft.rfft(signal))

    valid = (freqs >= 0.7) & (freqs <= 2.5)

    peak_freq = freqs[valid][np.argmax(mag[valid])]

    return peak_freq * 60.0


def get_in_channels(mode):
    if mode in ["raw", "masked"]:
        return 3
    if mode == "raw_masked":
        return 6
    if mode == "full":
        return 9
    raise ValueError(mode)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)

args = parser.parse_args()

data_root = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"

subjects = sorted(os.listdir(data_root))
test_subjects = subjects[32:]

dataset = SKIADataset(
    data_root,
    test_subjects,
    clip_len=128,
    stride=128,
    size=64,          # NHẸ HƠN
    mode=args.mode,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SKIANetLite(
    in_channels=get_in_channels(args.mode)
).to(device)

model.load_state_dict(torch.load(args.ckpt))
model.eval()

corrs = []
maes = []

with torch.no_grad():

    for x, gt in tqdm(loader):

        x = x.to(device)

        pred = model(x).cpu().numpy()
        gt = gt.numpy()

        for p, t in zip(pred, gt):

            p = bandpass(p)
            t = bandpass(t)

            corr = np.corrcoef(p, t)[0,1]

            bpm_p = estimate_bpm(p)
            bpm_t = estimate_bpm(t)

            mae = abs(bpm_p - bpm_t)

            if not np.isnan(corr):
                corrs.append(corr)

            maes.append(mae)

print("\n===== RESULTS =====")

print(f"Mode: {args.mode}")
print(f"Mean Pearson = {np.mean(corrs):.4f}")
print(f"HR MAE       = {np.mean(maes):.4f} bpm")