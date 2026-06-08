import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import butter, filtfilt

from skia_dataset import SKIADataset
from skia_model import SKIANetLite


FS = 30


def bandpass(x):
    b, a = butter(4, [0.7/(FS/2), 2.5/(FS/2)], btype='band')
    return filtfilt(b, a, x)


# =========================
# LOAD DATASET
# =========================

data_root = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"

subjects = sorted(os.listdir(data_root))
test_subjects = subjects[32:]

dataset = SKIADataset(
    data_root,
    test_subjects,
    clip_len=128,
    stride=128,
    size=128,
    mode="raw_masked",
)

# =========================
# LOAD MODEL
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SKIANetLite(in_channels=6).to(device)

model.load_state_dict(
    torch.load("../checkpoints/skia_raw_masked_best.pt")
)

model.eval()

# =========================
# SELECT SAMPLE
# =========================

best_corr = -999

best_x = None
best_gt = None
best_pred = None

for idx in range(min(len(dataset), 20)):

    x, gt = dataset[idx]

    with torch.no_grad():

        pred = model(
            x.unsqueeze(0).to(device)
        ).squeeze(0).cpu().numpy()

    gt_np = gt.numpy()

    pred_bp = bandpass(pred)
    gt_bp = bandpass(gt_np)

    corr = np.corrcoef(pred_bp, gt_bp)[0,1]

    if not np.isnan(corr) and corr > best_corr:

        best_corr = corr

        best_x = x
        best_gt = gt_bp
        best_pred = pred_bp

print(f"Best correlation found: {best_corr:.4f}")

x = best_x
gt = best_gt
pred = best_pred

# =========================
# VISUALIZATION
# =========================

# x shape = (6,T,H,W)

raw = x[:3].permute(1,2,3,0).numpy()
masked = x[3:].permute(1,2,3,0).numpy()

frame_idx = 30

raw_frame = raw[frame_idx]
masked_frame = masked[frame_idx]

# normalize for display
raw_frame = (raw_frame - raw_frame.min()) / (raw_frame.max() - raw_frame.min())
masked_frame = (masked_frame - masked_frame.min()) / (masked_frame.max() - masked_frame.min())

# create mask visualization
mask_vis = np.mean(masked_frame, axis=2)

fig = plt.figure(figsize=(14,7))

# =========================
# IMAGE ROW
# =========================

ax1 = plt.subplot(2,3,1)
ax1.imshow(raw_frame)
ax1.set_title("Original Frame")
ax1.axis("off")

ax2 = plt.subplot(2,3,2)
ax2.imshow(mask_vis, cmap="gray")
ax2.set_title("Static Mask")
ax2.axis("off")

ax3 = plt.subplot(2,3,3)
ax3.imshow(masked_frame)
ax3.set_title("Masked Frame")
ax3.axis("off")

# =========================
# WAVEFORM
# =========================

ax4 = plt.subplot(2,1,2)

gt = (gt - gt.mean()) / (gt.std() + 1e-6)
pred = (pred - pred.mean()) / (pred.std() + 1e-6)

ax4.plot(gt, label="GT BVP", linewidth=2)
ax4.plot(pred, label="Predicted BVP", linewidth=2)

ax4.set_title("Waveform Reconstruction")

ax4.legend()

plt.tight_layout()

plt.savefig("qualitative_figure.png", dpi=300)

print("saved qualitative_figure.png")
