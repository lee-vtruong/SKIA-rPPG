import os
import numpy as np
import matplotlib.pyplot as plt


DATA_ROOT = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"

subject = "subject1"

frame_idx = 300

subj_dir = os.path.join(DATA_ROOT, subject)

frames = np.load(
    os.path.join(subj_dir, "frames.npy"),
    mmap_mode="r"
)

frame = np.asarray(frames[frame_idx], dtype=np.float32)

H, W = frame.shape[:2]

mask = np.ones((H, W), dtype=np.float32)

# static ratio = 0.7
mask[int(H * 0.7):, :] = 0.0

masked_frame = frame * mask[..., None]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(frame)
ax[0].set_title("Original Frame")
ax[0].axis("off")

ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Static Mask")
ax[1].axis("off")

ax[2].imshow(masked_frame)
ax[2].set_title("Masked Frame")
ax[2].axis("off")

plt.tight_layout()

plt.savefig("mask_visualization.png", dpi=300)

print("saved")
