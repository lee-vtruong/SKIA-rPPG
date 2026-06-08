import os
import numpy as np

data_root = "/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1"

subjects = sorted(os.listdir(data_root))

for subj in subjects:
    mask_path = os.path.join(data_root, subj, "mask.npy")
    if not os.path.exists(mask_path):
        continue

    m = np.load(mask_path, mmap_mode="r")
    m_small = np.asarray(m[:min(len(m), 1500)])

    print(
        f"{subj}: "
        f"mean={m_small.mean():.4f}, "
        f"min={m_small.min():.4f}, "
        f"max={m_small.max():.4f}, "
        f"p1={np.percentile(m_small, 1):.4f}, "
        f"p5={np.percentile(m_small, 5):.4f}, "
        f"p50={np.percentile(m_small, 50):.4f}"
    )
