import matplotlib.pyplot as plt
import numpy as np

methods = [
    "CHROM",
    "CHROM\n+Mask",
    "SKIA Raw",
    "SKIA\nRaw+Mask",
]

pearson = [0.3507, 0.4166, 0.6320, 0.6844]
hr_mae = [33.50, 23.44, 8.99, 4.97]

x = np.arange(len(methods))

plt.figure(figsize=(8,5))
plt.bar(x, pearson)
plt.xticks(x, methods)
plt.ylabel("Pearson Correlation ↑")
plt.title("Waveform Reconstruction Performance")

for i, v in enumerate(pearson):
    plt.text(i, v + 0.015, f"{v:.3f}", ha="center")

plt.ylim(0, 0.8)
plt.tight_layout()
plt.savefig("final_pearson_results.png", dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.bar(x, hr_mae)
plt.xticks(x, methods)
plt.ylabel("HR MAE ↓ (bpm)")
plt.title("Heart Rate Estimation Error")

for i, v in enumerate(hr_mae):
    plt.text(i, v + 0.8, f"{v:.2f}", ha="center")

plt.ylim(0, 38)
plt.tight_layout()
plt.savefig("final_hr_mae_results.png", dpi=300)
plt.close()

print("saved final_pearson_results.png")
print("saved final_hr_mae_results.png")
