import numpy as np
import matplotlib.pyplot as plt

baseline = np.array([
0.717,0.333,0.323,0.025,0.171,0.140,0.183,0.178,0.107,0.704,
0.285,0.173,0.552,0.405,0.106,0.067,0.057,0.729,0.037,0.363,
0.139,0.625,0.544,0.829,0.585,0.486,0.451,0.266,0.744,0.507,
0.168,0.539,0.240,0.213,0.159,0.223,0.384,0.358,0.394,0.388,
0.441,0.389
])

masked = np.array([
0.752,0.439,0.304,0.045,0.200,0.181,0.239,0.236,0.161,0.766,
0.338,0.259,0.575,0.511,0.136,0.071,0.040,0.693,0.106,0.430,
0.136,0.773,0.526,0.863,0.722,0.550,0.523,0.370,0.806,0.547,
0.181,0.750,0.295,0.231,0.171,0.265,0.504,0.555,0.466,0.735,
0.513,0.533
])

plt.figure(figsize=(6,6))

plt.scatter(baseline, masked)

mx = max(baseline.max(), masked.max())

plt.plot([0,mx],[0,mx],'r--')

plt.xlabel("Baseline Pearson")
plt.ylabel("Static Mask Pearson")

plt.title("Subject-level Improvement")

plt.grid(True)

plt.tight_layout()

plt.savefig("scatter_subjects.png", dpi=300)

print("saved scatter_subjects.png")
