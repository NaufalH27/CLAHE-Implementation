# download LOLDataset first, read README.md for instruction

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity as SSIM
from libs import clahe_bilinear, ahe_bilinear


def apply_ahe(img):
    ycb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y = ycb[:, :, 0]
    ycb2 = ycb.copy()
    ycb2[:, :, 0] = ahe_bilinear(Y)
    return cv2.cvtColor(ycb2, cv2.COLOR_YCrCb2RGB)


def apply_clahe(img, alpha=200, smax=3):
    ycb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y = ycb[:, :, 0]
    ycb2 = ycb.copy()
    ycb2[:, :, 0] = clahe_bilinear(Y, alpha=alpha, smax=smax)
    return cv2.cvtColor(ycb2, cv2.COLOR_YCrCb2RGB)


def apply_hist_eq(img):
    ycb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycb[:, :, 0] = cv2.equalizeHist(ycb[:, :, 0])
    return cv2.cvtColor(ycb, cv2.COLOR_YCrCb2RGB)


low_dir = "LOLdataset/low"
high_dir = "LOLdataset/high"
files = sorted(os.listdir(low_dir))

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

csv_path = os.path.join(save_dir, "results_ssim.csv")
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(["Image", "Method", "SSIM"])

results = { "AHE": [], "CLAHE": [], "HE": [] }


for f in files:
    low_path = os.path.join(low_dir, f)
    high_path = os.path.join(high_dir, f)

    if not os.path.exists(high_path):
        print(f"[SKIPPED] No GT for {f}")
        continue

    low  = io.imread(low_path)
    high = io.imread(high_path)

    outputs = {
        "AHE":   apply_ahe(low),
        "CLAHE": apply_clahe(low),
        "HE":    apply_hist_eq(low),
    }

    print(f"{f}:")
    for method, out in outputs.items():
        ssim = SSIM(high, out, channel_axis=-1, data_range=255)
        results[method].append(ssim)

        print(f"  {method:<8} SSIM={ssim:.3f}")
        cv2.imwrite(os.path.join(save_dir, f"{method}_{f}"),
                    cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

        with open(csv_path, "a", newline="") as file:
            csv.writer(file).writerow([f, method, ssim])


print("\nAVERAGE SSIM RESULTS")
for method, arr in results.items():
    print(f"{method:<8} Avg SSIM = {np.mean(arr):.4f}")


DIST = 5
N = 5

scores = [(files[i], results["CLAHE"][i]) for i in range(len(results["CLAHE"]))]
scores = sorted(scores, key=lambda x: x[1], reverse=True)

selected = []
used = []

for fname,score in scores:
    idx = int(fname.split(".")[0])
    if any(abs(idx - k) <= DIST for k in used):
        continue
    selected.append((fname, score))
    used.append(idx)
    if len(selected) >= N:
        break

print("\nFINAL SELECTED IMAGES:")
for name,score in selected:
    print(f"{name}  SSIM={score:.4f}")


rows = 5
cols = len(selected)
plt.figure(figsize=(4 * cols, 15))

labels = ["Low", "High GT", "HE", "AHE", "CLAHE"]

for c,(fname,_) in enumerate(selected):
    low  = io.imread(os.path.join(low_dir, fname))
    high = io.imread(os.path.join(high_dir, fname))

    H = apply_hist_eq(low)
    A = apply_ahe(low)
    C = apply_clahe(low, alpha=100)

    imgs = [low, high, H, A, C]

    for r in range(rows):
        plt.subplot(rows, cols, r*cols + c + 1)
        plt.imshow(imgs[r])
        plt.axis("off")
        if c == 0: plt.ylabel(labels[r], fontsize=14, weight="bold")

plt.suptitle("TOP-5 CLAHE Images (Filtered)", fontsize=20, weight="bold")
plt.tight_layout()
plt.show()
