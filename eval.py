# download LOLDataset first, read README.md for instruction

import csv
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM
from skimage import io
from libs import clahe_bilinear, ahe_bilinear

def hist_equalization(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def gamma_correction(img, gamma=1.8):
    img_float = img.astype(np.float32) / 255.0
    corrected = np.power(img_float, 1/gamma)
    return (corrected*255).astype(np.uint8)

low_dir = "LOLdataset/low"
high_dir = "LOLdataset/high"

files = sorted(os.listdir(low_dir))
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)
csv_path = "results_psnr.csv"
with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Method", "SSIM"]) 


results = {} 

for f in files:
    low_path  = os.path.join(low_dir, f)
    high_path = os.path.join(high_dir, f)

    if not os.path.exists(high_path):
        print(f"[SKIPPED] No matching GT for {f}")
        continue

    low  = io.imread(low_path)
    high = io.imread(high_path)

    ycb = cv2.cvtColor(low, cv2.COLOR_RGB2YCrCb)
    ycb0 = ycb[:,:,0].copy()

    out_ahe_bilinear   = ycb.copy(); out_ahe_bilinear[:,:,0]   = ahe_bilinear(ycb0)
    out_clahe_bilinear = ycb.copy(); out_clahe_bilinear[:,:,0] = clahe_bilinear(ycb0, alpha=200, smax=3)
    out_ahe_bilinear   = cv2.cvtColor(out_ahe_bilinear,   cv2.COLOR_YCrCb2RGB)
    out_clahe_bilinear = cv2.cvtColor(out_clahe_bilinear, cv2.COLOR_YCrCb2RGB)

    out_histeq   = hist_equalization(low)

    methods = {
        "AHE_Bilinear"  : out_ahe_bilinear,
        "CLAHE_Bilinear": out_clahe_bilinear,
        "HE"        : out_histeq,
    }

    print(f"{f}:")
    for name,out in methods.items():
        ssim = SSIM(high, out, channel_axis=-1, data_range=255)
        print(f"  {name:<15}  SSIM={ssim:.3f}")
        cv2.imwrite(f"{save_dir}/{name}_{f}", cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
        results.setdefault(name, []).append(ssim)

        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([f, name, ssim])


print("\nAVERAGE SSIM RESULTS")
for name, scores in results.items():
    print(f"{name:<15}  Avg SSIM = {np.mean(scores):.4f}")

clahe_scores = [(files[i], results["CLAHE_Bilinear"][i]) for i in range(len(results["CLAHE_Bilinear"]))]
top5 = sorted(clahe_scores, key=lambda x: x[1], reverse=True)[:5]

import matplotlib.pyplot as plt

selected = []
last_index = -999

for img,score in top5:
    idx = int(img.split(".")[0])  
    
    if abs(idx - last_index) <= 5:
        print(f"Removed {img} (too close to previous {last_index})")
        continue
    
    selected.append((img, score))
    last_index = idx

selected = selected[:5]  
print("\nFINAL SELECTED TOP-5")
for img,s in selected:
    print(f" {img} → SSIM={s:.4f}")

N = 5   
DIST = 5  

clahe_scores = [(files[i], results["CLAHE_Bilinear"][i]) for i in range(len(files))]
clahe_scores = sorted(clahe_scores, key=lambda x:x[1], reverse=True)

selected = []
used = []

for fname,score in clahe_scores:
    idx = int(fname.split(".")[0])
    if any(abs(idx - u) <= DIST for u in used):  
        continue  
    selected.append((fname,score))
    used.append(idx)
    if len(selected) >= N:
        break

print("\nFINAL SELECTED IMAGES:")
for name,score in selected: print(f"{name}  SSIM={score:.4f}")


rows = 5
cols = len(selected)
plt.figure(figsize=(4*cols,15))

for c,(fname,_) in enumerate(selected):
    print(f"{fname}")

    low  = io.imread(os.path.join(low_dir,fname))
    high = io.imread(os.path.join(high_dir,fname))

    # process
    ycb = cv2.cvtColor(low, cv2.COLOR_RGB2YCrCb)
    Y = ycb[:,:,0]

    A = ycb.copy(); A[:,:,0] = ahe_bilinear(Y)
    C = ycb.copy(); C[:,:,0] = clahe_bilinear(Y, alpha=100, smax=3)
    A = cv2.cvtColor(A, cv2.COLOR_YCrCb2RGB)
    C = cv2.cvtColor(C, cv2.COLOR_YCrCb2RGB)
    H = hist_equalization(low)

    images = [low, high, H, A, C]
    labels = ["Low", "High GT", "HE", "AHE", "CLAHE"]

    for name,img in zip(labels[2:], [H,A,C]):
        ssim = SSIM(high, img, channel_axis=-1, data_range=255)
        print(f"  SSIM {name:<8} = {ssim:.4f}")

    for r in range(rows):
        i = r*cols + c + 1
        plt.subplot(rows, cols, i)
        plt.imshow(images[r])
        plt.axis("off")
        if c == 0: plt.ylabel(labels[r], fontsize=14, weight='bold')

plt.suptitle("TOP-5 CLAHE Images (Auto-Filtered, No Nearest Index)", fontsize=20, weight='bold')
plt.tight_layout()
plt.show()