# CLAHE menggunakan gaussian, gambar 3

import cv2
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

def extract_patches(img, patch_size, overlap):
    h, w = img.shape
    patches = []
    coords = []

    step = patch_size - overlap
    
    for i in range(0, h, step):
        for j in range(0, w, step):
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)
            patch = img[i:i_end, j:j_end]
            patches.append(patch)
            coords.append((i, j))
    return patches, coords

def bilinear_merge(patches, coords, img_shape, patch_size, overlap):
    h, w = img_shape
    result = np.zeros((h, w), np.float32)
    weight_sum = np.zeros((h, w), np.float32)

    # Create bilinear kernel
    ax = np.linspace(0, 1, patch_size)
    bx = 1 - np.abs(ax - 0.5) * 2
    kernel = np.outer(bx, bx) 
    kernel /= kernel.max()

    step = patch_size - overlap

    for patch, (i, j) in zip(patches, coords):
        ph, pw = patch.shape
        k = kernel[:ph, :pw]
        result[i:i+ph, j:j+pw] += patch.astype(np.float32) * k
        weight_sum[i:i+ph, j:j+pw] += k

    return np.clip(result / weight_sum, 0, 255).astype(np.uint8)


def clahe(image, alpha, smax, L=256, patch_size=32, overlap=25):
    patches, coords = extract_patches(image, patch_size, overlap)
    for i, p in enumerate(patches):
        h,w = p.shape
        total_pixels = h * w 
        cl = total_pixels/L * (1 + (alpha/100)*(smax-1))
        hist, _ = np.histogram(p, bins=L, range=(0,L))
        excess = np.sum(np.maximum(hist - cl, 0))
        hist_clipped = np.minimum(hist, cl)
        hist_clipped += excess/L
        cdf = np.cumsum(hist_clipped) / p.size
        lut = np.floor((L-1) * cdf).astype(np.uint8)
        patches[i] = lut[p]

    return bilinear_merge(patches, coords, image.shape, patch_size, overlap)


img_rgb = skimage.io.imread('dataset/225_low.png')
img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
Y = img_ycbcr[:, :, 0]

Y_clahe = clahe(Y, alpha=200, smax=3, patch_size=64, overlap=32)
img_ycbcr[:, :, 0] = Y_clahe
img = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2RGB)

fig, ax = plt.subplots(1, 3, figsize=(10, 5), gridspec_kw={'wspace': 0.02})

ax[0].imshow(img)
ax[0].axis("off")
ax[0].text(0.5, -0.05, "(a)", ha='center', va='top', transform=ax[0].transAxes, fontsize=14)

ax[1].imshow(Y_clahe, cmap='gray')
ax[1].axis("off")
ax[1].text(0.5, -0.05, "(b)", ha='center', va='top', transform=ax[1].transAxes, fontsize=14)

ax[2].imshow(img_rgb)
ax[2].axis("off")
ax[2].text(0.5, -0.05, "(c)", ha='center', va='top', transform=ax[2].transAxes, fontsize=14)

plt.show()