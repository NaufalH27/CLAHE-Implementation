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

def gaussian_merge(patches, coords, img_shape, patch_size, overlap):
    h, w = img_shape
    result = np.zeros((h,w), np.float32)
    weight_sum = np.zeros((h,w), np.float32)

    sigma = patch_size / 2
    ax = np.arange(patch_size) - patch_size//2
    gauss = np.exp(-(ax**2)/(2*(sigma**2)))
    kernel = np.outer(gauss, gauss)
    kernel /= kernel.max()

    step = patch_size - overlap

    for (patch,(i,j)) in zip(patches,coords):
        ph,pw = patch.shape
        k = kernel[:ph,:pw]
        result[i:i+ph,j:j+pw] += patch.astype(np.float32) * k
        weight_sum[i:i+ph,j:j+pw] += k

    return np.clip(result/weight_sum, 0, 255).astype(np.uint8)


def gaussian_clahe(image, alpha, smax, L=256, patch_size=32, overlap=25):
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

    return gaussian_merge(patches, coords, image.shape, patch_size, overlap)


img_rgb = skimage.io.imread('dataset/642.png')
img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
Y = img_ycbcr[:, :, 0]

Y_clahe = gaussian_clahe(Y, alpha=60, smax=10, patch_size=64, overlap=60)
img_ycbcr[:, :, 0] = Y_clahe
img = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2RGB)

plt.imshow(img)
plt.axis('off')
plt.show()