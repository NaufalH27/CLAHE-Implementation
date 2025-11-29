# ahe menggunakan gaussian, gambar 2

import cv2
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

def he(image, L=256):
    hist, _ = np.histogram(image, bins=L, range=(0,L))
    cdf = np.cumsum(hist) / image.size
    lut = np.floor((L-1) * cdf).astype(np.uint8)
    return lut[image]

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


def ahe(image, L=256, patch_size=64, overlap=32):
    patches, coords = extract_patches(image, patch_size, overlap)
    he_patches = [he(p,L) for p in patches]
    return bilinear_merge(he_patches, coords, image.shape, patch_size, overlap)

img_rgb = skimage.io.imread('dataset/225_low.png')
img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
Y = img_ycbcr[:, :, 0]

plt.imshow(ahe(Y), cmap="gray")
plt.axis('off')
plt.show()