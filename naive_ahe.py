# kode menghasilkan gambar 1 a dan b

import cv2
import matplotlib.pyplot as plt
import skimage.io
import numpy as np

img_rgb = skimage.io.imread('dataset/225_low.png')

img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
Y = img_ycbcr[:, :, 0]

def he(image, L):
    hist, bins = np.histogram(image, bins=L, range=(0, L))
    p = hist / image.size
    cdf = p.cumsum()
    lut = np.floor((L - 1) * cdf).astype(np.uint8)
    image_eq = lut[image]
    return image_eq

def grayscale_patches(image, patch_size):
    h, w = image.shape
    patches = []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)

            patch = image[i:i_end, j:j_end]
            patches.append(patch)

    return patches

def merge_patches(patches, image_shape, patch_size):
    h, w = image_shape
    merged = np.zeros(image_shape, dtype=patches[0].dtype)

    patch_idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)

            patch = patches[patch_idx]
            merged[i:i_end, j:j_end] = patch
            patch_idx += 1

    return merged

patch_size = 32

patches = grayscale_patches(Y, patch_size)
he_patches = [he(p,256) for p in patches]

ahe_image = merge_patches(he_patches, Y.shape, patch_size)

plt.figure(figsize=(10, 5))

fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace': 0.02})

ax[0].imshow(Y, cmap='gray')
ax[0].axis("off")
ax[0].text(0.5, -0.05, "(a)", ha='center', va='top', transform=ax[0].transAxes, fontsize=14)

ax[1].imshow(ahe_image, cmap='gray')
ax[1].axis("off")
ax[1].text(0.5, -0.05, "(b)", ha='center', va='top', transform=ax[1].transAxes, fontsize=14)

plt.show()