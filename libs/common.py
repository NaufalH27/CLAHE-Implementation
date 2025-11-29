import numpy as np

def he(image, L):
    hist, _ = np.histogram(image, bins=L, range=(0, L))
    p = hist / image.size
    cdf = p.cumsum()
    return np.floor((L - 1) * cdf).astype(np.uint8)

def clipped_he(image, alpha, smax, L=256):
    h, w = image.shape
    total_pixels = h * w
    clip_limit = total_pixels / L * (1 + (alpha/100)*(smax-1))
    hist, _ = np.histogram(image, bins=L, range=(0, L))
    excess = np.sum(np.maximum(hist - clip_limit, 0))
    hist_clipped = np.minimum(hist, clip_limit)
    hist_clipped += excess / L
    cdf = np.cumsum(hist_clipped) / total_pixels
    return np.floor((L-1) * cdf).astype(np.uint8)

def extract_patches(image, patch_size):
    h, w = image.shape
    patches = []
    positions = []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)
            patch = image[i:i_end, j:j_end]
            patches.append(patch)
            positions.append((i, j, i_end, j_end)) 

    return patches, positions