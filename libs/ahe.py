# gambar 3

import numpy as np
from .common import extract_patches, he

def ahe_bilinear(image, L=256, patch_size=64):
    patches, pos = extract_patches(image, patch_size)
    luts = [he(p, L) for p in patches]

    nH = len(set([p[0] for p in pos]))
    nW = len(set([p[1] for p in pos]))

    out = np.zeros_like(image, dtype=np.uint8)

    for idx,(i1,j1,i2,j2) in enumerate(pos):
        tI = i1//patch_size
        tJ = j1//patch_size

        A=idx
        B=idx+1 if (tJ+1)<nW else idx
        C=idx+nW if (tI+1)<nH else idx
        D=C+1 if (tI+1)<nH and (tJ+1)<nW else C

        gA,gB,gC,gD = [luts[x].astype(np.float32) for x in [A,B,C,D]]

        H,W = i2-i1, j2-j1

        fy = np.linspace(0,1,H, dtype=np.float32)[:,None]
        fx = np.linspace(0,1,W, dtype=np.float32)[None,:]

        wA = (1-fy)*(1-fx)
        wB = (1-fy)*(fx)
        wC = (fy)*(1-fx)
        wD = (fy)*(fx)

        block = image[i1:i2, j1:j2]
        out[i1:i2, j1:j2] = (
              wA * gA[block]
            + wB * gB[block]
            + wC * gC[block]
            + wD * gD[block]
        ).astype(np.uint8)

    return out

def ahe_naive(image, L=256, patch_size=64):
    patches, positions = extract_patches(image, patch_size)
    he_patches = [he(p, L)[p] for p in patches]
    merged = np.zeros_like(image)

    idx = 0
    for (i, j, i_end, j_end) in positions:
        merged[i:i_end, j:j_end] = he_patches[idx]
        idx += 1

    return merged
