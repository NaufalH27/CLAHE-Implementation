import cv2
import matplotlib.pyplot as plt
import numpy as np

img_rgb = cv2.imread('dataset/42_low.png')
print(img_rgb)

img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
Y = img_ycbcr[:, :, 0]
clahe = cv2.createCLAHE(clipLimit=5)
clahe_img = np.clip(clahe.apply(Y) + 30, 0, 255).astype(np.uint8)
img_ycbcr[:, :, 0] = clahe_img
img_rgb_processed = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2RGB)
cv2.imshow("test", img_rgb_processed)
cv2.waitKey(0)