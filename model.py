import cv2
import matplotlib.pyplot as plt

img_rgb = cv2.imread('225_low.png')
print(img_rgb)

img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
Y = img_ycbcr[:, :, 0]
plt.imshow(Y, cmap='gray')
plt.show()