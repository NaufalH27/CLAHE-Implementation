import cv2
import matplotlib.pyplot as plt
import skimage.io

img_gbr = skimage.io.imread('dataset/642.png')

img_ycbcr = cv2.cvtColor(img_gbr, cv2.COLOR_RGB2YCrCb)
Y = img_ycbcr[:, :, 0]
clahe = cv2.createCLAHE(clipLimit=4)
clahe_img = clahe.apply(Y)
img_ycbcr[:, :, 0] = clahe_img
img_rgb_processed = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2RGB)
plt.imshow(img_rgb_processed)
plt.show()