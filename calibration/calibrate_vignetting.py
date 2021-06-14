import numpy as np
import sys
import cv2

blur = 7

source = sys.argv[1]
result = sys.argv[2]

img = np.load(source)["arr_0"]

img = np.sum(img, axis=2).astype(np.float32)
img = cv2.GaussianBlur(img, (blur, blur), 0)

img = img / np.amax(img)
img = 1/img

np.savez_compressed(result, img)

