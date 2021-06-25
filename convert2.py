import numpy as np
import imageio
import sys

import normalize
import common

mul = 1
power = 1

img = np.load(sys.argv[1])
try:
	img = img["arr_0"]
except:
	pass

if len(img.shape) == 3:
	img = normalize.normalize(img)
	img = img[:,:,0:3]

img = np.clip(img, 0, 1e12)

amax = np.amax(img)
img = img/amax*65535
img = np.clip(img, 0, 65535).astype('uint16')

imageio.imwrite(sys.argv[2], img)

