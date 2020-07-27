import numpy as np
import imageio
import sys

import common

mul = 1
power = 1

img = np.load(sys.argv[1])

img = img[:,:,0:3]
#img = img[:,:,::-1]

img = np.clip(img, 0, 1e6)
img = np.power(img, power)

amax = np.amax(img)
img = img/amax*65536*mul
img = np.clip(img, 0, 65535).astype('uint16')

imageio.imwrite(sys.argv[2], img)

