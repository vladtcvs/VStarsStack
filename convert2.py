import numpy as np
import imageio
import sys

import common

mul = 2

img = np.load(sys.argv[1])

img = np.clip(img, 0, 1e6)
img = np.power(img, 0.5)

amax = np.amax(img)
img = (img/amax - 0.02)*65536*mul
img = np.clip(img, 0, 65535).astype('uint16')

imageio.imwrite(sys.argv[2], img)

