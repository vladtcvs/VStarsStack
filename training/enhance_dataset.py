import common
import sys
import numpy as np
import os
import imageio

path = sys.argv[1]
files = common.listfiles(path, ".png")
for name, filename in files:
	image = imageio.imread(filename)
	image = np.swapaxes(image, 0, 1)
#	print(image.shape)
	imageio.imwrite(os.path.join(path, name + ".t.png"), image)

