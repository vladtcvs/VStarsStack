import rawpy
import numpy as np
import imageio
import sys

import common

image = np.load(sys.argv[1])

path = sys.argv[2]
files = common.listfiles(path, ".nef")
image0 = rawpy.imread(files[0][1])

shape = image.shape
newimage = np.zeros((shape[0]*2,shape[1]*2))
for y in range(shape[0]):
	for x in range(shape[1]):
		r = image[y][x][0]
		g = image[y][x][1]
		b = image[y][x][2]
		newimage[y*2+1][x*2] = r
		newimage[y*2][x*2+1] = b
		newimage[y*2][x*2] = int(g/2)
		newimage[y*2+1][x*2+1] = int(g/2)

image0.raw_image_visible[:] = newimage / len(files)
rgb = image0.postprocess()

imageio.imwrite(sys.argv[3], rgb)

