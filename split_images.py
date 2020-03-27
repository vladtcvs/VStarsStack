import sys
import os
import common
import numpy as np
import imageio

w = 28
h = 28

images = common.listfiles(sys.argv[1], ".npy")
for name, filename in images:
	print(name)
	image = np.load(filename)
	ny = int(image.shape[0] / h)
	nx = int(image.shape[1] / w)
	for y in range(ny):
		for x in range(nx):
			print(y,x)
			sub = image[y*h:y*h+h, x*w:x*w+w]
			sub = (sub * 255 / np.amax(sub)).astype('uint8')
			path = os.path.join(sys.argv[2], "%s.%03i.%03i.png" % (name, y, x))
			imageio.imwrite(path, sub)


