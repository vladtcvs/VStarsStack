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
	for x in range(nx):
		for y in range(ny):
			print(y,x)
			sub = image[y*h:y*h+h, x*w:x*w+w, 0:3]
			sub = (sub * 255 / np.amax(sub)).astype('uint8')
			path = os.path.join(sys.argv[2], "%03i.%03i.%s.png" % (x, y, name))
			imageio.imwrite(path, sub)


