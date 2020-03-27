import numpy as np
import sys

import common
import matplotlib.pyplot as plt

path_images=sys.argv[1]
out=sys.argv[2]

imgs = common.listfiles(path_images, ".npy")

images = {}
shape = None

for name, filename in imgs:
	print(name)
	image = np.load(filename)
#	print(image.shape)
	if shape is None:
		shape = image.shape
	images[name] = image

summary = np.zeros(shape)
for name in images:
	summary = summary + images[name]

for y in range(shape[0]):
	for x in range(shape[1]):
		n = summary[y][x][3]
		if n > 0:
			summary[y][x] /= n

summary = summary[:,:,0:3]
np.save(out, summary)

