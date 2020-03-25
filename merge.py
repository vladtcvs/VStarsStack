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
	images[name] = np.load(filename)
	if shape is None:
		shape = images[name].shape

summary = np.zeros(shape)
for name in images:
	summary += images[name]


np.save(out, summary)

