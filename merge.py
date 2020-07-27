import numpy as np
import sys

import common
import matplotlib.pyplot as plt

def run(argv):
	path_images = argv[0]
	out = argv[1]

	imgs = common.listfiles(path_images, ".npy")

	images = {}
	shape = None

	for name, filename in imgs:
		image = np.load(filename)
		if shape is None:
			shape = image.shape
			break

	summary = np.zeros(shape)
	for name, filename in imgs:
		print(name)
		image = np.load(filename)
		summary = summary + image

	for y in range(shape[0]):
		for x in range(shape[1]):
			if shape[2] == 4:
				n = summary[y][x][3]
			else:
				n = 1
			if n > 0:
				summary[y][x] /= n

	summary = summary[:,:,0:3]
	np.save(out, summary)

if __name__ == "__main__":
	run(sys.argv[1:])

