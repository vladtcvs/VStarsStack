import os
import rawpy
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import common

def readnef(filename):
	image = rawpy.imread(fname).raw_image_visible

	shape = image.shape
	cshape = (int(shape[0]/2), int(shape[1]/2), 3)

	post = np.zeros(cshape)

	for y in range(cshape[0]):
		for x in range(cshape[1]):
			post[y][x][0] = image[y*2 + 1][x*2]
			post[y][x][1] = image[y*2][x*2+1]
			post[y][x][2] = image[y*2+1][x*2+1] + image[y*2][x*2]
	return post


if __name__ == "__main__":

	path=sys.argv[1]

	if len(sys.argv) > 2:
		npypath = sys.argv[2]
	else:
		npypath = path

	files = common.listfiles(path, ".nef")

	for name, fname in files:
		print(name)
		post = readnef(fname)
		np.save(os.path.join(npypath, name + ".npy"), post)

