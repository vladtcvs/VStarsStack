import numpy as np
import math
import common

def shift_image(image, t, name):
	shape = image.shape
	h = shape[0]
	w = shape[1]
	n = shape[2]
	shifted = np.zeros((h, w, 4))
	for y in range(h):
		if y % 200 == 0:
			print(name, ":", y, "/", h)
		xs = range(w)
		ys = []
		for i in range(w):
			ys.append(y)
		positions = list(zip(ys, xs))
		npositions = t.reverse(positions)
		for x in xs:
			oy = npositions[x][0]
			ox = npositions[x][1]
			_, pixel = common.getpixel(image, oy, ox)
			shifted[y][x] = pixel

	return shifted

