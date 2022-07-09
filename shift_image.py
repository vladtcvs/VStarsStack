import numpy as np
import math
import common

def shift_image(image, t, inmask=None, name=None):
	shape = image.shape
	h = shape[0]
	w = shape[1]
	
	shifted = np.zeros(shape)
	if inmask is None:
		inmask = np.ones(shape)

	mask = np.zeros(shape)
	for y in range(h):
		if name and y % 200 == 0:
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
			res, pixel = common.getpixel(image, oy, ox, False)
			resmask, pixelmask = common.getpixel(image, oy, ox, False)
			
			shifted[y][x] = pixel
			if res and resmask and pixelmask > 1-1e-6:
				mask[y][x] = 1

	return shifted, mask
