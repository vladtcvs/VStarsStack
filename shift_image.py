import numpy as np
import math
import common

def shift_image(image, t, image_weight_layer=None, name=None, image_weight=1):
	shape = image.shape
	h = shape[0]
	w = shape[1]
	
	shifted = np.zeros(shape)
	shifted_weight_layer = np.zeros(shape)

	if image_weight_layer is None:
		image_weight_layer = np.ones(shape)*image_weight

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
			_, pixel        = common.getpixel(image, oy, ox, False)
			_, pixel_weight = common.getpixel(image_weight_layer, oy, ox, False)

			shifted[y][x] = pixel
			shifted_weight_layer[y][x] = pixel_weight

	return shifted, shifted_weight_layer
