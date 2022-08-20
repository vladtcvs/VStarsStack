import numpy as np
import math
import common

def shift_image(image, t, proj, image_weight_layer=None, image_weight=1):
	shape = image.shape
	h = shape[0]
	w = shape[1]
	
	shifted = np.zeros(shape)
	shifted_weight_layer = np.zeros(shape)

	if image_weight_layer is None:
		image_weight_layer = np.ones(shape)*image_weight

	for y in range(h):
		positions = []
		for x in range(w):
			positions.append((y,x))
		original_positions = t.reverse(positions, proj)
		for i in range(len(positions)):
			y,x = positions[i]
			orig_y,orig_x   = original_positions[i]
			_, pixel        = common.getpixel(image, orig_y, orig_x, False)
			_, pixel_weight = common.getpixel(image_weight_layer, orig_y, orig_x, False)

			shifted[y][x] = pixel
			shifted_weight_layer[y][x] = pixel_weight

	return shifted, shifted_weight_layer
