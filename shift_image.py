import numpy as np
import math

def shift_image(image, t):
	shape = image.shape
	h = shape[0]
	w = shape[1]
	n = shape[2]
	shifted = np.zeros((h, w, 4))
	for y in range(h):
		if y % 200 == 0:
			print(y, "/", h)
		xs = range(w)
		ys = []
		for i in range(w):
			ys.append(y)
		positions = list(zip(ys, xs))
		npositions = t.reverse(positions)
		for x in range(w):
			oy = npositions[x][0]
			ox = npositions[x][1]
			oy_f = math.floor(oy)
			ox_f = math.floor(ox)
			oy_c = math.ceil(oy)
			ox_c = math.ceil(ox)
			if oy_f < 0 or ox_f < 0 or oy_c >= h or ox_c >= w:
				continue
			p1 = image[oy_f][ox_f]
			p2 = image[oy_f][ox_c]
			p3 = image[oy_c][ox_f]
			p4 = image[oy_c][ox_c]

			ky = oy - oy_f
			kx = ox - ox_f

			p = (1-ky) * ((1-kx)*p1 + kx*p2) + ky * ((1-kx)*p3 + kx*p4)

			shifted[y][x][0:3] = p
			shifted[y][x][3] = 1
			
	return shifted

