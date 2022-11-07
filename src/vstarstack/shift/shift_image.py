#
# Copyright (c) 2022 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import math
import vstarstack.common

def generate_points(h, w, len0):
	points = []
	for y in range(h):
		for x in range(w):
			points.append((y,x))
			if len(points) >= len0:
				yield points
				points = []
	if len(points) > 0:
		yield points

def shift_image(image, t, proj, image_weight_layer=None, image_weight=1):
	shape = image.shape
	h = shape[0]
	w = shape[1]
	
	shifted = np.zeros(shape)
	shifted_weight_layer = np.zeros(shape)

	if image_weight_layer is None:
		image_weight_layer = np.ones(shape)*image_weight

	for positions in generate_points(h, w, w*4):
		original_positions = t.reverse(positions, proj)
		for position, original_position in zip(positions, original_positions):
			y,x = position
			orig_y,orig_x   = original_position
			_, pixel        = vstarstack.common.getpixel(image, orig_y, orig_x, False)
			_, pixel_weight = vstarstack.common.getpixel(image_weight_layer, orig_y, orig_x, False)

			shifted[y][x] = pixel
			shifted_weight_layer[y][x] = pixel_weight

	return shifted, shifted_weight_layer
