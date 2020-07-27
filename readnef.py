import os
import rawpy
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import common

masks = np.array([
			[
				[
					[[0, 2], [0, 0]], # red
					[[1, 0], [0, 1]], # green
					[[0, 0], [2, 0]], # blue
				],
				[
					[[2, 0], [0, 0]], # red
					[[0, 1], [1, 0]], # green
					[[0, 0], [0, 2]], # blue
				]
			],
			[
				[
					[[0, 0], [0, 2]], # red
					[[0, 1], [1, 0]], # green
					[[2, 0], [0, 0]], # blue
				],
				[
					[[0, 0], [2, 0]], # red
					[[1, 0], [0, 1]], # green
					[[0, 2], [0, 0]], # blue
				]
			]
])

def getcolor(img, mask):
	return np.sum(img*mask)

def readnef(filename):
	options = {
		"half_size" : True,
		"four_color_rgb" : False,
		"use_camera_wb" : False,
                "use_auto_wb" : False,
		"user_wb" : (1,1,1,1),
		"user_flip" : 0,
		"output_color" : rawpy.ColorSpace.raw,
		"output_bps" : 16,
		"user_black" : None,
		"user_sat" : None,
		"no_auto_bright" : True,
		"auto_bright_thr" : 0.0,
		"adjust_maximum_thr" : 0,
		"bright" : 100.0,
		"highlight_mode" : rawpy.HighlightMode.Ignore,
		"exp_shift" : None,
		"exp_preserve_highlights" : 0.0,
		"no_auto_scale" : True,
		"gamma" : (1, 1),
		"chromatic_aberration" : None,
		"bad_pixels_path" : None
	}

	image = rawpy.imread(filename)
	rgb = image.postprocess(**options)
	return rgb

#	image = rawpy.imread(filename).raw_image_visible
#	shape = image.shape
#	cshape = (shape[0]-1, shape[1]-1, 4)
#	post = np.zeros(cshape)

#	for y in range(cshape[0]):
#		if y % 10 == 0:
#			print(y, cshape[0])
#		sy = y % 2
#		for x in range(cshape[1]):
#			sx = x % 2
#			cut = image[y:y+2, x:x+2]
#			mask = masks[sy][sx]
#			post[y][x][0] = getcolor(cut, mask[0])
#			post[y][x][1] = getcolor(cut, mask[1])
#			post[y][x][2] = getcolor(cut, mask[2])
#			post[y][x][3] = 1
#	return post


def usage():
	print("readnef nef_path npy_path")

def run(argv):

	if argv[0] == "help":
		usage()
		return

	path = argv[0]

	if len(sys.argv) > 1:
		npypath = argv[1]
	else:
		npypath = path

	files = common.listfiles(path, ".nef")

	for name, fname in files:
		print(name)
		post = readnef(fname)
		np.save(os.path.join(npypath, name + ".npy"), post)

if __name__ == "__main__":
	run(sys.argv[1:])

