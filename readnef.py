import os
import rawpy
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import common
import usage

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

def process_file(argv):
	input = argv[0]
	output = argv[1]
	post = readnef(input)
	np.savez_compressed(output, post)

def process_path(argv):
	input = argv[0]
	output = argv[1]
	files = common.listfiles(input, ".nef")
	for name, fname in files:
		print(name)
		post = readnef(fname)
		np.savez_compressed(os.path.join(output, name + ".npz"), post)

commands = {
	"file" : (process_file, "read single file", "input.NEF output.npz"),
	"path" : (process_path, "read all files in path", "input/ output/"),
}

def run(argv):
	usage.run(argv, "readnef", commands)

