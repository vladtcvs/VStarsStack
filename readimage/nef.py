import os
import rawpy
import exifread
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import cfg
import common
import usage
import multiprocessing as mp

mask = np.array([
			[[0, 0], [2, 0]], # red
			[[1, 0], [0, 1]], # green
			[[0, 2], [0, 0]], # blue
		])

def getcolor(img, mask):
	return np.sum(img*mask)

def readnef_postprocess(filename):
	options = {
		"half_size" : True,
		"use_camera_wb" : False,
		"use_auto_wb" : False,
		"gamma" : (1,1), 
		"no_auto_bright" : True,
		"output_bps" : 16,
		"no_auto_bright" : True,
		"four_color_rgb" : False,
		"user_wb" : (1,1,1,1),
	}

	image = rawpy.imread(filename)
	rgb = image.postprocess(**options)
	shape = rgb.shape
	shape = (shape[0], shape[1], shape[2]+1)
	rgba = np.zeros(shape)
	rgba[:,:,0:3] = rgb
	rgba[:,:,3] = 1
	return rgba

def readnef_manual(filename):
	img = rawpy.imread(filename)
	image = img.raw_image_visible
	shape = image.shape
	cshape = (int(shape[0]/2), int(shape[1]/2), 4)
	post = np.zeros(cshape)

	f = open(filename, 'rb')
	tags = exifread.process_file(f)
	f.close()

	shutter = float(tags["EXIF ExposureTime"].values[0])
	print("shutter = %f" % shutter)
        

	for y in range(cshape[0]):
		for x in range(cshape[1]):
			cut = image[2*y:2*y+2, 2*x:2*x+2]
			post[y][x][0] = getcolor(cut, mask[0])
			post[y][x][1] = getcolor(cut, mask[1])
			post[y][x][2] = getcolor(cut, mask[2])
			post[y][x][3] = shutter
	return post

readnef = readnef_manual

def work(input, output):
	print(input)
	post = readnef(input)
	np.savez_compressed(output, post)

def process_file(argv):
	input = argv[0]
	output = argv[1]
	work(input, output)

def process_path(argv):
	input = argv[0]
	output = argv[1]
	files = common.listfiles(input, ".nef")
	ncpu = max(int(mp.cpu_count())-1, 1)
	pool = mp.Pool(ncpu)
	pool.starmap(work, [(filename, os.path.join(output, name + ".npz")) for name, filename in files])
	pool.close()

def process(argv):
	if len(argv) > 0:
		input = argv[0]
		if os.path.isdir(input):
			process_path(argv)
		else:
			process_file(argv)
	else:
		process_path([cfg.config["paths"]["original"], cfg.config["paths"]["npy-orig"]])

commands = {
	"*" : (process, "read NEF to npz", "(input.NEF output.npz | [original/ npy/])"),
}

def run(argv):
	usage.run(argv, "readimage nef", commands, autohelp=False)

