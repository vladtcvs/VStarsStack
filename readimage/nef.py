import os
import rawpy
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import json
import cfg
import common
import usage
import multiprocessing as mp

import readimage.tags

mask = np.array([
			[[0, 0], [2, 0]], # red
			[[1, 0], [0, 1]], # green
			[[0, 2], [0, 0]], # blue
		])

def getcolor(img, mask):
	return np.sum(img*mask)

def readnef(filename, output):
	img = rawpy.imread(filename)
	image = img.raw_image_visible
	shape = image.shape
	
	tags = readimage.tags.read_tags(filename)
	
	params = {
		"originalW" : shape[1],
		"originalH" : shape[0],
	}

	exposure = np.ones(shape) * tags["shutter"]*tags["iso"]

	data = common.data_create(tags, params)
	common.data_add_channel(data, image, "raw")
	common.data_add_channel(data, exposure, "exposure")
	common.data_store(data, output)


def debayer():
	for y in range(cshape[0]):
		for x in range(cshape[1]):
			cut = image[2*y:2*y+2, 2*x:2*x+2]
			post[y][x][0] = getcolor(cut, mask[0])
			post[y][x][1] = getcolor(cut, mask[1])
			post[y][x][2] = getcolor(cut, mask[2])
	return post, tags

def work(input, output, metaoutput):
	print(input)
	readnef(input, output)

def process_file(argv):
	input = argv[0]
	output = argv[1]
	meta = argv[2]
	work(input, output, meta)

def process_path(argv):
	input = argv[0]
	output = argv[1]
	meta = argv[2]
	files = common.listfiles(input, ".nef")
	ncpu = max(int(mp.cpu_count())-1, 1)
	pool = mp.Pool(ncpu)
	pool.starmap(work, [(filename, os.path.join(output, name + ".npz"), os.path.join(meta, name + ".json")) for name, filename in files])
	pool.close()

def process(argv):
	if len(argv) > 0:
		input = argv[0]
		if os.path.isdir(input):
			process_path(argv)
		else:
			process_file(argv)
	else:
		process_path([cfg.config["paths"]["original"], cfg.config["paths"]["npy-orig"], cfg.config["paths"]["meta"]])

commands = {
	"*" : (process, "read NEF to npz", "(input.NEF output.npz meta.json | [original/ npy/ meta/])"),
}

def run(argv):
	usage.run(argv, "readimage nef", commands, autohelp=False)

