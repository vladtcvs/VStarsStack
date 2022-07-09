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


def readnef(filename, output):
	img = rawpy.imread(filename)
	image = img.raw_image_visible
	shape = image.shape
	
	tags = readimage.tags.read_tags(filename)
	
	params = {
		"originalW" : shape[1],
		"originalH" : shape[0],
	}

	exposure = tags["shutter"]*tags["iso"]

	data = common.data_create(tags, params)
	common.data_add_channel(data, image, "raw", encoded=True)
	common.data_add_parameter(data, exposure, "exposure")
	common.data_store(data, output)



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

