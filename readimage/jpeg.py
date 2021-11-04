import os
import exifread
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import json
from PIL import Image
import cfg
import common
import usage

import readimage.tags

def readjpeg(fname):
	rgb = np.asarray(Image.open(fname))
	shape = rgb.shape
	shape = (shape[0], shape[1], shape[2]+1)

	tags = readimage.tags.read_tags(fname)
        
	rgba = np.zeros(shape)
	rgba[:,:,0:3] = rgb
	rgba[:,:,3] = tags["shutter"]*tags["iso"]
	return rgba, tags

def process_file(argv):
	fname = argv[0]
	output = argv[1]
	metaoutput = argv[2]
	post, meta = readjpeg(fname)
	np.savez_compressed(output, post)
	with open(metaoutput, "w") as f:
		json.dump(meta, f, indent=4, ensure_ascii=False)

def process_path(argv):
	input = argv[0]
	output = argv[1]
	metaoutput = argv[2]

	files = common.listfiles(input, ".jpg")
	for name, fname in files:
		print(name)
		post, meta = readjpeg(fname)
		np.savez_compressed(os.path.join(output, name + ".npz"), post)
		with open(metaoutput, "w") as f:
			json.dump(meta, f, indent=4, ensure_ascii=False)

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
	"*" : (process, "read JPEG to npz", "(input.jpg output.npz meta.json | [original/ npy/ meta/])"),
}

def run(argv):
	usage.run(argv, "readimage jpeg", commands, autohelp=False)

