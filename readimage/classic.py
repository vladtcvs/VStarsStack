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

import zipfile

def readjpeg(fname):
	rgb = np.asarray(Image.open(fname)).astype(np.float32)
	shape = rgb.shape
	shape = (shape[0], shape[1])

	tags = readimage.tags.read_tags(fname)
	params = {
		"originalW" : shape[1],
		"originalH" : shape[0], 
	}

	try:
		e = tags["shutter"]*tags["iso"]
	except:
		e = 1

	data = common.data_create(tags, params)
	if len(rgb.shape) == 3:
		common.data_add_channel(data, rgb[:,:,0], "red")
		common.data_add_channel(data, rgb[:,:,1], "green")
		common.data_add_channel(data, rgb[:,:,2], "blue")
		
	else:
		common.data_add_channel(data, rgb, "gray")
	
	common.data_add_channel(data, np.ones(shape).astype(np.float32)*e, "exposure")
	return data

def process_file(argv):
	fname = argv[0]
	output = argv[1]
	data = readjpeg(fname)
	common.data_store(data, output)

def process_path(argv):
	input = argv[0]
	output = argv[1]

	files = common.listfiles(input, ".jpg") + common.listfiles(input, ".png") + common.listfiles(input, ".tiff")
	for name, fname in files:
		print(name)
		process_file((fname, os.path.join(output, name + '.zip')))

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
	"*" : (process, "read JPEG to npy", "(input.jpg output.zip | [original/ npy/])"),
}

def run(argv):
	usage.run(argv, "readimage jpeg", commands, autohelp=False)
