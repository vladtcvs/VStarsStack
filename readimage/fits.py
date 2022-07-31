import os
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import json
import cfg
import common
import usage

import math

import zipfile
from astropy.io import fits

def process_file(argv):
	fname = argv[0]
	output = argv[1]
	name = argv[2]

	print("Opening %s"% fname)
	images = fits.open(fname)

	for id in range(1):
		print(id)
		image = images[id]

		tags = {}
		for key in image.header:
			val = str(image.header[key])
			tags[key] = val
		data = common.data_create(tags=tags)
		
		common.data_add_parameter(data, image.data.shape[0], "originalH")
		common.data_add_parameter(data, image.data.shape[1], "originalW")
		common.data_add_parameter(data, image.header["EXPTIME"], "exposure")
		

		common.data_add_channel(data, image.data, "Y")

		framename = os.path.join(output, "%s.zip" % (name))
		common.data_store(data, framename)

def process_path(argv):
	input = argv[0]
	output = argv[1]

	files = common.listfiles(input, ".fits")
	for name, fname in files:
		print(name)
		process_file((fname, output, name))

def process(argv):
	if len(argv) > 0:
		input = argv[0]
		output = argv[1]
		if os.path.isdir(input):
			process_path((input, output))
		else:
			name = os.path.splitext(os.path.basename(input))[0]
			process_file((input, output, name))
	else:
		process_path([cfg.config["paths"]["original"], cfg.config["paths"]["npy-orig"]])

commands = {
	"*" : (process, "read SER to npy", "(input.ser output/ | [original/ npy/])"),
}

def run(argv):
	usage.run(argv, "readimage ser", commands, autohelp=False)
