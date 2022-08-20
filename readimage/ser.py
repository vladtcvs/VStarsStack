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

import math
import readimage.tags

import zipfile

def serread(f, bs, le):
	block = list(f.read(bs))
	if le:
		block = block[::-1]
	val = 0
	for i in range(bs):
		val *= 256
		val += block[i]
	return val

def serread4(f):
	return serread(f, 4, True)

def readser(fname):
	with open(fname, "rb") as f:
		fileid = f.read(14)
		if fileid != b'LUCAM-RECORDER':
			print("Invalid header, skipping")
			return []
		luid = serread4(f)
		colorid = serread4(f)
		if colorid != 0:
			print("Unsupported colorid = %i" % colorid)
			return []
		le16bit = serread4(f)
		width = serread4(f)
		height = serread4(f)
		depth = serread4(f)
		bpp = (int)(math.ceil(depth / 8))
		frames = serread4(f)
		observer = f.read(40).decode('utf8')
		instrume = f.read(40).decode('utf8')
		telescope = f.read(40).decode('utf8')
		datetime = serread(f, 8, True)
		datetimeUTC = serread(f, 8, True)

		tags = {
			"depth" : depth,
			"observer" : observer,
			"instrument" : instrume,
			"telescope" : telescope,
			"dateTime" : datetime,
			"dateTimeUTC" : datetimeUTC,
		}
		params = {
			"w" : cfg.camerad["w"],
			"h" : cfg.camerad["h"],
			"projection" : "perspective",
			"perspective_kh" : cfg.camerad["H"] / cfg.camerad["h"],
			"perspective_kw" : cfg.camerad["W"] / cfg.camerad["w"],
			"perspective_F" : cfg.camerad["F"],
		}

		for id in range(frames):
			print("\tprocessing frame %i" % id)
			frame = np.zeros((height, width), dtype=np.float32)
			for y in range(height):
				for x in range(width):
					frame[y,x] = serread(f, bpp, le16bit)
			data = common.data_create(tags, params)
			common.data_add_channel(data, frame, "raw", encoded=True)
			common.data_add_parameter(data, 1, "weight")
			yield id, data

def process_file(argv):
	fname = argv[0]
	output = argv[1]
	name = argv[2]

	for id, data in readser(fname):
		framename = os.path.join(output, "%s_%05i.zip" % (name, id))
		common.data_store(data, framename)

def process_path(argv):
	input = argv[0]
	output = argv[1]

	files = common.listfiles(input, ".ser")
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
