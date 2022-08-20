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


def readyuv(fname, W, H, format):
	frame_len = int(W*H*2)
	shape = (int(H*2), W)

	exposure = np.ones((H, W))

	with open(fname, "rb") as f:
		tags = {
			"depth" : 8,
		}

		params = {
			"w" : cfg.camerad["w"],
			"h" : cfg.camerad["h"],
			"projection" : "perspective",
			"perspective_kh" : cfg.camerad["H"] / cfg.camerad["h"],
			"perspective_kw" : cfg.camerad["W"] / cfg.camerad["w"],
			"perspective_F" : cfg.camerad["F"],
		}

		id = 0
		while True:
			frame = f.read(frame_len)
			if not frame:
				break
			yuv = np.frombuffer(frame, dtype=np.uint8)
			try:
				yuv = yuv.reshape(shape)
				print("\tprocessing frame %i" % id)

				data = common.data_create(tags, params)
				common.data_add_channel(data, yuv, "raw", encoded=True)
				common.data_add_parameter(data, 1, "exposure")
				yield id, data
				id += 1
			except:
				break

	
def process_file(argv):
	fname = argv[0]
	output = argv[1]
	name = argv[2]

	W = cfg.camerad["w"]
	H = cfg.camerad["h"]
	fmt = cfg.camerad["format"]

	for i, data in readyuv(fname, W, H, fmt):
		framename = os.path.join(output, "%s_%05i.zip" % (name, i))
		common.data_store(data, framename)

def process_path(argv):
	input = argv[0]
	output = argv[1]

	files = common.listfiles(input, ".yuv")
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
	"*" : (process, "read SER to npy", "(input.yuv output/ | [original/ npy/])"),
}

def run(argv):
	usage.run(argv, "readimage yuv", commands, autohelp=False)
