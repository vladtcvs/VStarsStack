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

	try:
		e = tags["shutter"]*tags["iso"]
	except:
		e = 1

	if len(rgb.shape) == 3:
		data = {
			"channels" : { 
				"red"   : rgb[:,:,0],
				"green" : rgb[:,:,1],
				"blue"  : rgb[:,:,2],
				"exposure" : np.ones(shape).astype(np.float32)*e
			},
			"meta" : {
				"channels" : ["red", "green", "blue", "exposure"],
				"tags" : tags,
			}
		}
	else:
		data = {
			"channels" : { 
				"gray"  : rgb,
				"exposure" : np.ones(shape).astype(np.float32)*e
			},
			"meta" : {
				"channels" : ["gray", "exposure"],
				"tags" : tags,
			}
		}
	return data

def process_file(argv):
	fname = argv[0]
	output = argv[1]
	data = readjpeg(fname)
	with zipfile.ZipFile(output, "w") as zf:
		with zf.open("meta.json", "w") as f:
			f.write(bytes(json.dumps(data["meta"], indent=4, ensure_ascii=False), "utf8"))
		for channel in data["channels"]:
			with zf.open(channel+".npy", "w") as f:
				np.save(f, data["channels"][channel])

def process_path(argv):
	input = argv[0]
	output = argv[1]

	files = common.listfiles(input, ".jpg")
	for name, fname in files:
		print(name)
		process_file((fname, os.path.join(output, name + '.npz')))

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
	"*" : (process, "read JPEG to npz", "(input.jpg output.zip | [original/ npy/])"),
}

def run(argv):
	usage.run(argv, "readimage jpeg", commands, autohelp=False)
