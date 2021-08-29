import os
import exifread
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

from PIL import Image
import common
import usage

def readjpeg(fname):
	rgb = np.asarray(Image.open(fname))
	shape = rgb.shape
	shape = (shape[0], shape[1], shape[2]+1)

	f = open(fname, 'rb')
	tags = exifread.process_file(f)
	f.close()
	
	shutter = float(tags["EXIF ExposureTime"].values[0])
	print("shutter = %f" % shutter)
        
	rgba = np.zeros(shape)
	rgba[:,:,0:3] = rgb
	rgba[:,:,3] = shutter
	return rgba

def process_file(argv):
	fname = argv[0]
	output = argv[1]
	post = readjpeg(fname)
	np.savez_compressed(output, post)

def process_path(argv):
	input = argv[0]
	output = argv[1]
	files = common.listfiles(input, ".jpg")
	for name, fname in files:
		print(name)
		post = readjpeg(fname)
		np.savez_compressed(os.path.join(output, name + ".npz"), post)

def process(argv):
	input = argv[0]
	if os.path.isdir(input):
		process_path(argv)
	else:
		process_file(argv)

commands = {
	"*" : (process, "read JPEG to npz", "(input.jpg output.npz | original/ npy/)"),
}

def run(argv):
	usage.run(argv, "readimage jpeg", commands)

