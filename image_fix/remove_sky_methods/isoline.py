import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import sys
import cv2
import usage
import os
import common
import multiprocessing as mp

import cfg
import sky_model.isoline_model
import stars.detect

ncpu = 11
k = 4
build_model = False
sky_blur = 251

def fun(l):
	return 2/(l**2+1) - 1

def interp(y, x, h, w):
	L = ((w/2)**2+(h/2)**2)**0.5
	x -= w/2
	y -= h/2
	l = (x**2+y**2)**0.5
	return fun(l/L)

def remove_sky(name, infname, outfname, make_model):
	print(name)

	img = common.data_load(fname)
	for channel in img["meta"]["channels"]:
		if channel in img["meta"]["encoded_channels"]:
			continue
		image = img["channels"][channel]

		sky = sky_model.isoline_model.build_sky_model(image)
		if make_model:
			result = sky
		else:
			result = image - sky

		common.data_add_channel(img, result, channel)
		
	common.data_store(img, outfname)

def process_file(argv, make_model=False):
	infname = argv[0]
	outfname = argv[1]
	name = os.path.splitext(os.path.basename(infname))[0]
	remove_sky(name, infname, outfname, make_model)

def process_dir(argv, make_model=False):
	inpath = argv[0]
	outpath = argv[1]
	files = common.listfiles(inpath, ".zip")
	pool = mp.Pool(ncpu)
	pool.starmap(remove_sky, [(name, fname, os.path.join(outpath, name + ".zip"), make_model) for name, fname in files])
	pool.close()

def process(argv):
	if len(argv) > 0:
		if os.path.isdir(argv[0]):
			process_dir(argv)
		else:
			process_file(argv)
	else:
		process_dir([cfg.config["paths"]["npy-fixed"], cfg.config["paths"]["npy-fixed"]])

def process_sky(argv):
	if os.path.isdir(argv[0]):
		process_dir(argv, True)
	else:
		process_file(argv, True)

commands = {
	"*" : (process, "Remove sky from image", "(input.file output.file | input/ output/)"),
	"sky" : (process_sky, "Build sky from image", "(input.file output.file | input/ output/)"),
}

def run(argv):
	usage.run(argv, "image-fix remove-sky isoline", commands, "")
