import cfg
import sys
import common
import projection
import math
import os
import numpy as np
import multiprocessing as mp
import usage
import cv2

ncpu = max(1, mp.cpu_count()-1)

def flatten(name, fname, out):
	print(name)
	
	img = common.data_load(fname)
	for channel in img["meta"]["channels"]:
		if channel in img["meta"]["encoded_channels"]:
			continue
		image = img["channels"][channel]

		if channel in cfg.flat["meta"]["channels"]:
			image = image / cfg.get_flat()["channels"][channel]

		common.data_add_channel(img, image, channel)

	common.data_store(img, out)
	
def process_file(argv):
	infname = argv[0]
	outfname = argv[1]
	name = os.path.splitext(os.path.basename(infname))[0]
	flatten(name, infname, outfname)

def process_dir(argv):
	inpath = argv[0]
	outpath = argv[1]
	files = common.listfiles(inpath, ".zip")
	pool = mp.Pool(ncpu)
	pool.starmap(flatten, [(name, fname, os.path.join(outpath, name + ".zip")) for name, fname in files])
	pool.close()

def process(argv):
	if len(argv) > 0:
		if os.path.isdir(argv[0]):
			process_dir(argv)
		else:
			process_file(argv)
	else:
		process_dir([cfg.config["paths"]["npy-fixed"], cfg.config["paths"]["npy-fixed"]])

def normalize_flat(argv):
	fname = argv[0]
	img = common.data_load(fname)
	for channel in img["meta"]["channels"]:
		img["channels"][channel] = cv2.GaussianBlur(img["channels"][channel], (5, 5), 0)
		img["channels"][channel] = img["channels"][channel] / np.amax(img["channels"][channel])
	common.data_store(img, fname)

commands = {
	"normalize-flat" : (normalize_flat, "flatten", "flat.zip"),
	"*" : (process, "flatten", "(input.file output.file | input/ output/)"),
}

def run(argv):
	usage.run(argv, "image-fix flatten", commands, autohelp=False)
