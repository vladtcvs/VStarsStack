import cfg
import sys
import common
import data
import math
import os
import numpy as np
import multiprocessing as mp
import usage

ncpu = max(1, mp.cpu_count()-1)

def flatten(name, fname, out, flat_file):
	print(name)

	img = common.data_load(fname)
	flat_img = common.data_load(flat_file)

	for channel in img["meta"]["channels"]:
		if channel in img["meta"]["encoded_channels"]:
			continue
		image = img["channels"][channel]

		if channel in flat_img["meta"]["channels"]:
			image = image / flat_img["channels"][channel]

		common.data_add_channel(img, image, channel)

	common.data_store(img, out)

def process_file(input, output, flat_file):
	name = os.path.splitext(os.path.basename(input))[0]
	flatten(name, input, output, flat_file)

def process_dir(input, output, flat_file):
	files = common.listfiles(input, ".zip")
	pool = mp.Pool(ncpu)
	args = [(name, fname, os.path.join(output, name + ".zip"), flat_file)
				for name, fname in files]
	pool.starmap(flatten, args)
	pool.close()

def process(argv):
	if len(argv) >= 3:
		input = argv[0]
		output = argv[1]
		flat_file = argv[2]
	else:
		input = cfg.config["paths"]["npy-fixed"]
		output = cfg.config["paths"]["npy-fixed"]
		flat_file = cfg.config["calibration"]["flat"]["path"]

	if os.path.isdir(input):
		process_dir(input, output, flat_file)
	else:
		process_file(input, output, flat_file)


def prepare_flats(argv):
	if len(argv) >= 2:
		npys = argv[0]
		result = argv[1]
	else:
		npys = cfg.config["calibration"]["flat"]["npy"]
		result = cfg.config["calibration"]["flat"]["path"]
	
	channels = {}
	files = common.listfiles(npys, ".zip")
	for _, fname in files:
		dark_frame = data.data_load(fname)
		for channel in dark_frame["meta"]["channels"]:
			if channel in dark_frame["meta"]["encoded_channels"]:
				continue
			if channel in ["weight"]:
				continue
			image = dark_frame["channels"][channel]
			if channel not in channels:
				channels[channel] = []
			channels[channel].append(image)

	result_image = data.data_create()
	for channel in channels:
		s = sum(channels[channel])
		s = s / np.amax(s)
		data.data_add_channel(result_image, s, channel)
	data.data_store(result_image, result)

commands = {
	"prepare" : (prepare_flats, "flat prepare", "prepare flat frames"),
	"*" : (process, "flat", "(input.file output.file | input/ output/) flat.zip"),
}

def run(argv):
	usage.run(argv, "image-fix flat", commands, autohelp=False)
