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

def remove_dark(name, fname, out, dark_file):
	print(name)

	img = data.DataFrame.load(fname)
	dark_img = data.DataFrame.load(dark_file)

	for channel in img.get_channels():
		image,opts = img.get_channel(channel)
		if not opts["brightness"]:
			continue

		if channel in dark_img.get_channels():
			image = image - dark_img.get_channel(channel)[0]

		img.add_channel(image, channel, **opts)

	img.store(out)

def process_file(input, output, flat_file):
	name = os.path.splitext(os.path.basename(input))[0]
	remove_dark(name, input, output, flat_file)

def process_dir(input, output, flat_file):
	files = common.listfiles(input, ".zip")
	pool = mp.Pool(ncpu)
	args = [(name, fname, os.path.join(output, name + ".zip"), flat_file)
				for name, fname in files]
	pool.starmap(remove_dark, args)
	pool.close()

def process(argv):
	if len(argv) >= 3:
		input = argv[0]
		output = argv[1]
		flat_file = argv[2]
	else:
		input = cfg.config["paths"]["npy-fixed"]
		output = cfg.config["paths"]["npy-fixed"]
		flat_file = cfg.config["calibration"]["dark"]["path"]

	if os.path.isdir(input):
		process_dir(input, output, flat_file)
	else:
		process_file(input, output, flat_file)

def prepare_darks(argv):
	if len(argv) >= 2:
		npys = argv[0]
		result = argv[1]
	else:
		npys = cfg.config["calibration"]["dark"]["npy"]
		result = cfg.config["calibration"]["dark"]["path"]
	
	channels = {}
	files = common.listfiles(npys, ".zip")
	for name, fname in files:
		print(name)
		dark_frame = data.DataFrame.load(fname)
		for channel in dark_frame.get_channels():
			image, options = dark_frame.get_channel(channel)
			if not options["brightness"]:
				continue
			if channel not in channels:
				channels[channel] = []
			channels[channel].append(image)

	result_image = data.DataFrame()
	for channel in channels:
		avg = sum(channels[channel]) / len(channels[channel])
		result_image.add_channel(avg, channel, brightness=True)
	result_image.store(result)

commands = {
	"prepare" : (prepare_darks, "dark prepare", "prepare dark frames"),
	"*" : (process, "dark", "(input.file output.file | input/ output/) dark.zip"),
}

def run(argv):
	usage.run(argv, "image-fix dark", commands, autohelp=False)
