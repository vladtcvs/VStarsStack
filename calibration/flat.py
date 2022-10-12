import cfg
import sys
import common
import data
import math
import os
import numpy as np
import multiprocessing as mp
import usage
import cv2

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
		s = cv2.GaussianBlur(s, (51, 51), 0)
		s = s / np.amax(s)
		data.data_add_channel(result_image, s, channel)
	data.data_store(result_image, result)

def prepare_sky(argv):
	out = argv[0]
	imgs = argv[1]
	files = common.listfiles(imgs, ".zip")
	channels = {}
	for _, fname in files:
		frame = data.data_load(fname)
		for channel in frame["meta"]["channels"]:
			if channel in frame["meta"]["encoded_channels"]:
				continue
			if channel in ["weight"]:
				continue
			image = frame["channels"][channel]
			image = cv2.GaussianBlur(image, (5, 5), 0)
			if channel not in channels:
				channels[channel] = []
			channels[channel].append(image / np.amax(image))

	thr = 0.006
	S = 101
	S2 = 301
	kh=1.1
	kl=0.9

	result_image = data.data_create()
	for channel in channels:
		avg = sum(channels[channel]) / len(channels[channel])
		skyes = []
		for i in range(len(channels[channel])):
			img = channels[channel][i]
			diff = abs(img - avg)
			mask = diff > thr
			
			sky = (1-mask)*img
			sky = sky / np.amax(sky)

			sky_avg = np.average(sky)
			sky_h = sky_avg * kh
			sky_l = sky_avg * kl
			sky255 = (sky - sky_l) / (sky_h - sky_l)
			sky255 = np.clip(sky255, 0, 1)

			sky_fixed = cv2.medianBlur((sky255*255).astype('uint8'), S)
			sky_fixed = sky_fixed.astype('float32')/255 * (sky_h - sky_l) + sky_l
			sky_fixed = cv2.GaussianBlur(sky_fixed, (S2, S2), 0)

			skyes.append(sky_fixed)

		sky_fixed = sum(skyes) / len(skyes)
		sky_fixed = sky_fixed / np.amax(sky_fixed)
		data.data_add_channel(result_image, sky_fixed, channel)
	data.data_store(result_image, out)

commands = {
	"prepare" : (prepare_flats, "flat prepare", "prepare flat frames"),
	"prepare-starsky" : (prepare_sky, "flat prepare-starsky output.zip inputs/", "prepare flat frames from N images with stars"),
	"*" : (process, "flat", "(input.file output.file | input/ output/) flat.zip"),
}

def run(argv):
	usage.run(argv, "image-fix flat", commands, autohelp=False)
