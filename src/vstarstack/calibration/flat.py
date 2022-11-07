#
# Copyright (c) 2022 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import vstarstack.cfg
import vstarstack.common
import vstarstack.data
import vstarstack.usage

import math
import os
import numpy as np
import multiprocessing as mp
import cv2
import scipy.signal

ncpu = max(1, mp.cpu_count()-1)

def flatten(name, fname, out, flat_file):
	print(name)

	img = vstarstack.data.DataFrame.load(fname)
	flat_img = vstarstack.data.DataFrame.load(flat_file)

	for channel in img.get_channels():
		image, opts = img.get_channel(channel)
		if not opts["brightness"]:
			continue

		if channel in flat_img.get_channels():
			image = image / flat_img.get_channel(channel)[0]

		img.add_channel(image, channel, **opts)
	img.store(out)

def process_file(input, output, flat_file):
	name = os.path.splitext(os.path.basename(input))[0]
	flatten(name, input, output, flat_file)

def process_dir(input, output, flat_file):
	files = vstarstack.common.listfiles(input, ".zip")
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
		input = vstarstack.cfg.config["paths"]["npy-fixed"]
		output = vstarstack.cfg.config["paths"]["npy-fixed"]
		flat_file = vstarstack.cfg.config["calibration"]["flat"]["path"]

	if os.path.isdir(input):
		process_dir(input, output, flat_file)
	else:
		process_file(input, output, flat_file)

def prepare_flats(argv):
	if len(argv) >= 2:
		npys = argv[0]
		result = argv[1]
	else:
		npys = vstarstack.cfg.config["calibration"]["flat"]["npy"]
		result = vstarstack.cfg.config["calibration"]["flat"]["path"]
	
	channels = {}
	files = vstarstack.common.listfiles(npys, ".zip")
	for _, fname in files:
		flat_frame = vstarstack.data.DataFrame.load(fname)
		for channel in flat_frame.get_channels():
			image, opts = flat_frame.get_channel(channel)
			if not opts["brightness"]:
				continue
			if channel not in channels:
				channels[channel] = []
			channels[channel].append(image)

	result_image = vstarstack.data.DataFrame()
	for channel in channels:
		s = sum(channels[channel])
		s = cv2.GaussianBlur(s, (51, 51), 0)
		s = s / np.amax(s)
		result_image.add_channel(s, channel, brightness=True)
	result_image.store(result)

def prepare_sky(argv):
	out = argv[1]
	imgs = argv[0]
	files = vstarstack.common.listfiles(imgs, ".zip")
	channels = {}
	for _, fname in files:
		frame = vstarstack.data.DataFrame.load(fname)
		for channel in frame.get_channels():
			image, opts = frame.get_channel(channel)
			if not opts["brightness"]:
				continue

			image = cv2.GaussianBlur(image, (5, 5), 0)
			if channel not in channels:
				channels[channel] = []
			channels[channel].append(image / np.amax(image))

	thr = 0.006
	S = 31
	S2 = 301

	result_image = vstarstack.data.DataFrame()
	for channel in channels:
		print(channel)
		avg = sum(channels[channel]) / len(channels[channel])
		skyes = []
		for i in range(len(channels[channel])):
			img = channels[channel][i]
			diff = abs(img - avg)
			mask_idx = np.where(diff > thr)

			sky = img
			sky[mask_idx] = np.average(sky)
			print("Apply median filter")
			sky_fixed = scipy.signal.medfilt2d(sky, S)
			print("\tDone")
			sky_fixed = cv2.GaussianBlur(sky_fixed, (S2, S2), 0)

			skyes.append(sky_fixed)

		sky_fixed = sum(skyes) / len(skyes)
		sky_fixed = sky_fixed / np.amax(sky_fixed)
		result_image.add_channel(sky_fixed, channel, brightness=True)
	result_image.store(out)

commands = {
	"prepare" : (prepare_flats, "flat prepare", "prepare flat frames"),
	"prepare-starsky" : (prepare_sky, "flat prepare-starsky inputs/ output.zip", "prepare flat frames from N images with stars"),
	"*" : (process, "flat", "(input.file output.file | input/ output/) flat.zip"),
}

def run(argv):
	vstarstack.usage.run(argv, "image-fix flat", commands, autohelp=False)
