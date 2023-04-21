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

import os
import exifread
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import json
from PIL import Image
import vstarstack.cfg
import vstarstack.common
import vstarstack.usage
import vstarstack.data

import math
import vstarstack.readimage.tags

import zipfile


def readyuv(fname, W, H):
	frame_len = int(W*H*2)
	shape = (int(H*2), W)

	exposure = np.ones((H, W))

	with open(fname, "rb") as f:
		tags = {
			"depth" : 8,
		}

		params = {
			"w" : W,
			"h" : H,
			"projection" : "perspective",
			"perspective_F" : vstarstack.cfg.scope.F,
			"perspective_kh" : vstarstack.cfg.camera.kh,
			"perspective_kw" : vstarstack.cfg.camera.kw,
			"format" : vstarstack.cfg.camera.format,
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

				dataframe = vstarstack.data.DataFrame(params, tags)
				exptime = 1
				weight = np.ones(frame.data.shape)*exptime

				dataframe.add_channel(yuv, "raw", encoded=True)
				dataframe.add_channel(weight, "weight")
				dataframe.add_channel_link("raw", "weight", "weight")
				yield id, dataframe
				id += 1
			except:
				break

	
def process_file(argv):
	fname = argv[0]
	output = argv[1]
	name = argv[2]

	W = vstarstack.cfg.camera.w
	H = vstarstack.cfg.camera.h

	for i, dataframe in readyuv(fname, W, H):
		framename = os.path.join(output, "%s_%05i.zip" % (name, i))
		dataframe.store(framename)

def process_path(argv):
	input = argv[0]
	output = argv[1]

	files = vstarstack.common.listfiles(input, ".yuv")
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
		process_path([vstarstack.cfg.config["paths"]["original"], vstarstack.cfg.config["paths"]["npy-orig"]])

commands = {
	"*" : (process, "read SER to npy", "(input.yuv output/ | [original/ npy/])"),
}

def run(argv):
	vstarstack.usage.run(argv, "readimage yuv", commands, autohelp=False)
