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

def serread(f, bs, le):
	block = list(f.read(bs))
	if le:
		block = block[::-1]
	val = 0
	for i in range(bs):
		val *= 256
		val += block[i]
	return val

def serread4(f):
	return serread(f, 4, True)

def pixelread(f, bpp, le, colorid):
	if colorid == 0:
		return serread(f, bpp, le)
	elif colorid == 100:
		return np.array([serread(f, bpp, le), serread(f, bpp, le), serread(f, bpp, le)])
	elif colorid == 101:
		return np.array([serread(f, bpp, le), serread(f, bpp, le), serread(f, bpp, le)])

def read_to_npy(f, bpp, le, shape):
	num = 1
	for i in range(len(shape)):
		num *= shape[i]
	num_b = bpp * num
	block = np.array(list(f.read(num_b)), dtype=np.uint32)
	block = block.reshape((num, bpp))
	for i in range(bpp):
		if le:
			block[:,i] *= 2**(8*i)
		else:
			block[:,i] *= 2**(8*(bpp-i))
	block = np.sum(block, axis=1)
	block = block.reshape(shape)
	return block

def readser(fname):
	with open(fname, "rb") as f:
		fileid = f.read(14)
		if fileid != b'LUCAM-RECORDER':
			print("Invalid header, skipping")
			return []
		luid = serread4(f)
		colorid = serread4(f)
		le16bit = serread4(f)
		width = serread4(f)
		height = serread4(f)
		depth = serread4(f)
		bpp = (int)(math.ceil(depth / 8))
		frames = serread4(f)
		observer = f.read(40).decode('utf8')
		instrume = f.read(40).decode('utf8')
		telescope = f.read(40).decode('utf8')
		datetime = serread(f, 8, True)
		datetimeUTC = serread(f, 8, True)

		if colorid == 0:
			shape = (height, width, 1)
			channels = ["Y"]
		elif colorid == 100:
			shape = (height, width, 3)
			channels = ["R", "G", "B"]
		elif colorid == 101:
			shape = (height, width, 3)
			channels = ["B", "G", "R"]
		else:
			print("Unsupported colorid = %i" % colorid)
			return []

		tags = {
			"depth" : depth,
			"observer" : observer,
			"instrument" : instrume,
			"telescope" : telescope,
			"dateTime" : datetime,
			"dateTimeUTC" : datetimeUTC,
		}

		params = {
			"w" : width,
			"h" : height,
			"projection" : "perspective",
			"perspective_F" : vstarstack.cfg.scope.F,
			"perspective_kh" : vstarstack.cfg.camera.kh,
			"perspective_kw" : vstarstack.cfg.camera.kw,
			"format" : vstarstack.cfg.camera.format,
		}

		for id in range(frames):
			print("\tprocessing frame %i" % id)
			frame = read_to_npy(f, bpp, le16bit, shape)
			dataframe = vstarstack.data.DataFrame(params, tags)
			exptime = 1
			weight = np.ones(frame.data.shape)*exptime
			index = 0
			for index in range(len(channels)):
				channel = channels[index]
				dataframe.add_channel(frame[:,:,index], channel)
				dataframe.add_channel(weight, "weight-"+channel)
				dataframe.add_channel_link(channel, "weight-"+channel, "weight")
			yield id, dataframe

def process_file(argv):
	fname = argv[0]
	output = argv[1]
	name = argv[2]

	for id, dataframe in readser(fname):
		framename = os.path.join(output, "%s_%05i.zip" % (name, id))
		dataframe.store(framename)

def process_path(argv):
	input = argv[0]
	output = argv[1]

	files = vstarstack.common.listfiles(input, ".ser")
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
	"*" : (process, "read SER to npy", "(input.ser output/ | [original/ npy/])"),
}

def run(argv):
	vstarstack.usage.run(argv, "readimage ser", commands, autohelp=False)
