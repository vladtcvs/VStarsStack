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
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import json
import vstarstack.cfg
import vstarstack.common
import vstarstack.usage
import vstarstack.data

import math

import zipfile
from astropy.io import fits

def process_file(argv):
	fname = argv[0]
	output = argv[1]
	name = argv[2]

	print("Opening %s"% fname)
	try:
		images = fits.open(fname)
	except Exception as e:
		print("Error reading file: %s" % e)
		return

	for id in range(1):
		print(id)
		image = images[id]

		tags = {}
		for key in image.header:
			val = str(image.header[key])
			tags[key] = val

		shape = image.data.shape
		if len(shape) == 2:
			original = image.data.reshape((1, shape[0], shape[1]))
		else:
			original = image.data
		shape = original.shape

		params = {
			"w" : shape[2],
			"h" : shape[1],
			"projection" : "perspective",
			"perspective_F" : vstarstack.cfg.scope.F,
			"perspective_kh" : vstarstack.cfg.camera.kh,
			"perspective_kw" : vstarstack.cfg.camera.kw,
			"format" : vstarstack.cfg.camera.format,
		}

		dataframe = vstarstack.data.DataFrame(params, tags)

		exptime = image.header["EXPTIME"]

		slices = []

		weight_channel_name = "weight"
		weight = np.ones((shape[1], shape[2]))*exptime
		dataframe.add_channel(weight, weight_channel_name, weight=True)

		if shape[0]==1:
			if "FILTER" in image.header:
				channel_name = image.header["FILTER"].strip()
			else:
				channel_name = "Y"
			slices.append(channel_name)
		elif shape[0] == 3:
			slices.append('R')
			slices.append('G')
			slices.append('B')
		else:
			print("Unknown image format, skip")
			return

		for i in range(len(slices)):
			dataframe.add_channel(original[i,:,:], slices[i], brightness=True)
			dataframe.add_channel_link(slices[i], weight_channel_name, "weight")

		framename = os.path.join(output, "%s.zip" % (name))
		dataframe.store(framename)

def process_path(argv):
	input = argv[0]
	output = argv[1]

	files = vstarstack.common.listfiles(input, ".fits")
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
	"*" : (process, "read FITS to npy", "(input.fits output/ | [original/ npy/])"),
}

def run(argv):
	vstarstack.usage.run(argv, "readimage fits", commands, autohelp=False)
