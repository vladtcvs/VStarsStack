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
import cfg
import common
import usage
import data

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

		dataframe = data.DataFrame(tags=tags)

		exptime = image.header["EXPTIME"]

		pixh = cfg.camerad["H"] / cfg.camerad["h"]
		pixw = cfg.camerad["W"] / cfg.camerad["w"]
		F = cfg.scope["F"]

		dataframe.add_parameter(image.data.shape[0], "h")
		dataframe.add_parameter(image.data.shape[1], "w")
		dataframe.add_parameter("perspective", "projection")
		dataframe.add_parameter(pixh, "perspective_kh")
		dataframe.add_parameter(pixw, "perspective_kw")
		dataframe.add_parameter(F, "perspective_F")

		if "FILTER" in image.header:
			channel_name = image.header["FILTER"].strip()
		else:
			channel_name = "Y"

		weight_channel_name = "weight-%s" % channel_name
		weight = np.ones(image.data.shape)*exptime

		dataframe.add_channel(weight, weight_channel_name, weight=True)
		dataframe.add_channel(image.data, channel_name, brightness=True)
		dataframe.add_channel_link(channel_name, weight_channel_name, "weight")

		framename = os.path.join(output, "%s.zip" % (name))
		dataframe.store(framename)

def process_path(argv):
	input = argv[0]
	output = argv[1]

	files = common.listfiles(input, ".fits")
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
		process_path([cfg.config["paths"]["original"], cfg.config["paths"]["npy-orig"]])

commands = {
	"*" : (process, "read FITS to npy", "(input.fits output/ | [original/ npy/])"),
}

def run(argv):
	usage.run(argv, "readimage fits", commands, autohelp=False)
