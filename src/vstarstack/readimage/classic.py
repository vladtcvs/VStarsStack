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

import vstarstack.readimage.tags

import zipfile

def readjpeg(fname):
	rgb = np.asarray(Image.open(fname)).astype(np.float32)
	shape = rgb.shape
	shape = (shape[0], shape[1])

	tags = vstarstack.readimage.tags.read_tags(fname)
	params = {
		"w" : shape[1],
		"h" : shape[0],
		"projection" : "perspective",
		"perspective_F" : vstarstack.cfg.scope.F,
		"perspective_kh" : vstarstack.cfg.camera.kh,
		"perspective_kw" : vstarstack.cfg.camera.kw,
		"format" : vstarstack.cfg.camera.format,
	}

	try:
		e = tags["shutter"]*tags["iso"]
	except:
		e = 1

	weight = np.ones((shape[0], shape[1]))*e

	dataframe = vstarstack.data.DataFrame(params, tags)
	dataframe.add_channel(weight, "weight", weight=True)

	if len(rgb.shape) == 3:
		dataframe.add_channel(rgb[:,:,0], "R", brightness=True)
		dataframe.add_channel(rgb[:,:,1], "G", brightness=True)
		dataframe.add_channel(rgb[:,:,2], "B", brightness=True)
		dataframe.add_channel_link("R", "weight", "weight")
		dataframe.add_channel_link("G", "weight", "weight")
		dataframe.add_channel_link("B", "weight", "weight")
	elif len(rgb.shape) == 2:
		dataframe.add_channel(rgb[:,:], "Y", weight_name="weight", brightness=True)
		dataframe.add_channel_link("Y", "weight", "weight")
	else:
		# unknown shape!
		pass
	return dataframe

def process_file(argv):
	fname = argv[0]
	output = argv[1]
	dataframe = readjpeg(fname)
	dataframe.store(output)

def process_path(argv):
	input = argv[0]
	output = argv[1]

	files = vstarstack.common.listfiles(input, ".jpg") + vstarstack.common.listfiles(input, ".png") + vstarstack.common.listfiles(input, ".tiff")
	for name, fname in files:
		print(name)
		process_file((fname, os.path.join(output, name + '.zip')))

def process(argv):
	if len(argv) > 0:
		input = argv[0]
		if os.path.isdir(input):
			process_path(argv)
		else:
			process_file(argv)
	else:
		process_path([vstarstack.cfg.config["paths"]["original"], vstarstack.cfg.config["paths"]["npy-orig"]])

commands = {
	"*" : (process, "read JPEG to npy", "(input.jpg output.zip | [original/ npy/])"),
}

def run(argv):
	vstarstack.usage.run(argv, "readimage jpeg", commands, autohelp=False)
