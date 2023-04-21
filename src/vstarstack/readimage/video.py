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

import cv2
import zipfile

def read_video(fname):
	vidcap = cv2.VideoCapture(fname)
	id = 0

	#vidcap.set(cv2.CAP_PROP_FORMAT, -1)
	while True:
		success, frame = vidcap.read()
		if not success:
			break	

		tags = {
			"depth" : 8,
		}

		params = {
			"w" : frame.shape[1],
			"h" : frame.shape[0],
			"projection" : "perspective",
			"perspective_F" : vstarstack.cfg.scope.F,
			"perspective_kh" : vstarstack.cfg.camera.kh,
			"perspective_kw" : vstarstack.cfg.camera.kw,
			"format" : vstarstack.cfg.camera.format,
		}

		print("\tprocessing frame %i" % id)

		exptime = 1
		weight = np.ones((frame.shape[0], frame.shape[1]))*exptime

		dataframe = vstarstack.data.DataFrame(params, tags)
		dataframe.add_channel(frame[:,:,0], "R")
		dataframe.add_channel(frame[:,:,1], "G")
		dataframe.add_channel(frame[:,:,2], "B")
		dataframe.add_channel(weight, "weight")
		dataframe.add_channel_link("R", "weight", "weight")
		dataframe.add_channel_link("G", "weight", "weight")
		dataframe.add_channel_link("B", "weight", "weight")
		yield id, dataframe
		id += 1
	
def process_file(argv):
	fname = argv[0]
	output = argv[1]
	name = argv[2]

	for i, dataframe in read_video(fname):
		framename = os.path.join(output, "%s_%05i.zip" % (name, i))
		dataframe.store(framename)

def process_path(argv):
	input = argv[0]
	output = argv[1]

	files = vstarstack.common.listfiles(input)
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
	"*" : (process, "read Video to npy", "(input.video output/ | [original/ npy/])"),
}

def run(argv):
	vstarstack.usage.run(argv, "readimage video", commands, autohelp=False)
