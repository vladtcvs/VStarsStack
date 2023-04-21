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
import rawpy
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import json
import vstarstack.cfg
import vstarstack.common
import vstarstack.data
import vstarstack.usage
import multiprocessing as mp

import vstarstack.readimage.tags


def readnef(filename, output):
	img = rawpy.imread(filename)
	image = img.raw_image_visible
	
	tags = vstarstack.readimage.tags.read_tags(filename)
	
	params = {
		"w" : image.data.shape[1],
		"h" : image.data.shape[0],
		"projection" : "perspective",
		"perspective_F" : vstarstack.cfg.scope.F,
		"perspective_kh" : vstarstack.cfg.camera.kh,
		"perspective_kw" : vstarstack.cfg.camera.kw,
		"format" : vstarstack.cfg.camera.format,
	}

	exptime = tags["shutter"]*tags["iso"]

	dataframe = vstarstack.data.DataFrame(params, tags)

	weight = np.ones(image.data.shape)*exptime

	dataframe.add_channel(image, "raw", encoded=True)
	dataframe.add_channel(weight, "weight")
	dataframe.add_channel_link("raw", "weight", "weight")

	dataframe.store(output)

def work(input, output):
	print(input)
	readnef(input, output)

def process_file(argv):
	input = argv[0]
	output = argv[1]
	work(input, output)

def process_path(argv):
	input = argv[0]
	output = argv[1]
	files = vstarstack.common.listfiles(input, ".nef")
	ncpu = max(int(mp.cpu_count())-1, 1)
	pool = mp.Pool(ncpu)
	pool.starmap(work, [(filename, os.path.join(output, name + ".zip")) for name, filename in files])
	pool.close()

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
	"*" : (process, "read NEF to npy", "(input.NEF output.zip| [original/ npy/])"),
}

def run(argv):
	vstarstack.usage.run(argv, "readimage nef", commands, autohelp=False)
