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
import cfg
import common
import data
import usage
import multiprocessing as mp

import readimage.tags


def readnef(filename, output):
	img = rawpy.imread(filename)
	image = img.raw_image_visible
	
	tags = readimage.tags.read_tags(filename)
	
	params = {}

	exptime = tags["shutter"]*tags["iso"]

	dataframe = data.DataFrame(params, tags)

	weight = np.ones(image.data.shape)*exptime

	dataframe.add_channel(image, "raw", encoded=True)
	dataframe.add_channel(weight, "weight")
	dataframe.add_channel_link("raw", "weight", "weight")

	dataframe.add_parameter(image.data.shape[0], "h")
	dataframe.add_parameter(image.data.shape[1], "w")
	dataframe.add_parameter("perspective", "projection")
	dataframe.add_parameter(cfg.camerad["H"] / cfg.camerad["h"], "perspective_kh")
	dataframe.add_parameter(cfg.camerad["W"] / cfg.camerad["w"], "perspective_kw")
	dataframe.add_parameter(cfg.scope["F"], "perspective_F")
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
	files = common.listfiles(input, ".nef")
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
		process_path([cfg.config["paths"]["original"], cfg.config["paths"]["npy-orig"]])

commands = {
	"*" : (process, "read NEF to npy", "(input.NEF output.zip| [original/ npy/])"),
}

def run(argv):
	usage.run(argv, "readimage nef", commands, autohelp=False)

