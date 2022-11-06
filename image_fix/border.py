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

import cfg
import sys
import numpy as np
import usage
import os
import multiprocessing as mp
import common
import data

bw = 60
ncpu = max(1, mp.cpu_count()-1)

def border(name, fname, outname, bw_left, bw_top, bw_right, bw_bottom):
	print(name)

	img = data.DataFrame.load(fname)

	for channel in img.get_channels():
		image,opts = img.get_channel(channel)
		if opts["encoded"]:
			continue
		if opts["weight"]:
			continue

		w_channel = img.links["weight"][channel]
		weight,_ = img.get_channel(w_channel)

		w = image.shape[1]
		h = image.shape[0]

		image[0:bw_top,:,:] = 0
		image[(h-bw_bottom):h,:,:] = 0

		image[:, 0:bw_left,:] = 0
		image[:, (w-bw_right):w,:] = 0

		weight[0:bw_top,:,:] = 0
		weight[(h-bw_bottom):h,:,:] = 0

		weight[:, 0:bw_left,:] = 0
		weight[:, (w-bw_right):w,:] = 0

		img.add_channel(image, channel, **opts)
		img.add_channel(weight, w_channel, weight=True)
		img.add_channel_link(channel, w_channel, "weight")

	img.store(outname)

def process_file(argv):
	infile = argv[0]
	outfile = argv[1]
	bbw = argv[2:]
	if len(bbw) >= 4:
		brd_left = int(bbw[0])
		brd_top = int(bbw[1])
		brd_right = int(bbw[2])
		brd_bottom = int(bbw[3])
	elif len(bbw) > 0:
		brd_left = int(bbw[0])
		brd_top = int(bbw[0])
		brd_right = int(bbw[0])
		brd_bottom = int(bbw[0])
	else:
		brd_left = bw
		brd_top = bw
		brd_right = bw
		brd_bottom = bw

	name = os.path.splitext(os.path.basename(infile))[0]

	border(name, infile, outfile, brd_left, brd_top, brd_right, brd_bottom)

def process_dir(argv):
	inpath = argv[0]
	outpath = argv[1]
	bbw = argv[2:]
	if len(bbw) >= 4:
		brd_left = int(bbw[0])
		brd_top = int(bbw[1])
		brd_right = int(bbw[2])
		brd_bottom = int(bbw[3])
	elif len(bbw) > 0:
		brd_left = int(bbw[0])
		brd_top = int(bbw[0])
		brd_right = int(bbw[0])
		brd_bottom = int(bbw[0])
	else:
		brd_left = bw
		brd_top = bw
		brd_right = bw
		brd_bottom = bw

	files = common.listfiles(inpath, ".zip")
	pool = mp.Pool(ncpu)
	pool.starmap(border, [(name, fname, os.path.join(outpath, name + ".zip"), brd_left, brd_top, brd_right, brd_bottom) for name, fname in files])
	pool.close()

def process(argv):
	if len(argv) > 0:
		if os.path.isdir(argv[0]):
			process_dir(argv)
		else:
			process_file(argv)
	else:
		process_dir([cfg.config["paths"]["npy-fixed"], cfg.config["paths"]["npy-fixed"]])

commands = {
	"*" : (process, "remove border", "(input.zip output.zip | [input/ output/])"),
}

def run(argv):
	usage.run(argv, "image-fix difference", commands)

