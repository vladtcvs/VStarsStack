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

import usage
import os
import common
import data
import numpy as np
import cfg

mask = np.array([
			[[0, 0], [1, 0]], # red
			[[1, 0], [0, 1]], # green
			[[0, 1], [0, 0]], # blue
		])

def getcolor(img, mask):
	return np.sum(img*mask)

def debayer_process(image, weight):
	h = image.shape[0]
	w = image.shape[1]
	
	cshape = (int(h/2), int(w/2))
	R = np.zeros(cshape)
	G = np.zeros(cshape)
	B = np.zeros(cshape)

	wR = np.zeros(cshape)
	wG = np.zeros(cshape)
	wB = np.zeros(cshape)

	for y in range(int(h/2)):
		for x in range(int(w/2)):
			cut = image[2*y:2*y+2, 2*x:2*x+2]
			wcut = weight[2*y:2*y+2, 2*x:2*x+2]
			R[y][x] = getcolor(cut, mask[0])
			G[y][x] = getcolor(cut, mask[1])
			B[y][x] = getcolor(cut, mask[2])

			wR[y][x] = getcolor(wcut, mask[0])
			wG[y][x] = getcolor(wcut, mask[1])
			wB[y][x] = getcolor(wcut, mask[2])

	return R, G, B, wR, wG, wB
	
def process_file(argv):
	fname = argv[0]
	output = argv[1]

	dataframe = data.DataFrame.load(fname)

	raw,_ = dataframe.get_channel("raw")
	weight,_ = dataframe.get_channel(dataframe.links["weight"]["raw"])

	R, G, B, wR, wG, wB = debayer_process(raw, weight)
	dataframe.add_channel(R, "R", brightness=True)
	dataframe.add_channel(G, "G", brightness=True)
	dataframe.add_channel(B, "B", brightness=True)

	dataframe.add_channel(wR, "weight-R", weight=True)
	dataframe.add_channel(wG, "weight-G", weight=True)
	dataframe.add_channel(wB, "weight-B", weight=True)

	dataframe.add_channel_link("R", "weight-R", "weight")
	dataframe.add_channel_link("G", "weight-G", "weight")
	dataframe.add_channel_link("B", "weight-B", "weight")

	dataframe.add_parameter(R.shape[0], "h")
	dataframe.add_parameter(R.shape[1], "w")
	if dataframe.params["projection"] == "perspective":
		pixh = dataframe.params["perspective_kh"]*2
		pixw = dataframe.params["perspective_kw"]*2
		dataframe.add_parameter(pixh, "perspective_kh")
		dataframe.add_parameter(pixw, "perspective_kw")
	else:
		print("Unsupported projection: %s" % dataframe.params["projection"])

	dataframe.store(output)

def process_path(argv):
	print(argv)
	input = argv[0]
	output = argv[1]

	files = common.listfiles(input, ".zip")
	for name, fname in files:
		print(name)
		process_file((fname, os.path.join(output, name + ".zip")))

def process(argv):
	if len(argv) > 0:
		input = argv[0]
		output = argv[1]
		if os.path.isdir(input):
			process_path((input, output))
		else:
			process_file((input, output))
	else:
		process_path([cfg.config["paths"]["npy-orig"], cfg.config["paths"]["npy-fixed"]])

commands = {
	"*" : (process, "Consider RAW as bayered image", "(npy-input.zip npy-output.zip | [input/ output/])"),
}

def run(argv):
	usage.run(argv, "debayer bayer", commands, autohelp=False)
