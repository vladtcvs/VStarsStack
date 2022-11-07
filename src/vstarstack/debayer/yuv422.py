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

import vstarstack.usage
import os
import vstarstack.common
import vstarstack.cfg
import vstarstack.data

import numpy as np

def indexes(h, w, dy, dx, sy, sx):
	return range(sy,h+sy,dy), range(sx,w+sx,dx)

def yuv_422_split(frame):
	frame = frame.astype(np.float32)
	h = frame.shape[0]
	w = frame.shape[1]
	
	Y1y, Y1x = indexes(h, w, 2, 2, 0, 0)
	Y1 = frame[:, Y1x]
	Y1 = Y1[Y1y,:]
	
	Y2y, Y2x = indexes(h,w, 2, 2, 1,0)
	Y2 = frame[:, Y2x]
	Y2 = Y2[Y2y,:]
	
	Cb1y, Cb1x = indexes(h, w, 2,4, 0, 1)
	Cb1 = frame[:, Cb1x]
	Cb1 = Cb1[Cb1y,:]
	
	Cb2y, Cb2x = indexes(h, w, 2,4, 1, 1)
	Cb2 = frame[:, Cb2x]
	Cb2 = Cb2[Cb2y,:]
	
	Cr1y, Cr1x = indexes(h, w, 2,4, 0, 3)
	Cr1 = frame[:, Cr1x]
	Cr1 = Cr1[Cr1y,:]
	
	Cr2y, Cr2x = indexes(h, w, 2,4, 1, 3)
	Cr2 = frame[:, Cr2x]
	Cr2 = Cr2[Cr2y,:]

	Y = np.concatenate((Y1, Y2), axis=1)
	Cb = np.concatenate((Cb1, Cb2), axis=1)
	Cr = np.concatenate((Cr1, Cr2), axis=1)

	Cb = np.repeat(Cb, axis=1, repeats=2)
	Cr = np.repeat(Cr, axis=1, repeats=2)

	Cb = (Cb.astype(np.float32) - 128) / Y
	Cr = (Cr.astype(np.float32) - 128) / Y

	return Y, Cb, Cr
	
def process_file(fname, output):
	dataframe = vstarstack.data.DataFrame.load(fname)

	raw,_ = dataframe.get_channel("raw")
	Y, Cb, Cr = yuv_422_split(raw)

	R = Y * (1 + 1.403 * Cr)
	G = Y * (1 - 0.714 * Cr - 0.344 * Cb)
	B = Y * (1 + 1.773 * Cb)

	dataframe.add_channel(Y, "Y", brightness=True)
	dataframe.add_channel(Cb, "Cb")
	dataframe.add_channel(Cr, "Cr")

	dataframe.add_channel(R, "R", brightness=True)
	dataframe.add_channel(G, "G", brightness=True)
	dataframe.add_channel(B, "B", brightness=True)

	dataframe.add_parameter(Y.shape[0], "h")
	dataframe.add_parameter(Y.shape[1], "w")

	dataframe.store(output)

def process_path(input, output):
	files = vstarstack.common.listfiles(input, ".zip")
	for name, fname in files:
		print(name)
		process_file(fname, os.path.join(output, name + ".zip"))

def process(argv):
	if len(argv) > 0:
		input = argv[0]
		output = argv[1]
		if os.path.isdir(input):
			process_path(input, output)
		else:
			process_file(input, output)
	else:
		process_path(vstarstack.cfg.config["paths"]["npy-orig"],
						vstarstack.cfg.config["paths"]["npy-fixed"])

commands = {
	"*" : (process, "Consider RAW as YUV with 422 subsampling", "(npy-input.zip npy-output.zip | [input/ output/])"),
}

def run(argv):
	vstarstack.usage.run(argv, "debayer yuv422", commands, autohelp=False)
