import usage
import os
import common

import numpy as np

mask = np.array([
			[[0, 0], [2, 0]], # red
			[[1, 0], [0, 1]], # green
			[[0, 2], [0, 0]], # blue
		])

def getcolor(img, mask):
	return np.sum(img*mask)

def debayer_process(frame):
	h = frame.shape[0]
	w = frame.shape[1]
	
	chape = (int(h/2), int(w/2))
	R = np.zeros(cshape)
	G = np.zeros(cshape)
	B = np.zeros(cshape)

	for y in range(int(h/2)):
		for x in range(int(w/2)):
			cut = image[2*y:2*y+2, 2*x:2*x+2]
			R[y][x] = getcolor(cut, mask[0])
			G[y][x] = getcolor(cut, mask[1])
			B[y][x] = getcolor(cut, mask[2])

	return R, G, B
	
def process_file(argv):
	fname = argv[0]
	output = argv[1]

	data = common.data_load(fname)

	R, G, B = debayer_process(data["channels"]["raw"])
	common.data_add_channel(data, R, "R")
	common.data_add_channel(data, G, "G")
	common.data_add_channel(data, B, "B")

	common.data_store(data, output)

def process_path(argv):
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
