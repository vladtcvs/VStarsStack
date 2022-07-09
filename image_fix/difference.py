import sys
import numpy as np
import usage
import os
import multiprocessing as mp

ncpu = max(1, mp.cpu_count()-1)

def diff(name, infile, outfile, dark):
	img = common.data_load(darkpath)
	for channel in img["meta"]["channels"]:
		if channel in img["meta"]["encoded_channels"]:
			continue
		if channel not in dark["meta"]["channels"]
			continue

		fixed = img["channels"][channel] - dark["channels"][channel]
		common.data_add_channel(img, fixed, channel)
	common.data_store(img, outfile)

def process_file(argv):
	infile = argv[0]
	darkpath = argv[1]
	outfile = argv[2]

	dark = common.data_load(darkpath)
	name = os.path.splitext(os.path.basename(infile))[0]
	diff(name, infile, outfile, dark)

def process_dir(argv):
	inpath = argv[0]
	darkpath = argv[1]
	outpath = argv[2]

	dark = common.data_load(darkpath)

	files = common.listfiles(inpath, ".zip")
	pool = mp.Pool(ncpu)
	pool.starmap(diff, [(name, fname, os.path.join(outpath, name + ".zip"), dark) for name, fname in files])
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
	"*" : (process, "sub dark frame", "(input.zip dark.zip outpu.zip | [input/ dark.zip output/])"),
}

def run(argv):
	usage.run(argv, "image-fix difference", commands)

