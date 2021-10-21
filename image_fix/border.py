import sys
import numpy as np
import usage
import os
import multiprocessing as mp
import common

bw = 60
ncpu = max(1, mp.cpu_count()-1)

def diff(name, fname, outname, bw_left, bw_top, bw_right, bw_bottom):
	print(name)
	img = np.load(fname)["arr_0"]
	nch = img.shape[2]-1
	w = img.shape[1]
	h = img.shape[0]
	img[0:bw_top,:,:] = 0
	img[(h-bw_bottom):h,:,:] = 0

	img[:, 0:bw_left,:] = 0
	img[:, (w-bw_right):w,:] = 0

	np.savez_compressed(outname, img)

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

	diff(name, infile, outfile, brd_left, brd_top, brd_right, brd_bottom)

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

	files = common.listfiles(inpath, ".npz")
	pool = mp.Pool(ncpu)
	pool.starmap(diff, [(name, fname, os.path.join(outpath, name + ".npz"), brd_left, brd_top, brd_right, brd_bottom) for name, fname in files])
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
	"*" : (process, "remove border", "(input.npz output.npz | [input/ output/])"),
}

def run(argv):
	usage.run(argv, "image-fix difference", commands)

