import sys
import numpy as np
import usage
import os
import multiprocessing as mp

ncpu = max(1, mp.cpu_count()-1)

def diff(img, back):
	img = np.load(argv[0])["arr_0"]
	nch = img.shape[2]-1
	img[:,:,0:nch] -= back
	np.savez_compressed(argv[2], img)

def process_file(argv):
	infile = argv[0]
	darkpath = argv[1]
	outfile = argv[2]

	name = os.path.splitext(os.path.basename(infile))[0]
	dark = np.load(darkpath)["arr_0"]

	diff(name, infile, outfile, dark)

def process_dir(argv):
	inpath = argv[0]
	darkpath = argv[1]
	outpath = argv[2]

	dark = np.load(darkpath)["arr_0"]
	files = common.listfiles(inpath, ".npz")
	pool = mp.Pool(ncpu)
	pool.starmap(diff, [(name, fname, os.path.join(outpath, name + ".npz"), dark) for name, fname in files])
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
	"*" : (process, "sub dark frame", "(input.npz dark.npz outpu.npz | [input/ dark.npz output/])"),
}

def run(argv):
	usage.run(argv, "image-fix difference", commands)

