import sys
import numpy as np
import usage
import os
import multiprocessing as mp
import common

bw = 50
ncpu = max(1, mp.cpu_count()-1)

def diff(name, fname, outname, bw):
	print(name)
	img = np.load(fname)["arr_0"]
	nch = img.shape[2]-1
	w = img.shape[1]
	h = img.shape[0]
	img[0:bw,:,:] = 0
	img[(h-bw):h,:,:] = 0

	img[:, 0:bw,:] = 0
	img[:, (w-bw):w,:] = 0

	np.savez_compressed(outname, img)

def process_file(argv):
	infile = argv[0]
	outfile = argv[1]

	name = os.path.splitext(os.path.basename(infile))[0]

	diff(name, infile, outfile, bw)

def process_dir(argv):
	inpath = argv[0]
	outpath = argv[1]

	files = common.listfiles(inpath, ".npz")
	pool = mp.Pool(ncpu)
	pool.starmap(diff, [(name, fname, os.path.join(outpath, name + ".npz"), bw) for name, fname in files])
	pool.close()

def process(argv):
	if os.path.isdir(argv[0]):
		process_dir(argv)
	else:
		process_file(argv)

commands = {
	"*" : (process, "remove border", "(input.npz output.npz | input/ output/)"),
}

def run(argv):
	usage.run(argv, "image-fix difference", commands)

