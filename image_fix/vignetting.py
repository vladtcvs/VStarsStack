import cfg
import sys
import common
import projection
import math
import os
import numpy as np
import multiprocessing as mp
import usage
import numba

ncpu = max(1, mp.cpu_count()-1)

@numba.njit
def mul(img, vig):
	nch = img.shape[2]-1
	for i in range(nch):
		img[:,:,i] *= vig
	return img

def devig(name, fname, out):
	print(name)
	img = np.load(fname)["arr_0"].astype(np.float64)
	vig = cfg.vignetting
	img = mul(img, vig)
	np.savez_compressed(out, img)

def process_file(argv):
	infname = argv[0]
	outfname = argv[1]
	name = os.path.splitext(os.path.basename(infname))[0]
	devig(name, infname, outfname)

def process_dir(argv):
	inpath = argv[0]
	outpath = argv[1]
	files = common.listfiles(inpath, ".npz")
	pool = mp.Pool(ncpu)
	pool.starmap(devig, [(name, fname, os.path.join(outpath, name + ".npz")) for name, fname in files])
	pool.close()

def process(argv):
	if os.path.isdir(argv[0]):
		process_dir(argv)
	else:
		process_file(argv)

commands = {
	"*" : (process, "devignetting", "(input.file output.file | input/ output/)"),
}

def run(argv):
	usage.run(argv, "image-fix vignetting", commands)

