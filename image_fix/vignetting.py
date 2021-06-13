import cfg
import sys
import common
import projection
import math
import os
import numpy as np
import multiprocessing as mp
import usage

ncpu = max(1, mp.cpu_count()-1)

def devig(name, fname, out, proj):
	print(name)
	img = np.load(fname)["arr_0"].astype(np.uint32)
	nch = img.shape[2]-1
	vig = cfg.vignetting
	for i in nch:
		img[:,:,i] *= vig
	np.savez_compressed(out, img)

def process_file(argv):
	infname = argv[0]
	outfname = argv[1]
	name = os.path.splitext(os.path.basename(infname))[0]
	proj = projection.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], cfg.camerad["w"], cfg.camerad["h"])
	devig(name, infname, outfname, proj)

def process_dir(argv):
	inpath = argv[0]
	outpath = argv[1]
	proj = projection.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], cfg.camerad["w"], cfg.camerad["h"])
	files = common.listfiles(inpath, ".npz")
	pool = mp.Pool(ncpu)
	pool.starmap(devig, [(name, fname, os.path.join(outpath, name + ".npz"), proj) for name, fname in files])
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

