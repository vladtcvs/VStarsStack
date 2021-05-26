import cfg
import sys
import common
import projection
import math
import os
import numpy as np
import multiprocessing as mp
import usage

ncpu = 11

def devig(name, fname, out, proj):
	print(name)
	img = np.load(fname)["arr_0"].astype(np.uint32)
	fun = cfg.vignetting["function"]
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			lat, lon = proj.project(y, x)
			cosa = math.cos(lon)*math.cos(lat)
			v = fun(cosa)
			img[y][x][0] /= v
			img[y][x][1] /= v
			img[y][x][2] /= v
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


commands = {
	"file" : (process_file, "process single file", "input.file output.file"),
	"path" : (process_dir,  "process all files in dir", "input_path/ output_path/"),
}

def run(argv):
	usage.run(argv, "process image-fix vignetting", commands)

