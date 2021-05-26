import cfg
import sys
import common
import projection
import math
import os
import numpy as np
import multiprocessing as mp

ncpu = 11

def devig(name, fname, outpath, proj):
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
	np.savez_compressed(os.path.join(outpath, name + ".npz"), img)


def run(argv):
	proj = projection.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], cfg.camerad["w"], cfg.camerad["h"])
	path = argv[0]
	outpath = argv[1]
	files = common.listfiles(path, ".npz")
	pool = mp.Pool(ncpu)
	pool.starmap(devig, [(name, fname, outpath, proj) for name, fname in files])
	pool.close()
		
if __name__ == "__main__":
	run(sys.argv[1:])

