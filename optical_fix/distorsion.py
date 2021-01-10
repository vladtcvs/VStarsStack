import cfg
import sys
import common
import projection
import math
import os
import numpy as np
import multiprocessing as mp

ncpu = 10

def fix(img, proj):
	h = img.shape[0]
	w = img.shape[1]
	fixed = np.zeros((h, w, 4))
	a = cfg.distorsion["a"]
	b = cfg.distorsion["b"]
	c = cfg.distorsion["c"]

	v0 = np.zeros((3,))
	v0[0] = 1
	v0[1] = 0
	v0[2] = 0

	for y in range(h):
		for x in range(w):
			fixedlat, fixedlon = proj.project(y, x)
			fixedangle = math.acos(math.cos(fixedlat)*math.cos(fixedlon))
			k = a * fixedangle**2 + b * fixedangle + c
			angle = fixedangle * k
			if angle < 1e-12:
				lat = fixedlat
				lon = fixedlon
			else:
				p = np.empty((3,))
				p[0] = 0
				p[1] = math.cos(fixedlat) * math.sin(fixedlon)
				p[2] = math.sin(fixedlat)
				p /= (p[1]**2+p[2]**2)**0.5
				v = v0 * math.cos(angle) + p * math.sin(angle)
				lat = math.asin(v[2])
				lon = math.atan2(v[1], v[0])
			fy, fx = proj.reverse(lat, lon)
			res, pixel = common.getpixel(img, fy, fx)

			if img.shape[2] == 3:
				fixed[y][x][0:3] = pixel
				if res:
					fixed[y][x][3] = 1
			else:
				fixed[y][x] = pixel
	return fixed

def dedistorsion(name, fname, outpath, proj):
	print(name)
	img = np.load(fname)["arr_0"]
	fixed = fix(img, proj)
	np.savez_compressed(os.path.join(outpath, name + ".npz"), fixed)


def run(argv):
	proj = projection.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], cfg.camerad["w"], cfg.camerad["h"])
	path = argv[0]
	outpath = argv[1]
	files = common.listfiles(path, ".npz")
	pool = mp.Pool(ncpu)
	pool.starmap(dedistorsion, [(name, fname, outpath, proj) for name, fname in files])
	pool.close()
		
if __name__ == "__main__":
	run(sys.argv[1:])

