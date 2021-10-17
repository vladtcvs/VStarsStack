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

def fix(img, proj):
	if cfg.distorsion is None:
		return img

	h = img.shape[0]
	w = img.shape[1]
	nch = img.shape[2] - 1

	fixed = np.zeros((h, w, nch+1))
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
			res, pixel = common.getpixel(img, fy, fx, False)
			fixed[y][x][0:nch] = pixel[0:nch]
			if res:
				fixed[y][x][nch] = 1

	return fixed

def dedistorsion(name, fname, outfname, proj):
	print(name)
	img = np.load(fname)["arr_0"]
	fixed = fix(img, proj)
	np.savez_compressed(outfname, fixed)

def process_file(argv):
	proj = projection.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], cfg.camerad["w"], cfg.camerad["h"])
	infname = argv[0]
	outfname = argv[1]
	name = os.path.splitext(os.path.basename(infname))[0]
	dedistorsion(name, infname, outfname, proj)

def process_dir(argv):
	proj = projection.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], cfg.camerad["w"], cfg.camerad["h"])
	inpath = argv[0]
	outpath = argv[1]
	files = common.listfiles(inpath, ".npz")
	pool = mp.Pool(ncpu)
	pool.starmap(dedistorsion, [(name, fname, os.path.join(outpath, name + ".npz"), proj) for name, fname in files])
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
	"*" :  (process, "Remove distrosion", "(input.file output.file | [input/ output/])"),
}

def run(argv):
	usage.run(argv, "image-fix distorsion", commands)

