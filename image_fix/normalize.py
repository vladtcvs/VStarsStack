import usage
import os
import common
import data
import cfg
import numpy as np

import multiprocessing as mp
ncpu = max(int(mp.cpu_count())-1, 1)

def normalize(name, infname, outfname):
	print(name)
	img = data.DataFrame.load(infname)

	for channel in img.get_channels():
		image,opts = img.get_channel(channel)
		if "normalized" in opts and opts["normalized"]:
			continue
		if opts["weight"]:
			continue
		if opts["encoded"]:
			continue
		if channel not in img.links["weight"]:
			continue
		weight,_ = img.get_channel(img.links["weight"][channel])
		image = image / weight
		image[np.where(weight == 0)] = 0
		opts["normalized"] = True
		img.add_channel(image, channel, **opts)

	img.store(outfname)

def process_file(argv):
	infname = argv[0]
	outfname = argv[1]
	name = os.path.splitext(os.path.basename(infname))[0]
	normalize(name, infname, outfname)

def process_dir(argv):
	inpath = argv[0]
	outpath = argv[1]
	files = common.listfiles(inpath, ".zip")
	pool = mp.Pool(ncpu)
	pool.starmap(normalize, [(name, fname, os.path.join(outpath, name + ".zip")) for name, fname in files])
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
	"*" : (process, "normalize to weight"),
}

def run(argv):
	usage.run(argv, "image-fix normalize", commands, autohelp=True)
