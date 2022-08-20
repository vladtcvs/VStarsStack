import multiprocessing as mp

import sky_model.gradient
import sky_model.gauss
import sky_model.isoline

import usage
import os
import common
import stars
import cfg

def remove_sky(name, infname, outfname, model):
	print(name)

	img = common.data_load(infname)
	mask = stars.detect.detect(img)[1]

	for channel in img["meta"]["channels"]:
		if channel in img["meta"]["encoded_channels"]:
			continue

		if channel == "weight":
			continue

		print("\t%s" % channel)
		image = img["channels"][channel]
		sky = model(image, mask)

		result = image - sky
		common.data_add_channel(img, result, channel)
		
	common.data_store(img, outfname)

def process_file(argv, model):
	infname = argv[0]
	outfname = argv[1]
	name = os.path.splitext(os.path.basename(infname))[0]
	remove_sky(name, infname, outfname, model)

def process_dir(argv, model):
	inpath = argv[0]
	outpath = argv[1]
	files = common.listfiles(inpath, ".zip")
	pool = mp.Pool(ncpu)
	pool.starmap(remove_sky, [(name, fname, os.path.join(outpath, name + ".zip"), model) for name, fname in files])
	pool.close()

def process(argv, model):
	if len(argv) > 0:
		if os.path.isdir(argv[0]):
			process_dir(argv, model)
		else:
			process_file(argv, model)
	else:
		process_dir([cfg.config["paths"]["npy-fixed"], cfg.config["paths"]["npy-fixed"]], model)

commands = {
	"isoline"  : (lambda argv : process(argv, sky_model.isoline.model),  "use isoline model"),
	"gauss"    : (lambda argv : process(argv, sky_model.gauss.model),    "use gauss blur model"),
	"gradient" : (lambda argv : process(argv, sky_model.gradient.model), "use gradient model"),
}

def run(argv):
	usage.run(argv, "image-fix remove-sky", commands, autohelp=True)
