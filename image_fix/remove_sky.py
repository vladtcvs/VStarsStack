import multiprocessing as mp

import sky_model.gradient
import sky_model.gauss
import sky_model.isoline
import sky_model.quadratic

import usage
import os
import common
import cfg
import data

def remove_sky(name, infname, outfname, model):
	print(name)

	img = data.DataFrame.load(infname)

	for channel in img.get_channels():
		
		image, opts = img.get_channel(channel)
		if not opts["brightness"]:
			continue

		print("\t%s" % channel)
		sky = model(image)

		result = image - sky
		img.add_channel(result, channel, **opts)

	img.store(outfname)

def process_file(argv, model):
	infname = argv[0]
	outfname = argv[1]
	name = os.path.splitext(os.path.basename(infname))[0]
	remove_sky(name, infname, outfname, model)

def process_dir(argv, model):
	inpath = argv[0]
	outpath = argv[1]
	files = common.listfiles(inpath, ".zip")
	pool = mp.Pool(cfg.nthreads)
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
	"isoline"   : (lambda argv : process(argv, sky_model.isoline.model),  "use isoline model"),
	"gauss"     : (lambda argv : process(argv, sky_model.gauss.model),    "use gauss blur model"),
	"gradient"  : (lambda argv : process(argv, sky_model.gradient.model), "use gradient model"),
	"quadratic" : (lambda argv : process(argv, sky_model.quadratic.model), "use quadratic gradient model"),
}

def run(argv):
	usage.run(argv, "image-fix remove-sky", commands, autohelp=True)
