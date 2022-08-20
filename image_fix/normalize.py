import usage
import os
import common
import data
import cfg

def normalize(name, infname, outfname):
    print(name)
    img = common.data_load(infname)

    if "weight" not in img["meta"]["channels"]:
        data.data_store(img, outfname)
        return

    weight = img["channels"]["weight"]

    for channel in img["meta"]["channels"]:
        if channel in img["meta"]["encoded_channels"]:
            continue
        if channel == "weight":
            continue
        
        image = img["channels"][channel]
        image = image / weight
        img["channels"][channel] = image

    data.data_add_parameter(img, 1, "normalized")
    data.data_store(img, outfname)

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
