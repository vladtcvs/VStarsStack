import usage
import os
import json

def list_cfg(argv):
	pass

def dircheck(name):
	if not os.path.isdir(name):
		os.mkdir(name)

def configurate(argv):
	dir = argv[0]
	if len(argv) > 1:
		name = argv[1]
	else:
		name = "default"

	# create project directory
	dircheck(dir)

	# directory for original images (NEF, png, jpg, etc)
	dircheck(dir + "/orig")

	# directory for original images in NPZ format
	dircheck(dir + "/npy-orig")

	# directory for images after pre-processing (remove darks, sky, vignetting, distorsion, etc)
	dircheck(dir + "/npy")

	# directory for images after moving
	dircheck(dir + "/shifted")

	config = {
		"use_sphere" : True,
		"compress" : True,
		"paths" : {
			"original"  : "orig",
			"npy-orig"  : "npy-orig",
			"npy-fixed" : "npy",
			"shifted"   : "shifted",
			"shifts"    : "shifts.json",
			"output"    : "sum.zip",
		}
	}

	cfgdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
	telescope_fname = cfgdir + "/" + name + ".json"
	with open(os.path.join(cfgdir, telescope_fname)) as f:
		config["telescope"] = json.load(f)

	with open(dir + "/project.json", "w") as f:
		json.dump(config, f, indent=4, ensure_ascii=False)


commands = {
        "list" : (list_cfg, "list available telescope names", ""),
	"*" : (configurate, "create project", "project_dir telescope_name"),
}

def run(argv):
	usage.run(argv, "project", commands)

