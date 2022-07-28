import planets.projection
import cfg
import usage
import common
import os

def process_file(filename, descfilename):
    pass

def process_path(npys, descs):
	files = common.listfiles(npys, ".zip")
	for name, filename  in files:
		print(name)
		out = os.path.join(descs, name + ".json")
		process_file(filename, out)

def process(argv):
	if len(argv) > 0:
		input = argv[0]
		output = argv[1]
		if os.path.isdir(input):
			process_path(input, output)
		else:
			process_file(input, output)
	else:
		process_path(cfg.config["planets"]["paths"]["cutted"], cfg.config["planets"]["paths"]["maps"])


commands = {
	"*" : (process, "detect compact objects", "cutted/ maps/"),
}

def run(argv):
	usage.run(argv, "planets buildmap", commands)
