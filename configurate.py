import usage
import os
import shutil
import json

def list_cfg(argv):
	pass

def configurate(argv):
	dir = argv[0]
	name = argv[1]
	if not os.path.isdir(dir):
		os.mkdir(dir)
	cfgdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
	with open(os.path.join(cfgdir, "config.json")) as f:
		config = json.load(f)
	config["telescope"] = name + ".json"
	with open(dir + "/config.json", "w") as f:
		json.dump(config, f, indent=4, ensure_ascii=False)
	tname = os.path.join(cfgdir, config["telescope"])
	shutil.copy(tname, os.path.join(dir, config["telescope"]))
	

commands = {
        "list" : (list_cfg, "list available config names", ""),
	"*" : (configurate, "copy config to working directory", "save_config_path/ telescope_name"),
}

def run(argv):
	usage.run(argv, "configurate", commands)

